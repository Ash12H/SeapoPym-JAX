# SEAPOPYM-Message

**Distributed Population Dynamics Simulation using Ray and JAX**

A modern, scalable framework for simulating spatially-explicit population dynamics with event-driven architecture and composable computational kernels. Currently implements a 2-compartment zooplankton model with age-structured production and temperature-dependent processes.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)

---

## 🎯 Key Features

- **Event-Driven Architecture**: Asynchronous scheduler coordinating distributed workers
- **Composable Kernels**: Define models as ordered lists of modular units
- **Distributed Computing**: Ray-based workers with domain decomposition
- **High Performance**: JAX JIT compilation for numerical kernels
- **2D Spatial Grids**: Spherical (lat/lon) and plane (Cartesian) geometries
- **Centralized Transport**: Dedicated TransportWorker with mass-conservative schemes
- **Environmental Forcing**: Time-varying forcings via ForcingManager (xarray-based)
- **Scientific Rigor**: Type-checked, tested, and validated against SEAPODYM-LMTL

---

## 🏗️ Architecture Overview

```
EventScheduler
    ↓
┌─────────────────────────────────────┐
│  LOCAL PHASE (parallel on workers) │
│    CellWorker2D.step()              │
│    └─ Kernel.execute_local_phase()  │
│       └─ [biology units]            │
└─────────────────────────────────────┘
    ↓ [synchronization]
┌─────────────────────────────────────┐
│  TRANSPORT PHASE (centralized)      │
│    TransportWorker.transport_step() │
│    └─ Advection + Diffusion         │
└─────────────────────────────────────┘
    ↓
[redistribute fields to workers]
```

### Core Concepts

- **Unit**: Elementary computation (`@unit` decorator, local or global scope)
- **Kernel**: Ordered list of Units executed per timestep
- **Worker**: Ray actor managing a spatial patch with local state
- **Scheduler**: Event-driven orchestrator (no global time loop)
- **TransportWorker**: Centralized worker for physical transport (mass-conservative)
- **ForcingManager**: Manages time-varying environmental forcings (temperature, NPP, currents)
- **GridInfo**: Coordinate system information (SphericalGridInfo, PlaneGridInfo)

---

## 📦 Installation

### Using UV (recommended)

```bash
# Install UV if not already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/yourusername/seapopym-message.git
cd seapopym-message

# Create virtual environment and install
uv venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Optional Dependencies

```bash
# GPU support (CUDA 12)
uv pip install -e ".[gpu]"

# Visualization tools
uv pip install -e ".[viz]"

# All extras
uv pip install -e ".[all]"
```

---

## 🚀 Quick Start

### Example: Zooplankton Simulation

The framework currently implements a 2-compartment zooplankton model:
- **Adult biomass** B(x,y,t): age-independent, dB/dt = R - λB
- **Juvenile production** p(x,y,τ,t): age-structured, dp/dt + dp/dτ = -μp
- **Recruitment**: Total absorption at age τ_r(T) (temperature-dependent)
- **Transport**: Both compartments are advected and diffused

```python
import jax.numpy as jnp
import ray
from seapopym_message.core.kernel import Kernel
from seapopym_message.distributed.worker import CellWorker2D
from seapopym_message.distributed.scheduler import EventScheduler
from seapopym_message.distributed.transport_config import TransportConfig, FieldConfig
from seapopym_message.transport.worker import TransportWorker
from seapopym_message.kernels.zooplankton import (
    compute_recruitment,
    age_production,
    update_biomass,
)
from seapopym_message.utils.grid import SphericalGridInfo

# Initialize Ray
ray.init()

# Define grid
grid_info = SphericalGridInfo(
    lat_min=-20.0, lat_max=20.0,
    lon_min=140.0, lon_max=220.0,
    nlat=40, nlon=80
)

# Create kernel with zooplankton units
kernel = Kernel([
    compute_recruitment,  # Must be first
    age_production,       # Must be second
    update_biomass        # Must be third
])

# Model parameters
params = {
    "lambda_0": 1/150,      # Baseline mortality [day⁻¹]
    "gamma_lambda": 0.15,   # Thermal sensitivity mortality [°C⁻¹]
    "tau_r0": 10.38,        # Max recruitment age [days]
    "gamma_tau_r": 0.11,    # Thermal sensitivity age [°C⁻¹]
    "T_ref": 0.0,           # Reference temperature [°C]
    "E": 0.1668,            # NPP transfer efficiency
    "n_ages": 11,           # Number of age classes
}

# Initial state
initial_state = {
    "biomass": jnp.ones((40, 80)) * 100.0,  # Adult biomass [kg/m²]
    "production": jnp.zeros((11, 40, 80)),  # Juvenile production by age
}

# Create workers (domain decomposition)
workers = [
    CellWorker2D.remote(
        worker_id=i,
        kernel=kernel,
        params=params,
        initial_state=initial_state,
        grid_info=grid_info,
    )
    for i in range(4)  # 4 workers for simplicity
]

# Configure transport
transport_config = TransportConfig(
    fields=[
        FieldConfig(name="biomass", dims=["Y", "X"]),           # 2D
        FieldConfig(name="production", dims=["age", "Y", "X"]), # 3D
    ]
)

# Create centralized transport worker
transport_worker = TransportWorker.remote(
    grid_type="spherical",
    lat_min=-20.0, lat_max=20.0,
    lon_min=140.0, lon_max=220.0,
    nlat=40, nlon=80,
    lat_bc="closed",
    lon_bc="periodic",
)

# Run simulation
scheduler = EventScheduler(
    workers=workers,
    dt=86400.0,  # 1 day in seconds
    t_max=86400.0 * 30,  # 30 days
    transport_worker=transport_worker,
    transport_config=transport_config,
    forcing_params={
        "horizontal_diffusivity": 1000.0,  # [m²/s]
    },
    global_nlat=40,
    global_nlon=80,
)

# Execute simulation
diagnostics = scheduler.run()

print(f"✅ Simulation completed: {len(diagnostics)} timesteps")
```

### With Environmental Forcings

```python
import xarray as xr
from seapopym_message.forcing.manager import ForcingManager

# Load forcings from NetCDF
temperature_ds = xr.open_dataset("temperature.nc")
npp_ds = xr.open_dataset("npp.nc")

# Create forcing manager
forcing_manager = ForcingManager(
    datasets={
        "temperature": temperature_ds,
        "npp": npp_ds,
    },
    derived_forcings=["tau_r", "mortality"],  # Computed from temperature
)

# Pass to scheduler
scheduler = EventScheduler(
    workers=workers,
    dt=86400.0,
    t_max=86400.0 * 365,
    forcing_manager=forcing_manager,
    forcing_params={
        "horizontal_diffusivity": 1000.0,
        "tau_r0": 10.38,
        "gamma_tau_r": 0.11,
        "lambda_0": 1/150,
        "gamma_lambda": 0.15,
        "T_ref": 0.0,
    },
    transport_worker=transport_worker,
    transport_config=transport_config,
    global_nlat=40,
    global_nlon=80,
)
```

---

## 📚 Examples

Explore interactive Jupyter notebooks in the `examples/` directory:

### Available Notebooks

1. **`zooplankton_full_simulation.ipynb`**
   - Complete zooplankton model with circular ocean currents
   - Island masking demonstration
   - Conservation diagnostics

2. **`zooplankton_full_simulation_plane.ipynb`**
   - Same model on Cartesian (plane) grid
   - Ideal for testing and validation

3. **`transport_demo_island.ipynb`**
   - Pure transport demonstration (advection + diffusion)
   - Blob of biomass flowing around an island
   - Mass conservation verification

### Running Examples

```bash
# Launch Jupyter
jupyter lab

# Navigate to examples/ and open a notebook
```

---

## 🧬 Implemented Models

### Zooplankton Model (SEAPODYM-LMTL)

A 2-compartment model with:

**Adult Biomass** (age-independent):
```
dB/dt = R - λ(T) × B
```

**Juvenile Production** (age-structured):
```
dp/dt + dp/dτ = 0  (aging)
p(τ=0) = E × NPP   (source from primary production)
R = Σ p(τ) for τ ≥ τ_r(T)  (recruitment with total absorption)
```

**Temperature Dependencies**:
- Mortality: `λ(T) = λ₀ exp(γ_λ (T - T_ref))`
- Recruitment age: `τ_r(T) = τ_r0 exp(-γ_τr (T - T_ref))`

**Transport**: Both B and p(τ) are advected by ocean currents and diffused.

**Units Execution Order** (critical):
1. `compute_recruitment(production)` → recruitment
2. `age_production(production)` → production (aged, absorbed classes zeroed)
3. `update_biomass(biomass, recruitment)` → biomass

See `src/seapopym_message/kernels/zooplankton.py` for implementation.

---

## 🗺️ Grid Architecture

The framework uses a composition-based grid system:

### GridInfo Hierarchy

```python
from seapopym_message.utils.grid import SphericalGridInfo, PlaneGridInfo

# Spherical grid (geographic coordinates)
grid_info = SphericalGridInfo(
    lat_min=-20.0, lat_max=20.0,
    lon_min=140.0, lon_max=220.0,
    nlat=40, nlon=80
)

# Plane grid (Cartesian coordinates)
grid_info = PlaneGridInfo(
    dx=10e3,  # 10 km
    dy=10e3,  # 10 km
    nlat=100, nlon=100
)
```

### Grid with Mask

```python
import xarray as xr

# Create ocean/land mask
ocean_mask = xr.DataArray(
    mask_array,  # 1.0 = ocean, 0.0 = land
    coords={"lat": lats, "lon": lons},
    dims=["lat", "lon"],
)

# Pass to TransportWorker
transport_worker = TransportWorker.remote(
    grid_type="spherical",
    lat_min=-20.0, lat_max=20.0,
    lon_min=140.0, lon_max=220.0,
    nlat=40, nlon=80,
    mask=ocean_mask,  # Grid knows its mask
)
```

**Grid Composition**:
```
Grid
 ├─ grid_info: GridInfo (coordinates, dimensions)
 ├─ mask: xr.DataArray (ocean/land boundaries)
 └─ methods: cell_areas(), face_areas(), dx(), dy()
```

---

## 🌊 Transport System

### Mass-Conservative Schemes

The TransportWorker implements:

- **Advection**: Upwind flux-based finite volume method
  - Guarantees mass conservation by construction
  - Handles land/ocean boundaries with flux masking

- **Diffusion**: Explicit Euler with centered differences
  - Stability: `dt ≤ min(dx²)/(4D)`
  - Zero-flux boundary conditions at land

### Transport Configuration

```python
from seapopym_message.distributed.transport_config import TransportConfig, FieldConfig

# Define which fields to transport
transport_config = TransportConfig(
    fields=[
        FieldConfig(name="biomass", dims=["Y", "X"]),           # 2D field
        FieldConfig(name="production", dims=["age", "Y", "X"]), # 3D field
        FieldConfig(name="temperature", dims=["depth", "Y", "X"]), # 3D bathymetric
    ]
)
```

The scheduler automatically:
1. Collects fields from all workers (global assembly)
2. Transports each 2D slice (handles 3D fields via iteration)
3. Redistributes updated fields back to workers

---

## 🧪 Development

### Run Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit

# With coverage
pytest --cov=seapopym_message --cov-report=html
```

### Code Quality

```bash
# Format code
ruff format

# Lint and auto-fix
ruff check --fix

# Type check
mypy src/seapopym_message

# Run all pre-commit hooks
pre-commit run --all-files
```

### Using Makefile

```bash
# Install dependencies
make install

# Run tests
make test

# Format and lint
make format
make lint

# Type check
make typecheck

# All quality checks
make check

# Clean build artifacts
make clean
```

---

## 📐 Architecture Decisions

### Why Centralized Transport?

**Advantages**:
- ✅ Mass conservation guaranteed (no accumulation at worker boundaries)
- ✅ Single source of truth for transport physics
- ✅ Easier to implement complex schemes (higher-order, multi-step)
- ✅ Simplifies worker code (biology only)

**Trade-offs**:
- Communication overhead (assemble global grid → redistribute)
- Single bottleneck for transport step
- Mitigated by: JAX JIT compilation, efficient Ray object store

### Event-Driven vs Loop-Based

Traditional approach:
```python
for t in range(steps):  # ❌ Global synchronization
    for worker in workers:
        worker.step()
```

SEAPOPYM-Message approach:
```python
scheduler = EventScheduler(...)  # ✅ Asynchronous coordination
diagnostics = scheduler.run()
```

**Benefits**: Workers can run at different speeds, easier to add adaptive timestepping.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Before committing:**

- Ensure tests pass: `pytest`
- Run code quality checks: `make check`
- Pre-commit hooks will run automatically

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Ray** for distributed computing framework
- **JAX** for high-performance numerical computing
- **xarray** for labeled multi-dimensional arrays
- **SEAPODYM-LMTL** model (Lehodey et al.)

---

## 🗺️ Roadmap

**Completed**:
- [x] Core architecture (Kernel, Unit, Worker, Scheduler)
- [x] 2D spatial grids (spherical + plane)
- [x] Distributed workers with Ray
- [x] Event-driven scheduler
- [x] Centralized transport (mass-conservative)
- [x] GridInfo hierarchy (composition-based)
- [x] Ocean/land masking (xarray.DataArray)
- [x] ForcingManager (time-varying environmental forcings)
- [x] Zooplankton model (2-compartment, age-structured)
- [x] Temperature-dependent processes
- [x] Domain decomposition and redistribution

**In Progress**:
- [ ] NetCDF/Zarr forcing readers (temporal interpolation)
- [ ] Halo exchange for distributed transport
- [ ] Multi-species coupling
- [ ] GPU optimization (JAX device placement)

**Future**:
- [ ] 3D bathymetric grids (depth-varying masks)
- [ ] Adaptive timestepping
- [ ] Advanced transport schemes (WENO, TVD)
- [ ] Performance benchmarks
- [ ] Scientific validation suite
- [ ] Documentation website

---

**Built with ❤️ for marine ecosystem modeling**
