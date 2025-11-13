# SEAPOPYM-Message

**Distributed Population Dynamics Simulation using Ray and JAX**

A modern, scalable framework for simulating spatially-explicit population dynamics with event-driven architecture and composable computational kernels.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)

---

## 🎯 Key Features

- **Event-Driven Architecture**: Asynchronous scheduler with priority queue (no global time loop)
- **Composable Kernels**: Define models as lists of modular units (`[growth, mortality, transport]`)
- **Distributed Computing**: Ray-based workers with intelligent message passing
- **High Performance**: JAX JIT compilation for GPU/TPU acceleration
- **2D Spatial Grids**: Native lat/lon support with halo exchange
- **Flexible Transport**: Distributed (scalable) or centralized (precise) strategies
- **Scientific Rigor**: Type-checked, tested, and validated

---

## 🏗️ Architecture Overview

```
EventScheduler (PriorityQueue)
    ↓
Ray Workers (distributed)
    ↓
Kernel = [Unit₁, Unit₂, ..., Unitₙ]
    ├─ Local Phase (parallel)
    ├─ [synchronization]
    └─ Global Phase (with neighbors)
```

### Core Concepts

- **Unit**: Elementary computation (growth, mortality, diffusion)
- **Kernel**: Ordered list of Units to execute per timestep
- **Worker**: Ray actor managing a spatial patch (cells)
- **Scheduler**: Event-driven orchestrator (no `for` loop on time)

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

# JAX-CFD for advanced transport
uv pip install -e ".[jax-cfd]"

# Visualization tools
uv pip install -e ".[viz]"

# All extras
uv pip install -e ".[all]"
```

---

## 🚀 Quick Start

### Define a Simple Model

```python
from seapopym_message.core import Kernel, unit
from seapopym_message.distributed import CellWorker2D, EventScheduler
import jax.numpy as jnp

# Define computational units
@unit(name='recruitment', inputs=[], outputs=['R'], scope='local')
def compute_recruitment(params):
    return params['R']

@unit(name='mortality', inputs=['biomass', 'temperature'],
      outputs=['mortality_rate'], scope='local', compiled=True)
def compute_mortality(biomass, temperature, params):
    lambda_T = params['lambda_0'] * jnp.exp(params['k'] * temperature)
    return lambda_T * biomass

@unit(name='growth', inputs=['biomass', 'R', 'mortality_rate'],
      outputs=['biomass'], scope='local', compiled=True)
def compute_growth(biomass, R, mortality_rate, dt, params):
    return biomass + (R - mortality_rate) * dt

# Create kernel (model = list of units)
kernel = Kernel([
    compute_recruitment,
    compute_mortality,
    compute_growth
])
```

### Run Distributed Simulation

```python
import ray

ray.init()

# Setup grid and workers
from seapopym_message.utils import setup_simulation_2d

workers, grid_info = setup_simulation_2d(
    nlat_global=120,
    nlon_global=180,
    num_workers_lat=4,
    num_workers_lon=6,
    kernel=kernel,
    params={'R': 10.0, 'lambda_0': 0.01, 'k': 0.05},
    grid_bounds={'lat_min': -15, 'lat_max': 15,
                 'lon_min': 120, 'lon_max': -75}
)

# Run simulation (event-driven, no for loop!)
scheduler = EventScheduler.remote(workers, dt=0.1, t_end=365.0)
results = ray.get(scheduler.run.remote())

print(f"Simulation completed: {len(results)} workers finished")
```

### Visualize Results

```python
from seapopym_message.utils import reconstruct_global_grid, plot_global_field

global_state = reconstruct_global_grid(workers, grid_info)
plot_global_field(global_state['biomass'], grid_info, title='Biomass (g/m²)')
```

---

## 📚 Documentation

- **[Architecture Guide](docs/architecture.md)**: Detailed explanation of design principles
- **[API Reference](docs/api/)**: Complete API documentation
- **[Tutorials](notebooks/)**: Jupyter notebooks with examples
- **[Design Discussions](IA/)**: Architecture decisions and comparisons

---

## 🧪 Development

### Run Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit -m unit

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

# Run all pre-commit hooks manually
pre-commit run --all-files
```

### Using Makefile

```bash
# Install development dependencies
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

## 🎓 Examples

### 1. Simple 0D Model (Single Cell)

```python
# See notebooks/01_simple_0d_model.ipynb
```

### 2. 1D Grid with Diffusion

```python
# See notebooks/02_1d_grid_diffusion.ipynb
```

### 3. 2D Lat/Lon with Temperature Forcing

```python
# See notebooks/03_2d_forcing.ipynb
```

### 4. Distributed Scalability Test

```python
# See notebooks/04_scalability.ipynb
```

---

## 🏛️ Architecture Highlights

### Two Transport Strategies

#### Version 1: Distributed (Default)
- Transport in each worker (scalable)
- Minimal communication (halos only)
- Best for: prototyping, large grids, parameter studies

#### Version 2: Centralized (Advanced)
- Dedicated TransportWorker with JAX-CFD
- High precision scientific schemes
- Best for: publications, multi-model coupling

**Both are supported!** Switch with configuration.

### Event-Driven Scheduler

No global `for t in range(steps)` loop. Instead:
- Workers schedule their own events
- Priority queue manages execution order
- Asynchronous message passing
- Non-blocking communication

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
- **JAX-CFD** for computational fluid dynamics
- **xarray** for labeled multi-dimensional arrays

---

## 📧 Contact

For questions or discussions, please open an issue or contact [your.email@example.com](mailto:your.email@example.com).

---

## 🗺️ Roadmap

- [x] Core architecture (Kernel, Unit, Scheduler)
- [x] 2D spatial grids (lat/lon)
- [x] Distributed workers with Ray
- [x] Event-driven scheduler
- [ ] JAX-CFD integration (Version 2 transport)
- [ ] NetCDF/Zarr forcing readers
- [ ] Multi-species coupling
- [ ] GPU optimization
- [ ] Documentation website
- [ ] Performance benchmarks
- [ ] Scientific validation suite

---

**Built with ❤️ for marine ecosystem modeling**
