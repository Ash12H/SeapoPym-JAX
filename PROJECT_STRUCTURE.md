# Project Structure

Complete structure of the SEAPOPYM-Message project.

```
seapopym-message/
├── src/
│   └── seapopym_message/
│       ├── __init__.py              # Main package entry point
│       ├── core/                    # Core architecture
│       │   ├── __init__.py
│       │   ├── kernel.py            # Kernel class (orchestrates Units)
│       │   ├── unit.py              # Unit class and @unit decorator
│       │   └── state.py             # State management
│       ├── distributed/             # Distributed computing
│       │   ├── __init__.py
│       │   ├── worker.py            # CellWorker (Ray actor)
│       │   ├── scheduler.py         # EventScheduler (priority queue)
│       │   └── message.py           # Message passing utilities
│       ├── kernels/                 # Pre-defined Units
│       │   ├── __init__.py
│       │   ├── biology.py           # Mortality, growth, recruitment
│       │   ├── transport.py         # Diffusion, advection
│       │   └── forcing.py           # Temperature-dependent processes
│       ├── forcing/                 # Environmental forcing
│       │   ├── __init__.py
│       │   ├── readers.py           # NetCDF/Zarr readers
│       │   └── interpolation.py    # Spatial/temporal interpolation
│       └── utils/                   # Utilities
│           ├── __init__.py
│           ├── domain.py            # Domain splitting (2D patches)
│           ├── grid.py              # GridInfo, coordinate systems
│           └── viz.py               # Visualization helpers
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Pytest fixtures
│   ├── unit/                        # Unit tests
│   │   ├── __init__.py
│   │   ├── test_unit.py             # Test Unit class
│   │   ├── test_kernel.py           # Test Kernel class
│   │   ├── test_biology.py          # Test biology Units
│   │   └── test_transport.py        # Test transport Units
│   └── integration/                 # Integration tests
│       ├── __init__.py
│       ├── test_distributed.py      # Test distributed simulation
│       └── test_scalability.py      # Scalability benchmarks
│
├── notebooks/                       # Jupyter notebooks (examples)
│   ├── 01_simple_0d_model.ipynb     # Single cell model
│   ├── 02_1d_grid_diffusion.ipynb   # 1D grid with diffusion
│   ├── 03_2d_spatial_model.ipynb    # 2D lat/lon model
│   ├── 04_temperature_forcing.ipynb # With environmental forcing
│   └── 05_scalability_test.ipynb    # Performance benchmarks
│
├── docs/                            # Documentation (MkDocs)
│   ├── index.md                     # Documentation home
│   ├── getting-started/
│   ├── architecture/
│   ├── api/
│   ├── tutorials/
│   ├── development/
│   └── about/
│
├── IA/                              # Architecture discussions (already exists)
│   ├── jax-cfd-integration.md
│   ├── architecture-revisited.md
│   ├── architecture-2d.md
│   ├── transport-architecture-comparison.md
│   └── transport-strategy.md
│
├── discussion/                      # ChatGPT discussions (already exists)
│   ├── 1.md
│   ├── 2.md
│   ├── 3.md
│   ├── 4.md
│   └── 5.md
│
├── pyproject.toml                   # Project configuration (UV, Ruff, Mypy)
├── uv.lock                          # UV lock file (auto-generated)
├── .pre-commit-config.yaml          # Pre-commit hooks
├── .gitignore                       # Git ignore rules
├── LICENSE                          # MIT License
├── README.md                        # Main README
├── Makefile                         # Development commands
├── mkdocs.yml                       # MkDocs configuration
└── PROJECT_STRUCTURE.md             # This file
```

---

## Module Organization

### 📦 `src/seapopym_message/core/`

**Core architecture components.**

- **`kernel.py`**: `Kernel` class that orchestrates Units
  - Topological sorting of Units
  - Separation local/global phases
  - Execution logic

- **`unit.py`**: `Unit` class and `@unit` decorator
  - Unit definition (func, inputs, outputs, scope)
  - Dependency checking
  - Execution interface

- **`state.py`**: State management
  - State wrapper (dict of arrays)
  - Snapshot/restore
  - Validation

### ⚡ `src/seapopym_message/distributed/`

**Distributed computing with Ray.**

- **`worker.py`**: `CellWorker2D` Ray actor
  - Manages a spatial patch
  - Executes Kernel locally
  - Halo exchange with neighbors

- **`scheduler.py`**: `EventScheduler` Ray actor
  - Priority queue of events
  - No global time loop
  - Asynchronous orchestration

- **`message.py`**: Message passing utilities
  - Message definitions
  - Serialization helpers

### 🧬 `src/seapopym_message/kernels/`

**Pre-defined computational Units.**

- **`biology.py`**: Biological processes
  - `compute_recruitment`
  - `compute_mortality`
  - `compute_growth`

- **`transport.py`**: Physical transport
  - `compute_diffusion_2d`
  - `compute_advection_2d`

- **`forcing.py`**: Environment-dependent processes
  - `compute_recruitment_temperature`
  - `compute_mortality_temperature`

### 🌊 `src/seapopym_message/forcing/`

**Environmental forcing data.**

- **`readers.py`**: Data readers
  - `load_netcdf`
  - `load_zarr`

- **`interpolation.py`**: Interpolation
  - Spatial (lat/lon)
  - Temporal (time)

### 🛠️ `src/seapopym_message/utils/`

**Utilities and helpers.**

- **`domain.py`**: Domain decomposition
  - `split_domain_2d`
  - `split_domain_2d_periodic_lon`

- **`grid.py`**: Grid management
  - `GridInfo` dataclass
  - Coordinate systems
  - Metrics (dx, dy)

- **`viz.py`**: Visualization
  - `plot_global_field`
  - `reconstruct_global_grid`

---

## Configuration Files

### `pyproject.toml`

Main project configuration with:
- **UV**: Package management
- **Ruff**: Linting and formatting (rules: E, W, F, I, B, C4, UP, ARG, SIM)
- **Mypy**: Type checking (strict mode)
- **Pytest**: Testing configuration
- **Coverage**: Code coverage settings

### `.pre-commit-config.yaml`

Pre-commit hooks:
1. **Ruff**: Lint and format
2. **Mypy**: Type check
3. **Standard checks**: trailing whitespace, YAML/TOML validation, etc.
4. **Pydocstyle**: Docstring validation (Google style)
5. **Nbstripout**: Clean Jupyter notebooks

### `Makefile`

Convenient development commands:
- `make install`: Install package
- `make install-dev`: Install with dev dependencies
- `make test`: Run tests
- `make format`: Format code
- `make lint`: Lint code
- `make typecheck`: Type check
- `make check`: All quality checks
- `make clean`: Clean build artifacts
- `make docs`: Build documentation
- `make pre-commit`: Run pre-commit hooks

---

## Testing Strategy

### Unit Tests (`tests/unit/`)

Test individual components in isolation:
- Core classes (Unit, Kernel)
- Individual Units (biology, transport)
- Utilities (domain splitting, grid)

### Integration Tests (`tests/integration/`)

Test complete workflows:
- Distributed simulation (workers + scheduler)
- Scalability (1 → 10 → 100 workers)
- End-to-end simulation

### Markers

```bash
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m slow          # Slow tests
pytest -m gpu           # GPU tests
```

---

## Development Workflow

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/yourusername/seapopym-message.git
cd seapopym-message

# Install with UV
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 2. Make Changes

```bash
# Create branch
git checkout -b feature/my-feature

# Write code in src/seapopym_message/
# Write tests in tests/

# Run quality checks
make check
```

### 3. Commit

```bash
# Pre-commit hooks run automatically
git add .
git commit -m "Add my feature"

# If hooks fail, fix issues and re-commit
```

### 4. Test

```bash
# Run tests
make test

# With coverage
make test-cov
```

### 5. Document

```bash
# Add docstrings (Google style)
# Update docs/ if needed

# Build documentation
make docs

# Serve locally
make docs-serve
```

---

## Next Steps

### Immediate (Core Implementation)

1. ✅ Project structure created
2. ⏳ Implement `Unit` class (`src/seapopym_message/core/unit.py`)
3. ⏳ Implement `Kernel` class (`src/seapopym_message/core/kernel.py`)
4. ⏳ Write unit tests
5. ⏳ Implement basic Units (mortality, growth)

### Short-term (Prototype)

1. Implement `CellWorker2D` (`src/seapopym_message/distributed/worker.py`)
2. Implement `EventScheduler` (`src/seapopym_message/distributed/scheduler.py`)
3. Create minimal example (2×2 workers, simple model)
4. Integration tests
5. Documentation

### Medium-term (Features)

1. Forcing readers (NetCDF/Zarr)
2. Transport Units (diffusion, advection)
3. Temperature-dependent processes
4. Visualization utilities
5. Jupyter notebooks (examples)

### Long-term (Production)

1. JAX-CFD integration
2. TransportWorker (Version 2)
3. Multi-species coupling
4. GPU optimization
5. Performance benchmarks
6. Scientific validation

---

## Code Style

### Type Hints (Required)

```python
def compute_mortality(
    biomass: jnp.ndarray,
    temperature: jnp.ndarray,
    params: dict[str, float]
) -> jnp.ndarray:
    """Compute mortality rate."""
    ...
```

### Docstrings (Google Style)

```python
def split_domain_2d(
    nlat: int,
    nlon: int,
    num_workers_lat: int,
    num_workers_lon: int
) -> list[dict]:
    """Split 2D domain into patches for workers.

    Args:
        nlat: Number of latitude cells.
        nlon: Number of longitude cells.
        num_workers_lat: Number of workers in latitude direction.
        num_workers_lon: Number of workers in longitude direction.

    Returns:
        List of patch definitions with worker IDs and neighbors.

    Raises:
        ValueError: If grid cannot be evenly divided.
    """
    ...
```

### Formatting (Ruff)

- Line length: 100 characters
- Quotes: double quotes
- Indentation: 4 spaces
- Imports: sorted (isort)

---

## Resources

- **UV Docs**: https://docs.astral.sh/uv/
- **Ruff Docs**: https://docs.astral.sh/ruff/
- **Mypy Docs**: https://mypy.readthedocs.io/
- **Ray Docs**: https://docs.ray.io/
- **JAX Docs**: https://jax.readthedocs.io/
- **MkDocs Material**: https://squidfunk.github.io/mkdocs-material/
