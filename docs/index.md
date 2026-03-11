# SeapoPym

**SeapoPym** is a JAX-accelerated framework for Eulerian population dynamics on N-dimensional grids.

It uses a DAG-based blueprint architecture where biological and physical processes (movement, growth, mortality) are declared as connected nodes with flux edges. Models are defined in YAML, compiled into optimized JAX computation graphs, and executed on CPU or GPU.

## Why SeapoPym?

- **For scientists**: explicit numerical schemes, visual DAG of processes, YAML-based model declaration — easy to understand and modify.
- **For ML engineers**: JAX backend, differentiable end-to-end, GPU/TPU support, gradient-based optimization and backpropagation on mechanistic models.

## Key Features

- **Blueprint Architecture** — Declare models as YAML: state variables, parameters, forcings, and process DAG.
- **Strict Unit Validation** — Pint-based dimensional consistency enforced at compile time.
- **Automatic Vectorization** — `vmap` dispatch over non-core dimensions with canonical ordering `(E, T, F, C, Z, Y, X)`.
- **Pluggable Writers** — WriterRaw (JAX-traceable for optimization), MemoryWriter (`xr.Dataset`), DiskWriter (Zarr).
- **Flexible Forcings** — Lazy loading with temporal interpolation (`linear`, `nearest`, `ffill`).
- **Optimization** — Gradient descent (Optax), CMA-ES, Genetic Algorithm, IPOP-CMA-ES (evosax).

## Pipeline

```
Blueprint (YAML) + Config → compile_model() → CompiledModel → simulate() / run() → Outputs
```

## Installation

```bash
pip install git+https://github.com/Ash12H/SeapoPym-JAX.git
```

For GPU support:

```bash
pip install git+https://github.com/Ash12H/SeapoPym-JAX.git[gpu]
```

For development:

```bash
git clone https://github.com/Ash12H/SeapoPym-JAX.git
cd SeapoPym-JAX
uv sync --all-extras
```

## Quickstart

```python
from seapopym.models import LMTL_NO_TRANSPORT
from seapopym.blueprint import Config
from seapopym.compiler import compile_model
from seapopym.engine import simulate
import xarray as xr
import numpy as np

# 1. Load a pre-defined blueprint
blueprint = LMTL_NO_TRANSPORT

# 2. Configure the experiment
config = Config.from_dict({
    "parameters": {"growth_rate": {"value": 0.01}, ...},
    "forcings": {"temperature": xr.DataArray(...)},
    "initial_state": {"biomass": xr.DataArray(...)},
    "execution": {"time_start": "2020-01-01", "time_end": "2020-12-31", "dt": "1d"},
})

# 3. Compile and run
model = compile_model(blueprint, config)
state, outputs = simulate(model, chunk_size=365)

# outputs is an xr.Dataset with the exported variables
print(outputs)
```

## Links

- [GitHub Repository](https://github.com/Ash12H/SeapoPym-JAX)
- [Contributing Guide](https://github.com/Ash12H/SeapoPym-JAX/blob/main/CONTRIBUTING.md)
- [Changelog](https://github.com/Ash12H/SeapoPym-JAX/blob/main/CHANGELOG.md)
