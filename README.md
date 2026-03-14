# SeapoPym

![Tests](https://github.com/Ash12H/SeapoPym-JAX/actions/workflows/ci.yml/badge.svg)
![Docs](https://github.com/Ash12H/SeapoPym-JAX/actions/workflows/docs.yml/badge.svg)
[![Coverage](https://codecov.io/gh/Ash12H/SeapoPym-JAX/branch/main/graph/badge.svg)](https://codecov.io/gh/Ash12H/SeapoPym-JAX)
![Python](https://img.shields.io/badge/python-%E2%89%A53.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
![JAX](https://img.shields.io/badge/powered_by-JAX-red)

**SeapoPym** is a JAX-accelerated framework for differentiable simulation of dynamical systems on N-dimensional grids.

It uses a **[Directed Acyclic Graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph) (DAG) blueprint architecture** where processes are declared as connected nodes with flux edges. Models are defined in YAML, compiled into optimized JAX computation graphs, and executed on CPU or GPU.

![Simulate dynamical systems, compute exact gradients through physics, and optimize parameters — all in JAX.](docs/assets/hero.png)

> **Lotka-Volterra twin experiment** — A 2-species predator-prey ODE declared as a SeapoPym Blueprint. Gradient descent recovers α and γ from partial, noisy observations (prey only, 5% Gaussian noise) by back-propagating through the entire simulation via `jax.grad` — converging to <1% error in ~100 steps.

## Why SeapoPym?

**For scientists** — Explicit numerical schemes (Euler, first-order upwind finite volume), YAML model declaration, visual process DAG, strict unit validation (Pint), NaN rejection at compile time.

**For ML engineers** — Pure JAX backend (`jax.lax.scan`), end-to-end differentiable (`jax.grad` through physics), `jax.vmap` auto-vectorization, GPU/TPU support, built-in optimization (Optax, CMA-ES, GA via evosax).

## Quickstart

Logistic growth — `dN/dt = r·N·(1 − N/K)` — in 20 lines:

```python
import numpy as np
import xarray as xr
from seapopym.blueprint import Blueprint, Config, functional
from seapopym.compiler import compile_model
from seapopym.engine import simulate

# 1. Define the physics — one equation, two parameters
@functional(name="logistic", units={"N": "kg/m^2", "r": "1/s", "K": "kg/m^2", "return": "kg/m^2/s"})
def logistic_growth(N, r, K):
    return r * N * (1 - N / K)

# 2. Declare the model
blueprint = Blueprint.from_dict({
    "id": "logistic", "version": "1.0",
    "declarations": {
        "state": {"N": {"units": "kg/m^2", "dims": ["Y", "X"]}},
        "parameters": {"r": {"units": "1/s"}, "K": {"units": "kg/m^2"}},
        "forcings": {},
    },
    "process": [{"func": "logistic", "inputs": {"N": "state.N", "r": "parameters.r", "K": "parameters.K"}, "outputs": {"return": "derived.growth"}}],
    "tendencies": {"N": [{"source": "derived.growth"}]},
})

# 3. Configure, compile, run
DAY = 86400.0
config = Config.from_dict({
    "parameters": {"r": xr.DataArray(0.05 / DAY), "K": xr.DataArray(100.0)},
    "forcings": {},
    "initial_state": {"N": xr.DataArray(np.array([[1.0]]), dims=["Y", "X"])},
    "execution": {"time_start": "2000-01-01", "time_end": "2000-12-31", "dt": "1d"},
})
model = compile_model(blueprint, config)
_, outputs = simulate(model)

# 4. Result — logistic saturation toward K=100
N = outputs["N"].values[:, 0, 0]
print(f"Day 0: {N[0]:.0f} → Day 90: {N[90]:.0f} → Day 180: {N[180]:.0f} → Day 365: {N[-1]:.0f}")
# Day 0: 1 → Day 90: 47 → Day 180: 99 → Day 365: 100
```

## Installation

```bash
pip install git+https://github.com/Ash12H/SeapoPym-JAX.git
```

With GPU support:
```bash
pip install "git+https://github.com/Ash12H/SeapoPym-JAX.git[gpu]"
```

For development:
```bash
git clone https://github.com/Ash12H/SeapoPym-JAX.git
cd SeapoPym-JAX
uv sync --all-extras
```

## Documentation

Full documentation, conceptual guides, and runnable examples:

**[https://ash12h.github.io/SeapoPym-JAX/](https://ash12h.github.io/SeapoPym-JAX/)**

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

[MIT](LICENSE) — Jules Lehodey, 2024
