# Seapopym

**Seapopym** is a Python framework for modeling marine population dynamics with JAX-accelerated backend. It provides a blueprint-based approach to define complex biological models and executes them efficiently on CPU/GPU.

## Key Features

- **Blueprint Architecture**: Define models declaratively (State, Parameters, Processes).
- **Backend Agnostic**: Run on **NumPy** for debugging or **JAX** for high-performance and gradients.
- **Strict Unit Validation**: Ensures dimensional consistency at compile time (using `pint`).
- **Flexible Forcings**: Automatic temporal interpolation (`linear`, `nearest`, `ffill`).
- **Efficient I/O**: Asynchronous Zarr output streaming or in-memory execution.

## Quickstart

### Installation

```bash
pip install seapopym
```

### Minimal Example

```python
from seapopym.blueprint import Blueprint, Config, functional
from seapopym.compiler import compile_model
from seapopym.engine import StreamingRunner
import xarray as xr
import numpy as np

# 1. Define a biological function
@functional(name="growth", backend="numpy", units={"biomass": "count", "rate": "1/s", "return": "count/s"})
def growth(biomass, rate):
    return biomass * rate

# 2. visual Blueprint
bp = Blueprint.from_dict({
    "id": "simple_model",
    "version": "1.0",
    "declarations": {
        "state": {"biomass": {"units": "count"}},
        "parameters": {"rate": {"units": "1/s"}},
        "forcings": {"time_index": {"dims": ["T"]}}
    },
    "process": [{
        "func": "growth",
        "inputs": {"biomass": "state.biomass", "rate": "parameters.rate"},
        "outputs": {"tendency": {"target": "tendencies.biomass", "type": "tendency"}}
    }]
})

# 3. Configure
config = Config.from_dict({
    "parameters": {"rate": {"value": 0.1}},
    "forcings": {"time_index": xr.DataArray(np.arange(10), dims=["T"])},
    "initial_state": {"biomass": xr.DataArray(100.0)},
    "execution": {"dt": "1s", "output_path": None}
})

# 4. Compile & Run
model = compile_model(bp, config)
runner = StreamingRunner(model)
state, outputs = runner.run(output_path=None) # In-memory

print(state["biomass"])
```

## Important: Units & Timesteps

**Seapopym v1 enforces strict unit consistency.**

1. **Time convention**: The internal solver works in **seconds**. All rates must be provided in `1/s` (or `unit/s`).
2. **Blueprints**: Must declare units using standard notation (e.g., `1/s`, `m/s`, `count`).
3. **No implicit conversion**: You must provide parameter values that match the declared units.

**Migration from legacy:**
If you used `1/d` in the past, please convert your parameter values and blueprint declarations to `1/s`. A `UnitError` will be raised if a mismatch is detected.

## Forcing Interpolation

You can enable temporal interpolation for forcings that don't match the simulation timestep:

```python
config.execution.forcing_interpolation = "linear" # or "nearest", "ffill"
```

## Documentation

See the `docs/` folder for detailed guides.
