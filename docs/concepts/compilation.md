# Compilation & Config

The **compiler** bridges the gap between a declarative Blueprint and a JAX-executable model. It validates, infers shapes, prepares data, and produces a `CompiledModel`.

## Config

A **Config** provides the concrete data needed to run a Blueprint:

```python
from seapopym.blueprint import Config

config = Config.from_dict({
    "parameters": {
        "lambda_0": xr.DataArray([1e-7, 2e-7], dims=["F"]),
        "tau_r_0": xr.DataArray([8.64e6, 1.2e7], dims=["F"]),
        # ...
    },
    "forcings": {
        "temperature": xr.DataArray(..., dims=["T", "Z", "Y", "X"]),
        "primary_production": xr.DataArray(..., dims=["T", "Y", "X"]),
        # ...
    },
    "initial_state": {
        "biomass": xr.DataArray(..., dims=["F", "Y", "X"]),
        "production": xr.DataArray(..., dims=["F", "C", "Y", "X"]),
    },
    "execution": {
        "time_start": "2000-01-01",
        "time_end": "2001-12-31",
        "dt": "1d",
    },
})
```

### Config Sections

| Section | Type | Purpose |
|---------|------|---------|
| `parameters` | `dict[str, xr.DataArray]` | Static model constants (growth rates, mortality coefficients) |
| `forcings` | `dict[str, xr.DataArray]` | Time-varying input data (temperature, currents, NPP) |
| `initial_state` | `dict[str, xr.DataArray]` | Starting values for state variables |
| `execution` | `ExecutionParams` | Time range, timestep, interpolation method |

### Execution Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_start` | str | — | Simulation start (ISO format, e.g., `"2000-01-01"`) |
| `time_end` | str | — | Simulation end |
| `dt` | str | `"1d"` | Timestep as a Pint-compatible string (`"1d"`, `"6h"`, `"0.05d"`) |
| `forcing_interpolation` | str | `"constant"` | Temporal interpolation: `"linear"`, `"nearest"`, `"ffill"`, `"constant"` |
| `output_path` | str \| None | None | Zarr path for disk output (enables DiskWriter) |

### Dimension Mapping

If your data uses non-canonical dimension names, provide a mapping:

```python
config = Config.from_dict({
    # ...
    "dimension_mapping": {
        "lat": "Y",
        "lon": "X",
        "time": "T",
        "depth": "Z",
    },
})
```

This lets you use standard xarray conventions (`lat`, `lon`) without renaming your data.

## compile_model()

```python
from seapopym.compiler import compile_model

model = compile_model(blueprint, config)
```

The compiler runs an 8-step pipeline:

### 1. Validation

Validates both blueprint and config:

- **Blueprint validation** — Functions exist in registry, signatures match inputs, unit consistency across the process chain, tendency sources reference valid derived variables.
- **Config validation** — All required parameters/forcings/initial_state present, dimension names match declarations, forcings span the requested time range.

!!! warning "Strict NaN rejection"
    The compiler rejects any data containing NaN values. Clean your input data before compilation.

### 2. Time Grid

Parses `time_start`, `time_end`, and `dt` into a `TimeGrid`:

- Converts `dt` to seconds via Pint (e.g., `"1d"` → 86400 s)
- Computes `n_timesteps = duration / dt`
- Validates exact divisibility (no silent rounding)
- Generates datetime64 coordinate array

### 3. Dimension Extraction

Reads declared dimensions from the blueprint and maps variable paths to their dimension tuples.

### 4. Shape Inference

Reads xarray metadata (`.dims`, `.sizes`) from all data sources without loading arrays into memory. Validates that the same dimension name has the same size everywhere.

### 5. Dimension Mapping

Applies user-provided name remapping (e.g., `lat` → `Y`).

### 6. Forcing Preparation

- Separates **static** forcings (no T dimension, e.g., bathymetry) from **dynamic** ones (with T, e.g., temperature).
- Transposes to canonical dimension order `(E, T, F, C, Z, Y, X)`.
- Slices dynamic forcings to the simulation time window.
- Creates a `ForcingStore` for lazy materialization.

### 7. State Preparation

- Transposes initial state arrays to canonical order.
- Converts `xr.DataArray` → `jax.Array`.
- Validates no NaN values.

### 8. Parameter Preparation

Same as state: transpose, convert, validate.

## CompiledModel

The output of `compile_model()` is a `CompiledModel` dataclass containing everything needed for execution:

| Attribute | Type | Description |
|-----------|------|-------------|
| `blueprint` | Blueprint | Original model definition (read-only) |
| `compute_nodes` | list[ComputeNode] | Ordered list of process step functions |
| `data_nodes` | dict[str, DataNode] | Metadata for all variables (dims, units, type) |
| `tendency_map` | dict | Maps state variables → contributing flux sources |
| `state` | dict[str, Array] | Initial state as JAX arrays |
| `forcings` | ForcingStore | Lazy forcing loader |
| `parameters` | dict[str, Array] | Parameters as JAX arrays |
| `shapes` | dict[str, int] | Dimension sizes (e.g., `{"T": 365, "Y": 180, "X": 360}`) |
| `coords` | dict[str, Array] | Coordinate arrays for each dimension |
| `dt` | float | Timestep in seconds |
| `time_grid` | TimeGrid | Temporal grid metadata |

## ForcingStore

The `ForcingStore` implements **lazy loading** — forcing data stays as xarray DataArrays until actually needed at runtime.

```python
# At compile time: data is NOT loaded into GPU memory
model = compile_model(blueprint, config)

# At runtime: only the needed chunk is materialized
chunk = model.forcings.get_chunk(0, 10)  # loads timesteps 0-9 only
```

**Temporal interpolation** is applied on-the-fly when source temporal resolution differs from the simulation timestep:

| Method | Behavior |
|--------|----------|
| `"constant"` | Direct selection (source and target must align) |
| `"linear"` | Linear interpolation between source points |
| `"nearest"` | Nearest neighbor |
| `"ffill"` | Forward fill (last known value) |

This is especially useful for monthly forcing data interpolated to daily timesteps.

## Canonical Dimension Order

SeapoPym enforces a canonical dimension ordering throughout the pipeline:

```
(E, T, F, C, Z, Y, X)
```

| Dim | Name | Typical Use |
|-----|------|-------------|
| E | Ensemble | Ensemble runs (optimization batches) |
| T | Time | Temporal axis |
| F | Functional group | Species or size classes |
| C | Cohort | Age classes |
| Z | Depth | Vertical layers |
| Y | Latitude | Spatial grid |
| X | Longitude | Spatial grid |

All arrays are automatically transposed to this order during compilation. You never need to worry about axis ordering in your data — the compiler handles it.
