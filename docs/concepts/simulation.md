# Simulation & Engine

The engine executes a compiled model, stepping through time using `jax.lax.scan`. SeapoPym provides two levels of API: `simulate()` for convenience and `run()` for control.

## simulate() — High-Level API

```python
from seapopym.engine import simulate

state, outputs = simulate(model, chunk_size=365)
```

`simulate()` handles the full lifecycle:

1. Selects output variables (default: all state variables)
2. Builds the step function
3. Creates the appropriate writer (memory or disk)
4. Runs the simulation
5. Closes resources

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | CompiledModel | — | The compiled model |
| `chunk_size` | int \| None | None | Temporal chunk size (None = single chunk) |
| `output_path` | str \| None | None | Zarr path (enables DiskWriter) |
| `export_variables` | list[str] \| None | None | Variables to export (None = all state vars) |

**Returns:** `(final_state, outputs)` where outputs is an `xr.Dataset` (MemoryWriter) or `None` (DiskWriter).

## run() — Low-Level API

```python
from seapopym.engine import run, build_step_fn

step_fn = build_step_fn(model, export_variables=["biomass"])
state, outputs = run(step_fn, model, model.state, model.parameters, chunk_size=365)
```

`run()` is the pure execution engine. You control step function construction, state initialization, and writer selection.

**When to use `run()` instead of `simulate()`:**

- Custom step function wrapping (e.g., for gradient computation)
- Providing modified parameters (e.g., during optimization)
- Manual writer management
- Vmap over parameter ensembles

## Step Function

`build_step_fn()` compiles the process DAG into a single function compatible with `jax.lax.scan`:

```python
step_fn((state, params), forcings_t) → ((new_state, params), outputs)
```

Each timestep executes 4 phases:

### Phase 1: Compute Chain

Iterates through compute nodes in order. For each node:

1. Resolves inputs from state, forcings, parameters, or previously computed intermediates.
2. Applies auto-vmap for non-core dimensions (see below).
3. Executes the function.
4. Stores results in the intermediates dict.

### Phase 2: Euler Integration

Applies explicit Euler to all state variables:

$$
\text{state}_{t+1} = \max\left(\text{state}_t + \sum_i \text{source}_i \times \Delta t,\; 0\right)
$$

### Phase 3: Masking

Multiplies state by the spatial mask (1 = valid, 0 = land/boundary).

### Phase 4: Export

Extracts the requested variables (state + intermediates) for output.

## Chunked Execution

For long simulations, temporal chunking reduces memory usage:

```python
# Process 365 timesteps at a time
state, outputs = simulate(model, chunk_size=365)
```

The execution loop:

```
for chunk_start, chunk_end in chunk_ranges(n_timesteps, chunk_size):
    forcings_chunk = model.forcings.get_chunk(chunk_start, chunk_end)
    (state, params), outputs = jax.lax.scan(step_fn, (state, params), forcings_chunk)
    writer.append(outputs)
```

!!! tip "Chunk size and memory"
    Smaller chunks use less GPU memory but add overhead from chunk transitions. For most models, `chunk_size=365` (one year) is a good starting point.

## Automatic Vectorization (vmap)

Physics functions are written for their **core dimensions** only. The engine automatically wraps them with `jax.vmap` to broadcast over remaining dimensions.

**Example:** `mortality(biomass, temp, ...)` operates on scalars, but actual data has shape `(F, Y, X)`.

```
Function core_dims: none (scalar operation)
Input shape:        biomass (F, Y, X)

→ Engine wraps with vmap over F, Y, X
→ Function processes each (f, y, x) cell independently
→ Output shape: (F, Y, X)
```

For functions with core dimensions (e.g., `aging_flow` operates on the C cohort axis):

```
Function core_dims: {"production": ["C"]}
Input shape:        production (F, C, Y, X)

→ Engine wraps with vmap over F, Y, X (but NOT C)
→ Function processes each (f, y, x) cohort vector independently
→ Output shape: (F, C, Y, X)
```

This is transparent — you write functions for the dimensions they care about, and the engine handles broadcasting.

## Writers

Three output backends for different use cases:

### WriterRaw — JAX-Traceable

```python
from seapopym.engine import WriterRaw
```

- Stores outputs as Python list of JAX arrays.
- Compatible with `jax.grad` and `jax.vmap`.
- No coordinate metadata.

**Use case:** Optimization loops where you need gradients through the simulation.

### MemoryWriter — xarray Dataset

```python
# Used automatically by simulate() when output_path is None
state, outputs = simulate(model)
# outputs is an xr.Dataset with proper dimensions and coordinates
```

- Accumulates chunks in memory, concatenates on `finalize()`.
- Returns a fully labeled `xr.Dataset` with dimension names and coordinates.

**Use case:** Interactive analysis, notebooks, small to medium outputs.

### DiskWriter — Zarr Streaming

```python
# Used automatically when output_path is provided
state, _ = simulate(model, output_path="/results/sim.zarr")
# Data already written to disk
```

- Appends each chunk directly to a Zarr store.
- Constant memory usage regardless of simulation length.

**Use case:** Large simulations where outputs exceed RAM.

### Comparison

| Writer | Output Type | Memory | Gradient | Best For |
|--------|------------|--------|----------|----------|
| WriterRaw | dict[str, Array] | Grows with T | Yes | Optimization |
| MemoryWriter | xr.Dataset | Grows with T | No | Analysis |
| DiskWriter | Zarr on disk | Constant | No | Large runs |
