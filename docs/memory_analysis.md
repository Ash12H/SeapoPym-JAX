# Memory Analysis: LMTL Simulation with Transport

**Target configuration**: 180x360 grid, F=6 functional groups, C~11 cohorts, Z=3 layers, ~70K timesteps, chunk_size=1000, CPU, forcings from NetCDF/Zarr (lazy).

---

## A. Memory Estimates

### State (carry, overwritten each step)

| Variable   | Shape              | Size     |
|------------|--------------------|----------|
| biomass    | (6, 180, 360)      | 1.55 MB  |
| production | (6, 180, 360, 11)  | 17.1 MB  |
| **Total**  |                    | **18.65 MB** |

### Forcings per chunk (chunk_size=1000, post-interpolation)

| Forcing            | Shape                  | Size       |
|--------------------|------------------------|------------|
| temperature        | (1000, 3, 180, 360)    | 778 MB     |
| u                  | (1000, 3, 180, 360)    | 778 MB     |
| v                  | (1000, 3, 180, 360)    | 778 MB     |
| primary_production | (1000, 180, 360)       | 259 MB     |
| day_of_year        | (1000,)                | 4 KB       |
| Static (dx, dy, mask, etc.) | broadcast views | ~0 MB      |
| **Total dynamic**  |                        | **~2.59 GB** |

> `day_of_year` was reduced from dims `[T, Y, X]` (259 MB/chunk) to `[T]` (4 KB/chunk) by removing the unnecessary spatial broadcast.

### Interpolation - hidden cost

Source arrays loaded for `xarray.interp()`:

| Source             | Shape                  | Size     |
|--------------------|------------------------|----------|
| temperature        | (5000, 3, 180, 360)    | 3.89 GB  |
| u                  | (5000, 3, 180, 360)    | 3.89 GB  |
| v                  | (5000, 3, 180, 360)    | 3.89 GB  |
| primary_production | (5000, 180, 360)       | 1.30 GB  |
| day_of_year        | (5000,)                | 20 KB    |
| **Total sources**  |                        | **~13 GB** |

These are loaded at the first `get_chunk()` call and potentially cached in RAM by xarray for subsequent chunks. **Key question to measure**: do the sources persist in memory after `get_chunk()` returns?

### Scan output per chunk (export_variables=["biomass"])

| Variable   | Shape                   | Size/chunk   |
|------------|-------------------------|--------------|
| biomass    | (1000, 6, 180, 360)     | **1.56 GB**  |
| production | (1000, 6, 180, 360, 11) | **17.1 GB**  |

Including production in outputs is impractical at this scale.

### MemoryWriter accumulation (70 chunks)

| Metric                     | Estimate           |
|----------------------------|--------------------|
| Accumulated chunks         | 70 x 1.56 GB = **109 GB** |
| `finalize()` peak          | ~218 GB (concatenate + np.asarray) |
| **Conclusion**             | In-memory output is impractical at F=6, 70K timesteps |

Use `DiskWriter` (Zarr) for large simulations.

### JIT intermediates (per timestep, inside XLA)

- ~17 intermediate arrays, largest shape (F, Y, X, C) = 17.1 MB
- Peak estimated: ~100-150 MB, reused by XLA, does not persist between chunks

---

## B. Chronological Memory Profile

```
Phase                              Estimated RAM
--------------------------------------------------
Compilation                        < 100 MB
get_chunk() first                  +13 GB (interp sources)
                                   +2.6 GB (interpolated chunk)
Chunk loop (N=70)                  13 GB (sources) + 2.6 GB (chunk) + N x 1.56 GB (outputs)
Last chunk                         13 + 2.6 + 109 = ~125 GB
finalize()                         Peak ~230 GB
Post-finalize (xr.Dataset)        ~109 GB
```

---

## C. Simulation vs Optimization - Memory Requirements

|                    | Simulation       | Gradient (Adam)            | Evolutionary (CMA-ES)      |
|--------------------|------------------|----------------------------|-----------------------------|
| Backward pass      | No               | Yes -> stores O(T x state) | No                          |
| Forcings           | Chunk by chunk   | All in memory              | **Can be chunked** (below)  |
| Outputs            | Stream to disk   | Loss inside scan           | Loss inside scan            |
| 70K steps, F=6     | ~20 GB peak      | ~1.3 TB without checkpoint | ~15 GB with chunking        |

### Key Insight: Evolutionary Optimization CAN Use Chunking

The current pattern loads everything via `get_all()`:

```python
def run_one(single_free):
    forcings = model.forcings.get_all()  # ALL in memory
    (_, _), outputs = lax.scan(step_fn, init, forcings)
    return outputs
jax.vmap(run_one)(free_params)
```

**Chunked pattern** (vmap INSIDE the Python loop, not around it):

```python
state = broadcast_init_state(pop_size)
for chunk_idx in range(n_chunks):
    forcings_chunk = model.forcings.get_chunk(start, end)  # 1 chunk only

    def run_chunk(pop_state, pop_params):
        (new_state, _), outputs = lax.scan(step_fn, (pop_state, merged), forcings_chunk)
        return new_state, outputs

    state, outputs = jax.vmap(run_chunk)(state, free_params)
```

- `forcings_chunk` is in the closure -> broadcast, not duplicated per individual
- Reduces forcing memory from ~200 GB to ~2.6 GB per chunk
- vmap vectorizes over `(pop_state, pop_params)` inside each chunk

---

## D. Architectural Recommendations

These are identified improvements, not in the immediate scope but documented for future work:

1. **`fori_loop` for simulation** - When only the final state is needed, `lax.fori_loop` avoids accumulating outputs entirely. Already supported via `RunnerConfig.loop_mode="fori_loop"`.

2. **Output subsampling** - Export every N timesteps instead of every step. Reduces MemoryWriter accumulation by factor N.

3. **Loss-in-scan for optimization** - Compute the loss inside `step_fn`, return only a scalar. Eliminates output accumulation entirely for optimization runs.

4. **`jax.checkpoint`** - Makes gradient-based optimization tractable on long time series by trading compute for memory (recompute vs store intermediates).

5. **Chunked vmap** - The pattern described above for evolutionary optimization. Moves the Python loop outside vmap, keeping forcings chunked.

6. **DiskWriter as default for large grids** - Auto-detect grid size and switch to disk output when estimated accumulation exceeds a threshold (e.g., 10 GB).

---

## E. day_of_year Dimension Fix

`day_of_year` depends only on time, not on spatial coordinates. The previous declaration `dims: [T, Y, X]` caused:

- **Wasted memory**: 180 x 360 = 64,800x redundant copies per timestep
- **Per chunk**: 259 MB instead of 4 KB
- **Source data**: 1.30 GB instead of 20 KB

**Fix applied**: Changed `dims: [T, Y, X]` to `dims: [T]` in both YAML model files. The vmap system automatically broadcasts scalar forcings (those without Y/X dims) via `in_axes=None`, so `day_length(latitude, day_of_year)` receives a scalar `day_of_year` per timestep, broadcast over Y. This is correct since photoperiod depends on latitude and time, not longitude.

**Downstream impact**: `day_length` output changes from (Y, X) to (Y,). `layer_weighted_mean` receives `day_length(Y)` instead of `(Y, X)` -> the vmap over X broadcasts `day_length` via `in_axes=None`. Functionally equivalent, memory-optimal.
