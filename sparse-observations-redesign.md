# SparseObservations Redesign

## Current Limitations

`SparseObservations` (in `seapopym/optimization/gradient.py`) hardcodes a rigid `(T, Y, X)` indexing structure that limits its applicability to outputs with arbitrary shapes.

### 1. Fixed 3D Index Structure

The dataclass requires exactly three spatial/temporal index arrays (`times`, `y`, `x`), making it incompatible with:

- **1D or 2D outputs** — e.g., spatially-aggregated time series (shape `(T,)`) or latitude-only transects (shape `(T, Y)`).
- **Higher-dimensional outputs** — e.g., outputs with depth (`Z`), cohort (`C`), or functional group dimensions: `(T, Y, X, Z, C)`.

### 2. Arbitrary Cohort Selection (L.186)

When the model output is 4D `(T, Y, X, C)`, the loss function silently picks cohort 0:

```python
pred_sparse = pred_full[obs.times, obs.y, obs.x, 0]
```

This is an arbitrary choice. Users may need to:

- Compare against a specific cohort (not necessarily 0).
- Aggregate across cohorts (sum, mean) before comparison.
- Observe individual cohorts independently.

### 3. Unsupported Use Cases

| Use case | Why it fails |
|---|---|
| Aggregation over cohorts/functional groups | No aggregation axis or function specified |
| Extra dimensions (Z, C, species) | Only T/Y/X indices exist |
| Outputs without spatial axes (e.g., scalar metrics) | Y and X are mandatory |
| Weighted spatial aggregation | No weighting mechanism |

## Proposed Direction

Replace the fixed `(times, y, x)` fields with a generic indexing scheme:

- **`indices`**: a tuple of arrays, one per dimension of the target output, allowing arbitrary dimensionality.
- **`aggregate_dims`**: optional specification of dimensions to reduce before comparison (e.g., sum over cohorts).
- **`aggregate_fn`**: the reduction function to apply (`sum`, `mean`, etc.).

This would let users express observations like "total biomass at (t=10, y=5, x=3) summed over all cohorts" or "mean production at (t=20) averaged over the spatial domain."

### Sketch

```python
@dataclass
class SparseObservations:
    variable: str
    indices: tuple[Array, ...]       # one array per indexed dimension
    values: Array
    aggregate_dims: tuple[int, ...] | None = None  # dims to reduce before indexing
    aggregate_fn: str = "sum"        # "sum", "mean", etc.
```

This is a **breaking change** to the current API and should be planned alongside updates to `GradientRunner.make_loss_fn` that consumes the observations.
