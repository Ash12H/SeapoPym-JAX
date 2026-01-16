# Benchmarks and Profiling Tools

This directory contains tools for performance analysis and optimization verification.

## Available Tools

### `profile_production_dynamics.py`
Line-by-line profiling of `compute_production_dynamics` using `line_profiler`.

**Usage:**
```bash
uv run python benchmarks/profile_production_dynamics.py
```

**Output:**
- Detailed timing breakdown per line
- Identifies bottlenecks in the function
- Shows percentage of time spent on each operation

**Key findings:**
- `.sum(dim="cohort")`: 19.4% (unavoidable reduction)
- `.where()` operations: 34.3% (conditional masking)
- `.shift()`: 14.5% (cohort transitions)

### `inspect_chunking.py`
Inspect chunking behavior and Dask graph characteristics.

**Usage:**
```bash
uv run python benchmarks/inspect_chunking.py
```

**Output:**
- Verifies chunking preservation through functions
- Detects unwanted rechunking or transpositions
- Analyzes Dask graph size and complexity
- Tests different chunking strategies

**Key findings:**
- ✅ Chunking is preserved exactly
- ✅ No transpositions occur
- ✅ Graph expansion is reasonable (< 20×)
- ✅ `.shift()` preserves chunk structure

## Optimization Conclusions

After extensive profiling and benchmarking:

1. **Manual optimizations are counterproductive**: xarray/numpy implementations are already highly optimized
2. **The function is Dask-friendly**: No rechunking, no transpositions, reasonable graph size
3. **15% execution time is acceptable**: Due to inherent algorithmic complexity (7 vectorized operations)
4. **Focus on transport (70% of time)**: Data parallelism with `chunks={'cohort': 1}` is the correct strategy

## Dependencies

- `line_profiler`: For line-by-line profiling
  ```bash
  uv pip install line_profiler
  ```
