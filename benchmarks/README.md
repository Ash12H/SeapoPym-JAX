# Performance Benchmarking Guide

This directory contains comprehensive performance benchmarking tools for `seapopym-message`. The benchmarks are designed specifically for JAX code, handling JIT compilation, asynchronous execution, and proper synchronization.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Benchmark Tools](#benchmark-tools)
3. [Understanding Results](#understanding-results)
4. [Best Practices](#best-practices)
5. [Troubleshooting](#troubleshooting)

## Quick Start

### Installation

First, install the benchmark dependencies:

```bash
# Install development dependencies (includes pytest-benchmark)
pip install -e ".[dev]"

# Optional: Install matplotlib for plotting
pip install matplotlib

# Optional: Install tensorboard for profiling
pip install tensorboard
```

### Run Basic Benchmarks

```bash
# Run all benchmarks with pytest
pytest tests/benchmarks/ --benchmark-only

# Run custom benchmark script
python benchmarks/run_benchmarks.py

# Run scaling analysis
python benchmarks/scaling_analysis.py --plot
```

## Benchmark Tools

### 1. pytest-benchmark (Recommended for Testing)

**Location:** `tests/benchmarks/`

**Purpose:** Integrated benchmarks that run with pytest, perfect for CI/CD and regression testing.

**Features:**
- Automatic statistical analysis
- Comparison with previous runs
- Integration with pytest fixtures
- JSON/CSV export

**Usage:**

```bash
# Run all benchmarks
pytest tests/benchmarks/ --benchmark-only

# Run specific test file
pytest tests/benchmarks/test_benchmark_zooplankton.py --benchmark-only

# Save results to JSON
pytest tests/benchmarks/ --benchmark-only --benchmark-json=results.json

# Compare with previous run
pytest tests/benchmarks/ --benchmark-only --benchmark-compare=0001

# Only run small grid benchmarks (faster)
pytest tests/benchmarks/ --benchmark-only -k "small"
```

**Available Tests:**

- `tests/benchmarks/test_benchmark_zooplankton.py`
  - `age_production`: NPP-driven production aging
  - `compute_recruitment`: Recruitment calculation
  - `update_biomass`: Biomass dynamics

- `tests/benchmarks/test_benchmark_transport.py`
  - `advection_upwind_flux`: Advection scheme
  - `diffusion_explicit_spherical`: Diffusion scheme
  - Combined transport (advection + diffusion)

**Grid Sizes:**
- Small: 10×10 (quick tests)
- Medium: 100×100 (realistic)
- Large: 360×720 (production scale)

### 2. Custom Benchmark Script

**Location:** `benchmarks/run_benchmarks.py`

**Purpose:** Detailed performance analysis with custom metrics and flexible configuration.

**Features:**
- JAX-specific timing with `.block_until_ready()`
- JIT compilation warm-up
- Customizable grid sizes
- JSON export
- Device information

**Usage:**

```bash
# Run with default settings (120×360 grid)
python benchmarks/run_benchmarks.py

# Custom grid size
python benchmarks/run_benchmarks.py --grid-size 360x720

# More iterations for stable results
python benchmarks/run_benchmarks.py --iterations 50

# Only zooplankton functions
python benchmarks/run_benchmarks.py --zooplankton-only

# Only transport functions
python benchmarks/run_benchmarks.py --transport-only

# Save results to JSON
python benchmarks/run_benchmarks.py --save-json results.json
```

**Output Example:**

```
==================================================
Benchmarking Zooplankton Functions (grid: 120x360)
==================================================

1. age_production:
BenchmarkResult:
  Mean:   2.456 ms ± 0.123 ms
  Median: 2.441 ms
  Min:    2.312 ms
  Max:    2.789 ms
  P95:    2.678 ms
  P99:    2.745 ms
  Throughput: 1.76e+07 ops/s
```

### 3. Scaling Analysis

**Location:** `benchmarks/scaling_analysis.py`

**Purpose:** Analyze how performance scales with problem size (grid resolution).

**Features:**
- Multiple grid sizes tested automatically
- Scaling exponent calculation (time ∝ n_cells^α)
- Efficiency metrics
- Visualization (requires matplotlib)
- CSV export

**Usage:**

```bash
# Run with default grid sizes
python benchmarks/scaling_analysis.py

# Custom grid sizes
python benchmarks/scaling_analysis.py --grid-sizes "30x60,60x120,120x360,240x720"

# Generate plots
python benchmarks/scaling_analysis.py --plot

# Save plot to file
python benchmarks/scaling_analysis.py --save-plot scaling.png

# Save data to CSV
python benchmarks/scaling_analysis.py --save-csv scaling_data.csv
```

**Understanding Scaling Exponent (α):**

The scaling analysis computes how execution time scales with problem size:
- **time ∝ n_cells^α**

Interpretation:
- **α < 1.0**: Super-linear scaling (excellent!) - Better than expected, often due to cache effects
- **α ≈ 1.0**: Linear scaling (ideal) - Time doubles when grid doubles
- **α > 1.0**: Sub-linear scaling - Worse than expected, investigate bottlenecks
  - 1.0 < α < 1.15: Still acceptable
  - α > 1.3: Poor scaling, investigate memory bandwidth or algorithmic issues

**Example Output:**

```
SCALING ANALYSIS REPORT
==================================================

age_production:
  Scaling exponent (α): 0.987
  Interpretation: time ∝ n_cells^0.987
  Scaling quality: Linear (ideal)
  Efficiency: 101.32%

  Grid Size    Cells           Time (ms)   Throughput
  -------------------------------------------------------
  30x60        1,800           0.234       7.69e+06
  60x120       7,200           0.901       7.99e+06
  120x360      43,200          5.234       8.26e+06
  240x720      172,800         20.456      8.45e+06
```

### 4. JAX Profiler

**Location:** `benchmarks/profile_jax.py`

**Purpose:** Detailed performance profiling to identify bottlenecks at the operation level.

**Features:**
- Generates trace files for TensorBoard
- Perfetto traces for Chrome viewer
- Full simulation profiling (biology + transport)
- GPU profiling support (if CUDA available)

**Usage:**

```bash
# Profile transport operations
python benchmarks/profile_jax.py --function transport --output-dir ./profiles

# Profile zooplankton operations
python benchmarks/profile_jax.py --function zooplankton --output-dir ./profiles

# Profile full simulation step
python benchmarks/profile_jax.py --function full --output-dir ./profiles

# Custom grid size
python benchmarks/profile_jax.py --function full --grid-size 360x720
```

**Viewing Results:**

```bash
# Option 1: TensorBoard (recommended)
tensorboard --logdir ./profiles

# Then open http://localhost:6006 in your browser

# Option 2: Chrome Tracing
# 1. Open chrome://tracing in Chrome
# 2. Load the .json.gz file from ./profiles
```

**What to Look For in Profiles:**

1. **Time Distribution**: Which operations take the most time?
2. **Memory Transfers**: Excessive CPU↔GPU transfers?
3. **Kernel Launch Overhead**: Too many small operations?
4. **Compilation Time**: Is JIT compilation taking too long?

## Understanding Results

### JAX-Specific Considerations

1. **First Run vs Subsequent Runs**
   - First run includes JIT compilation time
   - All benchmarks perform warm-up to exclude compilation
   - Real performance is measured after warm-up

2. **Asynchronous Execution**
   - JAX uses asynchronous dispatch
   - All timing uses `.block_until_ready()` for accuracy
   - Without it, you'd measure dispatch time, not execution time

3. **Device Backend**
   - CPU backend: Single-threaded by default
   - GPU backend: Massive parallelism, but transfer overhead
   - Check device with: `python -c "import jax; print(jax.devices())"`

### Performance Metrics

**Mean Time**: Average execution time (most important metric)

**Standard Deviation**: Stability of measurements (should be < 10% of mean)

**P95/P99**: 95th/99th percentile (important for latency-sensitive applications)

**Throughput**: Operations per second (grid_cells / time)

**CFL Number**: Stability criterion for transport
- CFL < 1.0 for advection
- CFL < 0.25 for diffusion

## Best Practices

### 1. Benchmark Workflow

```bash
# Step 1: Quick check with pytest (fast)
pytest tests/benchmarks/ --benchmark-only -k "small"

# Step 2: Detailed analysis with custom script
python benchmarks/run_benchmarks.py --grid-size 120x360 --iterations 20

# Step 3: Scaling analysis
python benchmarks/scaling_analysis.py --plot --save-csv results.csv

# Step 4: Profile if needed
python benchmarks/profile_jax.py --function full --output-dir ./profiles
```

### 2. Ensuring Accurate Results

```python
# ✓ GOOD: Proper JAX benchmarking
def benchmark_function(func, *args):
    # Warm-up
    for _ in range(3):
        result = func(*args)
        result.block_until_ready()

    # Timing
    times = []
    for _ in range(10):
        start = time.perf_counter()
        result = func(*args)
        result.block_until_ready()  # Critical!
        times.append(time.perf_counter() - start)

    return np.mean(times)

# ✗ BAD: Missing synchronization
def bad_benchmark(func, *args):
    start = time.time()
    result = func(*args)
    # Missing .block_until_ready() - measures dispatch, not execution!
    return time.time() - start
```

### 3. Choosing Grid Sizes

- **Development**: 30×60 or 60×120 (fast iteration)
- **Testing**: 120×360 (realistic, still manageable)
- **Production**: 360×720 or larger (actual use case)

### 4. Comparing Optimizations

```bash
# Before optimization
python benchmarks/run_benchmarks.py --save-json before.json

# After optimization
python benchmarks/run_benchmarks.py --save-json after.json

# Compare (using pytest-benchmark)
pytest tests/benchmarks/ --benchmark-compare=before --benchmark-json=after.json
```

## Troubleshooting

### Issue: Unstable Benchmarks (High Std Dev)

**Symptoms:** Standard deviation > 20% of mean time

**Solutions:**
1. Increase number of iterations: `--iterations 50`
2. Close other applications
3. Disable CPU frequency scaling
4. Check for thermal throttling

### Issue: First Run Much Slower

**Cause:** JIT compilation included in measurement

**Solution:** Benchmarks already include warm-up phase. If still an issue:
```python
# Add more warm-up iterations
result = benchmark_jax_function(func, *args, n_warmup=10)
```

### Issue: GPU Not Used

**Check:**
```python
import jax
print(jax.devices())
# Should show GPU devices if available
```

**Solutions:**
1. Install JAX with GPU support: `pip install "jax[cuda12]"`
2. Check CUDA installation
3. Set environment variable: `export JAX_PLATFORMS=gpu`

### Issue: Out of Memory

**Solutions:**
1. Reduce grid size
2. Enable XLA memory optimization:
   ```bash
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   ```
3. Use smaller batch sizes in profiling

### Issue: Benchmarks Take Too Long

**Solutions:**
1. Run only small grid tests: `pytest tests/benchmarks/ -k "small"`
2. Reduce iterations: `--iterations 5`
3. Run specific functions only: `--zooplankton-only`

## Performance Targets

### Zooplankton Functions (120×360 grid)

- `age_production`: < 5 ms (target: ~2 ms)
- `compute_recruitment`: < 5 ms (target: ~2 ms)
- `update_biomass`: < 2 ms (target: ~0.5 ms)

### Transport Functions (120×360 grid)

- `advection_upwind_flux`: < 10 ms (target: ~5 ms)
- `diffusion_explicit_spherical`: < 10 ms (target: ~5 ms)
- Combined transport: < 20 ms (target: ~10 ms)

### Full Simulation Step

- Biology + Transport: < 30 ms per timestep (target: ~15 ms)

These targets assume CPU execution. GPU should be 5-10× faster for large grids.

## Additional Resources

- **JAX Profiling Guide**: https://jax.readthedocs.io/en/latest/profiling.html
- **pytest-benchmark Docs**: https://pytest-benchmark.readthedocs.io/
- **TensorBoard Guide**: https://www.tensorflow.org/tensorboard/get_started

## Contributing

When adding new functions, please:
1. Add pytest-benchmark tests to `tests/benchmarks/`
2. Include small, medium, and large grid sizes
3. Update scaling analysis if introducing new bottlenecks
4. Document expected performance targets
