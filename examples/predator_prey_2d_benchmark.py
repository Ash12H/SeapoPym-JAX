"""2D Predator-Prey Model: Benchmark (Numpy vs JAX).

Grid size: 10x10 cells.
Dynamics: Independent Lotka-Volterra in each cell (no transport).
"""

import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import xarray as xr

from seapopym.blueprint import Blueprint, Config, functional
from seapopym.compiler import compile_model
from seapopym.engine import StreamingRunner

# =============================================================================
# 1. Define Functions (Same as 0D)
# =============================================================================


@functional(
    name="demo:prey_growth",
    backend="numpy",
    units={
        "prey": "count",
        "growth_rate": "1/s",
        "predator": "count",
        "attack_rate": "1/(count*s)",
        "return": "count/s",
    },
)
def prey_growth_numpy(prey, growth_rate, predator, attack_rate):
    """Prey population change: growth - predation."""
    growth = growth_rate * prey
    predation = attack_rate * prey * predator
    return growth - predation


@functional(
    name="demo:prey_growth",
    backend="jax",
    units={
        "prey": "count",
        "growth_rate": "1/s",
        "predator": "count",
        "attack_rate": "1/(count*s)",
        "return": "count/s",
    },
)
def prey_growth_jax(prey, growth_rate, predator, attack_rate):
    """Prey population change: growth - predation (JAX version)."""
    growth = growth_rate * prey
    predation = attack_rate * prey * predator
    return growth - predation


@functional(
    name="demo:predator_dynamics",
    backend="numpy",
    units={
        "prey": "count",
        "predator": "count",
        "conversion_rate": "1/(count*s)",
        "mortality_rate": "1/s",
        "return": "count/s",
    },
)
def predator_dynamics_numpy(prey, predator, conversion_rate, mortality_rate):
    """Predator population change: conversion - mortality."""
    conversion = conversion_rate * prey * predator
    mortality = mortality_rate * predator
    return conversion - mortality


@functional(
    name="demo:predator_dynamics",
    backend="jax",
    units={
        "prey": "count",
        "predator": "count",
        "conversion_rate": "1/(count*s)",
        "mortality_rate": "1/s",
        "return": "count/s",
    },
)
def predator_dynamics_jax(prey, predator, conversion_rate, mortality_rate):
    """Predator population change: conversion - mortality (JAX version)."""
    conversion = conversion_rate * prey * predator
    mortality = mortality_rate * predator
    return conversion - mortality


# =============================================================================
# 2. Define Blueprint & Config (2D)
# =============================================================================

blueprint = Blueprint.from_dict(
    {
        "id": "lotka-volterra-2d",
        "version": "1.0.0",
        "declarations": {
            "state": {
                "prey": {"units": "count"},
                "predator": {"units": "count"},
            },
            "parameters": {
                "prey_growth_rate": {"units": "1/s"},
                "attack_rate": {"units": "1/(count*s)"},
                "conversion_rate": {"units": "1/(count*s)"},
                "predator_mortality": {"units": "1/s"},
            },
            "forcings": {
                # Time dimension + Space dimensions implied by data
                "time_index": {"dims": ["T"]},
            },
        },
        "process": [
            {
                "func": "demo:prey_growth",
                "inputs": {
                    "prey": "state.prey",
                    "growth_rate": "parameters.prey_growth_rate",
                    "predator": "state.predator",
                    "attack_rate": "parameters.attack_rate",
                },
                "outputs": {
                    "tendency": {"target": "tendencies.prey", "type": "tendency"},
                },
            },
            {
                "func": "demo:predator_dynamics",
                "inputs": {
                    "prey": "state.prey",
                    "predator": "state.predator",
                    "conversion_rate": "parameters.conversion_rate",
                    "mortality_rate": "parameters.predator_mortality",
                },
                "outputs": {
                    "tendency": {"target": "tendencies.predator", "type": "tendency"},
                },
            },
        ],
    }
)

# Configuration
n_days = 30 * 2  # 50 years
n_timesteps = int(n_days)
grid_size = (180 * 12, 360 * 12)

# Create spatial grid with slightly randomized initial conditions
lat = np.arange(grid_size[0])
lon = np.arange(grid_size[1])

# Base values
prey_base = 22.0
pred_base = 10.0

# Add noise: (Y, X)
np.random.seed(42)
prey_init = prey_base + np.random.uniform(-2, 2, size=(grid_size[0], grid_size[1]))
pred_init = pred_base + np.random.uniform(-1, 1, size=(grid_size[0], grid_size[1]))

config = Config.from_dict(
    {
        "parameters": {
            # NOTE: All rates in per-second (1/s) as per model convention
            # Values converted from daily rates: rate_per_day / 86400 = rate_per_second
            "prey_growth_rate": {"value": 0.05 / 86400},  # 5% daily → 5.8e-7 /s
            "attack_rate": {"value": 0.01 / 86400},  # 1% daily → 1.16e-7 /(count*s)
            "conversion_rate": {"value": 0.001 / 86400},  # 0.1% daily → 1.16e-8 /(count*s)
            "predator_mortality": {"value": 0.01 / 86400},  # 1% daily → 1.16e-7 /s
            # Parameters are scalar here (uniform over grid), but could be 2D DataArrays
        },
        "forcings": {
            "time_index": xr.DataArray(
                np.arange(n_timesteps),
                dims=["T"],
            ),
        },
        "initial_state": {
            "prey": xr.DataArray(prey_init, dims=["Y", "X"], coords={"Y": lat, "X": lon}),
            "predator": xr.DataArray(pred_init, dims=["Y", "X"], coords={"Y": lat, "X": lon}),
        },
        "execution": {
            "dt": "0.2d",
        },
    }
)

# =============================================================================
# 3. Benchmark Logic
# =============================================================================


def benchmark_backend(backend, iterations=10):
    """Benchmark a backend with multiple iterations.

    Args:
        backend: Backend name ("numpy" or "jax").
        iterations: Number of benchmark runs.

    Returns:
        Tuple of (average_time, compiled_model).
    """
    print(f"\n--- Benchmarking {backend.upper()} Backend ---")

    # Compile
    start_compile = time.perf_counter()
    compiled = compile_model(blueprint, config, backend=backend)
    end_compile = time.perf_counter()
    print(f"Compilation time: {end_compile - start_compile:.4f}s")

    times = []

    # Temporary directory
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"seapopym_bench_2d_{backend}_"))
    output_zarr = tmp_dir / "output.zarr"

    try:
        # Warmup
        if backend == "jax":
            print("Warming up (JIT compilation)...")
            runner = StreamingRunner(compiled, chunk_size=compiled.n_timesteps)
            runner.run(str(output_zarr))
            shutil.rmtree(tmp_dir)
            tmp_dir = Path(tempfile.mkdtemp(prefix=f"seapopym_bench_2d_{backend}_"))
            output_zarr = tmp_dir / "output.zarr"

        for i in range(iterations):
            # Single chunk for max throughput measurement
            runner = StreamingRunner(compiled, chunk_size=compiled.n_timesteps)

            t0 = time.perf_counter()
            runner.run(str(output_zarr))
            t1 = time.perf_counter()

            times.append(t1 - t0)
            print(f"  Run {i + 1}: {times[-1]:.4f}s")

            shutil.rmtree(output_zarr)

    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"Average execution time: {avg_time:.4f}s ± {std_time:.4f}s")
    return avg_time, compiled


# =============================================================================
# 4. Run Benchmarks
# =============================================================================

if __name__ == "__main__":
    print("Simulation configuration:")
    print(f"  Grid: {grid_size[0]}x{grid_size[1]} ({grid_size[0] * grid_size[1]} cells)")
    print(f"  Steps: {n_timesteps}")
    print(f"  Chunk Size: {n_timesteps} (Single chunk)")

    # Run Numpy
    numpy_time, _ = benchmark_backend("numpy", iterations=5)

    # Run JAX
    jax_time, jax_compiled = benchmark_backend("jax", iterations=5)

    print("\n" + "=" * 40)
    print(f"RESULTS SUMMARY (2D {grid_size[0]}x{grid_size[1]})")
    print("=" * 40)
    print(f"Numpy: {numpy_time:.4f}s")
    print(f"JAX  : {jax_time:.4f}s")
    if jax_time < numpy_time:
        print(f"Speedup: {numpy_time / jax_time:.2f}x (JAX is faster)")
    else:
        print(f"Slowdown: {jax_time / numpy_time:.2f}x (Numpy is faster)")
