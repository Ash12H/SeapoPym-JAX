"""Benchmark LMTL 2D: CPU vs GPU comparison.

Compares execution time of the LMTL model across different grid sizes
on CPU and GPU backends.
"""

import time

import jax
import numpy as np
import pandas as pd
import xarray as xr
from pandas.core.generic import dt

# Import LMTL functions
import seapopym.functions.lmtl  # noqa: F401
from seapopym.blueprint import Blueprint, Config
from seapopym.compiler import compile_model
from seapopym.engine import StreamingRunner

# =============================================================================
# PARAMETERS
# =============================================================================

# LMTL Biological Parameters
LMTL_E = 0.1668
LMTL_LAMBDA_0 = 1 / 150 / 86400.0  # 1/s
LMTL_GAMMA_LAMBDA = 0.15
LMTL_TAU_R_0 = 10.38 * 86400.0  # s
LMTL_GAMMA_TAU_R = 0.11
LMTL_T_REF = 0.0

# Grid sizes to benchmark
GRID_SIZES = [(1, 1), (90, 180), (180, 360), (180 * 4, 360 * 4), (180 * 12, 360 * 12)]

# Simulation parameters
SIM_DAYS = 20  # 1 year
DT = "1d"

# Memory budget for batch_size calculation (in bytes, ~8GB safe for 12GB GPU)
GPU_MEMORY_BUDGET = 8 * 1024**3


def compute_batch_size(grid_size: tuple[int, int], n_cohorts: int) -> int:
    """Compute optimal batch_size based on grid size and available memory.

    Estimates memory per timestep and adjusts batch_size to stay within budget.
    Larger batch_size = better GPU utilization, but must fit in memory.
    """
    ny, nx = grid_size
    n_points = ny * nx

    # Estimate memory per timestep (float32 = 4 bytes)
    # - state.biomass: n_points
    # - state.production: n_points * n_cohorts
    # - forcings (2): 2 * n_points
    # - derived + tendencies: ~3 * n_points * n_cohorts (conservative)
    bytes_per_timestep = 4 * n_points * (1 + n_cohorts + 2 + 3 * n_cohorts)

    # Compute max batch_size that fits in memory budget
    max_batch = int(max(1, int(GPU_MEMORY_BUDGET / bytes_per_timestep)))

    return max_batch


def create_blueprint():
    """Create the LMTL blueprint."""
    max_age_days = int(np.ceil(LMTL_TAU_R_0 / 86400.0))
    cohort_ages_sec = np.arange(0, max_age_days + 1) * 86400.0

    return Blueprint.from_dict(
        {
            "id": "lmtl-benchmark",
            "version": "1.0",
            "declarations": {
                "state": {
                    "biomass": {"units": "g/m^2", "dims": ["Y", "X"]},
                    "production": {"units": "g/m^2", "dims": ["Y", "X", "C"]},
                },
                "parameters": {
                    "lambda_0": {"units": "1/s"},
                    "gamma_lambda": {"units": "1/delta_degC"},
                    "tau_r_0": {"units": "s"},
                    "gamma_tau_r": {"units": "1/delta_degC"},
                    "t_ref": {"units": "degC"},
                    "efficiency": {"units": "dimensionless"},
                    "cohort_ages": {"units": "s", "dims": ["C"]},
                },
                "forcings": {
                    "temperature": {"units": "degC", "dims": ["T", "Y", "X"]},
                    "primary_production": {"units": "g/m^2/s", "dims": ["T", "Y", "X"]},
                },
            },
            "process": [
                {
                    "func": "lmtl:gillooly_temperature",
                    "inputs": {"temp": "forcings.temperature"},
                    "outputs": {"return": {"target": "derived.temp_norm", "type": "derived"}},
                },
                {
                    "func": "lmtl:recruitment_age",
                    "inputs": {
                        "temp": "derived.temp_norm",
                        "tau_r_0": "parameters.tau_r_0",
                        "gamma": "parameters.gamma_tau_r",
                        "t_ref": "parameters.t_ref",
                    },
                    "outputs": {"return": {"target": "derived.rec_age", "type": "derived"}},
                },
                {
                    "func": "lmtl:npp_injection",
                    "inputs": {
                        "npp": "forcings.primary_production",
                        "efficiency": "parameters.efficiency",
                        "production": "state.production",
                    },
                    "outputs": {"return": {"target": "tendencies.production", "type": "tendency"}},
                },
                {
                    "func": "lmtl:aging_flow",
                    "inputs": {
                        "production": "state.production",
                        "cohort_ages": "parameters.cohort_ages",
                        "rec_age": "derived.rec_age",
                    },
                    "outputs": {"return": {"target": "tendencies.production", "type": "tendency"}},
                },
                {
                    "func": "lmtl:recruitment_flow",
                    "inputs": {
                        "production": "state.production",
                        "cohort_ages": "parameters.cohort_ages",
                        "rec_age": "derived.rec_age",
                    },
                    "outputs": {
                        "prod_loss": {"target": "tendencies.production", "type": "tendency"},
                        "biomass_gain": {"target": "tendencies.biomass", "type": "tendency"},
                    },
                },
                {
                    "func": "lmtl:mortality",
                    "inputs": {
                        "biomass": "state.biomass",
                        "temp": "derived.temp_norm",
                        "lambda_0": "parameters.lambda_0",
                        "gamma": "parameters.gamma_lambda",
                        "t_ref": "parameters.t_ref",
                    },
                    "outputs": {"return": {"target": "tendencies.biomass", "type": "tendency"}},
                },
            ],
        }
    ), len(np.arange(0, int(np.ceil(LMTL_TAU_R_0 / 86400.0)) + 1))


def create_config(grid_size, n_cohorts):
    """Create configuration for a given grid size."""
    ny, nx = grid_size

    # Time
    start_date = "2000-01-01"
    end_date = "2000-01-01"
    end_pd = pd.to_datetime(start_date) + pd.Timedelta(days=SIM_DAYS)
    end_date = end_pd.strftime("%Y-%m-%d")

    dates = pd.date_range(start=start_date, periods=SIM_DAYS + 5, freq="D")

    # Grid
    lat = np.arange(ny)
    lon = np.arange(nx)

    # Forcings
    day_of_year = dates.dayofyear.values
    temp_c = 15.0 + 5.0 * np.sin(2 * np.pi * day_of_year / 365.0)
    temp_3d = np.broadcast_to(temp_c[:, None, None], (len(dates), ny, nx))
    temp_da = xr.DataArray(temp_3d.copy(), dims=["T", "Y", "X"], coords={"T": dates, "Y": lat, "X": lon})

    npp_day = 1.0 + 0.5 * np.sin(2 * np.pi * day_of_year / 365.0)
    npp_sec = npp_day / 86400.0
    npp_3d = np.broadcast_to(npp_sec[:, None, None], (len(dates), ny, nx))
    npp_da = xr.DataArray(npp_3d.copy(), dims=["T", "Y", "X"], coords={"T": dates, "Y": lat, "X": lon})

    cohort_ages_sec = np.arange(0, n_cohorts) * 86400.0

    # Compute optimal batch_size for this grid
    batch_size = compute_batch_size(grid_size, n_cohorts)

    return Config.from_dict(
        {
            "parameters": {
                "lambda_0": {"value": LMTL_LAMBDA_0},
                "gamma_lambda": {"value": LMTL_GAMMA_LAMBDA},
                "tau_r_0": {"value": LMTL_TAU_R_0},
                "gamma_tau_r": {"value": LMTL_GAMMA_TAU_R},
                "t_ref": {"value": LMTL_T_REF},
                "efficiency": {"value": LMTL_E},
                "cohort_ages": xr.DataArray(cohort_ages_sec, dims=["C"]),
            },
            "forcings": {"temperature": temp_da, "primary_production": npp_da},
            "initial_state": {
                "biomass": xr.DataArray(np.zeros((ny, nx)), dims=["Y", "X"], coords={"Y": lat, "X": lon}),
                "production": xr.DataArray(
                    np.zeros((ny, nx, n_cohorts)), dims=["Y", "X", "C"], coords={"Y": lat, "X": lon}
                ),
            },
            "execution": {
                "time_start": start_date,
                "time_end": end_date,
                "dt": DT,
                "forcing_interpolation": "linear",
                "batch_size": batch_size,
            },
        }
    )


def run_benchmark(grid_size, device, warmup: bool = True):
    """Run benchmark for a specific grid size and device.

    Args:
        grid_size: Tuple (ny, nx) for grid dimensions.
        device: "cpu" or "gpu".
        warmup: If True, run a short warmup to trigger JIT compilation
                before timing the actual run.
    """
    ny, nx = grid_size
    n_points = ny * nx

    # Force device
    if device == "cpu":
        jax.config.update("jax_default_device", jax.devices("cpu")[0])
    else:
        jax.config.update("jax_default_device", jax.devices("gpu")[0])

    # Create model
    blueprint, n_cohorts = create_blueprint()
    config = create_config(grid_size, n_cohorts)

    # Compile
    t_compile_start = time.time()
    model = compile_model(blueprint, config, backend="jax")
    t_compile = time.time() - t_compile_start

    # Warmup run (triggers JIT compilation, not timed)
    if warmup:
        runner_warmup = StreamingRunner(model)
        _, warmup_out = runner_warmup.run(export_variables=["biomass"])
        jax.block_until_ready(warmup_out["biomass"].values)
        del runner_warmup, warmup_out

    # Timed run
    runner = StreamingRunner(model)
    t_run_start = time.time()
    state, outputs = runner.run(export_variables=["biomass"])
    # Block until computation is done
    jax.block_until_ready(outputs["biomass"].values)
    t_run = time.time() - t_run_start

    n_timesteps = len(outputs["biomass"].coords["T"])
    batch_size = compute_batch_size(grid_size, n_cohorts)

    return {
        "grid_size": f"{ny}x{nx}",
        "n_points": n_points,
        "device": device,
        "batch_size": batch_size,
        "compile_time": t_compile,
        "run_time": t_run,
        "n_timesteps": n_timesteps,
        "timesteps_per_sec": n_timesteps / t_run,
        "points_timesteps_per_sec": n_points * n_timesteps / t_run,
    }


def main():
    print("=" * 80)
    print("LMTL 2D BENCHMARK: CPU vs GPU")
    print("=" * 80)
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Simulation: {SIM_DAYS} days, dt={DT}")
    print()

    # Check if GPU is available
    try:
        gpu_devices = jax.devices("gpu")
        has_gpu = len(gpu_devices) > 0
        if has_gpu:
            print(f"GPU detected: {gpu_devices[0]}")
    except RuntimeError:
        has_gpu = False
        print("No GPU detected, running CPU-only benchmark")

    results = []

    for grid_size in GRID_SIZES:
        ny, nx = grid_size
        print(f"\n--- Grid: {ny}x{nx} ({ny * nx} points) ---")

        # CPU benchmark
        print("  Running CPU...", end=" ", flush=True)
        try:
            res_cpu = run_benchmark(grid_size, "cpu")
            print(f"{res_cpu['run_time']:.2f}s")
            results.append(res_cpu)
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        # GPU benchmark
        if has_gpu:
            print("  Running GPU...", end=" ", flush=True)
            try:
                res_gpu = run_benchmark(grid_size, "gpu")
                print(f"{res_gpu['run_time']:.2f}s")
                results.append(res_gpu)

                # Speedup
                speedup = res_cpu["run_time"] / res_gpu["run_time"]
                print(f"  Speedup: {speedup:.1f}x")
            except Exception as e:
                print(f"FAILED: {e}")

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    header = (
        f"{'Grid':<12} {'Points':<10} {'Batch':<8} {'Device':<8} "
        f"{'Compile':<10} {'Run':<10} {'Steps/s':<12} {'Pts*Steps/s':<15}"
    )
    print(header)
    print("-" * 90)

    for r in results:
        print(
            f"{r['grid_size']:<12} {r['n_points']:<10} {r['batch_size']:<8} {r['device']:<8} "
            f"{r['compile_time']:<10.2f} {r['run_time']:<10.2f} "
            f"{r['timesteps_per_sec']:<12.0f} {r['points_timesteps_per_sec']:<15.0f}"
        )

    # Speedup summary
    if has_gpu:
        print("\n" + "-" * 80)
        print("GPU SPEEDUP")
        print("-" * 80)

        cpu_results = {r["grid_size"]: r for r in results if r["device"] == "cpu"}
        gpu_results = {r["grid_size"]: r for r in results if r["device"] == "gpu"}

        for grid in cpu_results:
            if grid in gpu_results:
                speedup = cpu_results[grid]["run_time"] / gpu_results[grid]["run_time"]
                print(f"  {grid}: {speedup:.1f}x faster on GPU")


if __name__ == "__main__":
    main()
