"""Benchmark GPU memory usage for LMTL 2D.

Profiles VRAM consumption to find optimal batch_size for different grid sizes.
"""

import gc
import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr

# Import LMTL functions
import seapopym.functions.lmtl  # noqa: F401
from seapopym.blueprint import Blueprint, Config
from seapopym.compiler import compile_model
from seapopym.engine import StreamingRunner

# =============================================================================
# PARAMETERS
# =============================================================================

LMTL_E = 0.1668
LMTL_LAMBDA_0 = 1 / 150 / 86400.0
LMTL_GAMMA_LAMBDA = 0.15
LMTL_TAU_R_0 = 10.38 * 86400.0
LMTL_GAMMA_TAU_R = 0.11
LMTL_T_REF = 0.0

# Test configurations
GRID_SIZES = [(90, 180), (180, 360), (360, 720)]
BATCH_SIZES = [50, 100, 200, 500, 1000]
SIM_DAYS = 365
DT = "6h"


def get_gpu_memory_info():
    """Get GPU memory usage using nvidia-ml-py or fallback."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "total_mb": info.total / 1024**2,
            "used_mb": info.used / 1024**2,
            "free_mb": info.free / 1024**2,
        }
    except ImportError:
        # Fallback: estimate from JAX
        return None


def estimate_memory_usage(grid_size, batch_size, n_cohorts=11, dtype_bytes=4):
    """Estimate theoretical GPU memory usage per chunk."""
    ny, nx = grid_size
    n_points = ny * nx

    # Per-timestep memory
    # State: biomass (Y,X) + production (Y,X,C)
    state_size = n_points * dtype_bytes + n_points * n_cohorts * dtype_bytes

    # Forcings per timestep: temperature (Y,X) + npp (Y,X)
    forcing_size = 2 * n_points * dtype_bytes

    # Outputs accumulated in lax.scan (same as state for each timestep)
    output_per_step = state_size

    # Total for chunk
    chunk_memory = (
        state_size +  # Current state
        forcing_size * batch_size +  # Forcings for chunk
        output_per_step * batch_size  # Accumulated outputs
    )

    return {
        "state_mb": state_size / 1024**2,
        "forcings_mb": (forcing_size * batch_size) / 1024**2,
        "outputs_mb": (output_per_step * batch_size) / 1024**2,
        "total_mb": chunk_memory / 1024**2,
    }


def create_blueprint():
    """Create LMTL blueprint."""
    max_age_days = int(np.ceil(LMTL_TAU_R_0 / 86400.0))
    return Blueprint.from_dict({
        "id": "lmtl-memory-test",
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
            {"func": "lmtl:gillooly_temperature", "inputs": {"temp": "forcings.temperature"}, "outputs": {"return": {"target": "derived.temp_norm", "type": "derived"}}},
            {"func": "lmtl:recruitment_age", "inputs": {"temp": "derived.temp_norm", "tau_r_0": "parameters.tau_r_0", "gamma": "parameters.gamma_tau_r", "t_ref": "parameters.t_ref"}, "outputs": {"return": {"target": "derived.rec_age", "type": "derived"}}},
            {"func": "lmtl:npp_injection", "inputs": {"npp": "forcings.primary_production", "efficiency": "parameters.efficiency", "production": "state.production"}, "outputs": {"return": {"target": "tendencies.production", "type": "tendency"}}},
            {"func": "lmtl:aging_flow", "inputs": {"production": "state.production", "cohort_ages": "parameters.cohort_ages", "rec_age": "derived.rec_age"}, "outputs": {"return": {"target": "tendencies.production", "type": "tendency"}}},
            {"func": "lmtl:recruitment_flow", "inputs": {"production": "state.production", "cohort_ages": "parameters.cohort_ages", "rec_age": "derived.rec_age"}, "outputs": {"prod_loss": {"target": "tendencies.production", "type": "tendency"}, "biomass_gain": {"target": "tendencies.biomass", "type": "tendency"}}},
            {"func": "lmtl:mortality", "inputs": {"biomass": "state.biomass", "temp": "derived.temp_norm", "lambda_0": "parameters.lambda_0", "gamma": "parameters.gamma_lambda", "t_ref": "parameters.t_ref"}, "outputs": {"return": {"target": "tendencies.biomass", "type": "tendency"}}},
        ],
    }), max_age_days + 1


def create_config(grid_size, n_cohorts, batch_size):
    """Create configuration."""
    ny, nx = grid_size
    start_date = "2000-01-01"
    end_pd = pd.to_datetime(start_date) + pd.Timedelta(days=SIM_DAYS)
    end_date = end_pd.strftime("%Y-%m-%d")
    dates = pd.date_range(start=start_date, periods=SIM_DAYS + 5, freq="D")

    lat, lon = np.arange(ny), np.arange(nx)
    day_of_year = dates.dayofyear.values

    temp_c = 15.0 + 5.0 * np.sin(2 * np.pi * day_of_year / 365.0)
    temp_3d = np.broadcast_to(temp_c[:, None, None], (len(dates), ny, nx)).copy()
    temp_da = xr.DataArray(temp_3d, dims=["T", "Y", "X"], coords={"T": dates, "Y": lat, "X": lon})

    npp_sec = (1.0 + 0.5 * np.sin(2 * np.pi * day_of_year / 365.0)) / 86400.0
    npp_3d = np.broadcast_to(npp_sec[:, None, None], (len(dates), ny, nx)).copy()
    npp_da = xr.DataArray(npp_3d, dims=["T", "Y", "X"], coords={"T": dates, "Y": lat, "X": lon})

    cohort_ages_sec = np.arange(0, n_cohorts) * 86400.0

    return Config.from_dict({
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
            "production": xr.DataArray(np.zeros((ny, nx, n_cohorts)), dims=["Y", "X", "C"], coords={"Y": lat, "X": lon}),
        },
        "execution": {
            "time_start": start_date,
            "time_end": end_date,
            "dt": DT,
            "forcing_interpolation": "linear",
            "batch_size": batch_size,
        },
    })


def clear_gpu_memory():
    """Clear GPU memory."""
    gc.collect()
    jax.clear_caches()
    # Force synchronization
    try:
        jnp.zeros(1).block_until_ready()
    except Exception:
        pass


def test_configuration(grid_size, batch_size, n_cohorts):
    """Test if a configuration fits in GPU memory."""
    clear_gpu_memory()

    try:
        blueprint, _ = create_blueprint()
        config = create_config(grid_size, n_cohorts, batch_size)
        model = compile_model(blueprint, config, backend="jax")

        runner = StreamingRunner(model, chunk_size=batch_size)

        t_start = time.time()
        state, outputs = runner.run(export_variables=["biomass"])
        jax.block_until_ready(outputs["biomass"].values)
        t_elapsed = time.time() - t_start

        return True, t_elapsed

    except Exception as e:
        error_msg = str(e).lower()
        if "out of memory" in error_msg or "oom" in error_msg or "resource" in error_msg:
            return False, None
        raise


def main():
    print("=" * 80)
    print("GPU MEMORY BENCHMARK - LMTL 2D")
    print("=" * 80)

    # Check GPU
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        print(f"GPU Total Memory: {gpu_info['total_mb']:.0f} MB")
        print(f"GPU Free Memory:  {gpu_info['free_mb']:.0f} MB")
    else:
        print("GPU memory monitoring not available (install pynvml)")

    print(f"\nSimulation: {SIM_DAYS} days, dt={DT}")
    print()

    # Theoretical estimates
    print("=" * 80)
    print("THEORETICAL MEMORY ESTIMATES (per chunk)")
    print("=" * 80)
    print(f"{'Grid':<12} {'Batch':<8} {'State':<10} {'Forcings':<12} {'Outputs':<12} {'Total':<12}")
    print("-" * 80)

    for grid_size in GRID_SIZES:
        for batch_size in BATCH_SIZES:
            est = estimate_memory_usage(grid_size, batch_size)
            print(f"{grid_size[0]}x{grid_size[1]:<6} {batch_size:<8} "
                  f"{est['state_mb']:<10.1f} {est['forcings_mb']:<12.1f} "
                  f"{est['outputs_mb']:<12.1f} {est['total_mb']:<12.1f}")

    # Actual tests
    print("\n" + "=" * 80)
    print("ACTUAL GPU TESTS")
    print("=" * 80)

    results = []
    n_cohorts = int(np.ceil(LMTL_TAU_R_0 / 86400.0)) + 1

    for grid_size in GRID_SIZES:
        print(f"\n--- Grid: {grid_size[0]}x{grid_size[1]} ---")

        max_working_batch = None

        for batch_size in BATCH_SIZES:
            est = estimate_memory_usage(grid_size, batch_size, n_cohorts)
            print(f"  Batch {batch_size}: Est. {est['total_mb']:.0f} MB ... ", end="", flush=True)

            success, elapsed = test_configuration(grid_size, batch_size, n_cohorts)

            if success:
                print(f"OK ({elapsed:.2f}s)")
                max_working_batch = batch_size
                results.append({
                    "grid": f"{grid_size[0]}x{grid_size[1]}",
                    "batch_size": batch_size,
                    "estimated_mb": est["total_mb"],
                    "elapsed_s": elapsed,
                    "status": "OK",
                })
            else:
                print("OOM")
                results.append({
                    "grid": f"{grid_size[0]}x{grid_size[1]}",
                    "batch_size": batch_size,
                    "estimated_mb": est["total_mb"],
                    "elapsed_s": None,
                    "status": "OOM",
                })
                # Skip larger batches for this grid
                break

        if max_working_batch:
            print(f"  => Max working batch_size: {max_working_batch}")

    # Summary
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR 12 GB VRAM")
    print("=" * 80)

    for grid_size in GRID_SIZES:
        grid_key = f"{grid_size[0]}x{grid_size[1]}"
        grid_results = [r for r in results if r["grid"] == grid_key and r["status"] == "OK"]

        if grid_results:
            best = max(grid_results, key=lambda x: x["batch_size"])
            print(f"  {grid_key}: batch_size={best['batch_size']} ({best['elapsed_s']:.2f}s)")
        else:
            print(f"  {grid_key}: Grid too large for GPU")


if __name__ == "__main__":
    main()
