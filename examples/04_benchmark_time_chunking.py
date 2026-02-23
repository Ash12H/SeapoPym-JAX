"""Benchmark LMTL: GPU execution time vs temporal chunk size.

Fixes a spatial grid and total simulation length, then varies the number
of timesteps processed per jax.lax.scan call (chunk size) using StreamingRunner.

Outputs:
- examples/images/04_benchmark_time_chunking.png
"""

import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import seapopym.functions.lmtl  # noqa: F401
from seapopym.blueprint import Config
from seapopym.compiler import compile_model
from seapopym.engine.runners import StreamingRunner
from seapopym.models import LMTL_NO_TRANSPORT

# =============================================================================
# CONFIGURATION
# =============================================================================

TRUE_PARAMS = {
    "lambda_0": 1 / 150 / 86400,
    "gamma_lambda": 0.15,
    "tau_r_0": 10.38 * 86400,
    "gamma_tau_r": 0.11,
    "efficiency": 0.1668,
    "t_ref": 0.0,
}

LATITUDE = 30.0

# Grid size
GRID_SIDE = 10

# Total simulation: use a power of 2 so all chunk sizes divide evenly
SIM_DAYS = 32
DT = "1d"

# Chunk sizes to benchmark (must divide SIM_DAYS)
CHUNK_SIZES = [1, 2, 4, 8, 16, 32]

N_REPEATS = 1

IMAGE_DIR = Path("examples/images")

# =============================================================================
# BUILD MODEL
# =============================================================================

blueprint = LMTL_NO_TRANSPORT

max_age_days = int(np.ceil(TRUE_PARAMS["tau_r_0"] / 86400))
cohort_ages_sec = np.arange(0, max_age_days + 1) * 86400.0
n_cohorts = len(cohort_ages_sec)

start_date = "2000-01-01"
end_date = str((pd.Timestamp(start_date) + pd.DateOffset(days=SIM_DAYS)).date())
start_pd = pd.to_datetime(start_date)
end_pd = pd.to_datetime(end_date)
n_days = (end_pd - start_pd).days + 5
dates = pd.date_range(start=start_pd, periods=n_days, freq="D")
day_of_year = dates.dayofyear.values

ny, nx = GRID_SIDE, GRID_SIDE
lat, lon = np.arange(ny), np.arange(nx)

doy_float = day_of_year.astype(float)
doy_3d = np.broadcast_to(doy_float[:, None, None], (len(dates), ny, nx))
doy_da = xr.DataArray(doy_3d, dims=["T", "Y", "X"], coords={"T": dates, "Y": lat, "X": lon})

temp_c = 15.0 + 5.0 * np.sin(2 * np.pi * day_of_year / 365.0)
temp_4d = np.broadcast_to(temp_c[:, None, None, None], (len(dates), 1, ny, nx))
temp_da = xr.DataArray(
    temp_4d, dims=["T", "Z", "Y", "X"], coords={"T": dates, "Z": np.arange(1), "Y": lat, "X": lon}
)

npp_sec = (1.0 + 0.5 * np.sin(2 * np.pi * day_of_year / 365.0)) / 86400.0
npp_3d = np.broadcast_to(npp_sec[:, None, None], (len(dates), ny, nx))
npp_da = xr.DataArray(npp_3d, dims=["T", "Y", "X"], coords={"T": dates, "Y": lat, "X": lon})

config = Config.from_dict(
    {
        "parameters": {
            "lambda_0": {"value": TRUE_PARAMS["lambda_0"]},
            "gamma_lambda": {"value": TRUE_PARAMS["gamma_lambda"]},
            "tau_r_0": {"value": TRUE_PARAMS["tau_r_0"]},
            "gamma_tau_r": {"value": TRUE_PARAMS["gamma_tau_r"]},
            "t_ref": {"value": TRUE_PARAMS["t_ref"]},
            "efficiency": {"value": TRUE_PARAMS["efficiency"]},
            "cohort_ages": xr.DataArray(cohort_ages_sec, dims=["C"]),
            "day_layer": xr.DataArray([0], dims=["F"]),
            "night_layer": xr.DataArray([0], dims=["F"]),
            "latitude": {"value": LATITUDE},
        },
        "forcings": {
            "temperature": temp_da,
            "primary_production": npp_da,
            "day_of_year": doy_da,
        },
        "initial_state": {
            "biomass": xr.DataArray(np.zeros((1, ny, nx)), dims=["F", "Y", "X"], coords={"Y": lat, "X": lon}),
            "production": xr.DataArray(
                np.zeros((1, ny, nx, n_cohorts)), dims=["F", "Y", "X", "C"], coords={"Y": lat, "X": lon}
            ),
        },
        "execution": {
            "time_start": start_date,
            "time_end": end_date,
            "dt": DT,
            "forcing_interpolation": "linear",
            "batch_size": 1000,
        },
    }
)

print(f"Compiling model ({ny}x{nx} grid, {SIM_DAYS} days)...")
model = compile_model(blueprint, config)
n_timesteps = model.n_timesteps
print(f"Model compiled: {n_timesteps} timesteps, {n_cohorts} cohorts")


# =============================================================================
# DATA SIZES
# =============================================================================


def _fmt_size(nbytes: int) -> str:
    if nbytes < 1024:
        return f"{nbytes} B"
    elif nbytes < 1024**2:
        return f"{nbytes / 1024:.1f} KB"
    elif nbytes < 1024**3:
        return f"{nbytes / 1024**2:.1f} MB"
    else:
        return f"{nbytes / 1024**3:.2f} GB"


print(f"\n{'='*60}")
print("DATA SIZES")
print(f"{'='*60}")
total_bytes = 0

print("\n  State:")
for name, arr in model.state.items():
    nb = arr.nbytes
    total_bytes += nb
    print(f"    {name:>20s}  {str(arr.shape):>25s}  {_fmt_size(nb)}")

print("\n  Parameters:")
for name, arr in model.parameters.items():
    nb = arr.nbytes
    total_bytes += nb
    print(f"    {name:>20s}  {str(arr.shape):>25s}  {_fmt_size(nb)}")

print("\n  Forcings (1 step):")
sample_chunk = model.forcings.get_chunk(0, 1)
forcings_per_step = 0
for name, arr in sample_chunk.items():
    nb = arr.nbytes
    forcings_per_step += nb
    print(f"    {name:>20s}  {str(arr.shape):>25s}  {_fmt_size(nb)}/step")

total_forcings = forcings_per_step * n_timesteps
total_bytes += total_forcings
print(f"\n  {'TOTAL':>22s}  {'':>25s}  {_fmt_size(total_bytes)}")
print(f"  {'Forcings/chunk(C)':>22s}  {'':>25s}  C × {_fmt_size(forcings_per_step)}")
print()


# =============================================================================
# BENCHMARK
# =============================================================================


def benchmark_chunk(chunk_size: int, device_name: str, n_repeats: int = 1) -> tuple[float, float]:
    """Benchmark the full simulation via StreamingRunner on a given device.

    Returns (mean_time, std_time) in seconds.
    """
    dev = jax.devices(device_name)[0]
    times = []
    for i in range(n_repeats + 1):
        with jax.default_device(dev):
            runner = StreamingRunner(model, chunk_size=chunk_size)
            t0 = time.time()
            runner.run()
            elapsed = time.time() - t0

        if i > 0:  # skip first run (warmup / JIT)
            times.append(elapsed)

    return float(np.mean(times)), float(np.std(times))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BENCHMARK: LMTL — execution time vs temporal chunk size")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Grid: {ny}x{nx} ({ny * nx:,} cells), {SIM_DAYS} days, repeats={N_REPEATS}")
    print(f"Chunk sizes: {CHUNK_SIZES}")
    print()

    # Detect devices
    devices = ["cpu"]
    try:
        gpu_devices = jax.devices("gpu")
        if gpu_devices:
            devices.append("gpu")
            print(f"GPU: {gpu_devices[0]}")
    except RuntimeError:
        print("No GPU detected — CPU-only benchmark")

    # Results: {device: {chunk_size: (mean, std)}}
    results: dict[str, dict[int, tuple[float, float]]] = {d: {} for d in devices}

    for cs in CHUNK_SIZES:
        n_chunks = n_timesteps // cs
        remainder = n_timesteps % cs
        total_chunks = n_chunks + (1 if remainder else 0)
        print(f"\n--- chunk_size={cs} ({total_chunks} chunks) ---")

        for dev_name in devices:
            label = dev_name.upper()
            print(f"  {label}...", end=" ", flush=True)
            try:
                mean, std = benchmark_chunk(cs, dev_name, N_REPEATS)
                results[dev_name][cs] = (mean, std)
                print(f"{mean:.4f}s ± {std:.4f}s")
            except Exception as e:
                print(f"FAILED: {e}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY  (mean ± std over {N_REPEATS} runs)")
    print(f"{'='*70}")

    header = f"{'Chunk':>6} {'Chunks':>6}"
    for dev_name in devices:
        header += f"  {dev_name.upper() + ' (s)':>18}"
    print(header)
    print("-" * len(header))

    for cs in CHUNK_SIZES:
        n_chunks = n_timesteps // cs
        remainder = n_timesteps % cs
        total_chunks = n_chunks + (1 if remainder else 0)
        row = f"{cs:>6} {total_chunks:>6}"
        for dev_name in devices:
            if cs in results[dev_name]:
                m, s = results[dev_name][cs]
                row += f"  {m:>8.4f} ± {s:.4f}"
            else:
                row += f"  {'—':>18}"
        print(row)

    # =========================================================================
    # PLOT
    # =========================================================================

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    styles = {"cpu": ("o-", "tab:blue"), "gpu": ("s-", "tab:red")}
    for dev_name in devices:
        if results[dev_name]:
            fmt, color = styles[dev_name]
            cs_list = sorted(results[dev_name].keys())
            means = [results[dev_name][cs][0] for cs in cs_list]
            stds = [results[dev_name][cs][1] for cs in cs_list]
            ax.errorbar(cs_list, means, yerr=stds, fmt=fmt, label=dev_name.upper(),
                         color=color, linewidth=2, capsize=3, markersize=8)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Chunk size (timesteps per lax.scan call)")
    ax.set_ylabel("Execution time (s)")
    ax.set_title(f"LMTL — Chunk size vs execution time ({ny}x{nx} grid, {SIM_DAYS} days, n={N_REPEATS})")
    ax.set_xticks(CHUNK_SIZES)
    ax.set_xticklabels([str(cs) for cs in CHUNK_SIZES])
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    plot_path = IMAGE_DIR / "04_benchmark_time_chunking.png"
    plt.savefig(str(plot_path), dpi=150)
    print(f"\nPlot saved: {plot_path}")
    plt.close(fig)
