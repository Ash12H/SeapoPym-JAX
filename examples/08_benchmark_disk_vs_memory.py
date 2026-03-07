# %% [markdown]
# # Benchmark: DiskWriter vs MemoryWriter
#
# Fixes temporal parameters (`SIM_DAYS`, `CHUNK_SIZE`) and varies the spatial
# grid size, comparing execution speed, peak host RAM, and numerical
# correctness between the two output backends.
#
# Output: `examples/images/08_benchmark_disk_vs_memory.png`

# %%
import gc
import shutil
import tempfile
import time
import tracemalloc
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import seapopym.functions.lmtl  # noqa: F401
from seapopym.blueprint import Config
from seapopym.compiler import compile_model
from seapopym.engine.runner import Runner
from seapopym.models import LMTL_NO_TRANSPORT

# %% [markdown]
# ## Configuration

# %%
TRUE_PARAMS = {
    "lambda_0": 1 / 150 / 86400,
    "gamma_lambda": 0.15,
    "tau_r_0": 10.38 * 86400,
    "gamma_tau_r": 0.11,
    "efficiency": 0.1668,
    "t_ref": 0.0,
}

LATITUDE = 30.0
CHUNK_SIZE = 32
SIM_DAYS = 64
DT = "1d"
N_REPEATS = 3

GRID_SIDES = [5, 10, 20, 50, 100]

IMAGE_DIR = Path("examples/images")

# %% [markdown]
# ## Build Model

# %%
def build_model(grid_side: int):
    """Build a compiled LMTL_NO_TRANSPORT model for a given spatial size."""
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

    ny, nx = grid_side, grid_side
    lat, lon = np.arange(ny), np.arange(nx)

    doy_float = day_of_year.astype(float)
    doy_da = xr.DataArray(doy_float, dims=["T"], coords={"T": dates})

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
                "lambda_0": {"value": [TRUE_PARAMS["lambda_0"]]},
                "gamma_lambda": {"value": [TRUE_PARAMS["gamma_lambda"]]},
                "tau_r_0": {"value": [TRUE_PARAMS["tau_r_0"]]},
                "gamma_tau_r": {"value": [TRUE_PARAMS["gamma_tau_r"]]},
                "t_ref": {"value": TRUE_PARAMS["t_ref"]},
                "efficiency": {"value": [TRUE_PARAMS["efficiency"]]},
                "cohort_ages": {"value": cohort_ages_sec.tolist()},
                "day_layer": {"value": [0]},
                "night_layer": {"value": [0]},
            },
            "forcings": {
                "latitude": xr.DataArray(np.full(ny, LATITUDE), dims=["Y"], coords={"Y": lat}),
                "temperature": temp_da,
                "primary_production": npp_da,
                "day_of_year": doy_da,
            },
            "initial_state": {
                "biomass": xr.DataArray(
                    np.zeros((1, ny, nx)), dims=["F", "Y", "X"], coords={"Y": lat, "X": lon}
                ),
                "production": xr.DataArray(
                    np.zeros((1, ny, nx, n_cohorts)), dims=["F", "Y", "X", "C"], coords={"Y": lat, "X": lon}
                ),
            },
            "execution": {
                "time_start": start_date,
                "time_end": end_date,
                "dt": DT,
                "forcing_interpolation": "linear",
            },
        }
    )

    return compile_model(LMTL_NO_TRANSPORT, config)


# %% [markdown]
# ## Benchmark Functions

# %%
def benchmark_memory_writer(model, n_repeats: int) -> dict:
    """Benchmark MemoryWriter across n_repeats (first iteration is warmup)."""
    times = []
    ram_peaks = []
    biomass = None

    for i in range(n_repeats + 1):
        gc.collect()
        runner = Runner.simulation(chunk_size=CHUNK_SIZE, output="memory")

        tracemalloc.start()
        t0 = time.time()
        final_state, ds = runner.run(model, export_variables=["biomass"])
        jax.block_until_ready(final_state)
        elapsed = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if i == 0:
            # warmup — discard timing but keep biomass for correctness
            biomass = np.asarray(ds["biomass"].values)
            continue

        times.append(elapsed)
        ram_peaks.append(peak / 1024 / 1024)  # bytes -> MB
        if biomass is None:
            biomass = np.asarray(ds["biomass"].values)

    return {
        "time_mean": float(np.mean(times)),
        "time_std": float(np.std(times)),
        "ram_peak_mean": float(np.mean(ram_peaks)),
        "ram_peak_std": float(np.std(ram_peaks)),
        "biomass": biomass,
    }


def benchmark_disk_writer(model, n_repeats: int) -> dict:
    """Benchmark DiskWriter across n_repeats (first iteration is warmup)."""
    times = []
    ram_peaks = []
    biomass = None

    for i in range(n_repeats + 1):
        gc.collect()
        tmp_dir = tempfile.mkdtemp(prefix="seapopym_bench_")
        try:
            runner = Runner.simulation(chunk_size=CHUNK_SIZE, output="disk")

            tracemalloc.start()
            t0 = time.time()
            final_state, _ = runner.run(model, output_path=tmp_dir, export_variables=["biomass"])
            jax.block_until_ready(final_state)
            elapsed = time.time() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            if i == 0:
                # warmup — read back for correctness via zarr directly
                import zarr

                store = zarr.open(tmp_dir, mode="r")
                biomass = np.asarray(store["biomass"][:])
                continue

            times.append(elapsed)
            ram_peaks.append(peak / 1024 / 1024)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "time_mean": float(np.mean(times)),
        "time_std": float(np.std(times)),
        "ram_peak_mean": float(np.mean(ram_peaks)),
        "ram_peak_std": float(np.std(ram_peaks)),
        "biomass": biomass,
    }


# %% [markdown]
# ## Main Loop

# %%
print("=" * 70)
print("BENCHMARK: DiskWriter vs MemoryWriter")
print("=" * 70)
print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")
print(f"SIM_DAYS={SIM_DAYS}, CHUNK_SIZE={CHUNK_SIZE}, N_REPEATS={N_REPEATS}")
print(f"Grid sides: {GRID_SIDES}")
print()

results = []

for grid_side in GRID_SIDES:
    n_cells = grid_side * grid_side
    print(f"\n--- grid_side={grid_side} ({n_cells} cells) ---")
    print(f"  Compiling model...", end=" ", flush=True)
    model = build_model(grid_side)
    print("done")

    row = {"grid_side": grid_side, "cells": n_cells}

    # Memory writer
    print(f"  MemoryWriter...", end=" ", flush=True)
    try:
        mem = benchmark_memory_writer(model, N_REPEATS)
        row["mem_time_mean"] = mem["time_mean"]
        row["mem_time_std"] = mem["time_std"]
        row["mem_ram_mean"] = mem["ram_peak_mean"]
        row["mem_ram_std"] = mem["ram_peak_std"]
        bio_mem = mem["biomass"]
        print(f"{mem['time_mean']:.4f}s +/- {mem['time_std']:.4f}s, "
              f"RAM {mem['ram_peak_mean']:.1f} MB")
    except Exception as e:
        print(f"FAILED: {e}")
        bio_mem = None
        row["mem_time_mean"] = np.nan
        row["mem_time_std"] = np.nan
        row["mem_ram_mean"] = np.nan
        row["mem_ram_std"] = np.nan

    # Disk writer
    print(f"  DiskWriter...", end=" ", flush=True)
    try:
        disk = benchmark_disk_writer(model, N_REPEATS)
        row["disk_time_mean"] = disk["time_mean"]
        row["disk_time_std"] = disk["time_std"]
        row["disk_ram_mean"] = disk["ram_peak_mean"]
        row["disk_ram_std"] = disk["ram_peak_std"]
        bio_disk = disk["biomass"]
        print(f"{disk['time_mean']:.4f}s +/- {disk['time_std']:.4f}s, "
              f"RAM {disk['ram_peak_mean']:.1f} MB")
    except Exception as e:
        print(f"FAILED: {e}")
        bio_disk = None
        row["disk_time_mean"] = np.nan
        row["disk_time_std"] = np.nan
        row["disk_ram_mean"] = np.nan
        row["disk_ram_std"] = np.nan

    # Correctness
    if bio_mem is not None and bio_disk is not None:
        max_diff = float(np.max(np.abs(bio_mem - bio_disk)))
        correct = np.allclose(bio_mem, bio_disk, atol=1e-6, rtol=1e-5)
        row["correct"] = correct
        row["max_diff"] = max_diff
        label = "PASS" if correct else "FAIL"
        print(f"  Correctness: {label} (max abs diff = {max_diff:.2e})")
    else:
        row["correct"] = None
        row["max_diff"] = np.nan

    results.append(row)

# %% [markdown]
# ## Summary Table

# %%
print(f"\n{'='*90}")
print(f"SUMMARY  (mean +/- std over {N_REPEATS} runs)")
print(f"{'='*90}")

header = (f"{'Grid':>6} {'Cells':>7} | {'Mem Time (s)':>14} {'Disk Time (s)':>14} | "
          f"{'Mem RAM (MB)':>14} {'Disk RAM (MB)':>14} | {'Correct':>7}")
print(header)
print("-" * len(header))

for row in results:
    mt = f"{row['mem_time_mean']:.3f}+/-{row['mem_time_std']:.3f}" if not np.isnan(row["mem_time_mean"]) else "FAILED"
    dt = f"{row['disk_time_mean']:.3f}+/-{row['disk_time_std']:.3f}" if not np.isnan(row["disk_time_mean"]) else "FAILED"
    mr = f"{row['mem_ram_mean']:.1f}+/-{row['mem_ram_std']:.1f}" if not np.isnan(row["mem_ram_mean"]) else "FAILED"
    dr = f"{row['disk_ram_mean']:.1f}+/-{row['disk_ram_std']:.1f}" if not np.isnan(row["disk_ram_mean"]) else "FAILED"
    cor = "PASS" if row["correct"] is True else ("FAIL" if row["correct"] is False else "N/A")
    print(f"{row['grid_side']:>6} {row['cells']:>7} | {mt:>14} {dt:>14} | {mr:>14} {dr:>14} | {cor:>7}")

# %% [markdown]
# ## Visualization

# %%
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

cells = [r["cells"] for r in results]

fig, (ax_time, ax_ram) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

# --- Top: Execution time ---
mem_valid = [r for r in results if not np.isnan(r["mem_time_mean"])]
disk_valid = [r for r in results if not np.isnan(r["disk_time_mean"])]

if mem_valid:
    ax_time.errorbar(
        [r["cells"] for r in mem_valid],
        [r["mem_time_mean"] for r in mem_valid],
        yerr=[r["mem_time_std"] for r in mem_valid],
        fmt="o-", color="tab:blue", label="MemoryWriter",
        linewidth=2, capsize=3, markersize=8,
    )
if disk_valid:
    ax_time.errorbar(
        [r["cells"] for r in disk_valid],
        [r["disk_time_mean"] for r in disk_valid],
        yerr=[r["disk_time_std"] for r in disk_valid],
        fmt="s-", color="tab:red", label="DiskWriter",
        linewidth=2, capsize=3, markersize=8,
    )

ax_time.set_xscale("log")
ax_time.set_ylabel("Execution time (s)")
ax_time.set_title(f"DiskWriter vs MemoryWriter — {SIM_DAYS} days, chunk_size={CHUNK_SIZE}, n={N_REPEATS}")
ax_time.legend()
ax_time.grid(True, alpha=0.3, which="both")

# --- Bottom: Peak host RAM ---
if mem_valid:
    ax_ram.errorbar(
        [r["cells"] for r in mem_valid],
        [r["mem_ram_mean"] for r in mem_valid],
        yerr=[r["mem_ram_std"] for r in mem_valid],
        fmt="o-", color="tab:blue", label="MemoryWriter",
        linewidth=2, capsize=3, markersize=8,
    )
if disk_valid:
    ax_ram.errorbar(
        [r["cells"] for r in disk_valid],
        [r["disk_ram_mean"] for r in disk_valid],
        yerr=[r["disk_ram_std"] for r in disk_valid],
        fmt="s-", color="tab:red", label="DiskWriter",
        linewidth=2, capsize=3, markersize=8,
    )

ax_ram.set_xscale("log")
ax_ram.set_xlabel("Number of grid cells")
ax_ram.set_ylabel("Peak host RAM (MB)")
ax_ram.legend()
ax_ram.grid(True, alpha=0.3, which="both")

fig.tight_layout()

plot_path = IMAGE_DIR / "08_benchmark_disk_vs_memory.png"
plt.savefig(str(plot_path), dpi=150)
print(f"\nPlot saved: {plot_path}")
plt.close(fig)
