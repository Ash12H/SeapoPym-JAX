# %% [markdown]
# # Benchmark: MemoryWriter vs DiskWriter (Zarr)
#
# Compares the two output backends on a large grid with **biomass only**:
# 1. **MemoryWriter** — accumulates outputs in RAM, returns xarray Dataset
# 2. **DiskWriter** — streams outputs to Zarr, constant memory footprint
#
# Each benchmark point runs in an **isolated subprocess** for accurate
# peak RSS measurement.
#
# Outputs:
# - `examples/images/04b_benchmark_writers.png`

# %%
import json
import shutil
import string
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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

GRID_SIDE = 512
SIM_DAYS = 365
DT = "1d"

SIM_CHUNK_SIZES = [4, 16, 64, 128, 365]

N_REPEATS = 1

IMAGE_DIR = Path("examples/images")
TMP_DIR = Path("examples/_tmp_writers")

# %% [markdown]
# ## Subprocess Worker Template

# %%
_WORKER_TEMPLATE = string.Template(r'''
import gc, json, resource, shutil, sys, time
import numpy as np
import pandas as pd
import xarray as xr
import seapopym.functions.lmtl  # noqa: F401
from seapopym.blueprint import Config
from seapopym.compiler import compile_model
from seapopym.engine import simulate
from seapopym.models import LMTL_NO_TRANSPORT

# --- Config (injected) ---
GRID_SIDE = $grid_side
SIM_DAYS = $sim_days
LATITUDE = $latitude
TRUE_PARAMS = $true_params
WRITER = "$writer"
CHUNK_SIZE = $chunk_size
N_REPEATS = $n_repeats
OUTPUT_PATH = $output_path

# --- Setup ---
max_age_days = int(np.ceil(TRUE_PARAMS["tau_r_0"] / 86400))
cohort_ages_sec = np.arange(0, max_age_days + 1) * 86400.0
n_cohorts = len(cohort_ages_sec)

start_date = "2000-01-01"
end_date = str((pd.Timestamp(start_date) + pd.DateOffset(days=SIM_DAYS)).date())
dates = pd.date_range(
    start=pd.to_datetime(start_date),
    periods=(pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 5,
    freq="D",
)
day_of_year = dates.dayofyear.values
ny, nx = GRID_SIDE, GRID_SIDE
lat, lon = np.arange(ny), np.arange(nx)

doy_float = day_of_year.astype(float)
temp_c = 15.0 + 5.0 * np.sin(2 * np.pi * day_of_year / 365.0)
npp_sec = (1.0 + 0.5 * np.sin(2 * np.pi * day_of_year / 365.0)) / 86400.0
temp_4d = np.broadcast_to(temp_c[:, None, None, None], (len(dates), 1, ny, nx))
npp_3d = np.broadcast_to(npp_sec[:, None, None], (len(dates), ny, nx))

config = Config(
    parameters={
        "lambda_0": xr.DataArray([TRUE_PARAMS["lambda_0"]], dims=["F"]),
        "gamma_lambda": xr.DataArray([TRUE_PARAMS["gamma_lambda"]], dims=["F"]),
        "tau_r_0": xr.DataArray([TRUE_PARAMS["tau_r_0"]], dims=["F"]),
        "gamma_tau_r": xr.DataArray([TRUE_PARAMS["gamma_tau_r"]], dims=["F"]),
        "t_ref": xr.DataArray(TRUE_PARAMS["t_ref"]),
        "efficiency": xr.DataArray([TRUE_PARAMS["efficiency"]], dims=["F"]),
        "cohort_ages": xr.DataArray(cohort_ages_sec, dims=["C"]),
        "day_layer": xr.DataArray([0], dims=["F"]),
        "night_layer": xr.DataArray([0], dims=["F"]),
    },
    forcings={
        "latitude": xr.DataArray(np.full(ny, LATITUDE), dims=["Y"], coords={"Y": lat}),
        "temperature": xr.DataArray(temp_4d.copy(), dims=["T", "Z", "Y", "X"],
            coords={"T": dates, "Z": np.arange(1), "Y": lat, "X": lon}),
        "primary_production": xr.DataArray(npp_3d.copy(), dims=["T", "Y", "X"],
            coords={"T": dates, "Y": lat, "X": lon}),
        "day_of_year": xr.DataArray(doy_float, dims=["T"], coords={"T": dates}),
    },
    initial_state={
        "biomass": xr.DataArray(
            np.zeros((1, ny, nx)), dims=["F", "Y", "X"], coords={"Y": lat, "X": lon}),
        "production": xr.DataArray(
            np.zeros((1, n_cohorts, ny, nx)), dims=["F", "C", "Y", "X"],
            coords={"Y": lat, "X": lon}),
    },
    execution={
        "time_start": start_date, "time_end": end_date,
        "dt": "1d", "forcing_interpolation": "linear",
    },
)

model = compile_model(LMTL_NO_TRANSPORT, config)

# --- Run ---
times = []
for i in range(N_REPEATS + 1):
    gc.collect()
    # Clean Zarr output between runs
    if WRITER == "disk" and OUTPUT_PATH is not None:
        import shutil as _sh
        _p = __import__("pathlib").Path(OUTPUT_PATH)
        if _p.exists():
            _sh.rmtree(_p)

    t0 = time.time()
    if WRITER == "memory":
        simulate(model, chunk_size=CHUNK_SIZE, export_variables=["biomass"])
    else:
        simulate(model, chunk_size=CHUNK_SIZE, output_path=OUTPUT_PATH,
                 export_variables=["biomass"])
    elapsed = time.time() - t0
    if i > 0:
        times.append(elapsed)

rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss_mb = rss / 1024**2 if sys.platform == "darwin" else rss / 1024
print(json.dumps({"mean": float(np.mean(times)), "std": float(np.std(times)), "rss": rss_mb}))
''')


def benchmark_subprocess(
    writer: str, chunk_size: int, n_repeats: int, output_path: Path | None = None,
) -> tuple[float, float, float]:
    """Run a single benchmark in an isolated subprocess.

    Returns (mean_time, std_time, peak_rss_mb).
    """
    script = _WORKER_TEMPLATE.substitute(
        grid_side=GRID_SIDE,
        sim_days=SIM_DAYS,
        latitude=LATITUDE,
        true_params=repr(TRUE_PARAMS),
        writer=writer,
        chunk_size=chunk_size,
        n_repeats=n_repeats,
        output_path=repr(str(output_path)) if output_path else "None",
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Worker failed (writer={writer}, chunk={chunk_size}):\n{result.stderr[-2000:]}"
        )
    data = json.loads(result.stdout.strip().split("\n")[-1])
    return data["mean"], data["std"], data["rss"]


# %% [markdown]
# ## Run Benchmark

# %%
import jax  # noqa: E402

print("\n" + "=" * 80)
print("BENCHMARK: MemoryWriter vs DiskWriter (Zarr)")
print("=" * 80)
print(f"JAX version: {jax.__version__}")
print(f"Grid: {GRID_SIDE}x{GRID_SIDE}, {SIM_DAYS} days, repeats={N_REPEATS}")
print(f"Export: biomass only (F, Y, X)")
print(f"Chunk sizes: {SIM_CHUNK_SIZES}")

biomass_per_step_mb = 1 * GRID_SIDE * GRID_SIDE * 4 / 1024**2
print(f"Biomass per timestep: {biomass_per_step_mb:.1f} MB")
print()

TMP_DIR.mkdir(parents=True, exist_ok=True)

results_memory: dict[int, tuple[float, float, float]] = {}
results_disk: dict[int, tuple[float, float, float]] = {}

n_timesteps_approx = SIM_DAYS + 1

# --- 1. MemoryWriter ---
print("--- MemoryWriter (in-memory accumulation) ---")
for cs in SIM_CHUNK_SIZES:
    n_chunks = n_timesteps_approx // cs + (1 if n_timesteps_approx % cs else 0)
    print(f"  chunk={cs:>4d} ({n_chunks:>3d} chunks)...", end=" ", flush=True)
    t0 = time.time()
    mean, std, rss = benchmark_subprocess("memory", cs, N_REPEATS)
    wall = time.time() - t0
    results_memory[cs] = (mean, std, rss)
    print(f"{mean:.2f}s  RSS={rss:.0f}MB  (wall={wall:.1f}s)")

# --- 2. DiskWriter ---
print("\n--- DiskWriter (Zarr streaming) ---")
for cs in SIM_CHUNK_SIZES:
    n_chunks = n_timesteps_approx // cs + (1 if n_timesteps_approx % cs else 0)
    print(f"  chunk={cs:>4d} ({n_chunks:>3d} chunks)...", end=" ", flush=True)
    zarr_path = TMP_DIR / f"output_chunk{cs}.zarr"
    t0 = time.time()
    mean, std, rss = benchmark_subprocess("disk", cs, N_REPEATS, zarr_path)
    wall = time.time() - t0
    results_disk[cs] = (mean, std, rss)
    speedup = results_memory[cs][0] / mean if mean > 0 else 0
    rss_ratio = results_memory[cs][2] / rss if rss > 0 else 0
    print(f"{mean:.2f}s  RSS={rss:.0f}MB  (x{speedup:.2f} speed, x{rss_ratio:.1f} RSS, wall={wall:.1f}s)")

# %% [markdown]
# ## Summary

# %%
print(f"\n{'='*80}")
print("SUMMARY  (mean time, isolated peak RSS)")
print(f"{'='*80}")

header = (
    f"{'Chunk':>6}  {'Memory time':>12} {'RSS':>7}  {'Disk time':>12} {'RSS':>7}"
    f"  {'Speed ratio':>11}  {'RSS ratio':>10}"
)
print(header)
print("-" * len(header))

for cs in SIM_CHUNK_SIZES:
    m_t, _, m_r = results_memory[cs]
    d_t, _, d_r = results_disk[cs]
    speed = m_t / d_t if d_t > 0 else 0
    rss_r = m_r / d_r if d_r > 0 else 0
    print(
        f"{cs:>6d}  {m_t:>10.2f}s {m_r:>6.0f}M  {d_t:>10.2f}s {d_r:>6.0f}M"
        f"  {speed:>10.2f}x  {rss_r:>9.1f}x"
    )

# %% [markdown]
# ## Visualization

# %%
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

cs_list = SIM_CHUNK_SIZES

fig, (ax_time, ax_rss) = plt.subplots(1, 2, figsize=(14, 5))

# --- Time ---
ax_time.plot(cs_list, [results_memory[cs][0] for cs in cs_list],
             "o-", label="MemoryWriter", color="tab:blue", linewidth=2, markersize=8)
ax_time.plot(cs_list, [results_disk[cs][0] for cs in cs_list],
             "s-", label="DiskWriter (Zarr)", color="tab:red", linewidth=2, markersize=8)
ax_time.set_xscale("log", base=2)
ax_time.set_xlabel("Simulation chunk size")
ax_time.set_ylabel("Execution time (s)")
ax_time.set_title(f"Execution time — {GRID_SIDE}x{GRID_SIDE}, {SIM_DAYS} days, biomass only")
ax_time.set_xticks(cs_list)
ax_time.set_xticklabels([str(cs) for cs in cs_list])
ax_time.legend()
ax_time.grid(True, alpha=0.3, which="both")
ax_time.set_ylim(bottom=0)

# --- RSS ---
ax_rss.plot(cs_list, [results_memory[cs][2] for cs in cs_list],
            "o-", label="MemoryWriter", color="tab:blue", linewidth=2, markersize=8)
ax_rss.plot(cs_list, [results_disk[cs][2] for cs in cs_list],
            "s-", label="DiskWriter (Zarr)", color="tab:red", linewidth=2, markersize=8)
ax_rss.set_xscale("log", base=2)
ax_rss.set_xlabel("Simulation chunk size")
ax_rss.set_ylabel("Peak RSS (MB) — isolated subprocess")
ax_rss.set_title(f"Peak memory — {GRID_SIDE}x{GRID_SIDE}, {SIM_DAYS} days, biomass only")
ax_rss.set_xticks(cs_list)
ax_rss.set_xticklabels([str(cs) for cs in cs_list])
ax_rss.legend()
ax_rss.grid(True, alpha=0.3, which="both")

fig.tight_layout()

plot_path = IMAGE_DIR / "04b_benchmark_writers.png"
plt.savefig(str(plot_path), dpi=150)
print(f"\nPlot saved: {plot_path}")
plt.close(fig)

# %% [markdown]
# ## Cleanup

# %%
print(f"\nCleaning up {TMP_DIR}...")
shutil.rmtree(TMP_DIR, ignore_errors=True)
print("Done.")
