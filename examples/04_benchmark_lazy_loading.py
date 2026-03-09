# %% [markdown]
# # Benchmark LMTL: In-memory vs lazy-loaded (Zarr) forcings
#
# Studies the interaction between **simulation chunk size** and **Zarr temporal
# chunk size** for lazy-loaded forcings, plus measures **peak RSS memory**.
#
# Each benchmark point runs in an **isolated subprocess** so that
# `ru_maxrss` (peak RSS) is not polluted by previous runs.
#
# Three modes compared:
# - **In-memory**: forcings are numpy arrays in RAM (baseline)
# - **Zarr (unaligned)**: Zarr stored with default chunking, loaded lazily
# - **Zarr (aligned)**: Zarr temporal chunks match simulation chunk size
#
# Outputs:
# - `examples/images/04_benchmark_lazy_loading.png` (time comparison)
# - `examples/images/04_benchmark_lazy_memory.png` (peak RSS)

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
import pandas as pd
import xarray as xr

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

GRID_SIDE = 1024
SIM_DAYS = 128
DT = "1d"

# Simulation chunk sizes to benchmark
SIM_CHUNK_SIZES = [4, 16, 32, 64, 128]

N_REPEATS = 1

IMAGE_DIR = Path("examples/images")
ZARR_DIR = Path("examples/_tmp_forcing_zarr")

# %% [markdown]
# ## Build & Write Forcings

# %%
max_age_days = int(np.ceil(TRUE_PARAMS["tau_r_0"] / 86400))
n_cohorts = max_age_days + 1

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
temp_c = 15.0 + 5.0 * np.sin(2 * np.pi * day_of_year / 365.0)
npp_sec = (1.0 + 0.5 * np.sin(2 * np.pi * day_of_year / 365.0)) / 86400.0

temp_4d = np.broadcast_to(temp_c[:, None, None, None], (len(dates), 1, ny, nx))
temp_da = xr.DataArray(
    temp_4d.copy(), dims=["T", "Z", "Y", "X"],
    coords={"T": dates, "Z": np.arange(1), "Y": lat, "X": lon},
)

npp_3d = np.broadcast_to(npp_sec[:, None, None], (len(dates), ny, nx))
npp_da = xr.DataArray(npp_3d.copy(), dims=["T", "Y", "X"], coords={"T": dates, "Y": lat, "X": lon})

doy_da = xr.DataArray(doy_float, dims=["T"], coords={"T": dates})

ds_forcings = xr.Dataset({
    "temperature": temp_da,
    "primary_production": npp_da,
    "day_of_year": doy_da,
})


def write_zarr(ds: xr.Dataset, path: Path, zarr_t_chunk: int | None = None) -> Path:
    """Write dataset to Zarr with optional temporal chunking."""
    if path.exists():
        shutil.rmtree(path)
    encoding = {}
    if zarr_t_chunk is not None:
        for var in ds.data_vars:
            dims = ds[var].dims
            chunks = {d: zarr_t_chunk if d == "T" else ds.sizes[d] for d in dims}
            encoding[var] = {"chunks": tuple(chunks[d] for d in dims)}
    ds.to_zarr(str(path), encoding=encoding)
    return path


# Write Zarr stores upfront (shared by all subprocesses)
ZARR_DIR.mkdir(parents=True, exist_ok=True)
zarr_default_path = write_zarr(ds_forcings, ZARR_DIR / "default.zarr")
print(f"Written: {zarr_default_path}")

zarr_aligned_paths = {}
for cs in SIM_CHUNK_SIZES:
    zarr_aligned_paths[cs] = write_zarr(ds_forcings, ZARR_DIR / f"aligned_{cs}.zarr", zarr_t_chunk=cs)
print(f"Written {len(zarr_aligned_paths)} aligned Zarr stores")

# %% [markdown]
# ## Subprocess Benchmark Runner

# %%
# Worker script template — each subprocess imports, compiles, runs, and
# measures peak RSS in complete isolation.
# Uses string.Template ($var) to avoid conflicts with Python dict literals.
_WORKER_TEMPLATE = string.Template(r'''
import gc, json, resource, sys, time
import numpy as np
import pandas as pd
import xarray as xr
import seapopym.functions.lmtl  # noqa: F401
from seapopym.blueprint import Config
from seapopym.compiler import compile_model
from seapopym.engine import run, build_step_fn
from seapopym.models import LMTL_NO_TRANSPORT

class NullWriter:
    """Writer that discards outputs to measure pure forcing memory."""
    def append(self, data): pass
    def finalize(self): return {}
    def close(self): pass

# --- Config (injected) ---
GRID_SIDE = $grid_side
SIM_DAYS = $sim_days
LATITUDE = $latitude
TRUE_PARAMS = $true_params
MODE = "$mode"
CHUNK_SIZE = $chunk_size
N_REPEATS = $n_repeats
ZARR_PATH = $zarr_path

# --- Setup ---
blueprint = LMTL_NO_TRANSPORT
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

parameters = {
    "lambda_0": xr.DataArray([TRUE_PARAMS["lambda_0"]], dims=["F"]),
    "gamma_lambda": xr.DataArray([TRUE_PARAMS["gamma_lambda"]], dims=["F"]),
    "tau_r_0": xr.DataArray([TRUE_PARAMS["tau_r_0"]], dims=["F"]),
    "gamma_tau_r": xr.DataArray([TRUE_PARAMS["gamma_tau_r"]], dims=["F"]),
    "t_ref": xr.DataArray(TRUE_PARAMS["t_ref"]),
    "efficiency": xr.DataArray([TRUE_PARAMS["efficiency"]], dims=["F"]),
    "cohort_ages": xr.DataArray(cohort_ages_sec, dims=["C"]),
    "day_layer": xr.DataArray([0], dims=["F"]),
    "night_layer": xr.DataArray([0], dims=["F"]),
}

initial_state = {
    "biomass": xr.DataArray(np.zeros((1, ny, nx)), dims=["F", "Y", "X"], coords={"Y": lat, "X": lon}),
    "production": xr.DataArray(
        np.zeros((1, n_cohorts, ny, nx)), dims=["F", "C", "Y", "X"], coords={"Y": lat, "X": lon}
    ),
}

execution = {
    "time_start": start_date,
    "time_end": end_date,
    "dt": "1d",
    "forcing_interpolation": "linear",
}

static_forcings = {
    "latitude": xr.DataArray(np.full(ny, LATITUDE), dims=["Y"], coords={"Y": lat}),
}

# --- Build forcings based on mode ---
if MODE == "memory":
    doy_float = day_of_year.astype(float)
    temp_c = 15.0 + 5.0 * np.sin(2 * np.pi * day_of_year / 365.0)
    npp_sec = (1.0 + 0.5 * np.sin(2 * np.pi * day_of_year / 365.0)) / 86400.0
    temp_4d = np.broadcast_to(temp_c[:, None, None, None], (len(dates), 1, ny, nx))
    npp_3d = np.broadcast_to(npp_sec[:, None, None], (len(dates), ny, nx))
    forcings = {
        **static_forcings,
        "temperature": xr.DataArray(temp_4d.copy(), dims=["T", "Z", "Y", "X"],
            coords={"T": dates, "Z": np.arange(1), "Y": lat, "X": lon}),
        "primary_production": xr.DataArray(npp_3d.copy(), dims=["T", "Y", "X"],
            coords={"T": dates, "Y": lat, "X": lon}),
        "day_of_year": xr.DataArray(doy_float, dims=["T"], coords={"T": dates}),
    }
else:
    ds = xr.open_zarr(ZARR_PATH)
    forcings = {
        **static_forcings,
        "temperature": ds["temperature"],
        "primary_production": ds["primary_production"],
        "day_of_year": ds["day_of_year"],
    }

# --- Compile & run (NullWriter: discard outputs to isolate forcing memory) ---
model = compile_model(blueprint, Config(
    parameters=parameters, forcings=forcings,
    initial_state=initial_state, execution=execution,
))
step_fn = build_step_fn(model, export_variables=[])

times = []
for i in range(N_REPEATS + 1):
    gc.collect()
    t0 = time.time()
    run(step_fn, model, dict(model.state), dict(model.parameters),
        chunk_size=CHUNK_SIZE, writer=NullWriter())
    elapsed = time.time() - t0
    if i > 0:
        times.append(elapsed)

rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss_mb = rss / 1024**2 if sys.platform == "darwin" else rss / 1024
print(json.dumps({"mean": float(np.mean(times)), "std": float(np.std(times)), "rss": rss_mb}))
''')


def benchmark_subprocess(
    mode: str, chunk_size: int, n_repeats: int, zarr_path: Path | None = None,
) -> tuple[float, float, float]:
    """Run a single benchmark in an isolated subprocess for accurate peak RSS.

    Returns (mean_time, std_time, peak_rss_mb).
    """
    script = _WORKER_TEMPLATE.substitute(
        grid_side=GRID_SIDE,
        sim_days=SIM_DAYS,
        latitude=LATITUDE,
        true_params=repr(TRUE_PARAMS),
        mode=mode,
        chunk_size=chunk_size,
        n_repeats=n_repeats,
        zarr_path=repr(str(zarr_path)) if zarr_path else "None",
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Worker failed (mode={mode}, chunk={chunk_size}):\n{result.stderr[-2000:]}")
    # Last line of stdout is the JSON result
    data = json.loads(result.stdout.strip().split("\n")[-1])
    return data["mean"], data["std"], data["rss"]


# %% [markdown]
# ## Run Benchmark

# %%
import jax  # noqa: E402 — only for version display in main process

print("\n" + "=" * 80)
print("BENCHMARK: In-memory vs Lazy (Zarr) — isolated subprocess RSS")
print("=" * 80)
print(f"JAX version: {jax.__version__}")
print(f"Grid: {ny}x{nx}, {SIM_DAYS} days, repeats={N_REPEATS}")
print(f"Sim chunk sizes: {SIM_CHUNK_SIZES}")
print()

# Results: {label: {sim_chunk: (mean, std, rss)}}
results: dict[str, dict[int, tuple[float, float, float]]] = {
    "memory": {},
    "zarr_default": {},
}
results_aligned: dict[int, tuple[float, float, float]] = {}

n_timesteps_approx = SIM_DAYS + 1  # approximate for display

# --- 1. In-memory baseline ---
print("--- In-memory baseline ---")
for cs in SIM_CHUNK_SIZES:
    n_chunks = n_timesteps_approx // cs + (1 if n_timesteps_approx % cs else 0)
    print(f"  chunk={cs:>4d} ({n_chunks:>3d} chunks)...", end=" ", flush=True)
    t0 = time.time()
    mean, std, rss = benchmark_subprocess("memory", cs, N_REPEATS)
    wall = time.time() - t0
    results["memory"][cs] = (mean, std, rss)
    print(f"{mean:.4f}s  RSS={rss:.0f}MB  (wall={wall:.1f}s)")

# --- 2. Zarr with default chunking ---
print("\n--- Zarr (default chunking) ---")
for cs in SIM_CHUNK_SIZES:
    n_chunks = n_timesteps_approx // cs + (1 if n_timesteps_approx % cs else 0)
    print(f"  chunk={cs:>4d} ({n_chunks:>3d} chunks)...", end=" ", flush=True)
    t0 = time.time()
    mean, std, rss = benchmark_subprocess("zarr_default", cs, N_REPEATS, zarr_default_path)
    wall = time.time() - t0
    results["zarr_default"][cs] = (mean, std, rss)
    overhead = mean / results["memory"][cs][0]
    print(f"{mean:.4f}s  RSS={rss:.0f}MB  (x{overhead:.2f}, wall={wall:.1f}s)")

# --- 3. Zarr with aligned chunking ---
print("\n--- Zarr (aligned chunking: zarr_T_chunk == sim_chunk) ---")
for cs in SIM_CHUNK_SIZES:
    print(f"  chunk={cs:>4d} (zarr_T={cs:>4d})...", end=" ", flush=True)
    t0 = time.time()
    mean, std, rss = benchmark_subprocess("zarr_aligned", cs, N_REPEATS, zarr_aligned_paths[cs])
    wall = time.time() - t0
    results_aligned[cs] = (mean, std, rss)
    overhead = mean / results["memory"][cs][0]
    print(f"{mean:.4f}s  RSS={rss:.0f}MB  (x{overhead:.2f}, wall={wall:.1f}s)")

# %% [markdown]
# ## Summary

# %%
print(f"\n{'='*90}")
print("SUMMARY  (mean time, isolated peak RSS)")
print(f"{'='*90}")

header = (
    f"{'SimChk':>6}  {'Memory':>12} {'RSS':>6}  {'Zarr default':>12} {'RSS':>6}"
    f"  {'Zarr aligned':>12} {'RSS':>6}  {'Ovh(def)':>8}  {'Ovh(aln)':>8}"
)
print(header)
print("-" * len(header))

for cs in SIM_CHUNK_SIZES:
    m_mem, _, r_mem = results["memory"][cs]
    m_def, _, r_def = results["zarr_default"].get(cs, (0, 0, 0))
    m_aln, _, r_aln = results_aligned.get(cs, (0, 0, 0))
    ovh_def = m_def / m_mem if m_mem > 0 else 0
    ovh_aln = m_aln / m_mem if m_mem > 0 else 0
    print(
        f"{cs:>6d}  {m_mem:>10.4f}s {r_mem:>5.0f}M  {m_def:>10.4f}s {r_def:>5.0f}M"
        f"  {m_aln:>10.4f}s {r_aln:>5.0f}M  x{ovh_def:>6.2f}  x{ovh_aln:>6.2f}"
    )

# %% [markdown]
# ## Visualization

# %%
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

cs_list = SIM_CHUNK_SIZES

# --- Plot 1: Execution time ---
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(cs_list, [results["memory"][cs][0] for cs in cs_list],
        "o-", label="In-memory", color="tab:blue", linewidth=2, markersize=8)
ax.plot(cs_list, [results["zarr_default"][cs][0] for cs in cs_list],
        "s--", label="Zarr (default chunks)", color="tab:orange", linewidth=2, markersize=8)
ax.plot(cs_list, [results_aligned[cs][0] for cs in cs_list],
        "^-", label="Zarr (aligned chunks)", color="tab:green", linewidth=2, markersize=8)

ax.set_xscale("log", base=2)
ax.set_xlabel("Simulation chunk size")
ax.set_ylabel("Execution time (s)")
ax.set_title(f"LMTL — Forcing loading strategy ({ny}x{nx} grid, {SIM_DAYS} days)")
ax.set_xticks(cs_list)
ax.set_xticklabels([str(cs) for cs in cs_list])
ax.legend()
ax.grid(True, alpha=0.3, which="both")
ax.set_ylim(bottom=0)
fig.tight_layout()

plot_path = IMAGE_DIR / "04_benchmark_lazy_loading.png"
plt.savefig(str(plot_path), dpi=150)
print(f"\nPlot saved: {plot_path}")
plt.close(fig)

# --- Plot 2: Peak RSS (isolated per subprocess) ---
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(cs_list, [results["memory"][cs][2] for cs in cs_list],
        "o-", label="In-memory", color="tab:blue", linewidth=2, markersize=8)
ax.plot(cs_list, [results["zarr_default"][cs][2] for cs in cs_list],
        "s--", label="Zarr (default)", color="tab:orange", linewidth=2, markersize=8)
ax.plot(cs_list, [results_aligned[cs][2] for cs in cs_list],
        "^-", label="Zarr (aligned)", color="tab:green", linewidth=2, markersize=8)

ax.set_xscale("log", base=2)
ax.set_xlabel("Simulation chunk size")
ax.set_ylabel("Peak RSS (MB) — isolated subprocess, no output accumulation")
ax.set_title(f"LMTL — Forcing memory footprint ({ny}x{nx} grid, {SIM_DAYS} days)")
ax.set_xticks(cs_list)
ax.set_xticklabels([str(cs) for cs in cs_list])
ax.legend()
ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()

rss_path = IMAGE_DIR / "04_benchmark_lazy_memory.png"
plt.savefig(str(rss_path), dpi=150)
print(f"Plot saved: {rss_path}")
plt.close(fig)

# %% [markdown]
# ## Cleanup

# %%
print(f"\nCleaning up {ZARR_DIR}...")
shutil.rmtree(ZARR_DIR)
print("Done.")
