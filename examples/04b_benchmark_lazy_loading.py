# %% [markdown]
# # Benchmark LMTL: In-memory vs lazy-loaded (Zarr) forcings
#
# Studies the interaction between **simulation chunk size** and **Zarr temporal
# chunk size** for lazy-loaded forcings, plus measures **peak RSS memory**.
#
# Three modes compared:
# - **In-memory**: forcings are numpy arrays in RAM (baseline)
# - **Zarr (unaligned)**: Zarr stored with default chunking, loaded lazily
# - **Zarr (aligned)**: Zarr temporal chunks match simulation chunk size
#
# Outputs:
# - `examples/images/04b_benchmark_lazy_loading.png` (time comparison)
# - `examples/images/04b_benchmark_lazy_memory.png` (peak RSS)

# %%
import gc
import resource
import shutil
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
from seapopym.engine import simulate
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

GRID_SIDE = 100
SIM_DAYS = 128
DT = "1d"

# Simulation chunk sizes to benchmark
SIM_CHUNK_SIZES = [4, 16, 32, 64, 128]

N_REPEATS = 1

IMAGE_DIR = Path("examples/images")
ZARR_DIR = Path("examples/_tmp_forcing_zarr")

# %% [markdown]
# ## Common Setup

# %%
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
temp_c = 15.0 + 5.0 * np.sin(2 * np.pi * day_of_year / 365.0)
npp_sec = (1.0 + 0.5 * np.sin(2 * np.pi * day_of_year / 365.0)) / 86400.0

common_parameters = {
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

common_initial_state = {
    "biomass": xr.DataArray(np.zeros((1, ny, nx)), dims=["F", "Y", "X"], coords={"Y": lat, "X": lon}),
    "production": xr.DataArray(
        np.zeros((1, n_cohorts, ny, nx)), dims=["F", "C", "Y", "X"], coords={"Y": lat, "X": lon}
    ),
}

common_execution = {
    "time_start": start_date,
    "time_end": end_date,
    "dt": DT,
    "forcing_interpolation": "linear",
}

# %% [markdown]
# ## Build Forcings (in-memory)

# %%
temp_4d = np.broadcast_to(temp_c[:, None, None, None], (len(dates), 1, ny, nx))
temp_da = xr.DataArray(
    temp_4d.copy(), dims=["T", "Z", "Y", "X"],
    coords={"T": dates, "Z": np.arange(1), "Y": lat, "X": lon},
)

npp_3d = np.broadcast_to(npp_sec[:, None, None], (len(dates), ny, nx))
npp_da = xr.DataArray(npp_3d.copy(), dims=["T", "Y", "X"], coords={"T": dates, "Y": lat, "X": lon})

doy_da = xr.DataArray(doy_float, dims=["T"], coords={"T": dates})

static_forcings = {
    "latitude": xr.DataArray(np.full(ny, LATITUDE), dims=["Y"], coords={"Y": lat}),
}

ds_forcings = xr.Dataset({
    "temperature": temp_da,
    "primary_production": npp_da,
    "day_of_year": doy_da,
})

# %% [markdown]
# ## Helpers

# %%
def get_peak_rss_mb() -> float:
    """Get current peak RSS in MB (macOS: bytes, Linux: KB)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS returns bytes, Linux returns KB
    import sys
    if sys.platform == "darwin":
        return usage / 1024**2
    return usage / 1024


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


def load_lazy_forcings(zarr_path: Path) -> dict[str, xr.DataArray]:
    """Load forcings lazily from Zarr."""
    ds = xr.open_zarr(str(zarr_path))
    return {
        **static_forcings,
        "temperature": ds["temperature"],
        "primary_production": ds["primary_production"],
        "day_of_year": ds["day_of_year"],
    }


def compile_with_forcings(forcings: dict[str, xr.DataArray]):
    """Compile model with given forcings."""
    return compile_model(blueprint, Config(
        parameters=common_parameters,
        forcings=forcings,
        initial_state=common_initial_state,
        execution=common_execution,
    ))


def benchmark_run(model, chunk_size: int, n_repeats: int = 1) -> tuple[float, float, float]:
    """Run benchmark. Returns (mean_time, std_time, peak_rss_mb)."""
    times = []
    peak_rss = 0.0
    for i in range(n_repeats + 1):
        gc.collect()
        t0 = time.time()
        simulate(model, chunk_size=chunk_size)
        elapsed = time.time() - t0
        rss = get_peak_rss_mb()

        if i > 0:
            times.append(elapsed)
            peak_rss = max(peak_rss, rss)

    return float(np.mean(times)), float(np.std(times)), peak_rss


# %% [markdown]
# ## Compile In-Memory Model

# %%
memory_forcings = {**static_forcings, **{k: ds_forcings[k] for k in ds_forcings.data_vars}}

print(f"Compiling in-memory model ({ny}x{nx} grid, {SIM_DAYS} days)...")
model_memory = compile_with_forcings(memory_forcings)
n_timesteps = model_memory.n_timesteps
print(f"  {n_timesteps} timesteps, {n_cohorts} cohorts")

# %% [markdown]
# ## Run Benchmark

# %%
print("\n" + "=" * 80)
print("BENCHMARK: In-memory vs Lazy (Zarr) — aligned vs unaligned chunking")
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
# For aligned: one entry per sim_chunk_size
results_aligned: dict[int, tuple[float, float, float]] = {}

# --- 1. In-memory baseline ---
print("--- In-memory baseline ---")
for cs in SIM_CHUNK_SIZES:
    n_chunks = n_timesteps // cs + (1 if n_timesteps % cs else 0)
    print(f"  chunk={cs:>4d} ({n_chunks:>3d} chunks)...", end=" ", flush=True)
    mean, std, rss = benchmark_run(model_memory, cs, N_REPEATS)
    results["memory"][cs] = (mean, std, rss)
    print(f"{mean:.4f}s  RSS={rss:.0f}MB")

# --- 2. Zarr with default chunking ---
print("\n--- Zarr (default chunking) ---")
zarr_default_path = write_zarr(ds_forcings, ZARR_DIR / "default.zarr")
print(f"  Written: {zarr_default_path}")
model_zarr_default = compile_with_forcings(load_lazy_forcings(zarr_default_path))

for cs in SIM_CHUNK_SIZES:
    n_chunks = n_timesteps // cs + (1 if n_timesteps % cs else 0)
    print(f"  chunk={cs:>4d} ({n_chunks:>3d} chunks)...", end=" ", flush=True)
    mean, std, rss = benchmark_run(model_zarr_default, cs, N_REPEATS)
    results["zarr_default"][cs] = (mean, std, rss)
    overhead = mean / results["memory"][cs][0]
    print(f"{mean:.4f}s  RSS={rss:.0f}MB  (x{overhead:.2f})")

# --- 3. Zarr with aligned chunking ---
print("\n--- Zarr (aligned chunking: zarr_T_chunk == sim_chunk) ---")
for cs in SIM_CHUNK_SIZES:
    zarr_aligned_path = write_zarr(ds_forcings, ZARR_DIR / f"aligned_{cs}.zarr", zarr_t_chunk=cs)
    model_aligned = compile_with_forcings(load_lazy_forcings(zarr_aligned_path))

    n_chunks = n_timesteps // cs + (1 if n_timesteps % cs else 0)
    print(f"  chunk={cs:>4d} (zarr_T={cs:>4d})...", end=" ", flush=True)
    mean, std, rss = benchmark_run(model_aligned, cs, N_REPEATS)
    results_aligned[cs] = (mean, std, rss)
    overhead = mean / results["memory"][cs][0]
    print(f"{mean:.4f}s  RSS={rss:.0f}MB  (x{overhead:.2f})")

# %% [markdown]
# ## Summary

# %%
print(f"\n{'='*90}")
print("SUMMARY  (mean time, peak RSS)")
print(f"{'='*90}")

header = (
    f"{'SimChk':>6}  {'Memory':>12}  {'Zarr default':>12}  {'Zarr aligned':>12}"
    f"  {'Ovh(def)':>8}  {'Ovh(aln)':>8}"
)
print(header)
print("-" * len(header))

for cs in SIM_CHUNK_SIZES:
    m_mem = results["memory"][cs][0]
    m_def = results["zarr_default"].get(cs, (0, 0, 0))[0]
    m_aln = results_aligned.get(cs, (0, 0, 0))[0]
    ovh_def = m_def / m_mem if m_mem > 0 else 0
    ovh_aln = m_aln / m_mem if m_mem > 0 else 0
    print(
        f"{cs:>6d}  {m_mem:>10.4f}s  {m_def:>10.4f}s  {m_aln:>10.4f}s"
        f"  x{ovh_def:>6.2f}  x{ovh_aln:>6.2f}"
    )

print(f"\n{'RSS (MB)':>6}  ", end="")
for label in ["memory", "zarr_default"]:
    rss_min = min(v[2] for v in results[label].values()) if results[label] else 0
    rss_max = max(v[2] for v in results[label].values()) if results[label] else 0
    print(f"  {rss_min:.0f}-{rss_max:.0f} MB   ", end="")
if results_aligned:
    rss_min = min(v[2] for v in results_aligned.values())
    rss_max = max(v[2] for v in results_aligned.values())
    print(f"  {rss_min:.0f}-{rss_max:.0f} MB", end="")
print()

# %% [markdown]
# ## Visualization

# %%
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# --- Plot 1: Execution time ---
fig, ax = plt.subplots(figsize=(9, 5))

cs_list = SIM_CHUNK_SIZES

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

plot_path = IMAGE_DIR / "04b_benchmark_lazy_loading.png"
plt.savefig(str(plot_path), dpi=150)
print(f"\nPlot saved: {plot_path}")
plt.close(fig)

# --- Plot 2: Peak RSS ---
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(cs_list, [results["memory"][cs][2] for cs in cs_list],
        "o-", label="In-memory", color="tab:blue", linewidth=2, markersize=8)
ax.plot(cs_list, [results["zarr_default"][cs][2] for cs in cs_list],
        "s--", label="Zarr (default)", color="tab:orange", linewidth=2, markersize=8)
ax.plot(cs_list, [results_aligned[cs][2] for cs in cs_list],
        "^-", label="Zarr (aligned)", color="tab:green", linewidth=2, markersize=8)

ax.set_xscale("log", base=2)
ax.set_xlabel("Simulation chunk size")
ax.set_ylabel("Peak RSS (MB)")
ax.set_title(f"LMTL — Peak memory usage ({ny}x{nx} grid, {SIM_DAYS} days)")
ax.set_xticks(cs_list)
ax.set_xticklabels([str(cs) for cs in cs_list])
ax.legend()
ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()

rss_path = IMAGE_DIR / "04b_benchmark_lazy_memory.png"
plt.savefig(str(rss_path), dpi=150)
print(f"Plot saved: {rss_path}")
plt.close(fig)

# %% [markdown]
# ## Cleanup

# %%
print(f"\nCleaning up {ZARR_DIR}...")
shutil.rmtree(ZARR_DIR)
print("Done.")
