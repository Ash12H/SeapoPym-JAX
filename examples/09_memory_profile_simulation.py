# %% [markdown]
# # Memory Profiling: LMTL Simulation with Transport
#
# Compares MemoryWriter vs DiskWriter memory profiles on a reduced grid.
# Forcings are written to NetCDF first and opened lazily — simulating the
# real use case where data comes from files, not from in-memory arrays.
#
# Usage:
#     uv run python examples/09_memory_profile_simulation.py

# %%
import gc
import logging
import math
import shutil
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import xarray as xr
from matplotlib.patches import Patch

import seapopym.functions.lmtl  # noqa: F401
import seapopym.functions.transport  # noqa: F401
from seapopym.blueprint import Config
from seapopym.compiler import compile_model
from seapopym.engine.io import DiskWriter, MemoryWriter, resolve_var_dims
from seapopym.engine.runner import _scan
from seapopym.engine.step import build_step_fn
from seapopym.models import LMTL

jax.config.update("jax_default_device", jax.devices("cpu")[0])

# %% [markdown]
# ## Section 1: Configuration

# %%
GRID_SIDE = 200
N_GROUPS = 6
N_LAYERS = 3
CHUNK_SIZE = 1000
SIM_DAYS = 2500
LATITUDE = 30.0
CELL_SIZE = 10_000.0

TARGET_NY, TARGET_NX = 180, 360
SOURCE_FREQ = "10D"

TMP_DIR = Path("examples/_tmp_profile")
FORCING_NC = TMP_DIR / "forcings.nc"
DISK_OUTPUT_PATH = TMP_DIR / "output_zarr"

# %%
NY, NX = GRID_SIDE, GRID_SIDE
scale_factor = (TARGET_NY * TARGET_NX) / (NY * NX)

print(f"Test grid: {NY}x{NX} = {NY * NX} cells")
print(f"Target grid: {TARGET_NY}x{TARGET_NX} = {TARGET_NY * TARGET_NX} cells")
print(f"Scale factor: {scale_factor:.1f}x")
print(f"Groups: {N_GROUPS}, Layers: {N_LAYERS}, Chunk: {CHUNK_SIZE}")

# %% [markdown]
# ## Section 2: Memory Instrumentation

# %%
process = psutil.Process()


def make_snapshotter():
    """Create a fresh snapshot collector with its own baseline."""
    records = []
    baseline = process.memory_info().rss / 1024**2

    def snap(label: str) -> dict:
        gc.collect()
        rss_mb = process.memory_info().rss / 1024**2
        record = {"label": label, "rss_mb": rss_mb, "delta_mb": rss_mb - baseline}
        records.append(record)
        print(f"  [{label:>30s}]  RSS = {rss_mb:8.1f} MB  (delta = {record['delta_mb']:+8.1f} MB)")
        return record

    return snap, records


# %% [markdown]
# ## Section 3: Write Forcings to NetCDF, Then Open Lazily

# %%
max_age_days = 60
cohort_ages_sec = np.arange(0, max_age_days + 1) * 86400.0
n_cohorts = len(cohort_ages_sec)

start_date = "2000-01-01"
end_date = str((pd.Timestamp(start_date) + pd.DateOffset(days=SIM_DAYS)).date())
source_dates = pd.date_range(start=start_date, end=end_date, freq=SOURCE_FREQ)

lat = np.arange(NY, dtype=float)
lon = np.arange(NX, dtype=float)

# --- Build forcing arrays in memory ---
doy = source_dates.dayofyear.values.astype(float)

temp_base = 15.0 + 5.0 * np.sin(2 * np.pi * doy / 365.0)
temp_layers = np.stack([temp_base - i * 3.0 for i in range(N_LAYERS)], axis=1)
temp_4d = np.broadcast_to(
    temp_layers[:, :, None, None], (len(source_dates), N_LAYERS, NY, NX)
).copy()  # .copy() to make contiguous for NetCDF write

npp_sec = (1.0 + 0.5 * np.sin(2 * np.pi * doy / 365.0)) / 86400.0
npp_3d = np.broadcast_to(npp_sec[:, None, None], (len(source_dates), NY, NX)).copy()

u_4d = np.full((len(source_dates), N_LAYERS, NY, NX), 0.02, dtype=np.float32)
v_4d = np.full((len(source_dates), N_LAYERS, NY, NX), 0.01, dtype=np.float32)

# --- Write to NetCDF ---
TMP_DIR.mkdir(parents=True, exist_ok=True)

ds_forcing = xr.Dataset(
    {
        "temperature": (["T", "Z", "Y", "X"], temp_4d),
        "primary_production": (["T", "Y", "X"], npp_3d),
        "day_of_year": (["T"], doy),
        "u": (["T", "Z", "Y", "X"], u_4d),
        "v": (["T", "Z", "Y", "X"], v_4d),
    },
    coords={
        "T": source_dates,
        "Z": np.arange(N_LAYERS),
        "Y": lat,
        "X": lon,
    },
)
ds_forcing.to_netcdf(FORCING_NC)
nc_size_mb = FORCING_NC.stat().st_size / 1024**2
print(f"Wrote forcings to {FORCING_NC} ({nc_size_mb:.1f} MB)")

# --- Free in-memory arrays ---
del ds_forcing, temp_4d, temp_layers, temp_base, npp_3d, npp_sec, u_4d, v_4d, doy
gc.collect()

# --- Open lazily ---
ds_lazy = xr.open_dataset(FORCING_NC)
print(f"Opened lazily: {list(ds_lazy.data_vars)}")
print(f"  temperature: {ds_lazy['temperature'].shape}, chunks=None (numpy-backed, lazy from NetCDF)")

# Static forcings (tiny, stay in-memory)
dx_arr = np.full((NY, NX), CELL_SIZE)
dy_arr = np.full((NY, NX), CELL_SIZE)

# %% [markdown]
# ## Section 4: Compile Model with Lazy Forcings

# %%
config = Config(
    parameters={
        "lambda_0": xr.DataArray([1 / 150 / 86400] * N_GROUPS, dims=["F"]),
        "gamma_lambda": xr.DataArray([0.15] * N_GROUPS, dims=["F"]),
        "tau_r_0": xr.DataArray([10.38 * 86400] * N_GROUPS, dims=["F"]),
        "gamma_tau_r": xr.DataArray([0.11] * N_GROUPS, dims=["F"]),
        "t_ref": xr.DataArray(0.0),
        "efficiency": xr.DataArray([0.1668] * N_GROUPS, dims=["F"]),
        "cohort_ages": xr.DataArray(cohort_ages_sec, dims=["C"]),
        "day_layer": xr.DataArray([i % N_LAYERS for i in range(N_GROUPS)], dims=["F"]),
        "night_layer": xr.DataArray([0] * N_GROUPS, dims=["F"]),
    },
    forcings={
        "latitude": xr.DataArray(np.full(NY, LATITUDE), dims=["Y"], coords={"Y": lat}),
        "temperature": ds_lazy["temperature"],
        "primary_production": ds_lazy["primary_production"],
        "day_of_year": ds_lazy["day_of_year"],
        "u": ds_lazy["u"],
        "v": ds_lazy["v"],
        "D": xr.DataArray(0.0),
        "dx": xr.DataArray(dx_arr, dims=["Y", "X"], coords={"Y": lat, "X": lon}),
        "dy": xr.DataArray(dy_arr, dims=["Y", "X"], coords={"Y": lat, "X": lon}),
        "face_height": xr.DataArray(dy_arr, dims=["Y", "X"], coords={"Y": lat, "X": lon}),
        "face_width": xr.DataArray(dx_arr, dims=["Y", "X"], coords={"Y": lat, "X": lon}),
        "cell_area": xr.DataArray(dx_arr * dy_arr, dims=["Y", "X"], coords={"Y": lat, "X": lon}),
        "mask": xr.DataArray(np.ones((NY, NX)), dims=["Y", "X"], coords={"Y": lat, "X": lon}),
        "bc_north": xr.DataArray(1), "bc_south": xr.DataArray(1),
        "bc_east": xr.DataArray(1), "bc_west": xr.DataArray(1),
    },
    initial_state={
        "biomass": xr.DataArray(
            np.zeros((N_GROUPS, NY, NX)), dims=["F", "Y", "X"], coords={"Y": lat, "X": lon}
        ),
        "production": xr.DataArray(
            np.zeros((N_GROUPS, NY, NX, n_cohorts)),
            dims=["F", "Y", "X", "C"], coords={"Y": lat, "X": lon},
        ),
    },
    execution={
        "time_start": start_date, "time_end": end_date,
        "dt": "1d", "forcing_interpolation": "linear",
    },
)

blueprint = LMTL
model = compile_model(blueprint, config)
n_timesteps = model.n_timesteps
chunk_size = CHUNK_SIZE
n_chunks = math.ceil(n_timesteps / chunk_size)
export_variables = ["biomass"]

print(f"Simulation: {n_timesteps} timesteps, {n_chunks} chunks")
print(f"Source timesteps: {len(source_dates)}")


# %% [markdown]
# ## Section 5: Simulation Loop (parameterized by writer)

# %%
def run_profiled(writer_name: str, writer):
    """Run chunked simulation with memory snapshots, return (DataFrame, elapsed)."""
    snap, records = make_snapshotter()
    snap(f"{writer_name}_start")

    step_fn = build_step_fn(model, export_variables=export_variables)
    writer_coords = dict(model.coords)
    if model.time_grid is not None:
        writer_coords["T"] = model.time_grid.coords[:n_timesteps]
    var_dims = resolve_var_dims(model.data_nodes, export_variables)
    writer.initialize(model.shapes, export_variables, coords=writer_coords, var_dims=var_dims)

    state = dict(model.state)
    params = dict(model.parameters)

    snap(f"{writer_name}_init")

    t_start = time.time()
    try:
        for chunk_idx in range(n_chunks):
            start_t = chunk_idx * chunk_size
            end_t = min(start_t + chunk_size, n_timesteps)
            chunk_len = end_t - start_t

            forcings_chunk = model.forcings.get_chunk(start_t, end_t)
            snap(f"c{chunk_idx}_get_chunk")

            (state, params), outputs = _scan(step_fn, (state, params), forcings_chunk, chunk_len)
            jax.block_until_ready(outputs)
            snap(f"c{chunk_idx}_scan")

            writer.append(outputs)
            snap(f"c{chunk_idx}_append")

            del forcings_chunk, outputs
            gc.collect()
            snap(f"c{chunk_idx}_cleanup")

        snap("pre_finalize")
        result = writer.finalize()
        snap("post_finalize")
    finally:
        writer.close()

    elapsed = time.time() - t_start

    del result
    gc.collect()
    snap("final_cleanup")

    return pd.DataFrame(records), elapsed


# %% [markdown]
# ## Section 6: Run Both Writers

# %%
# --- MemoryWriter ---
print("\n" + "=" * 70)
print("RUN 1: MemoryWriter (lazy forcings from NetCDF)")
print("=" * 70)
df_mem, t_mem = run_profiled("memory", MemoryWriter())

# Force cleanup between runs
gc.collect()
time.sleep(0.5)

# --- DiskWriter ---
print("\n" + "=" * 70)
print("RUN 2: DiskWriter (lazy forcings from NetCDF)")
print("=" * 70)
if DISK_OUTPUT_PATH.exists():
    shutil.rmtree(DISK_OUTPUT_PATH)
df_disk, t_disk = run_profiled("disk", DiskWriter(DISK_OUTPUT_PATH))

# %% [markdown]
# ## Section 7: Comparison Table

# %%
print("\n" + "=" * 70)
print("COMPARISON: MemoryWriter vs DiskWriter (lazy NetCDF forcings)")
print("=" * 70)

common_labels = [l for l in df_mem["label"] if not l.startswith("memory_")]
mem_lookup = dict(zip(df_mem["label"], df_mem["delta_mb"]))
disk_lookup = dict(zip(df_disk["label"], df_disk["delta_mb"]))

print(f"\n{'Phase':<25s} {'Memory (MB)':>14s} {'Disk (MB)':>14s} {'Diff':>10s}")
print("-" * 66)
for label in common_labels:
    m = mem_lookup.get(label, float("nan"))
    d = disk_lookup.get(label, float("nan"))
    diff = m - d
    print(f"{label:<25s} {m:>+12.1f}   {d:>+12.1f}   {diff:>+8.1f}")

mem_peak = df_mem["delta_mb"].max()
disk_peak = df_disk["delta_mb"].max()
print(f"\n{'PEAK DELTA':<25s} {mem_peak:>+12.1f}   {disk_peak:>+12.1f}   {mem_peak - disk_peak:>+8.1f}")
print(f"{'Time (s)':<25s} {t_mem:>12.2f}   {t_disk:>12.2f}")

# Key observation: does get_chunk memory grow across chunks? (source caching)
mem_gets = df_mem[df_mem["label"].str.contains("get_chunk")]["delta_mb"].values
disk_gets = df_disk[df_disk["label"].str.contains("get_chunk")]["delta_mb"].values
print(f"\nget_chunk deltas across chunks:")
print(f"  MemoryWriter: {['%+.1f' % v for v in mem_gets]}")
print(f"  DiskWriter:   {['%+.1f' % v for v in disk_gets]}")
if len(mem_gets) > 1:
    growth = mem_gets[-1] - mem_gets[0]
    print(f"  -> Memory growth c0->c{len(mem_gets)-1}: {growth:+.1f} MB", end="")
    if abs(growth) < 5:
        print(" (stable — sources likely NOT cached)")
    else:
        print(" (growing — sources likely cached or accumulating)")

# %% [markdown]
# ## Section 8: Extrapolation

# %%
print(f"\n{'=' * 70}")
print(f"EXTRAPOLATION — {TARGET_NY}x{TARGET_NX} (scale={scale_factor:.0f}x)")
print("=" * 70)

output_per_chunk_target = chunk_size * N_GROUPS * TARGET_NY * TARGET_NX * 4 / 1024**2
chunk_forcing_target = (
    chunk_size * N_LAYERS * TARGET_NY * TARGET_NX * 4 * 3
    + chunk_size * TARGET_NY * TARGET_NX * 4
    + chunk_size * 4
) / 1024**2
src_target_mb = 5000 * N_LAYERS * TARGET_NY * TARGET_NX * 4 * 3 / 1024**2
target_n_chunks = 70

print(f"\n{'Component':<40s} {'Size':>12s}")
print("-" * 55)
print(f"{'Forcing chunk (1000 steps)':<40s} {chunk_forcing_target:>10.0f} MB")
print(f"{'Output/chunk (biomass only)':<40s} {output_per_chunk_target:>10.0f} MB")
print(f"{'Interpolation sources (if cached)':<40s} {src_target_mb:>10.0f} MB")

mem_accum = output_per_chunk_target * target_n_chunks
mem_finalize = mem_accum * 2
mem_peak_target = src_target_mb + chunk_forcing_target + mem_finalize
disk_peak_target = src_target_mb + chunk_forcing_target + output_per_chunk_target
disk_peak_no_cache = chunk_forcing_target + output_per_chunk_target

print(f"\n{'':40s} {'MemoryWriter':>14s} {'DiskWriter':>14s}")
print("-" * 70)
print(f"{'Accumulated outputs':<40s} {mem_accum:>12.0f} MB {'0':>12s} MB")
print(f"{'finalize() peak':<40s} {mem_finalize:>12.0f} MB {'0':>12s} MB")
print(f"{'PEAK (sources cached)':<40s} {mem_peak_target:>12.0f} MB {disk_peak_target:>12.0f} MB")
print(f"{'':40s} {'(~' + str(int(mem_peak_target / 1024)) + ' GB)':>14s} {'(~' + str(int(disk_peak_target / 1024)) + ' GB)':>14s}")
print(f"{'PEAK (sources NOT cached)':<40s} {'':>14s} {disk_peak_no_cache:>12.0f} MB")
print(f"{'':40s} {'':>14s} {'(~' + str(int(disk_peak_no_cache / 1024)) + ' GB)':>14s}")

# %% [markdown]
# ## Section 9: Visualization

# %%
fig, axes = plt.subplots(2, 2, figsize=(18, 12))


def phase_color(label):
    if "get_chunk" in label:
        return "orange"
    if "_scan" in label:
        return "steelblue"
    if "append" in label:
        return "green"
    if "finalize" in label:
        return "red"
    if "cleanup" in label:
        return "mediumpurple"
    return "gray"


legend_elements = [
    Patch(facecolor="orange", alpha=0.7, label="get_chunk (interp)"),
    Patch(facecolor="steelblue", alpha=0.7, label="scan (JAX)"),
    Patch(facecolor="green", alpha=0.7, label="append (write)"),
    Patch(facecolor="mediumpurple", alpha=0.7, label="cleanup (gc)"),
    Patch(facecolor="red", alpha=0.7, label="finalize"),
    Patch(facecolor="gray", alpha=0.7, label="other"),
]

for col, (df, name, elapsed) in enumerate([
    (df_mem, "MemoryWriter", t_mem),
    (df_disk, "DiskWriter", t_disk),
]):
    ax = axes[0, col]
    x = range(len(df))
    ax.plot(x, df["rss_mb"], "o-", color="steelblue", linewidth=2, markersize=3)
    ax.set_ylabel("RSS (MB)")
    ax.set_title(f"{name} — RSS ({elapsed:.2f}s) [lazy NetCDF]")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"], rotation=90, fontsize=5)

    ax = axes[1, col]
    colors = [phase_color(l) for l in df["label"]]
    ax.bar(x, df["delta_mb"], color=colors, alpha=0.7)
    ax.set_ylabel("Delta from baseline (MB)")
    ax.set_title(f"{name} — Delta by phase")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"], rotation=90, fontsize=5)
    ax.legend(handles=legend_elements, loc="upper left", fontsize=6)

fig.suptitle(
    f"Memory Profile (lazy NetCDF) — {NY}x{NX}, {n_timesteps} steps, {n_chunks} chunks",
    fontsize=13,
)
fig.tight_layout()
plot_file = "examples/images/09_memory_profile.png"
Path(plot_file).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_file, dpi=150)
print(f"\nPlot saved to {plot_file}")

# Cleanup temp files
ds_lazy.close()
if TMP_DIR.exists():
    shutil.rmtree(TMP_DIR)
    print(f"Cleaned up {TMP_DIR}")
