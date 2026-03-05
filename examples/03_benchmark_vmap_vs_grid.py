# %% [markdown]
# # Benchmark LMTL: GPU speedup — vmap parallelism vs spatial grid scaling
#
# Compares two sources of parallelism at equivalent operation counts:
# - **vmap**: N independent simulations on a 1x1 grid
# - **grid**: 1 simulation on a NxN grid (N² cells)
#
# Outputs:
# - `examples/images/03_benchmark_vmap_vs_grid_time.png` (execution times)
# - `examples/images/03_benchmark_vmap_vs_grid_speedup.png` (GPU speedup comparison)

# %%
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import seapopym.functions.lmtl  # noqa: F401
from seapopym.blueprint import Config
from seapopym.compiler import compile_model
from seapopym.engine.step import build_step_fn
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

OPT_PARAMS = ["lambda_0", "gamma_lambda", "tau_r_0", "gamma_tau_r", "efficiency"]
FIXED_PARAMS = {"t_ref": TRUE_PARAMS["t_ref"]}
LATITUDE = 30.0

# Parallel operation counts to benchmark
N_OPS = [1, 100, 10000, 1000000]
# For grid: closest square side -> actual cells
GRID_SIDES = {n: int(np.round(np.sqrt(n))) for n in N_OPS}

SIM_DAYS = 30
DT = "1d"
N_REPEATS = 10

IMAGE_DIR = Path("examples/images")

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

# %% [markdown]
# ## Vmap Benchmark (1x1 grid, N parallel simulations)

# %%
# Build 1x1 model once for vmap
ny_1, nx_1 = 1, 1
lat_1, lon_1 = np.arange(ny_1), np.arange(nx_1)

doy_float = day_of_year.astype(float)
doy_3d = np.broadcast_to(doy_float[:, None, None], (len(dates), ny_1, nx_1))
doy_da_1 = xr.DataArray(doy_3d, dims=["T", "Y", "X"], coords={"T": dates, "Y": lat_1, "X": lon_1})

temp_c = 15.0 + 5.0 * np.sin(2 * np.pi * day_of_year / 365.0)
temp_4d = np.broadcast_to(temp_c[:, None, None, None], (len(dates), 1, ny_1, nx_1))
temp_da_1 = xr.DataArray(
    temp_4d, dims=["T", "Z", "Y", "X"], coords={"T": dates, "Z": np.arange(1), "Y": lat_1, "X": lon_1}
)

npp_sec = (1.0 + 0.5 * np.sin(2 * np.pi * day_of_year / 365.0)) / 86400.0
npp_3d = np.broadcast_to(npp_sec[:, None, None], (len(dates), ny_1, nx_1))
npp_da_1 = xr.DataArray(npp_3d, dims=["T", "Y", "X"], coords={"T": dates, "Y": lat_1, "X": lon_1})

config_1x1 = Config.from_dict(
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
            "latitude": xr.DataArray(np.full(ny_1, LATITUDE), dims=["Y"], coords={"Y": lat_1}),
            "temperature": temp_da_1,
            "primary_production": npp_da_1,
            "day_of_year": doy_da_1,
        },
        "initial_state": {
            "biomass": xr.DataArray(
                np.zeros((1, ny_1, nx_1)), dims=["F", "Y", "X"], coords={"Y": lat_1, "X": lon_1}
            ),
            "production": xr.DataArray(
                np.zeros((1, ny_1, nx_1, n_cohorts)), dims=["F", "Y", "X", "C"], coords={"Y": lat_1, "X": lon_1}
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

print("Compiling 1x1 model for vmap benchmark...")
_model_1x1 = compile_model(blueprint, config_1x1)
_step_fn_1x1 = build_step_fn(_model_1x1)
_n_timesteps_1x1 = _model_1x1.n_timesteps
_initial_state_1x1 = _model_1x1.state
_forcings_1x1 = _model_1x1.forcings.get_all()


def run_single_sim(params: dict) -> jnp.ndarray:
    """Run a single 1x1 simulation, return final mean biomass."""
    full_params = {**params, **{k: jnp.array(v) for k, v in FIXED_PARAMS.items()}}
    full_params["cohort_ages"] = _model_1x1.parameters["cohort_ages"]
    full_params["day_layer"] = _model_1x1.parameters["day_layer"]
    full_params["night_layer"] = _model_1x1.parameters["night_layer"]
    # latitude is now a forcing, not a parameter

    def scan_body(carry, t):
        state, p = carry
        forcings_t = {
            name: (arr[t] if arr.ndim > 0 and arr.shape[0] == _n_timesteps_1x1 else arr)
            for name, arr in _forcings_1x1.items()
        }
        (new_state, p), outputs = _step_fn_1x1((state, p), forcings_t)
        return (new_state, p), jnp.mean(outputs["biomass"])

    _, biomass = jax.lax.scan(scan_body, (_initial_state_1x1, full_params), jnp.arange(_n_timesteps_1x1))
    return biomass[-1]


_in_axes: dict = {k: None for k in OPT_PARAMS}
_in_axes["efficiency"] = 0
_run_batch = jax.jit(jax.vmap(run_single_sim, in_axes=(_in_axes,)))


def benchmark_vmap(n_sims: int, device_name: str, n_repeats: int = 1) -> tuple[float, float]:
    """Benchmark N parallel vmap simulations. Returns (mean, std)."""
    dev = jax.devices(device_name)[0]
    with jax.default_device(dev):
        params = {k: jnp.array(TRUE_PARAMS[k]) for k in OPT_PARAMS}
        params["efficiency"] = jnp.linspace(0.05, 0.30, n_sims)

        result = _run_batch(params)
        jax.block_until_ready(result)

        times = []
        for _ in range(n_repeats):
            t0 = time.time()
            result = _run_batch(params)
            jax.block_until_ready(result)
            times.append(time.time() - t0)

        return float(np.mean(times)), float(np.std(times))

# %% [markdown]
# ## Grid Benchmark (single simulation, NxN grid)

# %%
def build_grid_model(ny: int, nx: int):
    """Compile model for a given grid size."""
    lat = np.arange(ny)
    lon = np.arange(nx)

    doy_3d = np.broadcast_to(doy_float[:, None, None], (len(dates), ny, nx))
    doy_da = xr.DataArray(doy_3d, dims=["T", "Y", "X"], coords={"T": dates, "Y": lat, "X": lon})

    temp_4d = np.broadcast_to(temp_c[:, None, None, None], (len(dates), 1, ny, nx))
    temp_da = xr.DataArray(
        temp_4d, dims=["T", "Z", "Y", "X"], coords={"T": dates, "Z": np.arange(1), "Y": lat, "X": lon}
    )

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
            },
        }
    )

    model = compile_model(blueprint, config)
    step_fn = build_step_fn(model)
    return model, step_fn


def run_grid_jit(step_fn, model):
    """Build a JIT-compiled scan simulation for grid model."""
    n_timesteps = model.n_timesteps
    initial_state = model.state
    forcings = model.forcings.get_all()
    parameters = model.parameters

    @jax.jit
    def _run():
        def scan_body(carry, t):
            state, params = carry
            forcings_t = {
                name: (arr[t] if arr.ndim > 0 and arr.shape[0] == n_timesteps else arr)
                for name, arr in forcings.items()
            }
            (new_state, params), outputs = step_fn((state, params), forcings_t)
            return (new_state, params), jnp.mean(outputs["biomass"])

        _, biomass = jax.lax.scan(scan_body, (initial_state, parameters), jnp.arange(n_timesteps))
        return biomass[-1]

    return _run


def benchmark_grid(ny: int, nx: int, device_name: str, n_repeats: int = 1) -> tuple[float, float]:
    """Benchmark a single simulation on (ny, nx) grid. Returns (mean, std)."""
    dev = jax.devices(device_name)[0]
    with jax.default_device(dev):
        model, step_fn = build_grid_model(ny, nx)
        run_fn = run_grid_jit(step_fn, model)

        result = run_fn()
        jax.block_until_ready(result)

        times = []
        for _ in range(n_repeats):
            t0 = time.time()
            result = run_fn()
            jax.block_until_ready(result)
            times.append(time.time() - t0)

        return float(np.mean(times)), float(np.std(times))

# %% [markdown]
# ## Run Benchmark

# %%
print("=" * 70)
print("BENCHMARK: LMTL — vmap parallelism vs spatial grid scaling")
print("=" * 70)
print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")
print(f"Simulation: {SIM_DAYS} days, dt={DT}, repeats={N_REPEATS}")
print(f"Parallel ops: {N_OPS}")
print()

# Detect GPU
try:
    gpu_devices = jax.devices("gpu")
    has_gpu = len(gpu_devices) > 0
    if has_gpu:
        print(f"GPU detected: {gpu_devices[0]}")
except RuntimeError:
    has_gpu = False
    print("No GPU detected — CPU-only benchmark")

if not has_gpu:
    print("This benchmark requires a GPU to compare speedups. Exiting.")
    exit(0)

# Results: {n_ops: (mean, std)}
vmap_cpu: dict[int, tuple[float, float]] = {}
vmap_gpu: dict[int, tuple[float, float]] = {}
grid_cpu: dict[int, tuple[float, float]] = {}
grid_gpu: dict[int, tuple[float, float]] = {}

for n in N_OPS:
    side = GRID_SIDES[n]
    actual_cells = side * side
    print(f"\n{'='*50}")
    print(f"N = {n:,} parallel ops")
    print(f"{'='*50}")

    # --- VMAP ---
    print(f"\n  [vmap] {n} simulations on 1x1 grid")
    print(f"    CPU...", end=" ", flush=True)
    try:
        mean, std = benchmark_vmap(n, "cpu", N_REPEATS)
        vmap_cpu[n] = (mean, std)
        print(f"{mean:.4f}s ± {std:.4f}s")
    except Exception as e:
        print(f"FAILED: {e}")

    print(f"    GPU...", end=" ", flush=True)
    try:
        mean, std = benchmark_vmap(n, "gpu", N_REPEATS)
        vmap_gpu[n] = (mean, std)
        speedup = vmap_cpu[n][0] / mean if n in vmap_cpu and mean > 0 else 0
        print(f"{mean:.4f}s ± {std:.4f}s  (speedup: {speedup:.1f}x)")
    except Exception as e:
        print(f"FAILED: {e}")

    # --- GRID ---
    print(f"\n  [grid] 1 simulation on {side}x{side} grid ({actual_cells:,} cells)")
    print(f"    CPU...", end=" ", flush=True)
    try:
        mean, std = benchmark_grid(side, side, "cpu", N_REPEATS)
        grid_cpu[n] = (mean, std)
        print(f"{mean:.4f}s ± {std:.4f}s")
    except Exception as e:
        print(f"FAILED: {e}")

    print(f"    GPU...", end=" ", flush=True)
    try:
        mean, std = benchmark_grid(side, side, "gpu", N_REPEATS)
        grid_gpu[n] = (mean, std)
        speedup = grid_cpu[n][0] / mean if n in grid_cpu and mean > 0 else 0
        print(f"{mean:.4f}s ± {std:.4f}s  (speedup: {speedup:.1f}x)")
    except Exception as e:
        print(f"FAILED: {e}")

# %% [markdown]
# ## Summary

# %%
print(f"\n{'='*70}")
print(f"SUMMARY  (mean ± std over {N_REPEATS} runs)")
print(f"{'='*70}")

print(f"\n{'N ops':>10}  {'Mode':>6}  {'CPU (s)':>18}  {'GPU (s)':>18}  {'Speedup':>8}")
print("-" * 70)

for n in N_OPS:
    side = GRID_SIDES[n]
    # vmap row
    row = f"{n:>10,}  {'vmap':>6}"
    if n in vmap_cpu:
        m, s = vmap_cpu[n]
        row += f"  {m:>8.4f} ± {s:.4f}"
    else:
        row += f"  {'—':>18}"
    if n in vmap_gpu:
        m, s = vmap_gpu[n]
        row += f"  {m:>8.4f} ± {s:.4f}"
        if n in vmap_cpu:
            row += f"  {vmap_cpu[n][0] / m:>8.1f}x"
    else:
        row += f"  {'—':>18}  {'—':>8}"
    print(row)

    # grid row
    actual_cells = side * side
    row = f"{'':>10}  {f'{side}²':>6}"
    if n in grid_cpu:
        m, s = grid_cpu[n]
        row += f"  {m:>8.4f} ± {s:.4f}"
    else:
        row += f"  {'—':>18}"
    if n in grid_gpu:
        m, s = grid_gpu[n]
        row += f"  {m:>8.4f} ± {s:.4f}"
        if n in grid_cpu:
            row += f"  {grid_cpu[n][0] / m:>8.1f}x"
    else:
        row += f"  {'—':>18}  {'—':>8}"
    print(row)

# %% [markdown]
# ## Visualization

# %%
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# --- Plot 1: Execution times (4 curves) ---
fig, ax = plt.subplots(figsize=(9, 5))

common_vmap = sorted(set(vmap_cpu.keys()) & set(vmap_gpu.keys()))
common_grid = sorted(set(grid_cpu.keys()) & set(grid_gpu.keys()))

if vmap_cpu:
    ns = sorted(vmap_cpu.keys())
    ax.errorbar(ns, [vmap_cpu[n][0] for n in ns], yerr=[vmap_cpu[n][1] for n in ns],
                 fmt="o--", label="vmap · CPU", color="tab:blue", linewidth=1.5, capsize=3, alpha=0.7)
if vmap_gpu:
    ns = sorted(vmap_gpu.keys())
    ax.errorbar(ns, [vmap_gpu[n][0] for n in ns], yerr=[vmap_gpu[n][1] for n in ns],
                 fmt="o-", label="vmap · GPU", color="tab:blue", linewidth=2, capsize=3)
if grid_cpu:
    ns = sorted(grid_cpu.keys())
    ax.errorbar(ns, [grid_cpu[n][0] for n in ns], yerr=[grid_cpu[n][1] for n in ns],
                 fmt="s--", label="grid · CPU", color="tab:red", linewidth=1.5, capsize=3, alpha=0.7)
if grid_gpu:
    ns = sorted(grid_gpu.keys())
    ax.errorbar(ns, [grid_gpu[n][0] for n in ns], yerr=[grid_gpu[n][1] for n in ns],
                 fmt="s-", label="grid · GPU", color="tab:red", linewidth=2, capsize=3)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Parallel operations")
ax.set_ylabel("Execution time (s)")
ax.set_title(f"LMTL — Execution time: vmap vs grid (n={N_REPEATS})")
ax.legend()
ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()

time_path = IMAGE_DIR / "03_benchmark_vmap_vs_grid_time.png"
plt.savefig(str(time_path), dpi=150)
print(f"\nPlot saved: {time_path}")
plt.close(fig)

# %%
# --- Plot 2: Speedup comparison ---
fig, ax = plt.subplots(figsize=(9, 5))

if common_vmap:
    speedups_vmap = [vmap_cpu[n][0] / vmap_gpu[n][0] for n in common_vmap]
    ax.semilogx(common_vmap, speedups_vmap, "o-", label="vmap (N sims, 1x1)",
                 color="tab:blue", linewidth=2, markersize=8)

if common_grid:
    speedups_grid = [grid_cpu[n][0] / grid_gpu[n][0] for n in common_grid]
    ax.semilogx(common_grid, speedups_grid, "s-", label="grid (1 sim, NxN)",
                 color="tab:red", linewidth=2, markersize=8)

ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Parallel operations")
ax.set_ylabel("GPU speedup (CPU time / GPU time)")
ax.set_title(f"LMTL — GPU speedup: vmap vs grid (n={N_REPEATS})")
ax.legend()
ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()

speedup_path = IMAGE_DIR / "03_benchmark_vmap_vs_grid_speedup.png"
plt.savefig(str(speedup_path), dpi=150)
print(f"Plot saved: {speedup_path}")
plt.close(fig)
