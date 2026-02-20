"""Benchmark LMTL 0D: CPU vs GPU — vmap over N simultaneous simulations.

Measures execution time of the LMTL 0D model as a function of the number
of parallel simulations (vmap over N parameter configurations).
Grid is fixed at 1x1 to isolate the effect of parallelism.

Outputs:
- examples/images/03_benchmark_cpu_vs_gpu_time.png   (execution time)
- examples/images/03_benchmark_cpu_vs_gpu_speedup.png (GPU speedup)
"""

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

# =============================================================================
# CONFIGURATION
# =============================================================================

# Biological parameters (true values)
TRUE_PARAMS = {
    "lambda_0": 1 / 150 / 86400,  # 1/s
    "gamma_lambda": 0.15,  # 1/degC
    "tau_r_0": 10.38 * 86400,  # s
    "gamma_tau_r": 0.11,  # 1/degC
    "efficiency": 0.1668,  # dimensionless
    "t_ref": 0.0,  # degC
}

# Parameters optimized (vmapped over efficiency)
OPT_PARAMS = ["lambda_0", "gamma_lambda", "tau_r_0", "gamma_tau_r", "efficiency"]

# Fixed parameters (not vmapped)
FIXED_PARAMS = {"t_ref": TRUE_PARAMS["t_ref"]}

# Extra catalogue parameters
LATITUDE = 30.0
DAY_LAYER = 0
NIGHT_LAYER = 0

# Benchmark settings
N_SIMS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
SIM_DAYS = 365
DT = "1d"

# Output
IMAGE_DIR = Path("examples/images")

# =============================================================================
# BUILD MODEL (once, 0D grid)
# =============================================================================

blueprint = LMTL_NO_TRANSPORT

max_age_days = int(np.ceil(TRUE_PARAMS["tau_r_0"] / 86400))
cohort_ages_sec = np.arange(0, max_age_days + 1) * 86400.0
n_cohorts = len(cohort_ages_sec)

# Time axis
start_date = "2000-01-01"
end_date = str((pd.Timestamp(start_date) + pd.DateOffset(days=SIM_DAYS)).date())
start_pd = pd.to_datetime(start_date)
end_pd = pd.to_datetime(end_date)
n_days = (end_pd - start_pd).days + 5
dates = pd.date_range(start=start_pd, periods=n_days, freq="D")

# Grid 1x1
ny, nx = 1, 1
lat, lon = np.arange(ny), np.arange(nx)

# Forcings
day_of_year = dates.dayofyear.values

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
            "day_layer": xr.DataArray([DAY_LAYER], dims=["F"]),
            "night_layer": xr.DataArray([NIGHT_LAYER], dims=["F"]),
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

print("Compiling model...")
_model = compile_model(blueprint, config, backend="jax")
_step_fn = build_step_fn(_model, params_as_argument=True)
_n_timesteps = _model.n_timesteps
_initial_state = _model.state
_forcings = _model.forcings.get_all()

print(f"Model compiled: {_n_timesteps} timesteps, {n_cohorts} cohorts")

# =============================================================================
# SIMULATION FUNCTION (scan + vmap)
# =============================================================================


def run_simulation(params: dict) -> jnp.ndarray:
    """Run a single simulation, return final mean biomass (scalar)."""
    full_params = {**params, **{k: jnp.array(v) for k, v in FIXED_PARAMS.items()}}
    full_params["cohort_ages"] = _model.parameters["cohort_ages"]
    full_params["day_layer"] = _model.parameters["day_layer"]
    full_params["night_layer"] = _model.parameters["night_layer"]
    full_params["latitude"] = _model.parameters["latitude"]

    def scan_body(carry, t):
        state, p = carry
        forcings_t = {
            name: (arr[t] if arr.ndim > 0 and arr.shape[0] == _n_timesteps else arr)
            for name, arr in _forcings.items()
        }
        new_carry, outputs = _step_fn((state, p), forcings_t)
        new_state, _ = new_carry
        return (new_state, p), jnp.mean(outputs["biomass"])

    _, biomass = jax.lax.scan(scan_body, (_initial_state, full_params), jnp.arange(_n_timesteps))
    return biomass[-1]


# Vmap: efficiency varies (axis 0), rest broadcast (None)
_in_axes: dict = {k: None for k in OPT_PARAMS}
_in_axes["efficiency"] = 0
_run_batch = jax.jit(jax.vmap(run_simulation, in_axes=(_in_axes,)))


# =============================================================================
# BENCHMARK
# =============================================================================


def benchmark(n_sims: int, device_name: str) -> float:
    """Benchmark n_sims parallel simulations on the given device.

    Returns execution time in seconds (after warmup).
    """
    dev = jax.devices(device_name)[0]

    # Build params on the target device
    with jax.default_device(dev):
        params = {k: jnp.array(TRUE_PARAMS[k]) for k in OPT_PARAMS}
        params["efficiency"] = jnp.linspace(0.05, 0.30, n_sims)

        # Warmup (JIT compile)
        result = _run_batch(params)
        jax.block_until_ready(result)

        # Timed run
        t0 = time.time()
        result = _run_batch(params)
        jax.block_until_ready(result)
        return time.time() - t0


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BENCHMARK: LMTL 0D — CPU vs GPU (vmap over N simulations)")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Simulation: {SIM_DAYS} days, dt={DT}, grid=1x1")
    print(f"N_SIMS: {N_SIMS}")
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

    # Run benchmarks
    cpu_times: dict[int, float] = {}
    gpu_times: dict[int, float] = {}

    for n in N_SIMS:
        print(f"\n--- N = {n} ---")

        # CPU
        print(f"  CPU...", end=" ", flush=True)
        try:
            t = benchmark(n, "cpu")
            cpu_times[n] = t
            print(f"{t:.4f}s")
        except Exception as e:
            print(f"FAILED: {e}")

        # GPU
        if has_gpu:
            print(f"  GPU...", end=" ", flush=True)
            try:
                t = benchmark(n, "gpu")
                gpu_times[n] = t
                speedup = cpu_times.get(n, 0) / t if t > 0 else float("inf")
                print(f"{t:.4f}s  (speedup: {speedup:.1f}x)")
            except Exception as e:
                print(f"FAILED: {e}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    header = f"{'N':>6}  {'CPU (s)':>10}"
    if has_gpu:
        header += f"  {'GPU (s)':>10}  {'Speedup':>8}"
    print(header)
    print("-" * len(header))

    for n in N_SIMS:
        row = f"{n:>6}"
        if n in cpu_times:
            row += f"  {cpu_times[n]:>10.4f}"
        else:
            row += f"  {'—':>10}"
        if has_gpu:
            if n in gpu_times:
                row += f"  {gpu_times[n]:>10.4f}"
                if n in cpu_times:
                    row += f"  {cpu_times[n] / gpu_times[n]:>8.1f}x"
                else:
                    row += f"  {'—':>8}"
            else:
                row += f"  {'—':>10}  {'—':>8}"
        print(row)

    # =========================================================================
    # PLOTS
    # =========================================================================

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Execution time (log-log) ---
    fig, ax = plt.subplots(figsize=(8, 5))

    ns_cpu = sorted(cpu_times.keys())
    ax.loglog(ns_cpu, [cpu_times[n] for n in ns_cpu], "o-", label="CPU", color="tab:blue", linewidth=2)

    if gpu_times:
        ns_gpu = sorted(gpu_times.keys())
        ax.loglog(ns_gpu, [gpu_times[n] for n in ns_gpu], "s-", label="GPU", color="tab:red", linewidth=2)

    ax.set_xlabel("Number of parallel simulations (N)")
    ax.set_ylabel("Execution time (s)")
    ax.set_title("LMTL 0D — CPU vs GPU execution time (vmap)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    time_path = IMAGE_DIR / "03_benchmark_cpu_vs_gpu_time.png"
    plt.savefig(str(time_path), dpi=150)
    print(f"\nPlot saved: {time_path}")
    plt.close(fig)

    # --- Plot 2: Speedup (only if GPU data available) ---
    if gpu_times:
        common_ns = sorted(set(cpu_times.keys()) & set(gpu_times.keys()))
        speedups = [cpu_times[n] / gpu_times[n] for n in common_ns]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogx(common_ns, speedups, "D-", color="tab:green", linewidth=2, markersize=8)
        ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Number of parallel simulations (N)")
        ax.set_ylabel("Speedup (CPU time / GPU time)")
        ax.set_title("LMTL 0D — GPU speedup vs N simulations")
        ax.grid(True, alpha=0.3, which="both")
        fig.tight_layout()

        speedup_path = IMAGE_DIR / "03_benchmark_cpu_vs_gpu_speedup.png"
        plt.savefig(str(speedup_path), dpi=150)
        print(f"Plot saved: {speedup_path}")
        plt.close(fig)
    else:
        print("(Speedup plot skipped — no GPU)")
