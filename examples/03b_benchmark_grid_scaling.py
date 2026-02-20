"""Benchmark LMTL: CPU vs GPU — grid size scaling.

Measures execution time of the LMTL model as a function of spatial grid
size (Y x X). A single simulation is run per grid size (no vmap), isolating
the effect of spatial workload on CPU vs GPU performance.

Outputs:
- examples/images/03b_benchmark_grid_scaling_time.png    (execution time)
- examples/images/03b_benchmark_grid_scaling_speedup.png (GPU speedup)
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

TRUE_PARAMS = {
    "lambda_0": 1 / 150 / 86400,
    "gamma_lambda": 0.15,
    "tau_r_0": 10.38 * 86400,
    "gamma_tau_r": 0.11,
    "efficiency": 0.1668,
    "t_ref": 0.0,
}

LATITUDE = 30.0

# Grid sizes to benchmark (square grids: N x N)
GRID_SIDES = [1, 4, 8, 16, 32, 64, 128, 256, 360]

SIM_DAYS = 30
DT = "1d"

IMAGE_DIR = Path("examples/images")

# =============================================================================
# HELPERS
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


def build_model(ny: int, nx: int):
    """Compile model for a given grid size and return (step_fn, state, forcings, n_timesteps, params)."""
    lat = np.arange(ny)
    lon = np.arange(nx)

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

    model = compile_model(blueprint, config, backend="jax")
    step_fn = build_step_fn(model)
    return model, step_fn


def run_simulation_jit(step_fn, model):
    """Build a JIT-compiled scan simulation for the given model."""
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


def benchmark_grid(ny: int, nx: int, device_name: str) -> float:
    """Compile + benchmark a single simulation on a (ny, nx) grid.

    Returns execution time in seconds (after warmup).
    """
    dev = jax.devices(device_name)[0]

    with jax.default_device(dev):
        model, step_fn = build_model(ny, nx)
        run_fn = run_simulation_jit(step_fn, model)

        # Warmup (JIT compile)
        result = run_fn()
        jax.block_until_ready(result)

        # Timed run
        t0 = time.time()
        result = run_fn()
        jax.block_until_ready(result)
        return time.time() - t0


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BENCHMARK: LMTL — CPU vs GPU (grid size scaling)")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Simulation: {SIM_DAYS} days, dt={DT}")
    print(f"Grid sides: {GRID_SIDES}")
    grid_cells = [n * n for n in GRID_SIDES]
    print(f"Grid cells: {grid_cells}")
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

    for n in GRID_SIDES:
        cells = n * n
        print(f"\n--- Grid {n}x{n} ({cells:,} cells) ---")

        # CPU
        print(f"  CPU...", end=" ", flush=True)
        try:
            t = benchmark_grid(n, n, "cpu")
            cpu_times[cells] = t
            print(f"{t:.4f}s")
        except Exception as e:
            print(f"FAILED: {e}")

        # GPU
        if has_gpu:
            print(f"  GPU...", end=" ", flush=True)
            try:
                t = benchmark_grid(n, n, "gpu")
                gpu_times[cells] = t
                speedup = cpu_times.get(cells, 0) / t if t > 0 else float("inf")
                print(f"{t:.4f}s  (speedup: {speedup:.1f}x)")
            except Exception as e:
                print(f"FAILED: {e}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    header = f"{'Grid':>10} {'Cells':>8}  {'CPU (s)':>10}"
    if has_gpu:
        header += f"  {'GPU (s)':>10}  {'Speedup':>8}"
    print(header)
    print("-" * len(header))

    for n in GRID_SIDES:
        cells = n * n
        row = f"{n}x{n:>5} {cells:>8,}"
        if cells in cpu_times:
            row += f"  {cpu_times[cells]:>10.4f}"
        else:
            row += f"  {'—':>10}"
        if has_gpu:
            if cells in gpu_times:
                row += f"  {gpu_times[cells]:>10.4f}"
                if cells in cpu_times:
                    row += f"  {cpu_times[cells] / gpu_times[cells]:>8.1f}x"
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

    cells_cpu = sorted(cpu_times.keys())
    ax.loglog(cells_cpu, [cpu_times[c] for c in cells_cpu], "o-", label="CPU", color="tab:blue", linewidth=2)

    if gpu_times:
        cells_gpu = sorted(gpu_times.keys())
        ax.loglog(cells_gpu, [gpu_times[c] for c in cells_gpu], "s-", label="GPU", color="tab:red", linewidth=2)

    ax.set_xlabel("Grid cells (Y × X)")
    ax.set_ylabel("Execution time (s)")
    ax.set_title("LMTL — CPU vs GPU execution time (grid scaling)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    time_path = IMAGE_DIR / "03b_benchmark_grid_scaling_time.png"
    plt.savefig(str(time_path), dpi=150)
    print(f"\nPlot saved: {time_path}")
    plt.close(fig)

    # --- Plot 2: Speedup ---
    if gpu_times:
        common = sorted(set(cpu_times.keys()) & set(gpu_times.keys()))
        speedups = [cpu_times[c] / gpu_times[c] for c in common]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogx(common, speedups, "D-", color="tab:green", linewidth=2, markersize=8)
        ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Grid cells (Y × X)")
        ax.set_ylabel("Speedup (CPU time / GPU time)")
        ax.set_title("LMTL — GPU speedup vs grid size")
        ax.grid(True, alpha=0.3, which="both")
        fig.tight_layout()

        speedup_path = IMAGE_DIR / "03b_benchmark_grid_scaling_speedup.png"
        plt.savefig(str(speedup_path), dpi=150)
        print(f"Plot saved: {speedup_path}")
        plt.close(fig)
    else:
        print("(Speedup plot skipped — no GPU)")
