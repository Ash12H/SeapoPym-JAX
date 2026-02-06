"""Benchmark Sobol Runner: CPU vs GPU comparison.

Produces two summary figures:
  1. CPU vs GPU throughput (sim/s) by batch size
  2. GPU: estimated hours for 1M simulations × 5 years across (chunk, batch) configs

Model: LMTL + Transport 2D (20×20 grid, 17 cohorts, 1-year simulation).
5-year estimates are extrapolated linearly (×5).
"""

import gc
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import seapopym.functions.lmtl  # noqa: F401
import seapopym.functions.transport  # noqa: F401
from seapopym.blueprint import Blueprint, Config
from seapopym.compiler import compile_model
from seapopym.sensitivity.runner import SobolRunner

# =============================================================================
# Constants
# =============================================================================
NY, NX = 20, 20
DX_M = DY_M = 111_000.0
COHORT_AGES_SEC = np.arange(17) * 86400.0
N_COHORTS = 17
EXTRACTION_POINTS = [(5, 10), (10, 5), (10, 15), (15, 10)]
PARAM_BOUNDS = {
    "efficiency": (0.05, 0.30),
    "lambda_0": (3e-8, 3e-6),
    "tau_r_0": (5 * 86400.0, 15 * 86400.0),
}

# =============================================================================
# Model helpers (same as sobol_sensitivity_2d.py)
# =============================================================================
BLUEPRINT = Blueprint.from_dict({
    "id": "sobol-bench", "version": "1.0",
    "declarations": {
        "state": {
            "biomass": {"units": "g/m^2", "dims": ["Y", "X"]},
            "production": {"units": "g/m^2", "dims": ["Y", "X", "C"]},
        },
        "parameters": {
            "lambda_0": {"units": "1/s"}, "gamma_lambda": {"units": "1/delta_degC"},
            "tau_r_0": {"units": "s"}, "gamma_tau_r": {"units": "1/delta_degC"},
            "t_ref": {"units": "degC"}, "efficiency": {"units": "dimensionless"},
            "cohort_ages": {"units": "s", "dims": ["C"]},
        },
        "forcings": {
            "temperature": {"units": "degC", "dims": ["T", "Y", "X"]},
            "primary_production": {"units": "g/m^2/s", "dims": ["T", "Y", "X"]},
            "u": {"units": "m/s", "dims": ["Y", "X"]}, "v": {"units": "m/s", "dims": ["Y", "X"]},
            "D": {"units": "m^2/s", "dims": ["Y", "X"]},
            "dx": {"units": "m", "dims": ["Y", "X"]}, "dy": {"units": "m", "dims": ["Y", "X"]},
            "face_height": {"units": "m", "dims": ["Y", "X"]}, "face_width": {"units": "m", "dims": ["Y", "X"]},
            "cell_area": {"units": "m^2", "dims": ["Y", "X"]},
            "mask": {"units": "dimensionless", "dims": ["Y", "X"]},
        },
    },
    "process": [
        {"func": "lmtl:gillooly_temperature", "inputs": {"temp": "forcings.temperature"}, "outputs": {"return": {"target": "derived.temp_norm", "type": "derived"}}},
        {"func": "lmtl:recruitment_age", "inputs": {"temp": "derived.temp_norm", "tau_r_0": "parameters.tau_r_0", "gamma": "parameters.gamma_tau_r", "t_ref": "parameters.t_ref"}, "outputs": {"return": {"target": "derived.rec_age", "type": "derived"}}},
        {"func": "lmtl:npp_injection", "inputs": {"npp": "forcings.primary_production", "efficiency": "parameters.efficiency", "production": "state.production"}, "outputs": {"return": {"target": "tendencies.production", "type": "tendency"}}},
        {"func": "lmtl:aging_flow", "inputs": {"production": "state.production", "cohort_ages": "parameters.cohort_ages", "rec_age": "derived.rec_age"}, "outputs": {"return": {"target": "tendencies.production", "type": "tendency"}}},
        {"func": "lmtl:recruitment_flow", "inputs": {"production": "state.production", "cohort_ages": "parameters.cohort_ages", "rec_age": "derived.rec_age"}, "outputs": {"prod_loss": {"target": "tendencies.production", "type": "tendency"}, "biomass_gain": {"target": "tendencies.biomass", "type": "tendency"}}},
        {"func": "lmtl:mortality", "inputs": {"biomass": "state.biomass", "temp": "derived.temp_norm", "lambda_0": "parameters.lambda_0", "gamma": "parameters.gamma_lambda", "t_ref": "parameters.t_ref"}, "outputs": {"return": {"target": "tendencies.biomass", "type": "tendency"}}},
        {"func": "phys:transport_tendency", "inputs": {"state": "state.biomass", "u": "forcings.u", "v": "forcings.v", "D": "forcings.D", "dx": "forcings.dx", "dy": "forcings.dy", "face_height": "forcings.face_height", "face_width": "forcings.face_width", "cell_area": "forcings.cell_area", "mask": "forcings.mask"}, "outputs": {"advection_rate": {"target": "tendencies.biomass", "type": "tendency"}, "diffusion_rate": {"target": "tendencies.biomass", "type": "tendency"}}},
    ],
})


def build_config(start="2000-01-01", end="2001-01-01"):
    sp, ep = pd.to_datetime(start), pd.to_datetime(end)
    nd = (ep - sp).days + 5
    dates = pd.date_range(start=sp, periods=nd, freq="D")
    lat, lon = np.arange(NY, dtype=np.float32), np.arange(NX, dtype=np.float32)
    doy = dates.dayofyear.values
    t3d = (
        np.broadcast_to((15.0 + 5.0 * np.sin(2 * np.pi * doy / 365.0))[:, None, None], (nd, NY, NX)).copy()
        + (2.0 * np.arange(NY) / NY)[None, :, None]
    )
    rng = np.random.default_rng(42)
    npp = (300e-3 / 86400.0 * rng.normal(1.0, 0.2, size=(nd, NY, NX)).clip(0.5, 2.0)).astype(np.float32)
    o = np.ones((NY, NX), dtype=np.float32)
    forcings = {
        "temperature": xr.DataArray(t3d.astype(np.float32), dims=["T", "Y", "X"], coords={"T": dates, "Y": lat, "X": lon}),
        "primary_production": xr.DataArray(npp, dims=["T", "Y", "X"], coords={"T": dates, "Y": lat, "X": lon}),
        "u": xr.DataArray(o * 0.5, dims=["Y", "X"], coords={"Y": lat, "X": lon}),
        "v": xr.DataArray(o * 0.0, dims=["Y", "X"], coords={"Y": lat, "X": lon}),
        "D": xr.DataArray(o * 500.0, dims=["Y", "X"], coords={"Y": lat, "X": lon}),
        "dx": xr.DataArray(o * DX_M, dims=["Y", "X"], coords={"Y": lat, "X": lon}),
        "dy": xr.DataArray(o * DY_M, dims=["Y", "X"], coords={"Y": lat, "X": lon}),
        "face_height": xr.DataArray(o * DY_M, dims=["Y", "X"], coords={"Y": lat, "X": lon}),
        "face_width": xr.DataArray(o * DX_M, dims=["Y", "X"], coords={"Y": lat, "X": lon}),
        "cell_area": xr.DataArray(o * DX_M * DY_M, dims=["Y", "X"], coords={"Y": lat, "X": lon}),
        "mask": xr.DataArray(o, dims=["Y", "X"], coords={"Y": lat, "X": lon}),
    }
    return Config.from_dict({
        "parameters": {
            "lambda_0": {"value": 1 / 150 / 86400.0}, "gamma_lambda": {"value": 0.15},
            "tau_r_0": {"value": 10.38 * 86400.0}, "gamma_tau_r": {"value": 0.11},
            "t_ref": {"value": 0.0}, "efficiency": {"value": 0.1668},
            "cohort_ages": xr.DataArray(COHORT_AGES_SEC, dims=["C"]),
        },
        "forcings": forcings,
        "initial_state": {
            "biomass": xr.DataArray(np.zeros((NY, NX), dtype=np.float32), dims=["Y", "X"], coords={"Y": lat, "X": lon}),
            "production": xr.DataArray(np.zeros((NY, NX, N_COHORTS), dtype=np.float32), dims=["Y", "X", "C"], coords={"Y": lat, "X": lon}),
        },
        "execution": {"time_start": start, "time_end": end, "dt": "1d", "forcing_interpolation": "linear"},
    })


# =============================================================================
# Benchmark runner
# =============================================================================
def bench(device_name, batch_size, chunk_size, n_runs=3):
    """Run one benchmark config. Returns (jit_time, avg_time, sims_per_sec)."""
    device = jax.devices(device_name)[0]
    jax.config.update("jax_default_device", device)

    config = build_config()
    model = compile_model(BLUEPRINT, config, backend="jax")
    model.parameters = jax.device_put(model.parameters, device)
    model.forcings = jax.device_put(model.forcings, device)
    model.state = jax.device_put(model.state, device)

    runner = SobolRunner(model, EXTRACTION_POINTS, "biomass", chunk_size)
    rng = np.random.default_rng(0)
    pb = {k: jnp.broadcast_to(v, (batch_size,) + v.shape) for k, v in model.parameters.items()}
    for name, (lo, hi) in PARAM_BOUNDS.items():
        pb[name] = jnp.array(rng.uniform(lo, hi, size=batch_size))
    pb = jax.device_put(pb, device)

    timings = []
    for _ in range(n_runs):
        t0 = time.time()
        result = runner.run_batch(pb, batch_size)
        jax.block_until_ready(result)
        timings.append(time.time() - t0)

    del runner, pb, model, result, config
    jax.clear_caches()
    gc.collect()

    jit_t = timings[0]
    avg_t = float(np.mean(timings[1:]))
    return jit_t, avg_t, batch_size / avg_t


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SOBOL BENCHMARK: CPU vs GPU — LMTL+Transport 20x20, 1 year")
    print("=" * 70)
    print(f"JAX {jax.__version__} | Devices: {jax.devices()}")

    has_gpu = len(jax.devices("gpu")) > 0

    # ------------------------------------------------------------------
    # Part 1: CPU vs GPU (chunk=365, varying batch)
    # ------------------------------------------------------------------
    BATCH_SIZES_CMP = [64, 256]
    cpu_results = {}
    gpu_results_cmp = {}

    for bs in BATCH_SIZES_CMP:
        print(f"  CPU  batch={bs:5d} chunk=365 ... ", end="", flush=True)
        jit_t, avg_t, sps = bench("cpu", bs, 365)
        print(f"avg={avg_t:.2f}s  {sps:.0f} sim/s")
        cpu_results[bs] = sps

        if has_gpu:
            print(f"  GPU  batch={bs:5d} chunk=365 ... ", end="", flush=True)
            jit_t, avg_t, sps = bench("gpu", bs, 365)
            print(f"avg={avg_t:.2f}s  {sps:.0f} sim/s")
            gpu_results_cmp[bs] = sps

    # ------------------------------------------------------------------
    # Part 2: GPU sweep (chunk × batch)
    # ------------------------------------------------------------------
    # Safe configs for 12 GB VRAM (batch × chunk × 17 × 20 × 20 × 4 < ~10 GB)
    GPU_CONFIGS = [
        # (chunk, batch)
        (30, 64), (30, 256), (30, 512), (30, 1024),
        (60, 64), (60, 256), (60, 512), (60, 1024),
        (120, 64), (120, 256), (120, 512),
        (365, 64), (365, 256),
    ]

    gpu_sweep = {}  # (chunk, batch) -> sps
    if has_gpu:
        print("\n  GPU sweep: chunk × batch")
        for chunk, bs in GPU_CONFIGS:
            print(f"  GPU  batch={bs:5d} chunk={chunk:3d} ... ", end="", flush=True)
            try:
                _, avg_t, sps = bench("gpu", bs, chunk)
                print(f"avg={avg_t:.2f}s  {sps:.0f} sim/s")
                gpu_sweep[(chunk, bs)] = sps
            except Exception as e:
                print(f"OOM")
                gpu_sweep[(chunk, bs)] = None
                jax.clear_caches()
                gc.collect()

    # ------------------------------------------------------------------
    # Part 3: Figures
    # ------------------------------------------------------------------
    TARGET_SIMS = 1_000_000
    SCALE_5Y = 5  # 1-year benchmark → 5-year extrapolation

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Figure 1: CPU vs GPU bar chart ---
    x = np.arange(len(BATCH_SIZES_CMP))
    w = 0.35
    cpu_sps = [cpu_results[bs] for bs in BATCH_SIZES_CMP]
    gpu_sps = [gpu_results_cmp.get(bs, 0) for bs in BATCH_SIZES_CMP]

    bars_cpu = ax1.bar(x - w / 2, cpu_sps, w, label="CPU", color="#5C6BC0")
    bars_gpu = ax1.bar(x + w / 2, gpu_sps, w, label="GPU", color="#26A69A")

    # Speedup annotations
    for i, bs in enumerate(BATCH_SIZES_CMP):
        if bs in gpu_results_cmp and cpu_results[bs] > 0:
            speedup = gpu_results_cmp[bs] / cpu_results[bs]
            ax1.annotate(
                f"{speedup:.0f}x",
                xy=(x[i] + w / 2, gpu_sps[i]),
                ha="center", va="bottom", fontsize=11, fontweight="bold", color="#00695C",
            )
        # Hours for 1M × 5y
        for bar, sps, color in [(bars_cpu[i], cpu_sps[i], "#283593"), (bars_gpu[i], gpu_sps[i], "#004D40")]:
            if sps > 0:
                hours = TARGET_SIMS / sps * SCALE_5Y / 3600
                label = f"{hours:.1f}h" if hours >= 1 else f"{hours * 60:.0f}min"
                ax1.text(bar.get_x() + bar.get_width() / 2, sps / 2, label,
                         ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"batch={bs}" for bs in BATCH_SIZES_CMP])
    ax1.set_ylabel("Throughput (sim/s) — log scale")
    ax1.set_title("CPU vs GPU  (chunk=365)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3, which="both")

    # --- Figure 2: GPU heatmap — hours for 1M × 5y ---
    if has_gpu and gpu_sweep:
        chunks_all = sorted(set(c for c, _ in gpu_sweep))
        batches_all = sorted(set(b for _, b in gpu_sweep))

        hours_grid = np.full((len(chunks_all), len(batches_all)), np.nan)
        for ci, chunk in enumerate(chunks_all):
            for bi, bs in enumerate(batches_all):
                sps = gpu_sweep.get((chunk, bs))
                if sps is not None and sps > 0:
                    hours_grid[ci, bi] = TARGET_SIMS / sps * SCALE_5Y / 3600

        im = ax2.imshow(hours_grid, aspect="auto", cmap="RdYlGn_r", origin="lower")
        cbar = fig.colorbar(im, ax=ax2, label="Hours for 1M × 5y")

        # Annotate cells
        for ci in range(len(chunks_all)):
            for bi in range(len(batches_all)):
                val = hours_grid[ci, bi]
                if np.isnan(val):
                    ax2.text(bi, ci, "OOM", ha="center", va="center", fontsize=8, color="gray")
                else:
                    label = f"{val:.1f}h" if val >= 1 else f"{val * 60:.0f}min"
                    text_color = "white" if val > np.nanmedian(hours_grid) else "black"
                    ax2.text(bi, ci, label, ha="center", va="center", fontsize=9, fontweight="bold", color=text_color)

        ax2.set_xticks(range(len(batches_all)))
        ax2.set_xticklabels([str(b) for b in batches_all])
        ax2.set_yticks(range(len(chunks_all)))
        ax2.set_yticklabels([str(c) for c in chunks_all])
        ax2.set_xlabel("Batch size")
        ax2.set_ylabel("Chunk size (days)")
        ax2.set_title("GPU: time for 1M simulations × 5 years", fontsize=12, fontweight="bold")

    fig.suptitle(
        f"Sobol Runner — LMTL+Transport 2D ({NY}×{NX}, {N_COHORTS} cohorts)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out_path = "examples/benchmark_sobol_cpu_vs_gpu.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out_path}")
