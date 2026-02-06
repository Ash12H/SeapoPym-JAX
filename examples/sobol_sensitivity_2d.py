"""Example: Sobol Sensitivity Analysis — LMTL + Transport on a 20x20 grid.

Demonstrates variance-based global sensitivity analysis on the full
LMTL ecosystem model with advective transport:
- Sinusoidal temperature (15 +/- 5 degC)
- Noisy primary production (~300 mg/m2/day)
- Eastward current (0.5 m/s), closed boundaries
- Sobol analysis on 3 parameters: efficiency, lambda_0, tau_r_0

Part 1: Sobol analysis on 30-day simulation (quick)
Part 2: Benchmark — average batch time for 5-year simulation
Part 3: Summary plot

Expected runtime: ~2 min on CPU (N=64, D=3 -> 320 evaluations).
"""

import time

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Register LMTL and transport functions
import seapopym.functions.lmtl  # noqa: F401
import seapopym.functions.transport  # noqa: F401
from seapopym.blueprint import Blueprint, Config
from seapopym.compiler import compile_model
from seapopym.sensitivity import SobolAnalyzer
from seapopym.sensitivity.runner import SobolRunner

jax.config.update("jax_default_device", jax.devices("cpu")[0])

# =============================================================================
# 1. MODEL DEFINITION — LMTL + Transport
# =============================================================================

# LMTL parameters
LMTL_E = 0.1668
LMTL_LAMBDA_0 = 1 / 150 / 86400.0  # 1/s
LMTL_GAMMA_LAMBDA = 0.15  # 1/degC
LMTL_TAU_R_0 = 10.38 * 86400.0  # seconds
LMTL_GAMMA_TAU_R = 0.11  # 1/degC
LMTL_T_REF = 0.0  # degC

# Cohort system (sized for max tau_r_0 in Sobol bounds: 15 days)
MAX_AGE_DAYS = 16
COHORT_AGES_SEC = np.arange(MAX_AGE_DAYS + 1) * 86400.0
N_COHORTS = len(COHORT_AGES_SEC)

# Grid
NY, NX = 20, 20
DX_M = 111_000.0  # ~1 degree at equator in meters
DY_M = 111_000.0

blueprint = Blueprint.from_dict(
    {
        "id": "sobol-lmtl-transport-2d",
        "version": "1.0",
        "declarations": {
            "state": {
                "biomass": {"units": "g/m^2", "dims": ["Y", "X"]},
                "production": {"units": "g/m^2", "dims": ["Y", "X", "C"]},
            },
            "parameters": {
                "lambda_0": {"units": "1/s"},
                "gamma_lambda": {"units": "1/delta_degC"},
                "tau_r_0": {"units": "s"},
                "gamma_tau_r": {"units": "1/delta_degC"},
                "t_ref": {"units": "degC"},
                "efficiency": {"units": "dimensionless"},
                "cohort_ages": {"units": "s", "dims": ["C"]},
            },
            "forcings": {
                "temperature": {"units": "degC", "dims": ["T", "Y", "X"]},
                "primary_production": {"units": "g/m^2/s", "dims": ["T", "Y", "X"]},
                # Transport grid (static)
                "u": {"units": "m/s", "dims": ["Y", "X"]},
                "v": {"units": "m/s", "dims": ["Y", "X"]},
                "D": {"units": "m^2/s", "dims": ["Y", "X"]},
                "dx": {"units": "m", "dims": ["Y", "X"]},
                "dy": {"units": "m", "dims": ["Y", "X"]},
                "face_height": {"units": "m", "dims": ["Y", "X"]},
                "face_width": {"units": "m", "dims": ["Y", "X"]},
                "cell_area": {"units": "m^2", "dims": ["Y", "X"]},
                "mask": {"units": "dimensionless", "dims": ["Y", "X"]},
            },
        },
        "process": [
            # --- LMTL Biology ---
            {
                "func": "lmtl:gillooly_temperature",
                "inputs": {"temp": "forcings.temperature"},
                "outputs": {"return": {"target": "derived.temp_norm", "type": "derived"}},
            },
            {
                "func": "lmtl:recruitment_age",
                "inputs": {
                    "temp": "derived.temp_norm",
                    "tau_r_0": "parameters.tau_r_0",
                    "gamma": "parameters.gamma_tau_r",
                    "t_ref": "parameters.t_ref",
                },
                "outputs": {"return": {"target": "derived.rec_age", "type": "derived"}},
            },
            {
                "func": "lmtl:npp_injection",
                "inputs": {
                    "npp": "forcings.primary_production",
                    "efficiency": "parameters.efficiency",
                    "production": "state.production",
                },
                "outputs": {"return": {"target": "tendencies.production", "type": "tendency"}},
            },
            {
                "func": "lmtl:aging_flow",
                "inputs": {
                    "production": "state.production",
                    "cohort_ages": "parameters.cohort_ages",
                    "rec_age": "derived.rec_age",
                },
                "outputs": {"return": {"target": "tendencies.production", "type": "tendency"}},
            },
            {
                "func": "lmtl:recruitment_flow",
                "inputs": {
                    "production": "state.production",
                    "cohort_ages": "parameters.cohort_ages",
                    "rec_age": "derived.rec_age",
                },
                "outputs": {
                    "prod_loss": {"target": "tendencies.production", "type": "tendency"},
                    "biomass_gain": {"target": "tendencies.biomass", "type": "tendency"},
                },
            },
            {
                "func": "lmtl:mortality",
                "inputs": {
                    "biomass": "state.biomass",
                    "temp": "derived.temp_norm",
                    "lambda_0": "parameters.lambda_0",
                    "gamma": "parameters.gamma_lambda",
                    "t_ref": "parameters.t_ref",
                },
                "outputs": {"return": {"target": "tendencies.biomass", "type": "tendency"}},
            },
            # --- Transport (advection + diffusion on biomass) ---
            {
                "func": "phys:transport_tendency",
                "inputs": {
                    "state": "state.biomass",
                    "u": "forcings.u",
                    "v": "forcings.v",
                    "D": "forcings.D",
                    "dx": "forcings.dx",
                    "dy": "forcings.dy",
                    "face_height": "forcings.face_height",
                    "face_width": "forcings.face_width",
                    "cell_area": "forcings.cell_area",
                    "mask": "forcings.mask",
                },
                "outputs": {
                    "advection_rate": {"target": "tendencies.biomass", "type": "tendency"},
                    "diffusion_rate": {"target": "tendencies.biomass", "type": "tendency"},
                },
            },
        ],
    }
)


# =============================================================================
# 2. HELPER: Build config for a given time range
# =============================================================================


def build_config(start_date: str, end_date: str, dt: str = "1d") -> Config:
    """Build config with synthetic forcings for the given time range."""
    start_pd = pd.to_datetime(start_date)
    end_pd = pd.to_datetime(end_date)
    n_days = (end_pd - start_pd).days + 5  # margin
    dates = pd.date_range(start=start_pd, periods=n_days, freq="D")

    lat = np.arange(NY, dtype=np.float32)
    lon = np.arange(NX, dtype=np.float32)
    rng = np.random.default_rng(42)

    # Temperature: sinusoidal 15 +/- 5 degC with spatial gradient
    day_of_year = dates.dayofyear.values
    temp_base = 15.0 + 5.0 * np.sin(2 * np.pi * day_of_year / 365.0)
    # Add latitudinal gradient: +2 degC from north to south
    lat_gradient = 2.0 * np.arange(NY) / NY
    temp_3d = np.broadcast_to(
        temp_base[:, None, None], (n_days, NY, NX)
    ).copy() + lat_gradient[None, :, None]
    temp_da = xr.DataArray(temp_3d.astype(np.float32), dims=["T", "Y", "X"], coords={"T": dates, "Y": lat, "X": lon})

    # NPP: ~300 mg/m2/day = 3.47e-6 g/m2/s with noise
    npp_base = 300e-3 / 86400.0  # g/m2/s
    npp_noise = rng.normal(1.0, 0.2, size=(n_days, NY, NX)).clip(0.5, 2.0)
    npp_3d = npp_base * npp_noise
    npp_da = xr.DataArray(npp_3d.astype(np.float32), dims=["T", "Y", "X"], coords={"T": dates, "Y": lat, "X": lon})

    # Transport grid (static, uniform)
    ones = np.ones((NY, NX), dtype=np.float32)
    forcings = {
        "temperature": temp_da,
        "primary_production": npp_da,
        "u": xr.DataArray(ones * 0.5, dims=["Y", "X"], coords={"Y": lat, "X": lon}),  # 0.5 m/s east
        "v": xr.DataArray(ones * 0.0, dims=["Y", "X"], coords={"Y": lat, "X": lon}),
        "D": xr.DataArray(ones * 500.0, dims=["Y", "X"], coords={"Y": lat, "X": lon}),  # 500 m2/s
        "dx": xr.DataArray(ones * DX_M, dims=["Y", "X"], coords={"Y": lat, "X": lon}),
        "dy": xr.DataArray(ones * DY_M, dims=["Y", "X"], coords={"Y": lat, "X": lon}),
        "face_height": xr.DataArray(ones * DY_M, dims=["Y", "X"], coords={"Y": lat, "X": lon}),
        "face_width": xr.DataArray(ones * DX_M, dims=["Y", "X"], coords={"Y": lat, "X": lon}),
        "cell_area": xr.DataArray(ones * DX_M * DY_M, dims=["Y", "X"], coords={"Y": lat, "X": lon}),
        "mask": xr.DataArray(ones, dims=["Y", "X"], coords={"Y": lat, "X": lon}),
    }

    return Config.from_dict(
        {
            "parameters": {
                "lambda_0": {"value": LMTL_LAMBDA_0},
                "gamma_lambda": {"value": LMTL_GAMMA_LAMBDA},
                "tau_r_0": {"value": LMTL_TAU_R_0},
                "gamma_tau_r": {"value": LMTL_GAMMA_TAU_R},
                "t_ref": {"value": LMTL_T_REF},
                "efficiency": {"value": LMTL_E},
                "cohort_ages": xr.DataArray(COHORT_AGES_SEC, dims=["C"]),
            },
            "forcings": forcings,
            "initial_state": {
                "biomass": xr.DataArray(np.zeros((NY, NX), dtype=np.float32), dims=["Y", "X"], coords={"Y": lat, "X": lon}),
                "production": xr.DataArray(
                    np.zeros((NY, NX, N_COHORTS), dtype=np.float32), dims=["Y", "X", "C"], coords={"Y": lat, "X": lon}
                ),
            },
            "execution": {
                "time_start": start_date,
                "time_end": end_date,
                "dt": dt,
                "forcing_interpolation": "linear",
            },
        }
    )


# =============================================================================
# 3. SOBOL ANALYSIS (30 days)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SOBOL SENSITIVITY — LMTL + Transport (20x20 grid)")
    print("=" * 70)

    # --- Compile 30-day model ---
    config_30d = build_config("2000-01-01", "2000-01-31", dt="1d")
    model_30d = compile_model(blueprint, config_30d, backend="jax")
    print(f"Model compiled: {model_30d.n_timesteps} timesteps, grid {NY}x{NX}, {N_COHORTS} cohorts")

    # --- Extraction points ---
    extraction_points = [
        (5, 10),   # north-center
        (10, 5),   # center-west
        (10, 15),  # center-east
        (15, 10),  # south-center
    ]
    POINT_LABELS = ["N-center", "C-west", "C-east", "S-center"]

    # --- Sobol parameters ---
    N_SAMPLES = 64
    D = 3
    N_TOTAL = N_SAMPLES * (D + 2)
    PARAM_BOUNDS = {
        "efficiency": (0.05, 0.30),
        "lambda_0": (3e-8, 3e-6),
        "tau_r_0": (5 * 86400.0, 15 * 86400.0),
    }
    QOI_NAMES = ["mean", "var", "max"]

    print(f"\nSobol analysis: N={N_SAMPLES}, D={D} -> {N_TOTAL} evaluations")
    print(f"Parameters: {list(PARAM_BOUNDS.keys())}")
    print(f"QoI: {QOI_NAMES}")
    print(f"Extraction points: {dict(zip(POINT_LABELS, extraction_points))}")

    analyzer = SobolAnalyzer(model_30d)
    t0 = time.time()
    result = analyzer.analyze(
        param_bounds=PARAM_BOUNDS,
        extraction_points=extraction_points,
        output_variable="biomass",
        n_samples=N_SAMPLES,
        qoi=QOI_NAMES,
        batch_size=64,
        chunk_size=30,
    )
    t_sobol = time.time() - t0

    print(f"\nSobol analysis completed in {t_sobol:.1f}s")

    # --- Display results ---
    print("\n" + "=" * 70)
    print("FIRST-ORDER INDICES (S1)")
    print("=" * 70)
    print(result.S1.to_string(float_format="%.4f"))

    print("\n" + "=" * 70)
    print("TOTAL-ORDER INDICES (ST)")
    print("=" * 70)
    print(result.ST.to_string(float_format="%.4f"))

    # =============================================================================
    # 4. BENCHMARK — 5 years, time per batch
    # =============================================================================

    print("\n" + "=" * 70)
    print("BENCHMARK — 5-year simulation, batch timing")
    print("=" * 70)

    config_5y = build_config("2000-01-01", "2005-01-01", dt="1d")
    model_5y = compile_model(blueprint, config_5y, backend="jax")
    print(f"5-year model: {model_5y.n_timesteps} timesteps")

    import jax.numpy as jnp

    runner_5y = SobolRunner(model_5y, extraction_points, "biomass", chunk_size=365)
    batch_size = 64

    # Build a full batch of parameters (model defaults + varied params)
    rng = np.random.default_rng(0)
    params_batch = {k: jnp.broadcast_to(v, (batch_size,) + v.shape) for k, v in model_5y.parameters.items()}
    for name, (lo, hi) in PARAM_BOUNDS.items():
        params_batch[name] = jnp.array(rng.uniform(lo, hi, size=batch_size))

    # Run 3 batches (first includes JIT compilation)
    N_BENCH = 3
    timings = []
    for i in range(N_BENCH):
        t0 = time.time()
        _ = runner_5y.run_batch(params_batch, batch_size)
        jax.block_until_ready(_)
        elapsed = time.time() - t0
        timings.append(elapsed)
        print(f"  Batch {i + 1}/{N_BENCH}: {elapsed:.2f}s")

    print(f"\n  First batch (incl. JIT): {timings[0]:.2f}s")
    print(f"  Average (excl. JIT):     {np.mean(timings[1:]):.2f}s")
    print(f"  -> {batch_size} evaluations x {model_5y.n_timesteps} timesteps x {NY}x{NX} grid")

    # =============================================================================
    # 5. SUMMARY PLOT
    # =============================================================================

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    param_names = list(PARAM_BOUNDS.keys())
    param_labels = ["efficiency", r"$\lambda_0$", r"$\tau_{r,0}$"]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    n_points = len(extraction_points)
    bar_width = 0.18

    for ax_idx, qoi_name in enumerate(QOI_NAMES):
        ax = axes[ax_idx]
        x = np.arange(n_points)

        for p_idx, (p_name, p_label) in enumerate(zip(param_names, param_labels)):
            # Extract S1 values for this QoI across all points
            s1_vals = [result.S1.loc[(qoi_name, pt), p_name] for pt in range(n_points)]
            st_vals = [result.ST.loc[(qoi_name, pt), p_name] for pt in range(n_points)]

            offset = (p_idx - 1) * bar_width
            bars = ax.bar(x + offset, s1_vals, bar_width * 0.9, label=f"{p_label} (S1)", color=colors[p_idx], alpha=0.85)
            # ST as transparent overlay
            ax.bar(x + offset, st_vals, bar_width * 0.9, fill=False, edgecolor=colors[p_idx], linewidth=1.5, linestyle="--")

        ax.set_title(f"QoI: {qoi_name}", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(POINT_LABELS, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(-0.1, 1.3)
        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.axhline(y=1, color="gray", linewidth=0.5, linestyle=":")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Sobol Index", fontsize=12)

    # Custom legend
    from matplotlib.patches import Patch

    legend_elements = []
    for p_idx, (p_label, c) in enumerate(zip(param_labels, colors)):
        legend_elements.append(Patch(facecolor=c, alpha=0.85, label=f"{p_label} — S1"))
        legend_elements.append(Patch(facecolor="none", edgecolor=c, linestyle="--", linewidth=1.5, label=f"{p_label} — ST"))
    fig.legend(handles=legend_elements, loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(
        f"Sobol Sensitivity — LMTL+Transport 2D ({NY}x{NX})  |  N={N_SAMPLES}, {N_TOTAL} eval.",
        fontsize=14,
        fontweight="bold",
        y=1.08,
    )
    plt.tight_layout()
    plt.savefig("examples/sobol_sensitivity_2d.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: examples/sobol_sensitivity_2d.png")
