"""CMA-ES → NUTS Pipeline on LMTL 0D Model (Twin Experiment).

Two-stage Bayesian parameter estimation:
1. IPOP-CMA-ES: gradient-free global search to find a good point estimate
   for ALL parameters, especially tau_r_0/gamma_tau_r (weak gradient).
2. NUTS: Bayesian posterior sampling for the gradient-rich parameters
   (lambda_0, gamma_lambda, efficiency), initialized at the CMA-ES optimum.
   tau_r_0/gamma_tau_r are fixed at CMA-ES values.

Why a pipeline?
    The sigmoid recruitment has ~1000x weaker gradient sensitivity for
    tau_r_0/gamma_tau_r vs the other parameters (daily cohorts, ±1 day
    transition). NUTS alone produces wide, unreliable posteriors for these.
    CMA-ES handles them well (gradient-free), and NUTS refines the rest.
"""

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import seapopym.functions.lmtl  # noqa: F401
from seapopym.blueprint import Blueprint, Config
from seapopym.compiler import compile_model
from seapopym.engine.step import build_step_fn
from seapopym.optimization.ipop import run_ipop_cmaes
from seapopym.optimization.likelihood import make_log_posterior, reparameterize_log_posterior
from seapopym.optimization.nuts import run_nuts
from seapopym.optimization.prior import HalfNormal, PriorSet, Uniform

# Use GPU if available, fall back to CPU
print(f"JAX devices: {jax.devices()}")

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Stage 1: CMA-ES (Hansen defaults) ---
N_RESTARTS = 3
N_PARAMS = 5  # number of optimized parameters
INITIAL_POPSIZE = 4 + int(3 * np.log(N_PARAMS))  # Hansen: 4 + floor(3*ln(n))
N_GENERATIONS = int(100 + 150 * (N_PARAMS + 3) ** 2 / np.sqrt(INITIAL_POPSIZE))
DISTANCE_THRESHOLD = 0.1
CMAES_SEED = 42

# --- Stage 2: NUTS ---
N_WARMUP = 100
N_SAMPLES = 200
NUTS_SEED = 0

# Parameters estimated by NUTS (gradient-rich)
NUTS_PARAMS = ["lambda_0", "gamma_lambda", "efficiency"]
# Parameters fixed from CMA-ES (weak gradient)
CMAES_ONLY_PARAMS = ["tau_r_0", "gamma_tau_r"]

# --- Simulation ---
SPINUP_YEARS = 1
OPT_YEARS = 1
DT = "1d"

# --- Twin experiment ---
OBS_FRACTION = 0.1
INITIAL_GUESS_FACTOR = 1.5

# True biological parameters
TRUE_PARAMS = {
    "lambda_0": 1 / 150 / 86400,
    "gamma_lambda": 0.15,
    "tau_r_0": 10.38 * 86400,
    "gamma_tau_r": 0.11,
    "efficiency": 0.1668,
    "t_ref": 0.0,
}

ALL_OPT_PARAMS = ["lambda_0", "gamma_lambda", "tau_r_0", "gamma_tau_r", "efficiency"]

# Bounds (shared by CMA-ES and NUTS priors)
BOUNDS = {
    "lambda_0": (1e-10, 5 * TRUE_PARAMS["lambda_0"]),
    "gamma_lambda": (0.01, 5 * TRUE_PARAMS["gamma_lambda"]),
    "tau_r_0": (0.1 * TRUE_PARAMS["tau_r_0"], 5 * TRUE_PARAMS["tau_r_0"]),
    "gamma_tau_r": (0.01, 5 * TRUE_PARAMS["gamma_tau_r"]),
    "efficiency": (0.01, 5 * TRUE_PARAMS["efficiency"]),
}

# NUTS priors
# HalfNormal for parameters inside exponentials (concentrate mass near small values)
# scale chosen so that ~95% of mass is below 2× the true value
NUTS_PRIORS = PriorSet(
    {
        "lambda_0": HalfNormal(scale=3 * TRUE_PARAMS["lambda_0"]),
        "gamma_lambda": HalfNormal(scale=0.3),
        "efficiency": Uniform(*BOUNDS["efficiency"]),
    }
)

FIXED_PARAMS = {"t_ref": TRUE_PARAMS["t_ref"]}

# =============================================================================
# BLUEPRINT (0D LMTL)
# =============================================================================

max_age_days = int(np.ceil(TRUE_PARAMS["tau_r_0"] / 86400))
cohort_ages_sec = np.arange(0, max_age_days + 1) * 86400.0
n_cohorts = len(cohort_ages_sec)

blueprint = Blueprint.from_dict(
    {
        "id": "lmtl-pipeline-demo",
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
            },
        },
        "process": [
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
        ],
    }
)

# =============================================================================
# FORCINGS & CONFIG
# =============================================================================

total_years = SPINUP_YEARS + OPT_YEARS
start_date = "2000-01-01"
end_date = str((pd.Timestamp(start_date) + pd.DateOffset(years=total_years)).date())

dates = pd.date_range(
    start=pd.to_datetime(start_date),
    periods=(pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 5,
    freq="D",
)
ny, nx = 1, 1
lat, lon = np.arange(ny), np.arange(nx)

day_of_year = dates.dayofyear.values
temp_c = 15.0 + 5.0 * np.sin(2 * np.pi * day_of_year / 365.0)
temp_da = xr.DataArray(
    np.broadcast_to(temp_c[:, None, None], (len(dates), ny, nx)),
    dims=["T", "Y", "X"],
    coords={"T": dates, "Y": lat, "X": lon},
)
npp_sec = (1.0 + 0.5 * np.sin(2 * np.pi * day_of_year / 365.0)) / 86400.0
npp_da = xr.DataArray(
    np.broadcast_to(npp_sec[:, None, None], (len(dates), ny, nx)),
    dims=["T", "Y", "X"],
    coords={"T": dates, "Y": lat, "X": lon},
)

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
        },
        "forcings": {"temperature": temp_da, "primary_production": npp_da},
        "initial_state": {
            "biomass": xr.DataArray(np.zeros((ny, nx)), dims=["Y", "X"], coords={"Y": lat, "X": lon}),
            "production": xr.DataArray(
                np.zeros((ny, nx, n_cohorts)), dims=["Y", "X", "C"], coords={"Y": lat, "X": lon}
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

# =============================================================================
# COMPILE & BUILD SIMULATION
# =============================================================================

print("Compiling model...")
_model = compile_model(blueprint, config, backend="jax")
_step_fn = build_step_fn(_model, params_as_argument=True)
_n_timesteps = _model.n_timesteps
_initial_state = _model.state
_forcings_stacked = _model.forcings.get_all()

_spinup_steps = int(SPINUP_YEARS / total_years * _n_timesteps)


def run_simulation(params: dict) -> jnp.ndarray:
    """Run full simulation, return biomass time series."""
    full_params = {**params, **{k: jnp.array(v) for k, v in FIXED_PARAMS.items()}}
    full_params["cohort_ages"] = _model.parameters["cohort_ages"]

    def scan_body(carry, t):
        state, p = carry
        forcings_t = {}
        for name, arr in _forcings_stacked.items():
            if arr.ndim > 0 and arr.shape[0] == _n_timesteps:
                forcings_t[name] = arr[t]
            else:
                forcings_t[name] = arr
        new_carry, outputs = _step_fn((state, p), forcings_t)
        new_state, _ = new_carry
        return (new_state, p), jnp.mean(outputs["biomass"])

    _, biomass = jax.lax.scan(scan_body, (_initial_state, full_params), jnp.arange(_n_timesteps))
    return biomass


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CMA-ES → NUTS Pipeline on LMTL 0D (Twin Experiment)")
    print(f"  Spin-up: {SPINUP_YEARS}y | Optimization: {OPT_YEARS}y | dt: {DT}")
    print(f"  Stage 1 (CMA-ES): {N_RESTARTS} restarts, {N_GENERATIONS} gen")
    print(f"  Stage 2 (NUTS):   {N_WARMUP} warmup + {N_SAMPLES} samples")
    print(f"  NUTS params:      {NUTS_PARAMS}")
    print(f"  Fixed from CMA-ES: {CMAES_ONLY_PARAMS}")
    print("=" * 60)

    # ----- Generate observations -----
    print("\nGenerating observations with TRUE parameters...")
    t0 = time.time()
    true_params_jax = {k: jnp.array(TRUE_PARAMS[k]) for k in ALL_OPT_PARAMS}
    true_biomass_full = run_simulation(true_params_jax)

    true_biomass = true_biomass_full[_spinup_steps:]
    n_opt_steps = len(true_biomass)

    n_obs = max(1, int(OBS_FRACTION * n_opt_steps))
    rng = np.random.default_rng(CMAES_SEED)
    obs_local_indices = np.sort(rng.choice(n_opt_steps, size=n_obs, replace=False))
    obs_global_indices = obs_local_indices + _spinup_steps
    observations = true_biomass[obs_local_indices]
    obs_std = float(jnp.std(observations))
    print(f"  {n_obs} observations, std={obs_std:.6f}")

    def loss_fn(params: dict) -> jnp.ndarray:
        biomass_full = run_simulation(params)
        pred = biomass_full[obs_global_indices]
        rmse = jnp.sqrt(jnp.mean((pred - observations) ** 2))
        return rmse / obs_std

    true_loss = loss_fn(true_params_jax)
    print(f"  Sanity: loss(true) = {float(true_loss):.6e}")

    # =====================================================================
    # STAGE 1: IPOP-CMA-ES (all parameters)
    # =====================================================================
    print("\n" + "=" * 60)
    print("STAGE 1: IPOP-CMA-ES")
    print("=" * 60)

    initial_params = {k: jnp.array(INITIAL_GUESS_FACTOR * TRUE_PARAMS[k]) for k in ALL_OPT_PARAMS}
    print(f"Initial guess ({INITIAL_GUESS_FACTOR}x true), loss = {float(loss_fn(initial_params)):.6f}")

    t0 = time.time()
    cmaes_result = run_ipop_cmaes(
        loss_fn=loss_fn,
        initial_params=initial_params,
        bounds=BOUNDS,
        n_restarts=N_RESTARTS,
        initial_popsize=INITIAL_POPSIZE,
        n_generations=N_GENERATIONS,
        distance_threshold=DISTANCE_THRESHOLD,
        seed=CMAES_SEED,
        verbose=True,
    )
    cmaes_elapsed = time.time() - t0

    best_cmaes = cmaes_result.modes[0]
    print(f"\nCMA-ES completed in {cmaes_elapsed:.1f}s, best loss = {best_cmaes.loss:.6e}")
    print("CMA-ES best parameters:")
    for p in ALL_OPT_PARAMS:
        ratio = float(best_cmaes.params[p]) / TRUE_PARAMS[p]
        print(f"  {p:<14} = {float(best_cmaes.params[p]):>12.4g}  (true: {TRUE_PARAMS[p]:>12.4g}, ratio: {ratio:.4f})")

    # =====================================================================
    # STAGE 2: NUTS (gradient-rich parameters only)
    # =====================================================================
    print("\n" + "=" * 60)
    print("STAGE 2: NUTS")
    print("=" * 60)

    # Fix tau_r_0 and gamma_tau_r at CMA-ES values
    fixed_from_cmaes = {k: best_cmaes.params[k] for k in CMAES_ONLY_PARAMS}
    print("Fixed from CMA-ES:")
    for k, v in fixed_from_cmaes.items():
        print(f"  {k} = {float(v):.4g}")

    # Build loss function that only takes NUTS params
    def nuts_loss_fn(nuts_params: dict) -> jnp.ndarray:
        full_params = {**nuts_params, **fixed_from_cmaes}
        return loss_fn(full_params)

    # Log-posterior in unit space
    log_posterior = make_log_posterior(nuts_loss_fn, NUTS_PRIORS)
    log_posterior_unit = reparameterize_log_posterior(log_posterior, NUTS_PRIORS)

    # Initialize NUTS at CMA-ES optimum (in unit space)
    nuts_init_phys = {k: best_cmaes.params[k] for k in NUTS_PARAMS}
    nuts_init_unit = NUTS_PRIORS.to_unit(nuts_init_phys)

    print(f"\nNUTS initial (from CMA-ES):")
    for p in NUTS_PARAMS:
        print(f"  {p:<14} = {float(nuts_init_phys[p]):>12.4g} [unit: {float(nuts_init_unit[p]):.4f}]")

    print(f"\nRunning NUTS ({N_WARMUP} warmup + {N_SAMPLES} samples)...")
    t0 = time.time()

    nuts_result = run_nuts(
        log_posterior_fn=log_posterior_unit,
        initial_params=nuts_init_unit,
        n_warmup=N_WARMUP,
        n_samples=N_SAMPLES,
        seed=NUTS_SEED,
        target_acceptance_rate=0.85,
    )

    # Convert back to physical space
    nuts_result.samples = NUTS_PRIORS.from_unit(nuts_result.samples)

    nuts_elapsed = time.time() - t0
    print(f"Completed in {nuts_elapsed:.1f}s")
    print(f"  Acceptance rate: {nuts_result.acceptance_rate:.2%}")
    print(f"  Divergences: {int(jnp.sum(nuts_result.divergences))} / {N_SAMPLES}")

    # =====================================================================
    # RESULTS
    # =====================================================================
    print("\n" + "=" * 60)
    print("COMBINED RESULTS")
    print("=" * 60)

    header = f"{'Param':<14} {'True':>12} {'Estimate':>12} {'Std':>12} {'Source':>10}"
    print(header)
    print("-" * len(header))

    for p in ALL_OPT_PARAMS:
        true_val = TRUE_PARAMS[p]
        if p in NUTS_PARAMS:
            samples = nuts_result.samples[p]
            mean = float(jnp.mean(samples))
            std = float(jnp.std(samples))
            source = "NUTS"
        else:
            mean = float(best_cmaes.params[p])
            std = 0.0
            source = "CMA-ES"
        print(f"{p:<14} {true_val:>12.4g} {mean:>12.4g} {std:>12.4g} {source:>10}")

    # =====================================================================
    # VISUALIZATION
    # =====================================================================
    n_nuts = len(NUTS_PARAMS)
    fig, axes = plt.subplots(n_nuts + 1, 2, figsize=(14, 4 * (n_nuts + 1)))

    # --- Top row: Biomass comparison ---
    ax_bio = axes[0, 0]
    dt_seconds = _model.dt
    time_days = np.arange(n_opt_steps) * dt_seconds / 86400.0

    ax_bio.plot(time_days, np.array(true_biomass), "k-", linewidth=2, label="True", alpha=0.7)
    ax_bio.scatter(
        obs_local_indices * dt_seconds / 86400.0,
        np.array(observations),
        c="red",
        s=20,
        zorder=5,
        label="Observations",
    )

    # CMA-ES prediction
    cmaes_bio = run_simulation(best_cmaes.params)[_spinup_steps:]
    ax_bio.plot(time_days, np.array(cmaes_bio), "--", color="tab:orange", linewidth=1.5, label="CMA-ES best")

    # NUTS posterior mean prediction
    nuts_mean_params = {k: jnp.mean(nuts_result.samples[k]) for k in NUTS_PARAMS}
    combined_mean = {**nuts_mean_params, **fixed_from_cmaes}
    nuts_bio = run_simulation(combined_mean)[_spinup_steps:]
    ax_bio.plot(time_days, np.array(nuts_bio), "--", color="tab:green", linewidth=1.5, label="NUTS mean")

    ax_bio.set_xlabel("Day")
    ax_bio.set_ylabel("Biomass (g/m²)")
    ax_bio.set_title("Biomass: True vs CMA-ES vs NUTS")
    ax_bio.legend(fontsize=8)
    ax_bio.grid(True, alpha=0.3)

    # --- Top row right: Parameter recovery bar chart ---
    ax_bar = axes[0, 1]
    x_pos = np.arange(len(ALL_OPT_PARAMS))
    true_vals = np.array([TRUE_PARAMS[p] for p in ALL_OPT_PARAMS])

    cmaes_ratios = np.array([float(best_cmaes.params[p]) / TRUE_PARAMS[p] for p in ALL_OPT_PARAMS])
    combined_ratios = []
    for p in ALL_OPT_PARAMS:
        if p in NUTS_PARAMS:
            combined_ratios.append(float(jnp.mean(nuts_result.samples[p])) / TRUE_PARAMS[p])
        else:
            combined_ratios.append(float(best_cmaes.params[p]) / TRUE_PARAMS[p])
    combined_ratios = np.array(combined_ratios)

    bar_w = 0.35
    ax_bar.bar(x_pos - bar_w / 2, cmaes_ratios, bar_w, label="CMA-ES", color="tab:orange", alpha=0.7)
    ax_bar.bar(x_pos + bar_w / 2, combined_ratios, bar_w, label="Pipeline", color="tab:green", alpha=0.7)
    ax_bar.axhline(y=1, color="k", linestyle="--", alpha=0.5)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(ALL_OPT_PARAMS, rotation=30, ha="right", fontsize=9)
    ax_bar.set_ylabel("Ratio to true value")
    ax_bar.set_title("Parameter recovery")
    ax_bar.legend(fontsize=8)
    ax_bar.grid(True, alpha=0.3, axis="y")

    # --- NUTS parameter rows: trace + histogram ---
    for i, p in enumerate(NUTS_PARAMS):
        samples = np.array(nuts_result.samples[p].flatten())
        true_val = TRUE_PARAMS[p]

        # Trace
        ax_trace = axes[i + 1, 0]
        ax_trace.plot(samples, linewidth=0.5, alpha=0.7, color="steelblue")
        ax_trace.axhline(y=true_val, color="red", linewidth=1.5, linestyle="--", label="True")
        ax_trace.axhline(
            y=float(best_cmaes.params[p]),
            color="tab:orange",
            linewidth=1,
            linestyle=":",
            label="CMA-ES",
        )
        ax_trace.set_ylabel(p)
        if i == 0:
            ax_trace.set_title("NUTS trace plots")
        if i == n_nuts - 1:
            ax_trace.set_xlabel("Sample")
        ax_trace.legend(fontsize=7, loc="upper right")

        # Histogram
        ax_hist = axes[i + 1, 1]
        sample_range = float(np.ptp(samples))
        if sample_range > 0 and sample_range / (abs(np.mean(samples)) + 1e-30) > 1e-8:
            ax_hist.hist(samples, bins=30, density=True, alpha=0.7, color="steelblue", edgecolor="white")
        else:
            ax_hist.axvline(x=float(np.mean(samples)), color="steelblue", linewidth=4, alpha=0.7)
            ax_hist.text(
                0.5,
                0.5,
                "std ≈ 0\n(not explored)",
                transform=ax_hist.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="gray",
            )
        ax_hist.axvline(x=true_val, color="red", linewidth=1.5, linestyle="--", label="True")
        ax_hist.axvline(x=float(np.mean(samples)), color="tab:green", linewidth=1.5, label="NUTS mean")
        ax_hist.axvline(
            x=float(best_cmaes.params[p]),
            color="tab:orange",
            linewidth=1,
            linestyle=":",
            label="CMA-ES",
        )
        ax_hist.set_ylabel("Density")
        if i == 0:
            ax_hist.set_title("NUTS posterior distributions")
        if i == n_nuts - 1:
            ax_hist.set_xlabel(p)
        ax_hist.legend(fontsize=7)

    fig.suptitle(
        f"CMA-ES → NUTS Pipeline — "
        f"CMA-ES: {cmaes_elapsed:.0f}s, loss={best_cmaes.loss:.2e} | "
        f"NUTS: {nuts_elapsed:.0f}s, accept={nuts_result.acceptance_rate:.0%}, "
        f"div={int(jnp.sum(nuts_result.divergences))}",
        fontsize=11,
    )
    fig.tight_layout()
    plt.savefig("examples/pipeline_cmaes_nuts_lmtl_0d_results.png", dpi=150)
    print("\nPlot saved to examples/pipeline_cmaes_nuts_lmtl_0d_results.png")

    # =====================================================================
    # PAIRWISE LOSS LANDSCAPE (corner plot)
    # =====================================================================
    GRID_SIZE = 20  # points per axis (total evals per pair = GRID_SIZE²)
    print(f"\nComputing pairwise loss landscapes ({GRID_SIZE}x{GRID_SIZE} grid)...")

    n_all = len(ALL_OPT_PARAMS)
    loss_jit = jax.jit(loss_fn)

    # Precompute grids for each parameter
    param_grids = {}
    for p in ALL_OPT_PARAMS:
        low, high = BOUNDS[p]
        param_grids[p] = np.linspace(low, high, GRID_SIZE)

    fig2, axes2 = plt.subplots(n_all, n_all, figsize=(16, 16))

    # Hide upper triangle and diagonal
    for i in range(n_all):
        for j in range(n_all):
            if j >= i:
                axes2[i, j].set_visible(False)

    n_pairs = n_all * (n_all - 1) // 2
    pair_count = 0

    for i in range(1, n_all):
        for j in range(i):
            pair_count += 1
            p_y = ALL_OPT_PARAMS[i]  # row = y axis
            p_x = ALL_OPT_PARAMS[j]  # col = x axis
            print(f"  [{pair_count}/{n_pairs}] {p_x} vs {p_y}...", end="", flush=True)

            grid_x = param_grids[p_x]
            grid_y = param_grids[p_y]
            loss_grid = np.zeros((GRID_SIZE, GRID_SIZE))

            for iy in range(GRID_SIZE):
                for ix in range(GRID_SIZE):
                    test_params = {k: jnp.array(TRUE_PARAMS[k]) for k in ALL_OPT_PARAMS}
                    test_params[p_x] = jnp.array(grid_x[ix])
                    test_params[p_y] = jnp.array(grid_y[iy])
                    loss_grid[iy, ix] = float(loss_jit(test_params))

            ax = axes2[i, j]
            # Clip high loss values for better contrast
            vmax = min(float(np.percentile(loss_grid, 95)), 5.0)
            cf = ax.contourf(
                grid_x,
                grid_y,
                loss_grid,
                levels=20,
                cmap="viridis",
                vmin=0,
                vmax=vmax,
            )
            ax.contour(
                grid_x,
                grid_y,
                loss_grid,
                levels=10,
                colors="white",
                linewidths=0.3,
                alpha=0.5,
            )

            # Mark true values
            ax.plot(TRUE_PARAMS[p_x], TRUE_PARAMS[p_y], "r*", markersize=12, label="True")

            # Mark CMA-ES modes
            for k, mode in enumerate(cmaes_result.modes):
                marker = "o" if k == 0 else "s"
                ax.plot(
                    float(mode.params[p_x]),
                    float(mode.params[p_y]),
                    marker,
                    color="tab:orange",
                    markersize=7,
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                    label=f"CMA #{k + 1}" if i == 1 and j == 0 else None,
                )

            # Labels
            if i == n_all - 1:
                ax.set_xlabel(p_x, fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(p_y, fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=6)
            print(" done")

    # Legend on the first visible upper triangle cell
    axes2[0, 0].set_visible(True)
    axes2[0, 0].axis("off")
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="*", color="red", linestyle="None", markersize=12, label="True"),
        Line2D([0], [0], marker="o", color="tab:orange", linestyle="None", markersize=7, label="CMA-ES modes"),
    ]
    axes2[0, 0].legend(handles=legend_elements, loc="center", fontsize=10, frameon=False)

    fig2.suptitle(
        "Pairwise loss landscape (other params fixed at true values)",
        fontsize=13,
    )
    fig2.tight_layout()
    plt.savefig("examples/pipeline_loss_landscape.png", dpi=150)
    print("\nPlot saved to examples/pipeline_loss_landscape.png")
