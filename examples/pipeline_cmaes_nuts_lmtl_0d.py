"""CMA-ES → NUTS Pipeline on LMTL 0D Model (Twin Experiment).

Two-stage Bayesian parameter estimation:
1. IPOP-CMA-ES: gradient-free global search to find mode(s).
2. NUTS: one Markov chain per CMA-ES mode, sampling all 5 parameters.
   Initialized at each mode's optimum.

This allows comparing posteriors from different starting points to detect
multimodality and assess convergence across chains.
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

# Force CPU for 0D model (GPU overhead dominates for tiny workloads)
jax.config.update("jax_default_device", jax.devices("cpu")[0])
print(f"JAX device: {jax.devices('cpu')[0]}")

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

# --- Stage 2: NUTS (one chain per CMA-ES mode) ---
N_WARMUP = 200
N_SAMPLES = 500
NUTS_SEED = 0

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

# All parameters estimated by NUTS, initialized from CMA-ES modes
NUTS_PARAMS = ALL_OPT_PARAMS

# Bounds (shared by CMA-ES and NUTS priors)
BOUNDS = {
    "lambda_0": (1e-10, 2 * TRUE_PARAMS["lambda_0"]),
    "gamma_lambda": (0.01, 2 * TRUE_PARAMS["gamma_lambda"]),
    "tau_r_0": (0.1 * TRUE_PARAMS["tau_r_0"], 2 * TRUE_PARAMS["tau_r_0"]),
    "gamma_tau_r": (0.01, 2 * TRUE_PARAMS["gamma_tau_r"]),
    "efficiency": (0.01, 2 * TRUE_PARAMS["efficiency"]),
}

# NUTS priors — all 5 parameters
# HalfNormal for parameters inside exponentials (concentrate mass near small values)
# Uniform for bounded parameters without strong prior information
NUTS_PRIORS = PriorSet(
    {
        "lambda_0": HalfNormal(scale=3 * TRUE_PARAMS["lambda_0"]),
        "gamma_lambda": HalfNormal(scale=0.3),
        "tau_r_0": Uniform(*BOUNDS["tau_r_0"]),
        "gamma_tau_r": Uniform(*BOUNDS["gamma_tau_r"]),
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
    # STAGE 2: NUTS (one chain per CMA-ES mode, all 5 parameters)
    # =====================================================================
    print("\n" + "=" * 60)
    print("STAGE 2: NUTS")
    print(f"  {len(cmaes_result.modes)} chain(s), {N_WARMUP} warmup + {N_SAMPLES} samples each")
    print(f"  Parameters: {NUTS_PARAMS}")
    print("=" * 60)

    # Log-posterior in unit space (shared across chains)
    log_posterior = make_log_posterior(loss_fn, NUTS_PRIORS)
    log_posterior_unit = reparameterize_log_posterior(log_posterior, NUTS_PRIORS)

    # Run one chain per CMA-ES mode
    chain_results: list = []
    nuts_elapsed_total = 0.0

    for chain_idx, mode in enumerate(cmaes_result.modes):
        print(f"\n--- Chain {chain_idx + 1}/{len(cmaes_result.modes)} "
              f"(from CMA-ES mode #{chain_idx + 1}, loss={mode.loss:.4e}) ---")

        # Initialize at this mode's optimum (in unit space)
        init_phys = {k: mode.params[k] for k in NUTS_PARAMS}
        init_unit = NUTS_PRIORS.to_unit(init_phys)

        for p in NUTS_PARAMS:
            print(f"  {p:<14} = {float(init_phys[p]):>12.4g} [unit: {float(init_unit[p]):.4f}]")

        t0 = time.time()
        result = run_nuts(
            log_posterior_fn=log_posterior_unit,
            initial_params=init_unit,
            n_warmup=N_WARMUP,
            n_samples=N_SAMPLES,
            seed=NUTS_SEED + chain_idx,
            target_acceptance_rate=0.85,
        )
        # Convert back to physical space
        result.samples = NUTS_PRIORS.from_unit(result.samples)
        elapsed = time.time() - t0
        nuts_elapsed_total += elapsed

        n_div = int(jnp.sum(result.divergences))
        print(f"  Completed in {elapsed:.1f}s — "
              f"accept={result.acceptance_rate:.2%}, divergences={n_div}/{N_SAMPLES}")

        chain_results.append(result)

    # =====================================================================
    # RESULTS
    # =====================================================================
    print("\n" + "=" * 60)
    print("RESULTS (per chain)")
    print("=" * 60)

    for chain_idx, result in enumerate(chain_results):
        print(f"\n--- Chain {chain_idx + 1} ---")
        header = f"  {'Param':<14} {'True':>12} {'Mean':>12} {'Std':>12}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for p in NUTS_PARAMS:
            true_val = TRUE_PARAMS[p]
            samples = result.samples[p]
            mean = float(jnp.mean(samples))
            std = float(jnp.std(samples))
            print(f"  {p:<14} {true_val:>12.4g} {mean:>12.4g} {std:>12.4g}")

    # =====================================================================
    # VISUALIZATION
    # =====================================================================
    n_chains = len(chain_results)
    n_params = len(NUTS_PARAMS)
    chain_colors = plt.cm.tab10(np.linspace(0, 1, max(n_chains, 1)))

    fig, axes = plt.subplots(n_params + 1, 2, figsize=(14, 3 * (n_params + 1)))

    # --- Top row left: Biomass comparison ---
    ax_bio = axes[0, 0]
    dt_seconds = _model.dt
    time_days = np.arange(n_opt_steps) * dt_seconds / 86400.0

    ax_bio.plot(time_days, np.array(true_biomass), "k-", linewidth=2, label="True", alpha=0.7)
    ax_bio.scatter(
        obs_local_indices * dt_seconds / 86400.0, np.array(observations),
        c="red", s=20, zorder=5, label="Observations",
    )
    cmaes_bio = run_simulation(best_cmaes.params)[_spinup_steps:]
    ax_bio.plot(time_days, np.array(cmaes_bio), "--", color="tab:orange", linewidth=1.5, label="CMA-ES best")

    for ci, result in enumerate(chain_results):
        mean_params = {k: jnp.mean(result.samples[k]) for k in NUTS_PARAMS}
        bio = run_simulation(mean_params)[_spinup_steps:]
        ax_bio.plot(time_days, np.array(bio), "--", color=chain_colors[ci],
                    linewidth=1.5, label=f"Chain {ci + 1} mean")

    ax_bio.set_xlabel("Day")
    ax_bio.set_ylabel("Biomass (g/m²)")
    ax_bio.set_title("Biomass: True vs CMA-ES vs NUTS chains")
    ax_bio.legend(fontsize=7)
    ax_bio.grid(True, alpha=0.3)

    # --- Top row right: Parameter recovery bar chart ---
    ax_bar = axes[0, 1]
    x_pos = np.arange(n_params)
    bar_w = 0.8 / (n_chains + 1)

    # CMA-ES best
    cmaes_ratios = np.array([float(best_cmaes.params[p]) / TRUE_PARAMS[p] for p in NUTS_PARAMS])
    ax_bar.bar(x_pos - 0.4 + bar_w / 2, cmaes_ratios, bar_w,
               label="CMA-ES", color="tab:orange", alpha=0.7)

    # Each chain
    for ci, result in enumerate(chain_results):
        ratios = np.array([float(jnp.mean(result.samples[p])) / TRUE_PARAMS[p] for p in NUTS_PARAMS])
        ax_bar.bar(x_pos - 0.4 + (ci + 1.5) * bar_w, ratios, bar_w,
                   label=f"Chain {ci + 1}", color=chain_colors[ci], alpha=0.7)

    ax_bar.axhline(y=1, color="k", linestyle="--", alpha=0.5)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(NUTS_PARAMS, rotation=30, ha="right", fontsize=9)
    ax_bar.set_ylabel("Ratio to true value")
    ax_bar.set_title("Parameter recovery")
    ax_bar.legend(fontsize=7)
    ax_bar.grid(True, alpha=0.3, axis="y")

    # --- Parameter rows: trace + histogram (overlay all chains) ---
    for i, p in enumerate(NUTS_PARAMS):
        true_val = TRUE_PARAMS[p]

        ax_trace = axes[i + 1, 0]
        ax_hist = axes[i + 1, 1]

        for ci, result in enumerate(chain_results):
            samples = np.array(result.samples[p].flatten())
            ax_trace.plot(samples, linewidth=0.4, alpha=0.6, color=chain_colors[ci],
                          label=f"Chain {ci + 1}" if i == 0 else None)

            sample_range = float(np.ptp(samples))
            if sample_range > 0 and sample_range / (abs(np.mean(samples)) + 1e-30) > 1e-8:
                ax_hist.hist(samples, bins=30, density=True, alpha=0.4,
                             color=chain_colors[ci], edgecolor="white",
                             label=f"Chain {ci + 1}" if i == 0 else None)
            else:
                ax_hist.axvline(x=float(np.mean(samples)), color=chain_colors[ci],
                                linewidth=3, alpha=0.6)

        ax_trace.axhline(y=true_val, color="red", linewidth=1.5, linestyle="--", label="True" if i == 0 else None)
        ax_trace.set_ylabel(p)
        if i == 0:
            ax_trace.set_title("NUTS trace plots")
            ax_trace.legend(fontsize=7, loc="upper right")
        if i == n_params - 1:
            ax_trace.set_xlabel("Sample")

        ax_hist.axvline(x=true_val, color="red", linewidth=1.5, linestyle="--", label="True" if i == 0 else None)
        ax_hist.set_ylabel("Density")
        if i == 0:
            ax_hist.set_title("NUTS posterior distributions")
            ax_hist.legend(fontsize=7)
        if i == n_params - 1:
            ax_hist.set_xlabel(p)

    # Summary stats in suptitle
    total_div = sum(int(jnp.sum(r.divergences)) for r in chain_results)
    total_samples = N_SAMPLES * n_chains
    mean_accept = np.mean([r.acceptance_rate for r in chain_results])
    fig.suptitle(
        f"CMA-ES → NUTS Pipeline — "
        f"CMA-ES: {cmaes_elapsed:.0f}s | "
        f"NUTS: {nuts_elapsed_total:.0f}s, {n_chains} chain(s), "
        f"accept={mean_accept:.0%}, div={total_div}/{total_samples}",
        fontsize=11,
    )
    fig.tight_layout()
    plt.savefig("examples/pipeline_cmaes_nuts_lmtl_0d_results.png", dpi=150)
    print(f"\nPlot saved to examples/pipeline_cmaes_nuts_lmtl_0d_results.png")

    # =====================================================================
    # CORNER PLOT: Pairwise loss landscape (zoomed + log scale)
    # =====================================================================
    print("\nGenerating corner plot (pairwise loss landscape)...")

    # JIT-compile loss_fn once (avoids re-tracing the lax.scan at every call)
    jit_loss = jax.jit(loss_fn)
    _ = jit_loss({k: best_cmaes.params[k] for k in ALL_OPT_PARAMS}).block_until_ready()

    # Zoom: parameter ranges centered on CMA-ES best, width = 4× distance to true
    # This avoids the "all yellow" problem from using the full bounds.
    corner_params = ALL_OPT_PARAMS
    n_corner = len(corner_params)
    GRID_RES = 20  # grid resolution per axis (20² × 10 pairs ≈ 4000 evals)

    corner_ranges: dict[str, tuple[float, float]] = {}
    for p in corner_params:
        best_val = float(best_cmaes.params[p])
        true_val = TRUE_PARAMS[p]
        half_width = max(abs(best_val - true_val) * 4, abs(true_val) * 0.3)
        lo = max(BOUNDS[p][0], best_val - half_width)
        hi = min(BOUNDS[p][1], best_val + half_width)
        corner_ranges[p] = (lo, hi)

    fig_corner, axes_corner = plt.subplots(
        n_corner, n_corner, figsize=(3 * n_corner, 3 * n_corner),
    )

    for row in range(n_corner):
        for col in range(n_corner):
            ax = axes_corner[row, col]

            if col > row:
                ax.axis("off")
                continue

            p_row = corner_params[row]
            p_col = corner_params[col]

            if row == col:
                # Diagonal: 1D loss profile
                vals = np.linspace(*corner_ranges[p_row], GRID_RES)
                losses_1d = []
                for v in vals:
                    test_params = {k: best_cmaes.params[k] for k in ALL_OPT_PARAMS}
                    test_params[p_row] = jnp.array(v)
                    losses_1d.append(float(jit_loss(test_params)))
                losses_1d = np.array(losses_1d)
                ax.semilogy(vals, losses_1d, "b-", linewidth=1.5)
                ax.axvline(TRUE_PARAMS[p_row], color="red", linestyle="--", linewidth=1, label="True")
                ax.axvline(float(best_cmaes.params[p_row]), color="green", linestyle=":", linewidth=1, label="CMA-ES")
                ax.set_ylabel("Loss (log)")
                if row == 0:
                    ax.legend(fontsize=6)
            else:
                # Off-diagonal: 2D loss landscape
                vals_col = np.linspace(*corner_ranges[p_col], GRID_RES)
                vals_row = np.linspace(*corner_ranges[p_row], GRID_RES)
                loss_grid = np.full((GRID_RES, GRID_RES), np.nan)

                for ii, vr in enumerate(vals_row):
                    for jj, vc in enumerate(vals_col):
                        test_params = {k: best_cmaes.params[k] for k in ALL_OPT_PARAMS}
                        test_params[p_row] = jnp.array(vr)
                        test_params[p_col] = jnp.array(vc)
                        loss_grid[ii, jj] = float(jit_loss(test_params))

                # Log scale with contours
                log_loss = np.log10(np.clip(loss_grid, 1e-12, None))
                im = ax.contourf(vals_col, vals_row, log_loss, levels=20, cmap="viridis")
                ax.contour(vals_col, vals_row, log_loss, levels=10, colors="white", linewidths=0.3, alpha=0.5)
                ax.plot(TRUE_PARAMS[p_col], TRUE_PARAMS[p_row], "r*", markersize=10, zorder=10)
                ax.plot(float(best_cmaes.params[p_col]), float(best_cmaes.params[p_row]),
                        "g^", markersize=8, zorder=10)

            # Labels on edges only
            if row == n_corner - 1:
                ax.set_xlabel(p_col, fontsize=8)
            else:
                ax.set_xticklabels([])
            if col == 0 and row != col:
                ax.set_ylabel(p_row, fontsize=8)
            elif row == col:
                pass  # keep ylabel "Loss (log)"
            else:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=6)

    fig_corner.suptitle(
        "Pairwise loss landscape (log₁₀) — zoomed around CMA-ES best\n"
        "Red ★ = true | Green ▲ = CMA-ES best",
        fontsize=11,
    )
    fig_corner.tight_layout()
    plt.savefig("examples/pipeline_loss_landscape.png", dpi=150)
    print("Corner plot saved to examples/pipeline_loss_landscape.png")
