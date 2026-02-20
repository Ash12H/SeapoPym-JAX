"""CMA-ES → NUTS Pipeline on LMTL 0D Model (Twin Experiment).

Two-stage Bayesian parameter estimation:
1. IPOP-CMA-ES: gradient-free global search in 5D to find mode(s).
2. NUTS: one Markov chain per CMA-ES mode, sampling 3 model params + σ (4D).
   tau_r_0 and gamma_tau_r are fixed per mode from CMA-ES.
   Warmup (step size + mass matrix) is shared from chain 1 to subsequent chains.

Mode 2 likelihood: full Gaussian with free σ estimated jointly by NUTS.

NUTS tuning notes (from gradient diagnostics, see examples/09_gradient_diagnostic.py):
- The AD gradient through jax.lax.scan (731 steps) is correct (verified with central
  finite differences at multiple epsilon).
- The posterior has extreme curvature differences across parameters in unit space:
    gamma_lambda H_ii ≈ -2.1e6 | lambda_0 H_ii ≈ -1.2e6 | sigma H_ii ≈ 769
  This gives a condition number ≈ 2700, causing step-size collapse and max tree depth
  (1023 leapfrog/sample) with a naive diagonal mass matrix.
- Three fixes applied:
  (1) LogNormal prior for lambda_0 — equalizes curvature across orders of magnitude
      (lambda_0 ≈ 7.7e-8; HalfNormal compressed it into 10% of unit space).
  (2) Dense mass matrix — captures the lambda_0 ↔ gamma_lambda correlation induced
      by the mortality function λ₀·exp(γ_λ·T). The diagonal matrix cannot follow this
      "banana" and causes zigzagging.
  (3) 500 warmup steps — gives window_adaptation enough samples to estimate the 4×4
      dense mass matrix (10 free parameters vs 4 for diagonal).
"""

import logging
import math
import pickle
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

try:
    import corner  # type: ignore[import-untyped]
except ImportError:
    corner = None

from matplotlib.patches import Patch

import seapopym.functions.lmtl  # noqa: F401
from seapopym.blueprint import Config
from seapopym.compiler import compile_model
from seapopym.engine.step import build_step_fn
from seapopym.models import LMTL_NO_TRANSPORT
from seapopym.optimization.ipop import run_ipop_cmaes
from seapopym.optimization.likelihood import (
    GaussianLikelihood,
    make_log_posterior,
    reparameterize_log_posterior,
)
from seapopym.optimization.nuts import run_nuts
from seapopym.optimization.prior import HalfNormal, LogNormal, PriorSet, Uniform

# Force CPU for 0D model (GPU overhead dominates for tiny workloads)
jax.config.update("jax_default_device", jax.devices("cpu")[0])
print(f"JAX device: {jax.devices('cpu')[0]}")

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Stage 1: CMA-ES (Hansen defaults) ---
N_RESTARTS = 1
N_PARAMS = 5  # number of optimized parameters
INITIAL_POPSIZE = 4 + int(3 * np.log(N_PARAMS))
N_GENERATIONS = int(100 + 150 * (N_PARAMS + 3) ** 2 / np.sqrt(INITIAL_POPSIZE))
DISTANCE_THRESHOLD = 0.1
CMAES_SEED = 42

# --- Stage 2: NUTS (one chain per CMA-ES mode) ---
N_WARMUP = 200  # enough for dense mass matrix adaptation (10 free params for 4×4)
N_SAMPLES = 1000
NUTS_SEED = 0

# --- Simulation ---
SPINUP_YEARS = 1
OPT_YEARS = 1
DT = "1d"
LATITUDE = 30.0  # degrees N

# --- Twin experiment ---
OBS_FRACTION = 0.05
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

# NUTS samples only these 3 model parameters (+ sigma); tau_r_0 and gamma_tau_r fixed per mode
NUTS_SAMPLED_PARAMS = ["lambda_0", "gamma_lambda", "efficiency"]

# Bounds (shared by CMA-ES)
BOUNDS = {
    "lambda_0": (1e-10, 2 * TRUE_PARAMS["lambda_0"]),
    "gamma_lambda": (0.01, 2 * TRUE_PARAMS["gamma_lambda"]),
    "tau_r_0": (0.1 * TRUE_PARAMS["tau_r_0"], 2 * TRUE_PARAMS["tau_r_0"]),
    "gamma_tau_r": (0.01, 2 * TRUE_PARAMS["gamma_tau_r"]),
    "efficiency": (0.01, 2 * TRUE_PARAMS["efficiency"]),
}

FIXED_PARAMS = {"t_ref": TRUE_PARAMS["t_ref"]}

RESULTS_FILE = "examples/results/07_pipeline_results.pkl"
CORNER_PLOT_FILE = "examples/images/07_pipeline_corner.png"
BIOMASS_PLOT_FILE = "examples/images/07_pipeline_biomass.png"

# =============================================================================
# BLUEPRINT (0D LMTL from catalogue)
# =============================================================================

blueprint = LMTL_NO_TRANSPORT

max_age_days = int(np.ceil(TRUE_PARAMS["tau_r_0"] / 86400))
cohort_ages_sec = np.arange(0, max_age_days + 1) * 86400.0
n_cohorts = len(cohort_ages_sec)

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

# Day of year forcing (T, Y, X)
day_of_year = dates.dayofyear.values
doy_float = day_of_year.astype(float)
doy_3d = np.broadcast_to(doy_float[:, None, None], (len(dates), ny, nx))
doy_da = xr.DataArray(doy_3d, dims=["T", "Y", "X"], coords={"T": dates, "Y": lat, "X": lon})

# Forcing: Temperature (seasonal, T, Z=1, Y, X)
temp_c = 15.0 + 5.0 * np.sin(2 * np.pi * day_of_year / 365.0)
temp_4d = np.broadcast_to(temp_c[:, None, None, None], (len(dates), 1, ny, nx))
temp_da = xr.DataArray(
    temp_4d,
    dims=["T", "Z", "Y", "X"],
    coords={"T": dates, "Z": np.arange(1), "Y": lat, "X": lon},
)

# Forcing: NPP (seasonal)
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

# =============================================================================
# COMPILE & BUILD SIMULATION
# =============================================================================

print("Compiling model...")
_model = compile_model(blueprint, config, backend="jax")
_step_fn = build_step_fn(_model)
_n_timesteps = _model.n_timesteps
_initial_state = _model.state
_forcings_stacked = _model.forcings.get_all()

_spinup_steps = int(SPINUP_YEARS / total_years * _n_timesteps)


def run_simulation(params: dict) -> jnp.ndarray:
    """Run full simulation, return biomass time series."""
    full_params = {**params, **{k: jnp.array(v) for k, v in FIXED_PARAMS.items()}}
    full_params["cohort_ages"] = _model.parameters["cohort_ages"]
    full_params["day_layer"] = _model.parameters["day_layer"]
    full_params["night_layer"] = _model.parameters["night_layer"]
    full_params["latitude"] = _model.parameters["latitude"]

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


def make_predict_fn(fixed_params: dict, obs_indices: jnp.ndarray):
    """Factory: build a predict_fn with tau_r_0/gamma_tau_r fixed from a CMA-ES mode."""

    def predict_fn(params: dict) -> jnp.ndarray:
        full = {**fixed_params}
        for k in NUTS_SAMPLED_PARAMS:
            full[k] = params[k]
        return run_simulation(full)[obs_indices]

    return predict_fn


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CMA-ES → NUTS Pipeline on LMTL 0D (Twin Experiment)")
    print(f"  Spin-up: {SPINUP_YEARS}y | Optimization: {OPT_YEARS}y | dt: {DT}")
    print(f"  Stage 1 (CMA-ES): {N_RESTARTS} restarts, {N_GENERATIONS} gen")
    print(f"  Stage 2 (NUTS):   {N_WARMUP} warmup + {N_SAMPLES} samples")
    print(f"  NUTS sampled params: {NUTS_SAMPLED_PARAMS} + sigma")
    print("=" * 60)

    # ----- Generate observations (always needed for figures) -----
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

    # =====================================================================
    # LOAD OR COMPUTE
    # =====================================================================
    if Path(RESULTS_FILE).exists():
        print(f"\nLoading saved results from {RESULTS_FILE}...")
        with open(RESULTS_FILE, "rb") as f:
            checkpoint = pickle.load(f)
        cmaes_result = checkpoint["cmaes_result"]
        chain_results = checkpoint["chain_results"]
        cmaes_elapsed = checkpoint["cmaes_elapsed"]
        nuts_elapsed_total = checkpoint["nuts_elapsed_total"]
        print(f"  Loaded {len(chain_results)} chain(s), CMA-ES: {cmaes_elapsed:.0f}s, NUTS: {nuts_elapsed_total:.0f}s")
    else:

        def loss_fn(params: dict) -> jnp.ndarray:
            biomass_full = run_simulation(params)
            pred = biomass_full[obs_global_indices]
            rmse = jnp.sqrt(jnp.mean((pred - observations) ** 2))
            return rmse / obs_std

        true_loss = loss_fn(true_params_jax)
        print(f"  Sanity: loss(true) = {float(true_loss):.6e}")

        # =================================================================
        # STAGE 1: IPOP-CMA-ES (all 5 parameters)
        # =================================================================
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
            progress_bar=True,
        )
        cmaes_elapsed = time.time() - t0

        best_cmaes = cmaes_result.modes[0]
        print(f"\nCMA-ES completed in {cmaes_elapsed:.1f}s, best loss = {best_cmaes.loss:.6e}")
        print("CMA-ES best parameters:")
        for p in ALL_OPT_PARAMS:
            ratio = float(best_cmaes.params[p]) / TRUE_PARAMS[p]
            print(
                f"  {p:<14} = {float(best_cmaes.params[p]):>12.4g}  (true: {TRUE_PARAMS[p]:>12.4g}, ratio: {ratio:.4f})"
            )

        # =================================================================
        # STAGE 2: NUTS (one chain per CMA-ES mode, 3 params + σ)
        # =================================================================
        print("\n" + "=" * 60)
        print("STAGE 2: NUTS (Mode 2 — Gaussian likelihood, σ free)")
        print(f"  {len(cmaes_result.modes)} chain(s), {N_WARMUP} warmup (shared) + {N_SAMPLES} samples each")
        print(f"  Sampled: {NUTS_SAMPLED_PARAMS} + sigma")
        print(f"  Fixed per mode: tau_r_0, gamma_tau_r")
        print("=" * 60)

        # NUTS priors (runtime: sigma prior depends on obs_std)
        # LogNormal for lambda_0: equalizes curvature across orders of magnitude.
        # With HalfNormal, the unit-space value was compressed near 0 (unit ≈ 0.10),
        # creating extreme curvature (H_ii ≈ -1.2e6) vs sigma (H_ii ≈ 769).
        nuts_priors = PriorSet(
            {
                "lambda_0": LogNormal(mu=math.log(TRUE_PARAMS["lambda_0"]), sigma=1.0),
                "gamma_lambda": HalfNormal(scale=0.3),
                "efficiency": Uniform(*BOUNDS["efficiency"]),
                "sigma": HalfNormal(scale=obs_std),
            }
        )

        # Run one chain per CMA-ES mode (shared warmup: chain 0 warms up, others reuse)
        chain_results: list = []
        nuts_elapsed_total = 0.0
        shared_kernel_params: dict | None = None

        for chain_idx, mode in enumerate(cmaes_result.modes):
            print(
                f"\n--- Chain {chain_idx + 1}/{len(cmaes_result.modes)} "
                f"(from CMA-ES mode #{chain_idx + 1}, loss={mode.loss:.4e}) ---"
            )

            # Fixed params for this mode
            fixed_params = {
                "tau_r_0": mode.params["tau_r_0"],
                "gamma_tau_r": mode.params["gamma_tau_r"],
            }
            print(
                f"  Fixed: tau_r_0={float(fixed_params['tau_r_0']):.4g}, "
                f"gamma_tau_r={float(fixed_params['gamma_tau_r']):.4g}"
            )

            # Build predict_fn for this mode (tau_r_0/gamma_tau_r baked in)
            predict_fn = make_predict_fn(fixed_params, obs_global_indices)

            # Mode 2 log-posterior: Gaussian likelihood + σ free
            # sigma_prior=None because σ is already in nuts_priors.log_prob() (no double counting)
            log_posterior = make_log_posterior(
                loss_fn=None,
                prior_set=nuts_priors,
                likelihood=GaussianLikelihood(),
                sigma_prior=None,
                observations_for_likelihood=(predict_fn, observations),
            )
            log_posterior_unit = reparameterize_log_posterior(log_posterior, nuts_priors)

            # Initialize at CMA-ES mode (sampled params) + estimated σ
            cmaes_pred = run_simulation(mode.params)[obs_global_indices]
            init_sigma = max(float(jnp.std(cmaes_pred - observations)), 1e-6)

            init_phys = {k: mode.params[k] for k in NUTS_SAMPLED_PARAMS}
            init_phys["sigma"] = jnp.array(init_sigma)
            init_unit = nuts_priors.to_unit(init_phys)

            for p in list(NUTS_SAMPLED_PARAMS) + ["sigma"]:
                print(f"  {p:<14} = {float(init_phys[p]):>12.4g} [unit: {float(init_unit[p]):.4f}]")

            t0 = time.time()
            result = run_nuts(
                log_posterior_fn=log_posterior_unit,
                initial_params=init_unit,
                n_warmup=N_WARMUP,
                n_samples=N_SAMPLES,
                seed=NUTS_SEED + chain_idx,
                target_acceptance_rate=0.85,
                is_mass_matrix_diagonal=False,  # dense: captures λ₀ ↔ γ_λ correlation
                progress_bar=True,
                kernel_params=shared_kernel_params,  # None for chain 0, reused after
            )

            # Share warmup from first chain to all subsequent chains
            if shared_kernel_params is None:
                shared_kernel_params = result.kernel_params

            # Convert back to physical space
            result.samples = nuts_priors.from_unit(result.samples)
            elapsed = time.time() - t0
            nuts_elapsed_total += elapsed

            warmup_str = "warmup" if result.n_warmup > 0 else "shared"
            n_div = int(jnp.sum(result.divergences))
            print(
                f"  Completed in {elapsed:.1f}s ({warmup_str}) — accept={result.acceptance_rate:.2%}, divergences={n_div}/{N_SAMPLES}"
            )

            chain_results.append((result, fixed_params))

        # Save results to pickle
        checkpoint = {
            "cmaes_result": cmaes_result,
            "chain_results": chain_results,
            "cmaes_elapsed": cmaes_elapsed,
            "nuts_elapsed_total": nuts_elapsed_total,
        }
        Path(RESULTS_FILE).parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE, "wb") as f:
            pickle.dump(checkpoint, f)
        print(f"\nResults saved to {RESULTS_FILE}")

    # =====================================================================
    # RESULTS (console table per mode)
    # =====================================================================
    print("\n" + "=" * 60)
    print("RESULTS (per chain)")
    print("=" * 60)

    all_nuts_params = NUTS_SAMPLED_PARAMS + ["sigma"]

    for chain_idx, (result, fixed) in enumerate(chain_results):
        print(f"\n--- Chain {chain_idx + 1} ---")
        print(f"  Fixed: tau_r_0={float(fixed['tau_r_0']):.4g}, gamma_tau_r={float(fixed['gamma_tau_r']):.4g}")
        header = f"  {'Param':<14} {'True':>12} {'CMA-ES':>12} {'NUTS mean':>12} {'NUTS std':>12}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        mode = cmaes_result.modes[chain_idx]
        for p in all_nuts_params:
            true_str = f"{TRUE_PARAMS[p]:>12.4g}" if p in TRUE_PARAMS else f"{'N/A':>12}"
            cmaes_str = f"{float(mode.params[p]):>12.4g}" if p in mode.params else f"{'N/A':>12}"
            samples = result.samples[p]
            mean = float(jnp.mean(samples))
            std = float(jnp.std(samples))
            print(f"  {p:<14} {true_str} {cmaes_str} {mean:>12.4g} {std:>12.4g}")

    # =====================================================================
    # FIGURE 1: Corner plot (all modes superposed)
    # =====================================================================
    if corner is not None:
        print("\nGenerating corner plot...")

        n_chains = len(chain_results)
        chain_colors = plt.cm.tab10(np.linspace(0, 0.9, max(n_chains, 1)))

        corner_labels = [r"$\lambda_0$", r"$\gamma_\lambda$", "efficiency", r"$\sigma$"]
        truths = [TRUE_PARAMS["lambda_0"], TRUE_PARAMS["gamma_lambda"], TRUE_PARAMS["efficiency"], None]

        # Compute range from ALL chains combined (avoids crash on degenerate chains)
        all_samples = []
        for result, _ in chain_results:
            arr = np.column_stack([np.array(result.samples[p].flatten()) for p in all_nuts_params])
            all_samples.append(arr)
        all_combined = np.vstack(all_samples)

        corner_range = []
        for col in range(all_combined.shape[1]):
            lo, hi = float(np.min(all_combined[:, col])), float(np.max(all_combined[:, col]))
            if hi - lo < 1e-10:
                mid = (lo + hi) / 2
                lo, hi = mid - 1e-6, mid + 1e-6
            corner_range.append((lo, hi))

        fig_corner = None
        for ci, (result, _) in enumerate(chain_results):
            samples_array = np.column_stack([np.array(result.samples[p].flatten()) for p in all_nuts_params])
            fig_corner = corner.corner(
                samples_array,
                labels=corner_labels,
                truths=truths if ci == 0 else None,
                color=chain_colors[ci],
                plot_datapoints=True,
                no_fill_contours=False,
                fig=fig_corner,
                hist_kwargs={"density": True},
                range=corner_range,
            )

        # Manual legend
        legend_handles = [Patch(facecolor=chain_colors[ci], label=f"Mode {ci + 1}") for ci in range(n_chains)]
        fig_corner.legend(handles=legend_handles, loc="upper right", fontsize=10)

        Path(CORNER_PLOT_FILE).parent.mkdir(parents=True, exist_ok=True)
        fig_corner.savefig(CORNER_PLOT_FILE, dpi=150)
        print(f"Corner plot saved to {CORNER_PLOT_FILE}")
    else:
        print("\nSkipping corner plot (install 'corner' package to enable)")

    # =====================================================================
    # FIGURE 2: Biomass + σ envelopes
    # =====================================================================
    print("Generating biomass plot...")

    n_chains = len(chain_results)
    chain_colors = plt.cm.tab10(np.linspace(0, 0.9, max(n_chains, 1)))
    dt_seconds = _model.dt
    time_days = np.arange(n_opt_steps) * dt_seconds / 86400.0

    fig_bio, ax_bio = plt.subplots(figsize=(12, 5))

    # True biomass
    ax_bio.plot(time_days, np.array(true_biomass), "k-", linewidth=2, label="True", alpha=0.8)

    # Observations
    ax_bio.scatter(
        obs_local_indices * dt_seconds / 86400.0,
        np.array(observations),
        c="red",
        s=20,
        zorder=5,
        label=f"Observations ({OBS_FRACTION:.0%})",
    )

    # Per mode: CMA-ES best + σ envelope
    for ci, (result, fixed) in enumerate(chain_results):
        mode = cmaes_result.modes[ci]
        color = chain_colors[ci]

        # CMA-ES biomass for this mode
        cmaes_bio = run_simulation(mode.params)[_spinup_steps:]
        cmaes_bio_np = np.array(cmaes_bio)

        # Mean σ from NUTS posterior
        mean_sigma = float(jnp.mean(result.samples["sigma"]))

        ax_bio.plot(
            time_days,
            cmaes_bio_np,
            "-",
            color=color,
            linewidth=1.5,
            label=f"Mode {ci + 1} (CMA-ES, σ={mean_sigma:.2e})",
        )
        ax_bio.fill_between(
            time_days,
            cmaes_bio_np - mean_sigma,
            cmaes_bio_np + mean_sigma,
            color=color,
            alpha=0.15,
        )

    ax_bio.set_xlabel("Day")
    ax_bio.set_ylabel("Biomass (g/m²)")
    ax_bio.set_title(
        f"CMA-ES → NUTS Pipeline — CMA-ES: {cmaes_elapsed:.0f}s | NUTS: {nuts_elapsed_total:.0f}s, {n_chains} chain(s)"
    )
    ax_bio.legend(fontsize=8)
    ax_bio.grid(True, alpha=0.3)
    fig_bio.tight_layout()

    Path(BIOMASS_PLOT_FILE).parent.mkdir(parents=True, exist_ok=True)
    fig_bio.savefig(BIOMASS_PLOT_FILE, dpi=150)
    print(f"Biomass plot saved to {BIOMASS_PLOT_FILE}")
