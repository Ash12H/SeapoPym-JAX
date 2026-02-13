"""NUTS Sampling on LMTL 0D Model (Twin Experiment).

Demonstrates Bayesian parameter estimation via NUTS (No-U-Turn Sampler,
Hoffman & Gelman 2014) on a 0D LMTL ecosystem model.

Steps:
1. Spin-up: simulate SPINUP_YEARS to stabilize the system
2. Generate synthetic observations (twin experiment)
3. Build log-posterior = -loss + log_prior (proxy mode)
4. Reparameterize to unit space [0,1] for efficient sampling
5. Run NUTS to sample the posterior
6. Visualize: trace plots, posterior distributions, parameter recovery

Note on differentiability:
    NUTS requires informative gradients (d loss / d param) to explore.
    In the LMTL model, the recruitment threshold (cohort_ages >= rec_age)
    is a hard step function — its gradient w.r.t. rec_age is zero.
    Consequently:
    - lambda_0, gamma_lambda, efficiency: have non-zero gradients
      (they act on biomass/production without passing through the threshold).
      NUTS can estimate these correctly.
    - tau_r_0, gamma_tau_r: affect biomass ONLY through rec_age, which
      enters a >= comparison. Their gradient is exactly 0, so NUTS
      cannot explore them (random walk + divergences).
    For tau_r_0/gamma_tau_r, use a gradient-free method like CMA-ES.

Note on reparameterization:
    Parameter scales differ by ~10^6 (lambda_0 ~ 1e-8, tau_r_0 ~ 1e6).
    Without reparameterization, NUTS step-size adaptation cannot find a
    single epsilon that works for all parameters simultaneously.
    We map all parameters to unit space [0,1] via prior bounds so that
    the sampler sees similar scales (reparameterize_log_posterior).
"""

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Register LMTL functions (already defined in seapopym.functions.lmtl)
import seapopym.functions.lmtl  # noqa: F401
from seapopym.blueprint import Blueprint, Config
from seapopym.compiler import compile_model
from seapopym.engine.step import build_step_fn
from seapopym.optimization.likelihood import make_log_posterior, reparameterize_log_posterior
from seapopym.optimization.nuts import run_nuts
from seapopym.optimization.prior import PriorSet, Uniform

jax.config.update("jax_default_device", jax.devices("cpu")[0])

# =============================================================================
# CONFIGURATION — modify these to tune the experiment
# =============================================================================

# NUTS
N_WARMUP = 500  # Warmup steps (step size + mass matrix adaptation)
N_SAMPLES = 200  # Posterior samples to collect
SEED = 0
INIT_OFFSET = 0.2  # Fractional offset from true params (0 = exact, 0.2 = +20%)

# Simulation
SPINUP_YEARS = 1  # Stabilize the system before observations
OPT_YEARS = 1  # Period on which observations are sampled
DT = "1d"

# Twin experiment
OBS_FRACTION = 0.05  # Fraction of timesteps sampled as observations

# True biological parameters (to be recovered)
TRUE_PARAMS = {
    "lambda_0": 1 / 150 / 86400,  # 1/s — base mortality rate
    "gamma_lambda": 0.15,  # 1/degC — mortality temperature sensitivity
    "tau_r_0": 10.38 * 86400,  # s — base recruitment age
    "gamma_tau_r": 0.11,  # 1/degC — recruitment temperature sensitivity
    "efficiency": 0.1668,  # dimensionless — NPP-to-biomass efficiency
    "t_ref": 0.0,  # degC — reference temperature (fixed, not optimized)
}

# Priors (Uniform on same bounds as CMA-ES)
PRIORS = PriorSet(
    {
        "lambda_0": Uniform(1e-10, 5 * TRUE_PARAMS["lambda_0"]),
        "gamma_lambda": Uniform(0.01, 5 * TRUE_PARAMS["gamma_lambda"]),
        "tau_r_0": Uniform(0.1 * TRUE_PARAMS["tau_r_0"], 5 * TRUE_PARAMS["tau_r_0"]),
        "gamma_tau_r": Uniform(0.01, 5 * TRUE_PARAMS["gamma_tau_r"]),
        "efficiency": Uniform(0.01, 5 * TRUE_PARAMS["efficiency"]),
    }
)

# Fixed parameters (not optimized)
FIXED_PARAMS = {"t_ref": TRUE_PARAMS["t_ref"]}

# =============================================================================
# BLUEPRINT (0D LMTL)
# =============================================================================

max_age_days = int(np.ceil(TRUE_PARAMS["tau_r_0"] / 86400))
cohort_ages_days = np.arange(0, max_age_days + 1)
cohort_ages_sec = cohort_ages_days * 86400.0
n_cohorts = len(cohort_ages_sec)

blueprint = Blueprint.from_dict(
    {
        "id": "lmtl-nuts-demo",
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
# FORCINGS & CONFIG (spin-up + optimization period)
# =============================================================================

total_years = SPINUP_YEARS + OPT_YEARS
start_date = "2000-01-01"
end_date = str((pd.Timestamp(start_date) + pd.DateOffset(years=total_years)).date())

start_pd = pd.to_datetime(start_date)
end_pd = pd.to_datetime(end_date)
n_days = (end_pd - start_pd).days + 5
dates = pd.date_range(start=start_pd, periods=n_days, freq="D")

ny, nx = 1, 1
lat = np.arange(ny)
lon = np.arange(nx)

day_of_year = dates.dayofyear.values
temp_c = 15.0 + 5.0 * np.sin(2 * np.pi * day_of_year / 365.0)
temp_3d = np.broadcast_to(temp_c[:, None, None], (len(dates), ny, nx))
temp_da = xr.DataArray(temp_3d, dims=["T", "Y", "X"], coords={"T": dates, "Y": lat, "X": lon})

npp_day = 1.0 + 0.5 * np.sin(2 * np.pi * day_of_year / 365.0)
npp_sec = npp_day / 86400.0
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
    """Run full simulation (spin-up + opt period), return biomass time series."""
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
        biomass = jnp.mean(outputs["biomass"])
        return (new_state, p), biomass

    init_carry = (_initial_state, full_params)
    _, biomass_history = jax.lax.scan(scan_body, init_carry, jnp.arange(_n_timesteps))
    return biomass_history


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    param_names = list(PRIORS.priors.keys())

    print("=" * 60)
    print("NUTS on LMTL 0D (Twin Experiment)")
    print(f"  Spin-up: {SPINUP_YEARS} year(s)  |  Optimization: {OPT_YEARS} year(s)")
    print(f"  NUTS: {N_WARMUP} warmup + {N_SAMPLES} samples")
    print("=" * 60)

    # ----- Generate observations (year 2 only) -----
    print("\nGenerating observations with TRUE parameters...")
    t0 = time.time()
    true_params_jax = {k: jnp.array(TRUE_PARAMS[k]) for k in param_names}
    true_biomass_full = run_simulation(true_params_jax)

    true_biomass = true_biomass_full[_spinup_steps:]
    n_opt_steps = len(true_biomass)

    print(f"  Simulation: {time.time() - t0:.2f}s  ({_n_timesteps} total, {_spinup_steps} spin-up, {n_opt_steps} opt)")
    print(f"  Biomass range: [{float(jnp.min(true_biomass)):.4f}, {float(jnp.max(true_biomass)):.4f}]")

    # Sample observations
    n_obs = max(1, int(OBS_FRACTION * n_opt_steps))
    rng = np.random.default_rng(SEED)
    obs_local_indices = np.sort(rng.choice(n_opt_steps, size=n_obs, replace=False))
    obs_global_indices = obs_local_indices + _spinup_steps
    observations = true_biomass[obs_local_indices]
    obs_std = float(jnp.std(observations))
    print(f"  {n_obs} observations ({100 * n_obs / n_opt_steps:.1f}%), std={obs_std:.6f}")

    # ----- Loss function (NRMSE-std) -----
    def loss_fn(params: dict) -> jnp.ndarray:
        """NRMSE-std loss computed on optimization period only."""
        biomass_full = run_simulation(params)
        pred = biomass_full[obs_global_indices]
        rmse = jnp.sqrt(jnp.mean((pred - observations) ** 2))
        return rmse / obs_std

    # Sanity check
    true_loss = loss_fn(true_params_jax)
    print(f"  Sanity check: loss(true_params) = {float(true_loss):.6e}")

    # ----- Log-posterior (Mode 1: proxy = -loss + log_prior) -----
    log_posterior = make_log_posterior(loss_fn, PRIORS)

    # Reparameterize to unit space [0,1] so all parameters have similar scale.
    # Without this, raw gradients differ by ~10^6 (lambda_0 gradient ~ 1e8,
    # gamma_lambda gradient ~ 1e2) and NUTS cannot adapt a single step size.
    log_posterior_unit = reparameterize_log_posterior(log_posterior, PRIORS)

    # ----- Run NUTS -----
    # Initialize offset from true params to avoid gradient singularity at
    # the exact minimum (loss=0 → d(sqrt(x))/dx = 0/0), then map to unit space.
    initial_params_phys = {k: jnp.array(TRUE_PARAMS[k] * (1.0 + INIT_OFFSET)) for k in param_names}
    initial_params_unit = PRIORS.to_unit(initial_params_phys)

    print(f"\nInitial params ({INIT_OFFSET:+.0%} offset from true):")
    for p in param_names:
        print(f"  {p}: {float(initial_params_phys[p]):.4g} (true: {TRUE_PARAMS[p]:.4g}) [unit: {float(initial_params_unit[p]):.4f}]")
    print(f"\nRunning NUTS ({N_WARMUP} warmup + {N_SAMPLES} samples) in unit space...")
    t0 = time.time()

    result = run_nuts(
        log_posterior_fn=log_posterior_unit,
        initial_params=initial_params_unit,
        n_warmup=N_WARMUP,
        n_samples=N_SAMPLES,
        seed=SEED,
    )

    # Convert samples back to physical space
    result.samples = PRIORS.from_unit(result.samples)

    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s")
    print(f"  Acceptance rate: {result.acceptance_rate:.2%}")
    print(f"  Divergences: {int(jnp.sum(result.divergences))} / {N_SAMPLES}")

    # ----- Results -----
    print("\n" + "=" * 60)
    print("Posterior summary")
    print("=" * 60)

    header = f"{'Param':<14} {'True':>12} {'Mean':>12} {'Std':>12} {'Q5':>12} {'Q95':>12}"
    print(header)
    print("-" * len(header))

    for p in param_names:
        samples = result.samples[p]
        true_val = TRUE_PARAMS[p]
        mean = float(jnp.mean(samples))
        std = float(jnp.std(samples))
        q5 = float(jnp.percentile(samples, 5))
        q95 = float(jnp.percentile(samples, 95))
        print(f"{p:<14} {true_val:>12.4g} {mean:>12.4g} {std:>12.4g} {q5:>12.4g} {q95:>12.4g}")

    # ----- Visualization -----
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 2, figsize=(12, 3 * n_params))

    for i, p in enumerate(param_names):
        samples = np.array(result.samples[p].flatten())
        true_val = TRUE_PARAMS[p]

        # Left: trace plot
        ax_trace = axes[i, 0]
        ax_trace.plot(samples, linewidth=0.5, alpha=0.7, color="steelblue")
        ax_trace.axhline(y=true_val, color="red", linewidth=1.5, linestyle="--", label="True")
        ax_trace.set_ylabel(p)
        if i == 0:
            ax_trace.set_title("Trace plots")
        if i == n_params - 1:
            ax_trace.set_xlabel("Sample")
        ax_trace.legend(fontsize=7, loc="upper right")

        # Right: posterior histogram
        ax_hist = axes[i, 1]
        sample_range = float(np.ptp(samples))
        if sample_range > 0 and sample_range / (abs(np.mean(samples)) + 1e-30) > 1e-8:
            ax_hist.hist(samples, bins=30, density=True, alpha=0.7, color="steelblue", edgecolor="white")
        else:
            # Parameter not explored — show single bar at the constant value
            ax_hist.axvline(x=float(np.mean(samples)), color="steelblue", linewidth=4, alpha=0.7, label="Stuck")
            ax_hist.text(0.5, 0.5, f"std ≈ 0\n(not explored)", transform=ax_hist.transAxes,
                         ha="center", va="center", fontsize=9, color="gray")
        ax_hist.axvline(x=true_val, color="red", linewidth=1.5, linestyle="--", label="True")
        ax_hist.axvline(x=float(np.mean(samples)), color="orange", linewidth=1.5, label="Mean")
        ax_hist.set_ylabel("Density")
        if i == 0:
            ax_hist.set_title("Posterior distributions")
        if i == n_params - 1:
            ax_hist.set_xlabel(p)
        ax_hist.legend(fontsize=7)

    fig.suptitle(
        f"NUTS on LMTL 0D — {N_WARMUP} warmup + {N_SAMPLES} samples "
        f"(accept={result.acceptance_rate:.0%}, "
        f"div={int(jnp.sum(result.divergences))})",
        fontsize=12,
    )
    fig.tight_layout()
    plt.savefig("examples/nuts_lmtl_0d_results.png", dpi=150)
    print("\nPlot saved to examples/nuts_lmtl_0d_results.png")
