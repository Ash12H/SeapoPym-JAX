"""CMA-ES vs SimpleGA comparison on LMTL 0D (Twin Experiment).

Compares IPOP-CMA-ES and IPOP-SimpleGA on a 0D LMTL ecosystem model
across multiple noise levels. Both algorithms use the same IPOP restart
framework (Auger & Hansen 2005) and the same total evaluation budget,
isolating the effect of the optimization algorithm.

CMA-ES adapts a full covariance matrix capturing parameter correlations,
while SimpleGA uses elitist selection + uniform crossover + Gaussian mutation
without any covariance model.

Steps:
1. Spin-up: simulate SPINUP_YEARS to stabilize the system
2. For each noise level (0%, 10%, 20%):
   a. Generate synthetic observations + noise
   b. Run IPOP-CMA-ES
   c. Run IPOP-SimpleGA (same restarts, same budget)
   d. Record best loss and parameter recovery error
3. Visualize comparison across noise levels
"""

import logging
import math
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Register LMTL functions (already defined in seapopym.functions.lmtl)
import seapopym.functions.lmtl  # noqa: F401
from seapopym.blueprint import Config
from seapopym.compiler import compile_model
from seapopym.engine.step import build_step_fn
from seapopym.models import LMTL_NO_TRANSPORT
from seapopym.optimization.ipop import run_ipop

# Use GPU if available, fall back to CPU
print(f"JAX devices: {jax.devices()}")

# =============================================================================
# CONFIGURATION
# =============================================================================

N_PARAMS = 5

# IPOP settings (Hansen defaults for n=5)
INITIAL_POPSIZE = 4 + int(3 * math.log(N_PARAMS))  # 8 for n=5
N_GENERATIONS = int(100 + 150 * (N_PARAMS + 3) ** 2 / math.sqrt(INITIAL_POPSIZE))
N_RESTARTS = 5
DISTANCE_THRESHOLD = 0.1
SEED = 42

# Simulation
SPINUP_YEARS = 1
OPT_YEARS = 2
DT = "1d"
LATITUDE = 30.0

# Twin experiment
OBS_FRACTION = 0.1
INITIAL_GUESS_FACTOR = 1.5

# Noise sweep
NOISE_LEVELS = [0.0, 0.10, 0.20]

# True biological parameters (to be recovered)
TRUE_PARAMS = {
    "lambda_0": 1 / 150 / 86400,
    "gamma_lambda": 0.15,
    "tau_r_0": 10.38 * 86400,
    "gamma_tau_r": 0.11,
    "efficiency": 0.1668,
    "t_ref": 0.0,  # fixed, not optimized
}

# Bounds for optimized parameters
BOUNDS = {
    "lambda_0": (1e-10, 5 * TRUE_PARAMS["lambda_0"]),
    "gamma_lambda": (0.01, 5 * TRUE_PARAMS["gamma_lambda"]),
    "tau_r_0": (0.1 * TRUE_PARAMS["tau_r_0"], 5 * TRUE_PARAMS["tau_r_0"]),
    "gamma_tau_r": (0.01, 5 * TRUE_PARAMS["gamma_tau_r"]),
    "efficiency": (0.01, 5 * TRUE_PARAMS["efficiency"]),
}

FIXED_PARAMS = {"t_ref": TRUE_PARAMS["t_ref"]}

# SimpleGA hyperparameters
GA_CROSSOVER_RATE = 0.5
GA_MUTATION_STD = 0.1

PLOT_FILE = "examples/images/08_cmaes_vs_ga_lmtl_0d.png"

# =============================================================================
# BLUEPRINT (0D LMTL from catalogue)
# =============================================================================

blueprint = LMTL_NO_TRANSPORT

max_age_days = int(np.ceil(TRUE_PARAMS["tau_r_0"] / 86400))
cohort_ages_days = np.arange(0, max_age_days + 1)
cohort_ages_sec = cohort_ages_days * 86400.0
n_cohorts = len(cohort_ages_sec)

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
doy_float = day_of_year.astype(float)
doy_3d = np.broadcast_to(doy_float[:, None, None], (len(dates), ny, nx))
doy_da = xr.DataArray(doy_3d, dims=["T", "Y", "X"], coords={"T": dates, "Y": lat, "X": lon})

temp_c = 15.0 + 5.0 * np.sin(2 * np.pi * day_of_year / 365.0)
temp_4d = np.broadcast_to(temp_c[:, None, None, None], (len(dates), 1, ny, nx))
temp_da = xr.DataArray(
    temp_4d,
    dims=["T", "Z", "Y", "X"],
    coords={"T": dates, "Z": np.arange(1), "Y": lat, "X": lon},
)

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
    """Run full simulation (spin-up + opt period), return biomass time series."""
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
        biomass = jnp.mean(outputs["biomass"])
        return (new_state, p), biomass

    init_carry = (_initial_state, full_params)
    _, biomass_history = jax.lax.scan(scan_body, init_carry, jnp.arange(_n_timesteps))
    return biomass_history


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    param_names = list(BOUNDS.keys())
    true_vals = np.array([TRUE_PARAMS[p] for p in param_names])

    print("=" * 60)
    print("CMA-ES vs SimpleGA on LMTL 0D (Twin Experiment)")
    print(f"  Spin-up: {SPINUP_YEARS} year(s)  |  Optimization: {OPT_YEARS} year(s)")
    print(
        f"  IPOP: {N_RESTARTS} restarts, pop {INITIAL_POPSIZE}"
        f"->{INITIAL_POPSIZE * 2 ** (N_RESTARTS - 1)}, "
        f"{N_GENERATIONS} gen"
    )
    print(f"  Noise levels: {[f'{n*100:.0f}%' for n in NOISE_LEVELS]}")
    print("=" * 60)

    # ----- Generate TRUE biomass (once) -----
    print("\nGenerating observations with TRUE parameters...")
    t0 = time.time()
    true_params_jax = {k: jnp.array(TRUE_PARAMS[k]) for k in BOUNDS}
    true_biomass_full = run_simulation(true_params_jax)
    true_biomass = true_biomass_full[_spinup_steps:]
    n_opt_steps = len(true_biomass)
    print(f"  Simulation: {time.time() - t0:.2f}s  ({n_opt_steps} opt steps)")

    # Sample observation indices (shared across noise levels)
    n_obs = max(1, int(OBS_FRACTION * n_opt_steps))
    rng = np.random.default_rng(SEED)
    obs_local_indices = np.sort(rng.choice(n_opt_steps, size=n_obs, replace=False))
    obs_global_indices = obs_local_indices + _spinup_steps
    clean_observations = true_biomass[obs_local_indices]

    # Initial guess
    initial_params = {k: jnp.array(INITIAL_GUESS_FACTOR * TRUE_PARAMS[k]) for k in BOUNDS}

    # ----- Run comparison across noise levels -----
    results = {"cma_es": [], "simple_ga": []}

    for noise_level in NOISE_LEVELS:
        print(f"\n{'=' * 60}")
        print(f"Noise level: {noise_level * 100:.0f}%")
        print("=" * 60)

        # Add noise to observations
        if noise_level > 0:
            noise_rng = np.random.default_rng(SEED + int(noise_level * 1000))
            noise = jnp.array(
                noise_rng.normal(0, noise_level * np.array(clean_observations), size=clean_observations.shape)
            )
            observations = clean_observations + noise
        else:
            observations = clean_observations

        obs_std = float(jnp.std(observations))

        # Loss function (NRMSE-std)
        def _make_loss(obs, std):
            def _loss(params: dict) -> jnp.ndarray:
                biomass_full = run_simulation(params)
                pred = biomass_full[obs_global_indices]
                rmse = jnp.sqrt(jnp.mean((pred - obs) ** 2))
                return rmse / std
            return _loss

        loss_fn = _make_loss(observations, obs_std)

        for strategy, label in [("cma_es", "IPOP-CMA-ES"), ("simple_ga", "IPOP-SimpleGA")]:
            print(f"\n  Running {label}...")
            t0 = time.time()

            strategy_kwargs = {}
            if strategy == "simple_ga":
                strategy_kwargs = {"crossover_rate": GA_CROSSOVER_RATE, "mutation_std": GA_MUTATION_STD}

            result = run_ipop(
                loss_fn=loss_fn,
                initial_params=initial_params,
                bounds=BOUNDS,
                strategy=strategy,
                n_restarts=N_RESTARTS,
                initial_popsize=INITIAL_POPSIZE,
                n_generations=N_GENERATIONS,
                distance_threshold=DISTANCE_THRESHOLD,
                seed=SEED,
                progress_bar=True,
                **strategy_kwargs,
            )

            elapsed = time.time() - t0
            best = result.modes[0]

            # Parameter recovery error (relative)
            pred_vals = np.array([float(best.params[p]) for p in param_names])
            rel_error = np.mean(np.abs(pred_vals - true_vals) / true_vals)

            # Total function evaluations = sum(popsize_i * n_generations_i)
            total_evals = sum(
                INITIAL_POPSIZE * (2**i) * r.n_iterations
                for i, r in enumerate(result.all_results)
            )

            results[strategy].append({
                "noise": noise_level,
                "loss": best.loss,
                "rel_error": rel_error,
                "n_modes": len(result.modes),
                "total_evals": total_evals,
                "elapsed": elapsed,
                "params": {p: float(best.params[p]) for p in param_names},
            })

            print(f"    {label}: loss={best.loss:.6f}, rel_error={rel_error:.4f}, "
                  f"evals={total_evals:,}, modes={len(result.modes)}, time={elapsed:.1f}s")

    # ----- Summary table -----
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    header = (
        f"{'Noise':>6} | {'CMA-ES loss':>12} {'GA loss':>12} | "
        f"{'CMA-ES err':>12} {'GA err':>12} | {'CMA-ES evals':>13} {'GA evals':>13}"
    )
    print(header)
    print("-" * len(header))
    for i, noise in enumerate(NOISE_LEVELS):
        cma = results["cma_es"][i]
        ga = results["simple_ga"][i]
        print(
            f"{noise*100:5.0f}% | {cma['loss']:12.6f} {ga['loss']:12.6f} | "
            f"{cma['rel_error']:12.4f} {ga['rel_error']:12.4f} | "
            f"{cma['total_evals']:13,} {ga['total_evals']:13,}"
        )

    # ----- Visualization -----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    noise_pct = [n * 100 for n in NOISE_LEVELS]

    # Plot 1: Best loss vs noise
    ax1 = axes[0]
    cma_losses = [r["loss"] for r in results["cma_es"]]
    ga_losses = [r["loss"] for r in results["simple_ga"]]
    ax1.plot(noise_pct, cma_losses, "o-", color="steelblue", linewidth=2, markersize=8, label="IPOP-CMA-ES")
    ax1.plot(noise_pct, ga_losses, "s--", color="orangered", linewidth=2, markersize=8, label="IPOP-SimpleGA")
    ax1.set_xlabel("Noise level (%)")
    ax1.set_ylabel("Best loss (NRMSE-std)")
    ax1.set_title("Convergence quality vs noise")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Parameter recovery error vs noise
    ax2 = axes[1]
    cma_errors = [r["rel_error"] for r in results["cma_es"]]
    ga_errors = [r["rel_error"] for r in results["simple_ga"]]
    ax2.plot(noise_pct, cma_errors, "o-", color="steelblue", linewidth=2, markersize=8, label="IPOP-CMA-ES")
    ax2.plot(noise_pct, ga_errors, "s--", color="orangered", linewidth=2, markersize=8, label="IPOP-SimpleGA")
    ax2.set_xlabel("Noise level (%)")
    ax2.set_ylabel("Mean relative parameter error")
    ax2.set_title("Parameter recovery vs noise")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Total function evaluations vs noise
    ax3 = axes[2]
    cma_evals = [r["total_evals"] for r in results["cma_es"]]
    ga_evals = [r["total_evals"] for r in results["simple_ga"]]
    x = np.arange(len(NOISE_LEVELS))
    width = 0.35
    ax3.bar(x - width / 2, cma_evals, width, color="steelblue", label="IPOP-CMA-ES")
    ax3.bar(x + width / 2, ga_evals, width, color="orangered", label="IPOP-SimpleGA")
    ax3.set_xlabel("Noise level (%)")
    ax3.set_ylabel("Total function evaluations")
    ax3.set_title("Computational budget (with early stopping)")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{n*100:.0f}%" for n in NOISE_LEVELS])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    Path(PLOT_FILE).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_FILE, dpi=150)
    print(f"\nPlot saved to {PLOT_FILE}")
