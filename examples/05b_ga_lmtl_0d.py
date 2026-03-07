# %% [markdown]
# # Genetic Algorithm on LMTL 0D Model (Twin Experiment)
#
# Demonstrates SimpleGA (evosax) on a 0D LMTL ecosystem model.
#
# Runs **two experiments** with shared model setup:
# 1. Clean observations (no noise)
# 2. Noisy observations (15% Gaussian noise)
#
# Visualization focuses on **loss evolution**:
# - Best-of-generation (best descendant at each generation)
# - Elite (cumulative minimum — elitism)

# %%
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import seapopym.functions.lmtl  # noqa: F401
from seapopym.blueprint import Config
from seapopym.compiler import compile_model
from seapopym.engine import Runner
from seapopym.models import LMTL_NO_TRANSPORT
from seapopym.optimization import GAOptimizer, Objective

print(f"JAX devices: {jax.devices()}")

# %% [markdown]
# ## Configuration

# %%
POPSIZE = 64
N_GENERATIONS = 250
PATIENCE = 50
SEED = 42

SPINUP_YEARS = 1
OPT_YEARS = 2
DT = "1d"
LATITUDE = 30.0

OBS_FRACTION = 0.1

TRUE_PARAMS = {
    "lambda_0": 1 / 150 / 86400,
    "gamma_lambda": 0.15,
    "tau_r_0": 10.38 * 86400,
    "gamma_tau_r": 0.11,
    "efficiency": 0.1668,
    "t_ref": 0.0,
}

BOUNDS = {
    "lambda_0": (1e-10, 5 * TRUE_PARAMS["lambda_0"]),
    "gamma_lambda": (0.01, 5 * TRUE_PARAMS["gamma_lambda"]),
    "tau_r_0": (0.1 * TRUE_PARAMS["tau_r_0"], 5 * TRUE_PARAMS["tau_r_0"]),
    "gamma_tau_r": (0.01, 5 * TRUE_PARAMS["gamma_tau_r"]),
    "efficiency": (0.01, 5 * TRUE_PARAMS["efficiency"]),
}

FIXED_PARAMS = {"t_ref": TRUE_PARAMS["t_ref"]}

NOISE_LEVELS = [0.0, 0.15]

PLOT_FILE = "examples/images/05b_ga_lmtl_0d.png"

# %% [markdown]
# ## Forcings & Model Setup

# %%
blueprint = LMTL_NO_TRANSPORT

max_age_days = int(np.ceil(BOUNDS["tau_r_0"][1] / 86400))  # upper bound covers all optimizer trials
cohort_ages_days = np.arange(0, max_age_days + 1)
cohort_ages_sec = cohort_ages_days * 86400.0
n_cohorts = len(cohort_ages_sec)

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
doy_da = xr.DataArray(doy_float, dims=["T"], coords={"T": dates})

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

print("Compiling model...")
model = compile_model(blueprint, config)
runner = Runner.optimization()
n_timesteps = model.n_timesteps
spinup_steps = int(SPINUP_YEARS / total_years * n_timesteps)

# %% [markdown]
# ## Generate Synthetic Observations

# %%
print("=" * 60)
print("Genetic Algorithm on LMTL 0D (Twin Experiment)")
print(f"  Spin-up: {SPINUP_YEARS} year(s)  |  Optimization: {OPT_YEARS} year(s)")
print(f"  GA: popsize={POPSIZE}, {N_GENERATIONS} gen, patience={PATIENCE}")
print(f"  Noise levels: {NOISE_LEVELS}")
print("=" * 60)

print("\nGenerating observations with TRUE parameters...")
t0 = time.time()
true_params_jax = {k: jnp.array([TRUE_PARAMS[k]]) for k in BOUNDS}
outputs_true = runner(model, true_params_jax, export_variables=["biomass"])

biomass_true = outputs_true["biomass"]
true_biomass_full = jnp.mean(biomass_true, axis=tuple(range(1, biomass_true.ndim)))
true_biomass = true_biomass_full[spinup_steps:]
n_opt_steps = len(true_biomass)

print(
    f"  Simulation: {time.time() - t0:.2f}s  ({n_timesteps} total steps, "
    f"{spinup_steps} spin-up, {n_opt_steps} opt)"
)
print(f"  Year 2 biomass range: [{float(jnp.min(true_biomass)):.4f}, {float(jnp.max(true_biomass)):.4f}]")

n_obs = max(1, int(OBS_FRACTION * n_opt_steps))
rng = np.random.default_rng(SEED)
obs_local_indices = np.sort(rng.choice(n_opt_steps, size=n_obs, replace=False))
obs_global_indices = obs_local_indices + spinup_steps
obs_values_clean = true_biomass[obs_local_indices]
print(f"  {n_obs} observations ({100 * n_obs / n_opt_steps:.1f}%), std={float(jnp.std(obs_values_clean)):.6f}")


# %% [markdown]
# ## Shared helpers

# %%
def extract_predictions(outputs):
    biomass = outputs["biomass"]
    ts = jnp.mean(biomass, axis=tuple(range(1, biomass.ndim)))
    return ts[obs_global_indices]


param_names = list(BOUNDS.keys())
dt_seconds = model.dt
time_days = np.arange(n_opt_steps) * dt_seconds / 86400.0

# %% [markdown]
# ## Run GA for each noise level

# %%
all_experiment_results = {}

for noise_level in NOISE_LEVELS:
    label = "clean" if noise_level == 0.0 else f"noise={noise_level:.0%}"
    print(f"\n{'=' * 60}")
    print(f"Experiment: {label}")
    print(f"{'=' * 60}")

    if noise_level > 0:
        noise_rng = np.random.default_rng(SEED + 1)
        noise = jnp.array(noise_rng.normal(0, noise_level * np.array(obs_values_clean), size=obs_values_clean.shape))
        obs_values = obs_values_clean + noise
    else:
        obs_values = obs_values_clean

    objective = Objective(observations=obs_values, transform=extract_predictions)
    optimizer = GAOptimizer(
        runner=runner,
        objectives=[(objective, "nrmse", 1.0)],
        bounds=BOUNDS,
        popsize=POPSIZE,
        seed=SEED,
    )

    print(f"Running GA ({label})...")
    t0 = time.time()
    result = optimizer.run(model, n_generations=N_GENERATIONS, patience=PATIENCE, progress_bar=True)
    elapsed = time.time() - t0

    print(f"  Completed in {elapsed:.1f}s — {result.n_iterations} generations")
    print(f"  Best loss: {result.loss:.6e}")
    print(f"  {'Converged' if result.converged else 'Did not converge'}")

    all_experiment_results[noise_level] = {
        "result": result,
        "obs_values": obs_values,
        "elapsed": elapsed,
    }

    # Print parameter recovery
    print(f"\n  Parameter recovery:")
    for p in param_names:
        val = float(jnp.squeeze(result.params[p]))
        ratio = val / TRUE_PARAMS[p]
        print(f"    {p:<14} = {val:>12.4g}  (ratio = {ratio:.4f})")

# %% [markdown]
# ## Visualization

# %%
n_experiments = len(NOISE_LEVELS)
fig, axes = plt.subplots(n_experiments, 4, figsize=(20, 5 * n_experiments))
if n_experiments == 1:
    axes = axes[np.newaxis, :]

for row_idx, noise_level in enumerate(NOISE_LEVELS):
    label = "Clean" if noise_level == 0.0 else f"Noise {noise_level:.0%}"
    exp = all_experiment_results[noise_level]
    result = exp["result"]
    obs_values = exp["obs_values"]
    loss_history = result.loss_history

    # --- Col 0: Biomass time series ---
    ax = axes[row_idx, 0]
    ax.plot(time_days, true_biomass, "k-", linewidth=2, label="True", alpha=0.7)
    ax.scatter(
        obs_local_indices * dt_seconds / 86400.0, obs_values, c="red", s=20, zorder=5, label="Observations"
    )
    outputs_best = runner(model, result.params, export_variables=["biomass"])
    biomass_best = outputs_best["biomass"]
    pred_full = jnp.mean(biomass_best, axis=tuple(range(1, biomass_best.ndim)))
    pred = pred_full[spinup_steps:]
    ax.plot(time_days, pred, "--", color="orangered", linewidth=1.5, label=f"GA (loss={result.loss:.2e})")
    ax.set_xlabel("Day (year 2)")
    ax.set_ylabel("Biomass (g/m²)")
    ax.set_title(f"[{label}] Biomass trajectories")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Col 1: OBS vs PRED scatter ---
    ax = axes[row_idx, 1]
    pred_best = np.array(pred)[obs_local_indices - spinup_steps]
    ax.scatter(obs_values, pred_best, c="red", edgecolors="k", linewidths=0.5, s=30, zorder=5)
    obs_range = [min(np.min(obs_values), np.min(pred_best)), max(np.max(obs_values), np.max(pred_best))]
    ax.plot(obs_range, obs_range, "k--", alpha=0.5, label="1:1 line")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted (GA)")
    ax.set_title(f"[{label}] Obs vs Pred")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Col 2: Parameter recovery ---
    ax = axes[row_idx, 2]
    x = np.arange(len(param_names))
    true_vals = np.array([TRUE_PARAMS[p] for p in param_names])
    pred_vals = np.array([float(jnp.squeeze(result.params[p])) for p in param_names])
    ratios = pred_vals / true_vals

    ax.bar(x - 0.15, np.ones(len(param_names)), 0.3, label="True", color="black", alpha=0.3)
    ax.bar(x + 0.15, ratios, 0.3, label="GA", color="orangered", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([p[:8] for p in param_names], rotation=45, ha="right")
    ax.set_ylabel("Ratio to true value")
    ax.set_title(f"[{label}] Parameter recovery")
    ax.axhline(y=1, color="k", linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Col 3: Loss evolution ---
    ax = axes[row_idx, 3]
    generations = np.arange(1, len(loss_history) + 1)
    best_of_gen = np.array(loss_history)
    elite = np.minimum.accumulate(best_of_gen)
    ax.semilogy(generations, best_of_gen, color="steelblue", linewidth=0.8, alpha=0.5, label="Best of generation")
    ax.semilogy(generations, elite, color="orangered", linewidth=2, label="Elite (cumul. min)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Loss (NRMSE)")
    ax.set_title(f"[{label}] Loss evolution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
Path(PLOT_FILE).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(PLOT_FILE, dpi=150)
print(f"\nPlot saved to {PLOT_FILE}")
