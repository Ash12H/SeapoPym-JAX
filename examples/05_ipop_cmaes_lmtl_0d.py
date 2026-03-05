# %% [markdown]
# # IPOP-CMA-ES on LMTL 0D Model (Twin Experiment)
#
# Demonstrates multi-restart CMA-ES with increasing population (IPOP strategy,
# Auger & Hansen 2005) on a 0D LMTL ecosystem model.
#
# Steps:
# 1. Spin-up: simulate SPINUP_YEARS to stabilize the system
# 2. Optimization year: generate synthetic observations on OPT_YEARS
# 3. Run IPOP-CMA-ES to recover the parameters
# 4. Visualize modes found, convergence, and parameter recovery

# %%
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

import seapopym.functions.lmtl  # noqa: F401
from seapopym.blueprint import Config
from seapopym.compiler import compile_model
from seapopym.engine import Runner
from seapopym.models import LMTL_NO_TRANSPORT
from seapopym.optimization import IPOPCMAESOptimizer, Objective

print(f"JAX devices: {jax.devices()}")

# %% [markdown]
# ## Configuration

# %%
N_PARAMS = 5

# IPOP-CMA-ES — Hansen defaults (pycma, Auger & Hansen 2005)
INITIAL_POPSIZE = 4 + int(3 * math.log(N_PARAMS))  # 8 for n=5
N_GENERATIONS = int(100 + 150 * (N_PARAMS + 3) ** 2 / math.sqrt(INITIAL_POPSIZE))
N_RESTARTS = 8
DISTANCE_THRESHOLD = 0.1
SEED = 42

SPINUP_YEARS = 1
OPT_YEARS = 2
DT = "1d"
LATITUDE = 30.0

OBS_FRACTION = 0.1
INITIAL_GUESS_FACTOR = 1.5

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

PLOT_FILE = "examples/images/05_ipop_cmaes_lmtl_0d.png"

# %% [markdown]
# ## Forcings & Model Setup

# %%
blueprint = LMTL_NO_TRANSPORT

max_age_days = int(np.ceil(TRUE_PARAMS["tau_r_0"] / 86400))
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
print("IPOP-CMA-ES on LMTL 0D (Twin Experiment)")
print(f"  Spin-up: {SPINUP_YEARS} year(s)  |  Optimization: {OPT_YEARS} year(s)")
print(
    f"  IPOP: {N_RESTARTS} restarts, pop {INITIAL_POPSIZE}->{INITIAL_POPSIZE * 2 ** (N_RESTARTS - 1)}, "
    f"{N_GENERATIONS} gen"
)
print("=" * 60)

print("\nGenerating observations with TRUE parameters...")
t0 = time.time()
true_params_jax = {k: jnp.array(TRUE_PARAMS[k]) for k in BOUNDS}
outputs_true = runner(model, true_params_jax)

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
obs_values = true_biomass[obs_local_indices]
print(f"  {n_obs} observations ({100 * n_obs / n_opt_steps:.1f}%), std={float(jnp.std(obs_values)):.6f}")

# %% [markdown]
# ## Objective & Optimizer

# %%
def extract_predictions(outputs):
    biomass = outputs["biomass"]
    ts = jnp.mean(biomass, axis=tuple(range(1, biomass.ndim)))
    return ts[obs_global_indices]

objective = Objective(observations=obs_values, transform=extract_predictions)

optimizer = IPOPCMAESOptimizer(
    runner=runner,
    objectives=[(objective, "nrmse", 1.0)],
    bounds=BOUNDS,
    n_restarts=N_RESTARTS,
    initial_popsize=INITIAL_POPSIZE,
    n_generations=N_GENERATIONS,
    distance_threshold=DISTANCE_THRESHOLD,
    seed=SEED,
)

# %% [markdown]
# ## IPOP-CMA-ES

# %%
print("\nRunning IPOP-CMA-ES...")
t0 = time.time()

result = optimizer.run(model, progress_bar=True)

elapsed = time.time() - t0
print(f"\nCompleted in {elapsed:.1f}s")
print(f"Found {len(result.modes)} distinct mode(s) across {result.n_restarts} restarts")

# %% [markdown]
# ## Results

# %%
param_names = list(BOUNDS.keys())

print("\n" + "=" * 60)
print("Modes found (sorted by loss)")
print("=" * 60)

header = f"{'Mode':<6} {'Loss':>10}"
for p in param_names:
    header += f" {p[:8]:>10}"
print(header)
print("-" * len(header))

row = f"{'True':<6} {0.0:>10.6f}"
for p in param_names:
    row += f" {TRUE_PARAMS[p]:>10.4g}"
print(row)

for i, mode in enumerate(result.modes):
    row = f"{'#' + str(i + 1):<6} {mode.loss:>10.6f}"
    for p in param_names:
        row += f" {float(mode.params[p]):>10.4g}"
    print(row)

# %% [markdown]
# ## Visualization

# %%
n_modes = len(result.modes)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: Loss per restart
ax1 = axes[0]
restart_losses = [float(r.loss) for r in result.all_results]
mode_indices = [i for i, r in enumerate(result.all_results) if any(m.loss == r.loss for m in result.modes)]
ax1.bar(range(len(restart_losses)), restart_losses, color="steelblue", alpha=0.7)
ax1.bar(mode_indices, [restart_losses[i] for i in mode_indices], color="orangered", alpha=0.9)
ax1.set_xlabel("Restart")
ax1.set_ylabel("Loss (NRMSE-std)")
ax1.set_title("Loss per restart")
popsizes = [INITIAL_POPSIZE * 2**i for i in range(len(restart_losses))]
ax1.set_xticks(range(len(restart_losses)))
ax1.set_xticklabels([f"pop={p}" for p in popsizes], rotation=45, ha="right", fontsize=8)

# Plot 2: Biomass trajectories (year 2 only)
ax2 = axes[1]
dt_seconds = model.dt
time_days = np.arange(n_opt_steps) * dt_seconds / 86400.0
ax2.plot(time_days, true_biomass, "k-", linewidth=2, label="True", alpha=0.7)
ax2.scatter(obs_local_indices * dt_seconds / 86400.0, obs_values, c="red", s=20, zorder=5, label="Observations")

colors = plt.cm.Set1(np.linspace(0, 1, max(n_modes, 1)))
for i, mode in enumerate(result.modes):
    outputs_mode = runner(model, mode.params)
    biomass_mode = outputs_mode["biomass"]
    pred_full = jnp.mean(biomass_mode, axis=tuple(range(1, biomass_mode.ndim)))
    pred = pred_full[spinup_steps:]
    ax2.plot(time_days, pred, "--", color=colors[i], linewidth=1.5, label=f"Mode #{i + 1}")

ax2.set_xlabel("Day (year 2)")
ax2.set_ylabel("Biomass (g/m²)")
ax2.set_title("Biomass trajectories (year 2)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot 3: Parameter recovery
ax3 = axes[2]
x = np.arange(len(param_names))
width = 0.8 / (n_modes + 1)

ax3.bar(x - 0.4 + width / 2, np.ones(len(param_names)), width, label="True", color="black", alpha=0.3)
true_vals = np.array([TRUE_PARAMS[p] for p in param_names])

for i, mode in enumerate(result.modes):
    pred_vals = np.array([float(mode.params[p]) for p in param_names])
    ratios = pred_vals / true_vals
    ax3.bar(x - 0.4 + (i + 1.5) * width, ratios, width, label=f"Mode #{i + 1}", color=colors[i], alpha=0.7)

ax3.set_xticks(x)
ax3.set_xticklabels([p[:8] for p in param_names], rotation=45, ha="right")
ax3.set_ylabel("Ratio to true value")
ax3.set_title("Parameter recovery")
ax3.axhline(y=1, color="k", linestyle="--", alpha=0.5)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
Path(PLOT_FILE).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(PLOT_FILE, dpi=150)
print(f"\nPlot saved to {PLOT_FILE}")
