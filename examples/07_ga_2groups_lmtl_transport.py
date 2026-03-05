# %% [markdown]
# # GA with 2 Functional Groups on LMTL + Transport (Twin Experiment)
#
# Demonstrates SimpleGA (evosax) on a 2D LMTL model **with transport**:
#
# - 20x20 grid, open boundaries, Rankine vortex (stationary)
# - Localised NPP blob at centre (Gaussian)
# - 2 functional groups: surface (day=0, night=0) + DVM (day=1, night=0)
# - Observation: single spatial point, 5% of timesteps, day/night sensor
#
# Runs **two experiments** with shared model setup:
# 1. Clean observations (no noise)
# 2. Noisy observations (30% Gaussian noise)

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
import seapopym.functions.transport  # noqa: F401
from seapopym.blueprint import Config
from seapopym.compiler import compile_model
from seapopym.engine import Runner
from seapopym.models import LMTL
from seapopym.optimization import GAOptimizer, Objective

jax.config.update("jax_default_device", jax.devices("cpu")[0])
print(f"JAX device: {jax.devices('cpu')[0]}")

# %% [markdown]
# ## Configuration

# %%
POPSIZE = 128
N_GENERATIONS = 50
PATIENCE = 10
SEED = 42

SPINUP_YEARS = 1
OPT_YEARS = 2
DT = "1d"
OBS_FRACTION = 0.05
LATITUDE = 30.0

# Grid
NY, NX = 20, 20
N_LAYERS = 2
N_GROUPS = 2
CELL_SIZE = 10_000.0  # 10 km cells

# Rankine vortex
VORTEX_VMAX = 0.05  # m/s max tangential velocity
VORTEX_RADIUS = 4  # cells — radius of maximum velocity

# NPP blob
NPP_CENTRE = (NY // 2, NX // 2)
NPP_SIGMA = 3.0  # cells
NPP_PEAK = 2.0 / 86400.0  # g/m^2/s peak value

# Diffusion (0 = advection only)
DIFFUSIVITY = 0.0

# Observation point
OBS_Y, OBS_X = NY // 2 + 3, NX // 2 + 5  # offset from centre, avoids islands

TRUE_PARAMS = {
    "lambda_0": 1 / 150 / 86400,
    "gamma_lambda": 0.15,
    "tau_r_0": 10.38 * 86400,
    "gamma_tau_r": 0.11,
    "efficiency": 0.1668,
    "t_ref": 0.0,
}

ALL_OPT_PARAMS = ["lambda_0", "gamma_lambda", "tau_r_0", "gamma_tau_r", "efficiency"]

BOUNDS = {
    "lambda_0": (1e-10, 5 * TRUE_PARAMS["lambda_0"]),
    "gamma_lambda": (0.01, 5 * TRUE_PARAMS["gamma_lambda"]),
    "tau_r_0": (0.1 * TRUE_PARAMS["tau_r_0"], 5 * TRUE_PARAMS["tau_r_0"]),
    "gamma_tau_r": (0.01, 5 * TRUE_PARAMS["gamma_tau_r"]),
    "efficiency": (0.01, 5 * TRUE_PARAMS["efficiency"]),
}

FIXED_PARAMS = {"t_ref": TRUE_PARAMS["t_ref"]}

NOISE_LEVELS = [0.0, 0.30]

PLOT_FILE = "examples/images/07_ga_2groups_lmtl_transport.png"

# %% [markdown]
# ## Build Forcings (Rankine vortex + NPP blob)

# %%
blueprint = LMTL

max_age_days = int(np.ceil(TRUE_PARAMS["tau_r_0"] / 86400))
cohort_ages_sec = np.arange(0, max_age_days + 1) * 86400.0
n_cohorts = len(cohort_ages_sec)

total_years = SPINUP_YEARS + OPT_YEARS
start_date = "2000-01-01"
end_date = str((pd.Timestamp(start_date) + pd.DateOffset(years=total_years)).date())

dates = pd.date_range(
    start=pd.to_datetime(start_date),
    periods=(pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 5,
    freq="D",
)

lat = np.arange(NY, dtype=float)
lon = np.arange(NX, dtype=float)

# --- Day of year ---
doy = dates.dayofyear.values.astype(float)
doy_3d = np.broadcast_to(doy[:, None, None], (len(dates), NY, NX))
doy_da = xr.DataArray(doy_3d, dims=["T", "Y", "X"], coords={"T": dates, "Y": lat, "X": lon})

# --- Temperature (2 layers, seasonal) ---
temp_surface = 15.0 + 5.0 * np.sin(2 * np.pi * doy / 365.0)
temp_deep = 8.0 + 2.0 * np.sin(2 * np.pi * doy / 365.0)
temp_4d = np.stack(
    [
        np.broadcast_to(temp_surface[:, None, None], (len(dates), NY, NX)),
        np.broadcast_to(temp_deep[:, None, None], (len(dates), NY, NX)),
    ],
    axis=1,
)
temp_da = xr.DataArray(
    temp_4d,
    dims=["T", "Z", "Y", "X"],
    coords={"T": dates, "Z": np.arange(N_LAYERS), "Y": lat, "X": lon},
)

# --- NPP: localised Gaussian blob with seasonal modulation ---
yy, xx = np.meshgrid(lat, lon, indexing="ij")
npp_spatial = NPP_PEAK * np.exp(
    -((yy - NPP_CENTRE[0]) ** 2 + (xx - NPP_CENTRE[1]) ** 2) / (2 * NPP_SIGMA**2)
)
seasonal = 1.0 + 0.5 * np.sin(2 * np.pi * doy / 365.0)
npp_3d = seasonal[:, None, None] * npp_spatial[None, :, :]
npp_da = xr.DataArray(npp_3d, dims=["T", "Y", "X"], coords={"T": dates, "Y": lat, "X": lon})

# --- Rankine vortex (stationary, same for both layers) ---
cy, cx = NY / 2.0, NX / 2.0
dy_grid = (yy - cy) * CELL_SIZE
dx_grid = (xx - cx) * CELL_SIZE
r = np.sqrt(dx_grid**2 + dy_grid**2)
r = np.maximum(r, 1e-6)  # avoid division by zero at centre

# Rankine profile: solid-body inside, 1/r outside
r_max = VORTEX_RADIUS * CELL_SIZE
v_tangential = np.where(r <= r_max, VORTEX_VMAX * r / r_max, VORTEX_VMAX * r_max / r)

# Tangential velocity → (u, v) components: tangent = (-dy, dx) / r
u_field = -v_tangential * dy_grid / r  # zonal (eastward)
v_field = v_tangential * dx_grid / r  # meridional (northward)

# Broadcast to (T, Z, Y, X) — stationary
u_4d = np.broadcast_to(u_field[None, None, :, :], (len(dates), N_LAYERS, NY, NX))
v_4d = np.broadcast_to(v_field[None, None, :, :], (len(dates), N_LAYERS, NY, NX))
u_da = xr.DataArray(u_4d, dims=["T", "Z", "Y", "X"], coords={"T": dates, "Z": np.arange(N_LAYERS), "Y": lat, "X": lon})
v_da = xr.DataArray(v_4d, dims=["T", "Z", "Y", "X"], coords={"T": dates, "Z": np.arange(N_LAYERS), "Y": lat, "X": lon})

# --- Diffusivity (uniform, stationary) ---
D_4d = np.full((len(dates), N_LAYERS, NY, NX), DIFFUSIVITY)
D_da = xr.DataArray(D_4d, dims=["T", "Z", "Y", "X"], coords={"T": dates, "Z": np.arange(N_LAYERS), "Y": lat, "X": lon})

# --- Grid metrics (uniform rectangular grid) ---
dx_arr = np.full((NY, NX), CELL_SIZE)
dy_arr = np.full((NY, NX), CELL_SIZE)
cell_area = dx_arr * dy_arr
# --- Ocean mask with 2 islands ---
mask = np.ones((NY, NX))
# Island 1: 2x2 block, upper-left of centre
mask[6:8, 6:8] = 0.0
# Island 2: 2x2 block, lower-right of centre
mask[13:15, 13:15] = 0.0

dx_da = xr.DataArray(dx_arr, dims=["Y", "X"], coords={"Y": lat, "X": lon})
dy_da = xr.DataArray(dy_arr, dims=["Y", "X"], coords={"Y": lat, "X": lon})
face_height_da = xr.DataArray(dy_arr, dims=["Y", "X"], coords={"Y": lat, "X": lon})
face_width_da = xr.DataArray(dx_arr, dims=["Y", "X"], coords={"Y": lat, "X": lon})
cell_area_da = xr.DataArray(cell_area, dims=["Y", "X"], coords={"Y": lat, "X": lon})
mask_da = xr.DataArray(mask, dims=["Y", "X"], coords={"Y": lat, "X": lon})

# %% [markdown]
# ## Compile Model

# %%
config = Config.from_dict(
    {
        "parameters": {
            "lambda_0": {"value": [TRUE_PARAMS["lambda_0"]] * N_GROUPS},
            "gamma_lambda": {"value": [TRUE_PARAMS["gamma_lambda"]] * N_GROUPS},
            "tau_r_0": {"value": [TRUE_PARAMS["tau_r_0"]] * N_GROUPS},
            "gamma_tau_r": {"value": [TRUE_PARAMS["gamma_tau_r"]] * N_GROUPS},
            "t_ref": {"value": TRUE_PARAMS["t_ref"]},
            "efficiency": {"value": [TRUE_PARAMS["efficiency"]] * N_GROUPS},
            "cohort_ages": {"value": cohort_ages_sec.tolist()},
            "day_layer": {"value": [0, 1]},
            "night_layer": {"value": [0, 0]},
        },
        "forcings": {
            "latitude": xr.DataArray(np.full(NY, LATITUDE), dims=["Y"], coords={"Y": lat}),
            "temperature": temp_da,
            "primary_production": npp_da,
            "day_of_year": doy_da,
            "u": u_da,
            "v": v_da,
            "D": D_da,
            "dx": dx_da,
            "dy": dy_da,
            "face_height": face_height_da,
            "face_width": face_width_da,
            "cell_area": cell_area_da,
            "mask": mask_da,
            "bc_north": 1,  # OPEN
            "bc_south": 1,
            "bc_east": 1,
            "bc_west": 1,
        },
        "initial_state": {
            "biomass": xr.DataArray(
                np.zeros((N_GROUPS, NY, NX)), dims=["F", "Y", "X"], coords={"Y": lat, "X": lon}
            ),
            "production": xr.DataArray(
                np.zeros((N_GROUPS, NY, NX, n_cohorts)),
                dims=["F", "Y", "X", "C"],
                coords={"Y": lat, "X": lon},
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
print("GA 2 Groups + Transport (Twin Experiment)")
print(f"  Grid: {NY}x{NX}, cell={CELL_SIZE/1000:.0f}km, open BCs")
print(f"  Vortex: Rankine, Vmax={VORTEX_VMAX} m/s, R={VORTEX_RADIUS} cells")
print(f"  Spin-up: {SPINUP_YEARS}y | Optimization: {OPT_YEARS}y | dt: {DT}")
print(f"  Groups: G0=surface (0,0), G1=DVM (1,0)")
print(f"  Obs point: ({OBS_Y}, {OBS_X}), fraction={OBS_FRACTION}")
print(f"  GA: popsize={POPSIZE}, {N_GENERATIONS} gen, patience={PATIENCE}")
print(f"  Noise levels: {NOISE_LEVELS}")
print("=" * 60)

print("\nGenerating observations with TRUE parameters...")
t0 = time.time()
true_params_jax = {k: jnp.array([TRUE_PARAMS[k]] * N_GROUPS) for k in ALL_OPT_PARAMS}
outputs_true = runner(model, true_params_jax)

biomass_true = outputs_true["biomass"]  # (T, F, Y, X)
bio_g0_full = biomass_true[:, 0, OBS_Y, OBS_X]
bio_g1_full = biomass_true[:, 1, OBS_Y, OBS_X]

bio_g0 = bio_g0_full[spinup_steps:]
bio_g1 = bio_g1_full[spinup_steps:]
n_opt_steps = len(bio_g0)

print(f"  Simulation: {time.time() - t0:.2f}s ({n_timesteps} steps, {spinup_steps} spin-up)")
print(f"  Obs point G0 biomass: [{float(jnp.min(bio_g0)):.6f}, {float(jnp.max(bio_g0)):.6f}]")
print(f"  Obs point G1 biomass: [{float(jnp.min(bio_g1)):.6f}, {float(jnp.max(bio_g1)):.6f}]")

n_obs = max(2, int(OBS_FRACTION * n_opt_steps))
rng = np.random.default_rng(SEED)
obs_local_idx = np.sort(rng.choice(n_opt_steps, size=n_obs, replace=False))
obs_global_idx = obs_local_idx + spinup_steps

is_day = rng.random(n_obs) < 0.5
night_weight = (~is_day).astype(float)
n_day, n_night = int(np.sum(is_day)), int(np.sum(~is_day))

obs_values_clean = np.array(bio_g0[obs_local_idx]) + night_weight * np.array(bio_g1[obs_local_idx])
print(f"  {n_obs} observations ({n_day} day, {n_night} night), std={float(np.std(obs_values_clean)):.6f}")

# %% [markdown]
# ## Biomass Snapshots (12 months, year 2)

# %%
# Total biomass = G0 + G1, one snapshot per month during year 2
biomass_total_true = np.array(biomass_true[:, 0] + biomass_true[:, 1])  # (T, Y, X)
dt_sec_snap = float(model.dt)
month_indices = [spinup_steps + int(m * 30.44 * 86400 / dt_sec_snap) for m in range(12)]
month_indices = [min(i, n_timesteps - 1) for i in month_indices]

fig_snap, axes_snap = plt.subplots(3, 4, figsize=(16, 12))
vmin = float(np.min(biomass_total_true[month_indices]))
vmax = float(np.max(biomass_total_true[month_indices]))

for idx, (ax, t_idx) in enumerate(zip(axes_snap.flat, month_indices)):
    # Mask land cells as grey
    biomass_masked = np.where(mask, biomass_total_true[t_idx], np.nan)
    im = ax.imshow(
        biomass_masked, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis",
    )
    ax.contour(mask, levels=[0.5], colors="k", linewidths=1.5)
    ax.plot(OBS_X, OBS_Y, "r*", markersize=12, markeredgecolor="white", markeredgewidth=0.5)
    day_offset = (t_idx - spinup_steps) * dt_sec_snap / 86400.0
    ax.set_title(f"Month {idx + 1} (day {day_offset:.0f})", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

fig_snap.suptitle("True biomass (G0+G1) — monthly snapshots, year 2", fontsize=13)
fig_snap.colorbar(im, ax=axes_snap, label="Biomass (g/m²)", shrink=0.8)
snap_file = "examples/images/07_biomass_snapshots.png"
Path(snap_file).parent.mkdir(parents=True, exist_ok=True)
fig_snap.savefig(snap_file, dpi=150)
print(f"Snapshots saved to {snap_file}")

# %% [markdown]
# ## Shared helpers

# %%
_night_w = jnp.array(night_weight)


def extract_predictions(outputs):
    biomass = outputs["biomass"]  # (T, F, Y, X)
    g0 = biomass[:, 0, OBS_Y, OBS_X]
    g1 = biomass[:, 1, OBS_Y, OBS_X]
    return g0[obs_global_idx] + _night_w * g1[obs_global_idx]


dt_sec = model.dt
time_days = np.arange(n_opt_steps) * dt_sec / 86400.0

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
        noise = noise_rng.normal(0, noise_level * np.abs(obs_values_clean), size=obs_values_clean.shape)
        obs_values = obs_values_clean + noise
    else:
        obs_values = obs_values_clean

    objective = Objective(observations=jnp.array(obs_values), transform=extract_predictions)
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
    print(f"\n  Parameter recovery (per group):")
    for p in ALL_OPT_PARAMS:
        vals = result.params[p]
        for g in range(N_GROUPS):
            v = float(vals[g])
            ratio = v / TRUE_PARAMS[p]
            print(f"    {p:<14} G{g} = {v:>12.4g}  (ratio = {ratio:.4f})")

# %% [markdown]
# ## Visualization

# %%
# Build per-group parameter labels
param_labels = []
for p in ALL_OPT_PARAMS:
    for g in range(N_GROUPS):
        param_labels.append(f"{p[:10]} G{g}")
true_vals_flat = np.array([TRUE_PARAMS[p] for p in ALL_OPT_PARAMS for _ in range(N_GROUPS)])

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

    # --- Col 0: Biomass time series at obs point ---
    ax = axes[row_idx, 0]
    ax.plot(time_days, np.array(bio_g0), "k-", linewidth=2, label="True G0 (surface)", alpha=0.7)
    ax.plot(time_days, np.array(bio_g1), "k--", linewidth=2, label="True G1 (DVM)", alpha=0.7)
    ax.scatter(
        obs_local_idx[is_day] * dt_sec / 86400.0, obs_values[is_day],
        c="gold", s=25, zorder=5, label="Obs (day)", edgecolors="k", linewidths=0.5,
    )
    ax.scatter(
        obs_local_idx[~is_day] * dt_sec / 86400.0, obs_values[~is_day],
        c="navy", s=25, zorder=5, label="Obs (night)", edgecolors="k", linewidths=0.5,
    )
    outputs_best = runner(model, result.params)
    biomass_best = outputs_best["biomass"]
    bg0_best = biomass_best[:, 0, OBS_Y, OBS_X]
    bg1_best = biomass_best[:, 1, OBS_Y, OBS_X]
    ax.plot(time_days, np.array(bg0_best[spinup_steps:]), "-", color="orangered",
            linewidth=1.5, alpha=0.8, label="GA G0")
    ax.plot(time_days, np.array(bg1_best[spinup_steps:]), "--", color="orangered",
            linewidth=1.5, alpha=0.8, label="GA G1")
    ax.set_xlabel("Day")
    ax.set_ylabel("Biomass (g/m²)")
    ax.set_title(f"[{label}] Biomass at obs point ({OBS_Y},{OBS_X})")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Col 1: OBS vs PRED scatter ---
    ax = axes[row_idx, 1]
    bg0_opt = np.array(bg0_best[spinup_steps:])
    bg1_opt = np.array(bg1_best[spinup_steps:])
    pred_best = bg0_opt[obs_local_idx] + night_weight * bg1_opt[obs_local_idx]
    ax.scatter(obs_values, pred_best, c=np.where(is_day, "gold", "navy"),
               edgecolors="k", linewidths=0.5, s=30, zorder=5)
    obs_range = [min(obs_values.min(), pred_best.min()), max(obs_values.max(), pred_best.max())]
    ax.plot(obs_range, obs_range, "k--", alpha=0.5, label="1:1 line")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted (GA)")
    ax.set_title(f"[{label}] Obs vs Pred (gold=day, navy=night)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Col 2: Parameter recovery (10 individual params) ---
    ax = axes[row_idx, 2]
    n_all = len(param_labels)
    x_pos = np.arange(n_all)
    vals = np.array([float(result.params[p][g]) for p in ALL_OPT_PARAMS for g in range(N_GROUPS)])
    ratios = vals / true_vals_flat

    ax.bar(x_pos - 0.15, np.ones(n_all), 0.3, label="True", color="black", alpha=0.3)
    ax.bar(x_pos + 0.15, ratios, 0.3, label="GA", color="orangered", alpha=0.7)
    ax.axhline(y=1, color="k", linestyle="--", alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(param_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Ratio to true value")
    ax.set_title(f"[{label}] Parameter recovery (per group)")
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

fig.suptitle("SimpleGA 2 Groups + Transport — Clean vs Noisy", fontsize=13)
fig.tight_layout()
Path(PLOT_FILE).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(PLOT_FILE, dpi=150)
print(f"\nPlot saved to {PLOT_FILE}")
