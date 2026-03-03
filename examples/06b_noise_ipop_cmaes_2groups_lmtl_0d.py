"""IPOP-CMA-ES with 2 Functional Groups on LMTL 0D (Twin Experiment, Noisy Observations).

Demonstrates multi-group optimization using the native F dimension to run
2 groups in parallel, sharing the same 5 biological parameters, with 30%
Gaussian noise added to observations.

- Group 0 (surface): stays at surface (day_layer=0, night_layer=0)
- Group 1 (DVM):     diel vertical migration (day_layer=1, night_layer=0)

The DVM group experiences deeper (cooler) temperatures during daytime,
which affects mortality and recruitment rates via the Gillooly transform.

Observation model:
- Day:   surface sensor sees only group 0
- Night: surface sensor sees group 0 + group 1

Setup: 1y spin-up + 2y optimization, 5% observations, IPOP-CMA-ES.
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

import seapopym.functions.lmtl  # noqa: F401
from seapopym.blueprint import Config
from seapopym.compiler import compile_model
from seapopym.engine import Runner
from seapopym.models import LMTL_NO_TRANSPORT
from seapopym.optimization.ipop import run_ipop_cmaes

# Force CPU for 0D model (GPU overhead dominates for tiny workloads)
jax.config.update("jax_default_device", jax.devices("cpu")[0])
print(f"JAX device: {jax.devices('cpu')[0]}")

# =============================================================================
# CONFIGURATION
# =============================================================================

N_PARAMS = 5
INITIAL_POPSIZE = 4 + int(3 * math.log(N_PARAMS))  # Hansen: 4 + floor(3*ln(n))
N_GENERATIONS = int(100 + 150 * (N_PARAMS + 3) ** 2 / math.sqrt(INITIAL_POPSIZE))
N_RESTARTS = 5
DISTANCE_THRESHOLD = 0.1
SEED = 42

SPINUP_YEARS = 1
OPT_YEARS = 2
DT = "1d"
OBS_FRACTION = 0.05
INITIAL_GUESS_FACTOR = 1.5
LATITUDE = 30.0  # degrees N

# True biological parameters (shared between both groups)
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

FIXED_PARAMS = {"t_ref": TRUE_PARAMS["t_ref"], "latitude": LATITUDE}

NOISE_LEVEL = 0.30  # 30% Gaussian noise on observations

PLOT_FILE = "examples/images/06b_noise_ipop_cmaes_2groups_lmtl_0d.png"

# =============================================================================
# BLUEPRINT (LMTL no-transport, with native F dimension for 2 groups)
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
n_layers = 2
n_groups = 2

# Day of year forcing (T, Y, X) — broadcast spatially
doy = dates.dayofyear.values.astype(float)
doy_3d = np.broadcast_to(doy[:, None, None], (len(dates), ny, nx))
doy_da = xr.DataArray(doy_3d, dims=["T", "Y", "X"], coords={"T": dates, "Y": lat, "X": lon})

# Temperature: 2 depth layers
# Surface (Z=0): 15 +/- 5degC seasonal
# Deep    (Z=1):  8 +/- 2degC (cooler, damped seasonal)
temp_surface = 15.0 + 5.0 * np.sin(2 * np.pi * doy / 365.0)
temp_deep = 8.0 + 2.0 * np.sin(2 * np.pi * doy / 365.0)
temp_4d = np.stack(
    [
        np.broadcast_to(temp_surface[:, None, None], (len(dates), ny, nx)),
        np.broadcast_to(temp_deep[:, None, None], (len(dates), ny, nx)),
    ],
    axis=1,
)  # (T, Z=2, Y, X)
temp_da = xr.DataArray(
    temp_4d,
    dims=["T", "Z", "Y", "X"],
    coords={"T": dates, "Z": np.arange(n_layers), "Y": lat, "X": lon},
)

# NPP: surface only (no Z dimension)
npp_day = 1.0 + 0.5 * np.sin(2 * np.pi * doy / 365.0)
npp_sec = npp_day / 86400.0
npp_3d = np.broadcast_to(npp_sec[:, None, None], (len(dates), ny, nx))
npp_da = xr.DataArray(npp_3d, dims=["T", "Y", "X"], coords={"T": dates, "Y": lat, "X": lon})

# Group 0: surface only (day=0, night=0)
# Group 1: DVM (day=1, night=0)
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
            "day_layer": xr.DataArray([0, 1], dims=["F"]),
            "night_layer": xr.DataArray([0, 0], dims=["F"]),
            "latitude": {"value": LATITUDE},
        },
        "forcings": {
            "temperature": temp_da,
            "primary_production": npp_da,
            "day_of_year": doy_da,
        },
        "initial_state": {
            "biomass": xr.DataArray(
                np.zeros((n_groups, ny, nx)), dims=["F", "Y", "X"], coords={"Y": lat, "X": lon}
            ),
            "production": xr.DataArray(
                np.zeros((n_groups, ny, nx, n_cohorts)), dims=["F", "Y", "X", "C"], coords={"Y": lat, "X": lon}
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

# =============================================================================
# COMPILE & BUILD SIMULATION
# =============================================================================

print("Compiling model...")
_model = compile_model(blueprint, config)
_runner = Runner.optimization()
_n_timesteps = _model.n_timesteps
_spinup_steps = int(SPINUP_YEARS / total_years * _n_timesteps)


def run_simulation(params: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run full simulation, return per-group biomass time series (T,) each."""
    outputs = _runner(_model, params)
    # outputs["biomass"] has shape (T, F, Y, X) — extract per-group spatial mean
    biomass = outputs["biomass"]
    bio_g0 = jnp.mean(biomass[:, 0], axis=tuple(range(1, biomass.ndim - 1)))
    bio_g1 = jnp.mean(biomass[:, 1], axis=tuple(range(1, biomass.ndim - 1)))
    return bio_g0, bio_g1


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("IPOP-CMA-ES with 2 Functional Groups (Twin Experiment)")
    print(f"  Spin-up: {SPINUP_YEARS}y | Optimization: {OPT_YEARS}y | dt: {DT}")
    print(f"  Group 0: surface (day=0, night=0)")
    print(f"  Group 1: DVM     (day=1, night=0)")
    print(f"  Latitude: {LATITUDE}°N")
    print(f"  IPOP: {N_RESTARTS} restarts, pop {INITIAL_POPSIZE}→{INITIAL_POPSIZE * 2 ** (N_RESTARTS - 1)}")
    print("=" * 60)

    # ----- Generate observations with TRUE parameters -----
    print("\nGenerating observations with TRUE parameters...")
    t0 = time.time()
    true_params_jax = {k: jnp.array(TRUE_PARAMS[k]) for k in ALL_OPT_PARAMS}
    bio_g0_full, bio_g1_full = run_simulation(true_params_jax)

    # Post spin-up
    bio_g0 = bio_g0_full[_spinup_steps:]
    bio_g1 = bio_g1_full[_spinup_steps:]
    n_opt_steps = len(bio_g0)

    print(f"  Simulation: {time.time() - t0:.2f}s ({_n_timesteps} steps, {_spinup_steps} spin-up)")
    print(f"  Group 0 biomass: [{float(jnp.min(bio_g0)):.4f}, {float(jnp.max(bio_g0)):.4f}]")
    print(f"  Group 1 biomass: [{float(jnp.min(bio_g1)):.4f}, {float(jnp.max(bio_g1)):.4f}]")

    # Sample observations (5%)
    n_obs = max(2, int(OBS_FRACTION * n_opt_steps))
    rng = np.random.default_rng(SEED)
    obs_local_idx = np.sort(rng.choice(n_opt_steps, size=n_obs, replace=False))
    obs_global_idx = obs_local_idx + _spinup_steps

    # Randomly assign day/night (50/50)
    is_day = rng.random(n_obs) < 0.5
    night_weight = (~is_day).astype(float)
    n_day, n_night = int(np.sum(is_day)), int(np.sum(~is_day))

    # Observations: day = g0 only, night = g0 + g1
    obs_values = np.array(bio_g0[obs_local_idx]) + night_weight * np.array(bio_g1[obs_local_idx])
    noise = rng.normal(0, NOISE_LEVEL * np.abs(obs_values), size=obs_values.shape)
    obs_values = obs_values + noise
    obs_std = float(np.std(obs_values))
    print(f"  {n_obs} observations ({n_day} day, {n_night} night), std={obs_std:.6f}")

    # ----- Loss function -----
    _obs_global = jnp.array(obs_global_idx)
    _obs_values = jnp.array(obs_values)
    _night_w = jnp.array(night_weight)

    def loss_fn(opt_params: dict) -> jnp.ndarray:
        """NRMSE loss with day/night observation operator."""
        bg0, bg1 = run_simulation(opt_params)

        # Day: surface only (g0). Night: both groups (g0 + g1).
        pred = bg0[_obs_global] + _night_w * bg1[_obs_global]
        return jnp.sqrt(jnp.mean((pred - _obs_values) ** 2)) / obs_std

    # Sanity check
    true_loss = loss_fn(true_params_jax)
    print(f"  Sanity: loss(true) = {float(true_loss):.6e}")

    # ----- IPOP-CMA-ES -----
    initial_params = {k: jnp.array(INITIAL_GUESS_FACTOR * TRUE_PARAMS[k]) for k in ALL_OPT_PARAMS}
    print(f"\nInitial guess ({INITIAL_GUESS_FACTOR}x true), loss = {float(loss_fn(initial_params)):.6f}")

    print("\nRunning IPOP-CMA-ES...")
    t0 = time.time()
    result = run_ipop_cmaes(
        loss_fn=loss_fn,
        initial_params=initial_params,
        bounds=BOUNDS,
        n_restarts=N_RESTARTS,
        initial_popsize=INITIAL_POPSIZE,
        n_generations=N_GENERATIONS,
        distance_threshold=DISTANCE_THRESHOLD,
        seed=SEED,
        progress_bar=True,
    )
    elapsed = time.time() - t0

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Found {len(result.modes)} distinct mode(s) across {result.n_restarts} restarts")

    # ----- Results -----
    print("\n" + "=" * 60)
    print("Modes found (sorted by loss)")
    print("=" * 60)

    header = f"{'Mode':<6} {'Loss':>10}"
    for p in ALL_OPT_PARAMS:
        header += f" {p[:10]:>12}"
    print(header)
    print("-" * len(header))

    row = f"{'True':<6} {0.0:>10.6f}"
    for p in ALL_OPT_PARAMS:
        row += f" {TRUE_PARAMS[p]:>12.4g}"
    print(row)

    for i, mode in enumerate(result.modes):
        row = f"{'#' + str(i + 1):<6} {mode.loss:>10.6f}"
        for p in ALL_OPT_PARAMS:
            row += f" {float(mode.params[p]):>12.4g}"
        print(row)

    best = result.modes[0]
    print("\nParameter recovery (best mode):")
    for p in ALL_OPT_PARAMS:
        ratio = float(best.params[p]) / TRUE_PARAMS[p]
        print(f"  {p:<14} ratio = {ratio:.4f}")

    # ----- Visualization -----
    n_modes = len(result.modes)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.Set1(np.linspace(0, 1, max(n_modes, 1)))
    dt_sec = _model.dt
    time_days = np.arange(n_opt_steps) * dt_sec / 86400.0

    # --- Top left: Biomass trajectories (both groups) ---
    ax = axes[0, 0]
    ax.plot(time_days, np.array(bio_g0), "k-", linewidth=2, label="True G0 (surface)", alpha=0.7)
    ax.plot(time_days, np.array(bio_g1), "k--", linewidth=2, label="True G1 (DVM)", alpha=0.7)
    ax.scatter(
        obs_local_idx[is_day] * dt_sec / 86400.0,
        obs_values[is_day],
        c="gold", s=25, zorder=5, label="Obs (day)", edgecolors="k", linewidths=0.5,
    )
    ax.scatter(
        obs_local_idx[~is_day] * dt_sec / 86400.0,
        obs_values[~is_day],
        c="navy", s=25, zorder=5, label="Obs (night)", edgecolors="k", linewidths=0.5,
    )
    for i, mode in enumerate(result.modes):
        bg0_m, bg1_m = run_simulation(mode.params)
        ax.plot(time_days, np.array(bg0_m[_spinup_steps:]), "-", color=colors[i],
                linewidth=1.5, alpha=0.8, label=f"Mode #{i+1} G0")
        ax.plot(time_days, np.array(bg1_m[_spinup_steps:]), "--", color=colors[i],
                linewidth=1.5, alpha=0.8, label=f"Mode #{i+1} G1")
    ax.set_xlabel("Day")
    ax.set_ylabel("Biomass (g/m²)")
    ax.set_title("Biomass: True vs CMA-ES modes (G0=surface, G1=DVM)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Top right: Parameter recovery ---
    ax = axes[0, 1]
    x_pos = np.arange(len(ALL_OPT_PARAMS))
    width = 0.8 / (n_modes + 1)
    ax.bar(x_pos - 0.4 + width / 2, np.ones(len(ALL_OPT_PARAMS)), width,
           label="True", color="black", alpha=0.3)
    true_vals = np.array([TRUE_PARAMS[p] for p in ALL_OPT_PARAMS])
    for i, mode in enumerate(result.modes):
        ratios = np.array([float(mode.params[p]) for p in ALL_OPT_PARAMS]) / true_vals
        ax.bar(x_pos - 0.4 + (i + 1.5) * width, ratios, width,
               label=f"Mode #{i+1}", color=colors[i], alpha=0.7)
    ax.axhline(y=1, color="k", linestyle="--", alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([p[:10] for p in ALL_OPT_PARAMS], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Ratio to true value")
    ax.set_title("Parameter recovery")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Bottom left: Loss per restart ---
    ax = axes[1, 0]
    restart_losses = [float(r.loss) for r in result.all_results]
    mode_indices = [i for i, r in enumerate(result.all_results)
                    if any(m.loss == r.loss for m in result.modes)]
    ax.bar(range(len(restart_losses)), restart_losses, color="steelblue", alpha=0.7)
    ax.bar(mode_indices, [restart_losses[i] for i in mode_indices], color="orangered", alpha=0.9)
    ax.set_xlabel("Restart")
    ax.set_ylabel("Loss (NRMSE)")
    ax.set_title("Loss per restart")
    popsizes = [INITIAL_POPSIZE * 2**i for i in range(len(restart_losses))]
    ax.set_xticks(range(len(restart_losses)))
    ax.set_xticklabels([f"pop={p}" for p in popsizes], rotation=45, ha="right", fontsize=8)

    # --- Bottom right: Surface observation fit ---
    ax = axes[1, 1]
    bg0_best, bg1_best = run_simulation(best.params)
    bg0_opt = np.array(bg0_best[_spinup_steps:])
    bg1_opt = np.array(bg1_best[_spinup_steps:])
    pred_best = bg0_opt[obs_local_idx] + night_weight * bg1_opt[obs_local_idx]
    ax.scatter(obs_values, pred_best, c=np.where(is_day, "gold", "navy"),
               edgecolors="k", linewidths=0.5, s=30, zorder=5)
    obs_range = [min(obs_values.min(), pred_best.min()), max(obs_values.max(), pred_best.max())]
    ax.plot(obs_range, obs_range, "k--", alpha=0.5, label="1:1 line")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted (best mode)")
    ax.set_title("Observation fit (gold=day, navy=night)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"IPOP-CMA-ES 2 Groups — {elapsed:.0f}s, {n_modes} mode(s), "
        f"best loss={best.loss:.2e}",
        fontsize=12,
    )
    fig.tight_layout()
    Path(PLOT_FILE).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_FILE, dpi=150)
    print(f"\nPlot saved to {PLOT_FILE}")
