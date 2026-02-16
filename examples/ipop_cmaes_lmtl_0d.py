"""IPOP-CMA-ES on LMTL 0D Model (Twin Experiment).

Demonstrates multi-restart CMA-ES with increasing population (IPOP strategy,
Auger & Hansen 2005) on a 0D LMTL ecosystem model.

Steps:
1. Spin-up: simulate SPINUP_YEARS to stabilize the system
2. Optimization year: generate synthetic observations on OPT_YEARS
3. Run IPOP-CMA-ES to recover the parameters
4. Visualize modes found, convergence, and parameter recovery
"""

import math
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
from seapopym.optimization.ipop import run_ipop_cmaes

# Use GPU if available, fall back to CPU
print(f"JAX devices: {jax.devices()}")

# =============================================================================
# CONFIGURATION — modify these to tune the experiment
# =============================================================================

# Number of optimized parameters (used for Hansen defaults below)
N_PARAMS = 5

# IPOP-CMA-ES — Hansen defaults (pycma, Auger & Hansen 2005)
#   popsize:  4 + floor(3 * ln(n))
#   maxiter:  100 + 150 * (n+3)^2 / sqrt(popsize)
#   restarts: 9 (Auger & Hansen 2005 benchmark)
INITIAL_POPSIZE = 4 + int(3 * math.log(N_PARAMS))  # 8 for n=5
N_GENERATIONS = int(100 + 150 * (N_PARAMS + 3) ** 2 / math.sqrt(INITIAL_POPSIZE))
N_RESTARTS = 5
DISTANCE_THRESHOLD = 0.1
SEED = 42

# Simulation
SPINUP_YEARS = 1  # Stabilize the system before optimization
OPT_YEARS = 1  # Period on which observations are sampled
DT = "1d"

# Twin experiment
OBS_FRACTION = 0.1  # Fraction of year-2 timesteps sampled as observations
INITIAL_GUESS_FACTOR = 1  # Initial guess = factor × true values

# True biological parameters (to be recovered)
TRUE_PARAMS = {
    "lambda_0": 1 / 150 / 86400,  # 1/s — base mortality rate
    "gamma_lambda": 0.15,  # 1/degC — mortality temperature sensitivity
    "tau_r_0": 10.38 * 86400,  # s — base recruitment age
    "gamma_tau_r": 0.11,  # 1/degC — recruitment temperature sensitivity
    "efficiency": 0.1668,  # dimensionless — NPP-to-biomass efficiency
    "t_ref": 0.0,  # degC — reference temperature (fixed, not optimized)
}

# Bounds for optimized parameters
BOUNDS = {
    "lambda_0": (1e-10, 5 * TRUE_PARAMS["lambda_0"]),
    "gamma_lambda": (0.01, 5 * TRUE_PARAMS["gamma_lambda"]),
    "tau_r_0": (0.1 * TRUE_PARAMS["tau_r_0"], 5 * TRUE_PARAMS["tau_r_0"]),
    "gamma_tau_r": (0.01, 5 * TRUE_PARAMS["gamma_tau_r"]),
    "efficiency": (0.01, 5 * TRUE_PARAMS["efficiency"]),
}

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
        "id": "lmtl-ipop-demo",
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

# Grid 1x1 (0D)
ny, nx = 1, 1
lat = np.arange(ny)
lon = np.arange(nx)

# Forcing: Temperature (seasonal, 15 +/- 5 degC)
day_of_year = dates.dayofyear.values
temp_c = 15.0 + 5.0 * np.sin(2 * np.pi * day_of_year / 365.0)
temp_3d = np.broadcast_to(temp_c[:, None, None], (len(dates), ny, nx))
temp_da = xr.DataArray(temp_3d, dims=["T", "Y", "X"], coords={"T": dates, "Y": lat, "X": lon})

# Forcing: NPP (seasonal, 1 +/- 0.5 g/m^2/day)
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

# Spin-up / optimization split
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
    print("=" * 60)
    print("IPOP-CMA-ES on LMTL 0D (Twin Experiment)")
    print(f"  Spin-up: {SPINUP_YEARS} year(s)  |  Optimization: {OPT_YEARS} year(s)")
    print(
        f"  IPOP: {N_RESTARTS} restarts, pop {INITIAL_POPSIZE}→{INITIAL_POPSIZE * 2 ** (N_RESTARTS - 1)}, "
        f"{N_GENERATIONS} gen"
    )
    print("=" * 60)

    # ----- Generate observations with TRUE parameters (year 2 only) -----
    print("\nGenerating observations with TRUE parameters...")
    t0 = time.time()
    true_params_jax = {k: jnp.array(TRUE_PARAMS[k]) for k in BOUNDS}
    true_biomass_full = run_simulation(true_params_jax)

    # Split: year 1 = spin-up, year 2 = optimization
    true_biomass = true_biomass_full[_spinup_steps:]
    n_opt_steps = len(true_biomass)

    print(
        f"  Simulation: {time.time() - t0:.2f}s  ({_n_timesteps} total steps, "
        f"{_spinup_steps} spin-up, {n_opt_steps} opt)"
    )
    print(f"  Year 2 biomass range: [{float(jnp.min(true_biomass)):.4f}, {float(jnp.max(true_biomass)):.4f}]")

    # Sample observations on year 2
    n_obs = max(1, int(OBS_FRACTION * n_opt_steps))
    rng = np.random.default_rng(SEED)
    obs_local_indices = np.sort(rng.choice(n_opt_steps, size=n_obs, replace=False))
    obs_global_indices = obs_local_indices + _spinup_steps  # indices in full simulation
    observations = true_biomass[obs_local_indices]
    obs_std = float(jnp.std(observations))
    print(f"  {n_obs} observations ({100 * n_obs / n_opt_steps:.1f}%), std={obs_std:.6f}")

    # ----- Loss function (NRMSE-std, year 2 only) -----
    def loss_fn(params: dict) -> jnp.ndarray:
        """NRMSE-std loss computed on year 2 only."""
        biomass_full = run_simulation(params)
        pred = biomass_full[obs_global_indices]
        rmse = jnp.sqrt(jnp.mean((pred - observations) ** 2))
        return rmse / obs_std

    # Sanity check: true params should give loss ≈ 0
    true_loss = loss_fn(true_params_jax)
    print(f"  Sanity check: loss(true_params) = {float(true_loss):.6e}")

    # Initial guess
    initial_params = {k: jnp.array(INITIAL_GUESS_FACTOR * TRUE_PARAMS[k]) for k in BOUNDS}
    initial_loss = loss_fn(initial_params)
    print(f"\nInitial guess ({INITIAL_GUESS_FACTOR}x true), loss = {float(initial_loss):.6f}")

    # ----- IPOP-CMA-ES -----
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
        verbose=True,
    )

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Found {len(result.modes)} distinct mode(s) across {result.n_restarts} restarts")

    # ----- Results -----
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

    # ----- Visualization -----
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
    dt_seconds = _model.dt
    time_days = np.arange(n_opt_steps) * dt_seconds / 86400.0
    ax2.plot(time_days, true_biomass, "k-", linewidth=2, label="True", alpha=0.7)
    ax2.scatter(obs_local_indices * dt_seconds / 86400.0, observations, c="red", s=20, zorder=5, label="Observations")

    colors = plt.cm.Set1(np.linspace(0, 1, max(n_modes, 1)))
    for i, mode in enumerate(result.modes):
        pred_full = run_simulation(mode.params)
        pred = pred_full[_spinup_steps:]
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
    plt.savefig("examples/ipop_cmaes_lmtl_0d_results.png", dpi=150)
    print("\nPlot saved to examples/ipop_cmaes_lmtl_0d_results.png")
