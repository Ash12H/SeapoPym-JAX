"""Optimization Comparison for LMTL 0D Model.

Compares gradient-based (Adam) and CMA-ES optimization methods
on the LMTL (Low/Mid Trophic Level) model using twin experiments.

Grid: 1x1 (0D)
Setup: 1y spin-up + 1y optimization, 10% observations, dt=1d
Loss: NRMSE-std (RMSE normalized by observation std)
"""

import time
from pathlib import Path

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
from seapopym.optimization import EvolutionaryOptimizer, Optimizer

# Force CPU for 0D model (GPU overhead dominates for tiny workloads)
jax.config.update("jax_default_device", jax.devices("cpu")[0])
print(f"JAX device: {jax.devices('cpu')[0]}")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Simulation
SPINUP_YEARS = 1
OPT_YEARS = 1
DT = "1d"
LATITUDE = 30.0  # degrees N

# Twin experiment
OBS_FRACTION = 0.02
INITIAL_GUESS_FACTOR = 2  # Initial guess = factor × true values
SEED = 42

# True biological parameters (to be recovered)
TRUE_PARAMS = {
    "lambda_0": 1 / 150 / 86400,  # 1/s — base mortality rate
    "gamma_lambda": 0.15,  # 1/degC — mortality temperature sensitivity
    "tau_r_0": 10.38 * 86400,  # s — base recruitment age
    "gamma_tau_r": 0.11,  # 1/degC — recruitment temperature sensitivity
    "efficiency": 0.1668,  # dimensionless — NPP to biomass efficiency
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

# CMA-ES configuration (Hansen defaults for n=5)
N_PARAMS = 5
CMAES_POPSIZE = 4 + int(3 * np.log(N_PARAMS))  # 8 for n=5
CMAES_PATIENCE = 50

# Gradient configuration
GRAD_N_STEPS = 500
GRAD_LR = 0.01

PLOT_FILE = "examples/images/04_optimization_comparison_lmtl_0d.png"

# =============================================================================
# BLUEPRINT (0D LMTL from catalogue)
# =============================================================================

blueprint = LMTL_NO_TRANSPORT

max_age_days = int(np.ceil(TRUE_PARAMS["tau_r_0"] / 86400))
cohort_ages_sec = np.arange(0, max_age_days + 1) * 86400.0
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

# Grid 1x1 (0D)
ny, nx = 1, 1
lat = np.arange(ny)
lon = np.arange(nx)

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
    print("=" * 60)
    print("Optimization Comparison: Gradient (Adam) vs CMA-ES")
    print(f"  Spin-up: {SPINUP_YEARS}y | Optimization: {OPT_YEARS}y | dt: {DT}")
    print(f"  Gradient: Adam lr={GRAD_LR}, max {GRAD_N_STEPS} steps")
    print(f"  CMA-ES:   popsize={CMAES_POPSIZE}, patience={CMAES_PATIENCE}")
    print("=" * 60)

    # ----- Generate observations with TRUE parameters -----
    print("\nGenerating observations with TRUE parameters...")
    t0 = time.time()
    true_params_jax = {k: jnp.array(TRUE_PARAMS[k]) for k in BOUNDS}
    true_biomass_full = run_simulation(true_params_jax)

    # Split: spin-up + optimization
    true_biomass = true_biomass_full[_spinup_steps:]
    n_opt_steps = len(true_biomass)

    print(
        f"  Simulation: {time.time() - t0:.2f}s  ({_n_timesteps} total steps, "
        f"{_spinup_steps} spin-up, {n_opt_steps} opt)"
    )
    print(f"  Biomass range: [{float(jnp.min(true_biomass)):.4f}, {float(jnp.max(true_biomass)):.4f}]")

    # Sample observations
    n_obs = max(1, int(OBS_FRACTION * n_opt_steps))
    rng = np.random.default_rng(SEED)
    obs_local_indices = np.sort(rng.choice(n_opt_steps, size=n_obs, replace=False))
    obs_global_indices = obs_local_indices + _spinup_steps
    observations = true_biomass[obs_local_indices]
    obs_std = float(jnp.std(observations))

    print(f"  {n_obs} observations ({100 * n_obs / n_opt_steps:.1f}%), std={obs_std:.6f}")

    # ----- Loss function -----
    def loss_fn(params: dict) -> jnp.ndarray:
        """NRMSE-std loss computed on optimization period only."""
        biomass_full = run_simulation(params)
        pred = biomass_full[obs_global_indices]
        rmse = jnp.sqrt(jnp.mean((pred - observations) ** 2))
        return rmse / obs_std

    # Sanity check
    true_loss = loss_fn(true_params_jax)
    print(f"  Sanity: loss(true) = {float(true_loss):.6e}")

    # Initial guess
    initial_params = {k: jnp.array(INITIAL_GUESS_FACTOR * TRUE_PARAMS[k]) for k in BOUNDS}
    initial_loss = loss_fn(initial_params)
    print(f"\nInitial guess ({INITIAL_GUESS_FACTOR}x true), loss = {float(initial_loss):.6f}")

    # =================================================================
    # OPTIMIZATION
    # =================================================================

    results = {}
    param_names = list(BOUNDS.keys())

    # --- Gradient-based (Adam) ---
    print("\nRunning Gradient (Adam)...")
    t0 = time.time()
    grad_opt = Optimizer(
        algorithm="adam",
        learning_rate=GRAD_LR,
        bounds=BOUNDS,
        scaling="bounds",
    )
    results["gradient"] = grad_opt.run(loss_fn, initial_params, n_steps=GRAD_N_STEPS, verbose=True)
    grad_time = time.time() - t0
    print(f"  Time: {grad_time:.2f}s  |  Loss: {results['gradient'].loss:.6f}")

    # --- CMA-ES ---
    print("\nRunning CMA-ES...")
    t0 = time.time()
    evo_opt = EvolutionaryOptimizer(
        popsize=CMAES_POPSIZE,
        bounds=BOUNDS,
        seed=SEED,
    )
    results["cma_es"] = evo_opt.run(
        loss_fn,
        initial_params,
        n_generations=500,
        patience=CMAES_PATIENCE,
        verbose=True,
    )
    evo_time = time.time() - t0
    print(f"  Time: {evo_time:.2f}s  |  Loss: {results['cma_es'].loss:.6f}")

    # =================================================================
    # SUMMARY
    # =================================================================

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    header = f"{'Method':<12} {'Loss':>10}"
    for p in param_names:
        header += f" {p[:8]:>10}"
    header += f" {'Time':>8}"
    print(header)
    print("-" * len(header))

    row = f"{'True':<12} {0.0:>10.6f}"
    for p in param_names:
        row += f" {TRUE_PARAMS[p]:>10.4g}"
    row += f" {'-':>8}"
    print(row)

    times = {"gradient": grad_time, "cma_es": evo_time}
    for method, result in results.items():
        row = f"{method:<12} {result.loss:>10.6f}"
        for p in param_names:
            row += f" {float(result.params[p]):>10.4g}"
        row += f" {times[method]:>7.1f}s"
        print(row)

    print(f"\nFixed parameters: t_ref = {FIXED_PARAMS['t_ref']}")

    # =================================================================
    # VISUALIZATION
    # =================================================================

    dt_seconds = _model.dt
    time_days = np.arange(n_opt_steps) * dt_seconds / 86400.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Plot 1: Convergence ---
    ax1 = axes[0]
    for method, result in results.items():
        ax1.semilogy(result.loss_history, label=method, alpha=0.8)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss (NRMSE-std, log scale)")
    ax1.set_title("Convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Biomass trajectories (optimization period) ---
    ax2 = axes[1]
    ax2.plot(time_days, true_biomass, "k-", label="True", linewidth=2, alpha=0.7)
    ax2.scatter(
        obs_local_indices * dt_seconds / 86400.0,
        observations,
        c="red",
        s=20,
        zorder=5,
        label="Observations",
    )

    for method, result in results.items():
        pred_full = run_simulation(result.params)
        pred = pred_full[_spinup_steps:]
        ax2.plot(time_days, pred, "--", label=method, alpha=0.7)

    ax2.set_xlabel("Day (year 2)")
    ax2.set_ylabel("Biomass (g/m²)")
    ax2.set_title("Biomass Trajectories")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Parameter recovery ---
    ax3 = axes[2]
    x = np.arange(len(param_names))
    width = 0.25

    true_vals = np.array([TRUE_PARAMS[p] for p in param_names])
    ax3.bar(x - width, np.ones(len(param_names)), width, label="True", color="black", alpha=0.3)

    for i, (method, result) in enumerate(results.items()):
        pred_vals = np.array([float(result.params[p]) for p in param_names])
        ratios = pred_vals / true_vals
        ax3.bar(x + i * width, ratios, width, label=method, alpha=0.7)

    ax3.set_xticks(x)
    ax3.set_xticklabels([p[:8] for p in param_names], rotation=45, ha="right")
    ax3.set_ylabel("Ratio to True")
    ax3.set_title("Parameter Recovery")
    ax3.axhline(y=1, color="k", linestyle="--", alpha=0.5)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    Path(PLOT_FILE).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_FILE, dpi=150)
    print(f"\nPlot saved to {PLOT_FILE}")
