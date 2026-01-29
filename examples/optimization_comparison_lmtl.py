"""Optimization Comparison for LMTL Model.

Compares gradient-based, CMA-ES, and hybrid optimization methods
on the LMTL (Low/Mid Trophic Level) model using twin experiments.

Grid: 1x1 (0D-like, extensible to larger grids)
Duration: 1 year
Observations: 1% of timesteps (twin experiment with true parameters)
Loss: NRMSE-std (RMSE normalized by observation std)
"""

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from seapopym.blueprint import Blueprint, Config, functional
from seapopym.compiler import compile_model
from seapopym.engine.step import build_step_fn
from seapopym.optimization import EvolutionaryOptimizer, HybridOptimizer, Optimizer

# =============================================================================
# 1. TRUE PARAMETERS (for twin experiment)
# =============================================================================

TRUE_PARAMS = {
    "lambda_0": 1 / 150 / 86400,  # 1/s - base mortality rate
    "gamma_lambda": 0.15,  # 1/degC - mortality temperature sensitivity
    "tau_r_0": 10.38 * 86400,  # s - base recruitment age
    "gamma_tau_r": 0.11,  # 1/degC - recruitment temperature sensitivity
    "t_ref": 0.0,  # degC - reference temperature
    "efficiency": 0.1668,  # dimensionless - NPP to biomass efficiency
}

# Bounds: [~0, 5 * true] for each parameter
# t_ref is NOT optimized (fixed at true value)
BOUNDS = {
    "lambda_0": (1e-10, 5 * TRUE_PARAMS["lambda_0"]),
    "gamma_lambda": (0.01, 5 * TRUE_PARAMS["gamma_lambda"]),
    "tau_r_0": (0.1 * TRUE_PARAMS["tau_r_0"], 5 * TRUE_PARAMS["tau_r_0"]),
    "gamma_tau_r": (0.01, 5 * TRUE_PARAMS["gamma_tau_r"]),
    "efficiency": (0.01, 5 * TRUE_PARAMS["efficiency"]),
}

# Fixed parameters (not optimized)
FIXED_PARAMS = {
    "t_ref": TRUE_PARAMS["t_ref"],
}

# =============================================================================
# 2. LMTL FUNCTIONS (same as lmtl_2d.py)
# =============================================================================


@functional(name="lmtl:gillooly_temperature", backend="jax", units={"temp": "degC", "return": "degC"})
def gillooly_temperature(temp):
    """Normalize temperature using Gillooly et al. (2001)."""
    return temp / (1.0 + temp / 273.0)


@functional(
    name="lmtl:recruitment_age",
    backend="jax",
    units={"temp": "degC", "tau_r_0": "s", "gamma": "1/delta_degC", "t_ref": "degC", "return": "s"},
)
def recruitment_age(temp, tau_r_0, gamma, t_ref):
    """Compute recruitment age (time to recruitment)."""
    return tau_r_0 * jnp.exp(-gamma * (temp - t_ref))


@functional(
    name="lmtl:mortality",
    backend="jax",
    units={
        "biomass": "g/m^2",
        "temp": "degC",
        "lambda_0": "1/s",
        "gamma": "1/delta_degC",
        "t_ref": "degC",
        "return": "g/m^2/s",
    },
)
def mortality_tendency(biomass, temp, lambda_0, gamma, t_ref):
    """Compute mortality loss for biomass."""
    rate = lambda_0 * jnp.exp(gamma * (temp - t_ref))
    return -rate * biomass


@functional(
    name="lmtl:npp_injection",
    backend="jax",
    core_dims={"production": ["C"]},
    out_dims=["C"],
    units={
        "npp": "g/m^2/s",
        "efficiency": "dimensionless",
        "production": "g/m^2",
        "return": "g/m^2/s",
    },
)
def npp_injection(npp, efficiency, production):
    """Inject Primary Production into the first cohort (0)."""
    source_flux = npp * efficiency
    tendency = jnp.zeros_like(production)
    tendency = tendency.at[0].set(source_flux)
    return tendency


@functional(
    name="lmtl:aging_flow",
    backend="jax",
    core_dims={"production": ["C"], "cohort_ages": ["C"]},
    out_dims=["C"],
    units={"production": "g/m^2", "cohort_ages": "s", "rec_age": "s", "return": "g/m^2/s"},
)
def aging_flow(production, cohort_ages, rec_age):
    """Compute aging flux (transfer from C to C+1)."""
    d_tau_raw = cohort_ages[1:] - cohort_ages[:-1]
    last_d_tau = d_tau_raw[-1:]
    d_tau = jnp.concatenate([d_tau_raw, last_d_tau])
    aging_coef = 1.0 / d_tau
    base_outflow = production * aging_coef
    is_recruited = cohort_ages >= rec_age
    aging_outflow = jnp.where(is_recruited, 0.0, base_outflow)
    aging_outflow = aging_outflow.at[-1].set(0.0)
    loss = -aging_outflow
    gain = jnp.concatenate([jnp.zeros(1), aging_outflow[:-1]])
    return loss + gain


@functional(
    name="lmtl:recruitment_flow",
    backend="jax",
    core_dims={"production": ["C"], "cohort_ages": ["C"]},
    out_dims=["C"],
    outputs=["prod_loss", "biomass_gain"],
    units={
        "production": "g/m^2",
        "cohort_ages": "s",
        "rec_age": "s",
        "prod_loss": "g/m^2/s",
        "biomass_gain": "g/m^2/s",
    },
)
def recruitment_flow(production, cohort_ages, rec_age):
    """Compute recruitment flux (transfer from C to Biomass)."""
    d_tau_raw = cohort_ages[1:] - cohort_ages[:-1]
    last_d_tau = d_tau_raw[-1:]
    d_tau = jnp.concatenate([d_tau_raw, last_d_tau])
    aging_coef = 1.0 / d_tau
    base_outflow = production * aging_coef
    is_recruited = cohort_ages >= rec_age
    flux_to_biomass = jnp.where(is_recruited, base_outflow, 0.0)
    prod_loss = -flux_to_biomass
    biomass_gain = jnp.sum(flux_to_biomass)
    return prod_loss, biomass_gain


# =============================================================================
# 3. BLUEPRINT
# =============================================================================

# Cohort ages based on max recruitment age
max_age_days = int(np.ceil(TRUE_PARAMS["tau_r_0"] / 86400))
cohort_ages_days = np.arange(0, max_age_days + 1)
cohort_ages_sec = cohort_ages_days * 86400.0
n_cohorts = len(cohort_ages_sec)

blueprint = Blueprint.from_dict(
    {
        "id": "lmtl-optimization",
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
# 4. SIMULATION SETUP
# =============================================================================

# Time configuration
start_date = "2000-01-01"
end_date = "2001-01-01"  # 1 year
dt = "3h"

# Generate forcing dates
start_pd = pd.to_datetime(start_date)
end_pd = pd.to_datetime(end_date)
n_days = (end_pd - start_pd).days + 5
dates = pd.date_range(start=start_pd, periods=n_days, freq="D")

# Grid (1x1 for now, extensible)
grid_size = (1, 1)
ny, nx = grid_size
lat = np.arange(ny)
lon = np.arange(nx)

# Forcing: Temperature (seasonal)
day_of_year = dates.dayofyear.values
temp_c = 15.0 + 5.0 * np.sin(2 * np.pi * day_of_year / 365.0)
temp_3d = np.broadcast_to(temp_c[:, None, None], (len(dates), ny, nx))
temp_da = xr.DataArray(temp_3d, dims=["T", "Y", "X"], coords={"T": dates, "Y": lat, "X": lon})

# Forcing: NPP (seasonal)
npp_day = 1.0 + 0.5 * np.sin(2 * np.pi * day_of_year / 365.0)
npp_sec = npp_day / 86400.0
npp_3d = np.broadcast_to(npp_sec[:, None, None], (len(dates), ny, nx))
npp_da = xr.DataArray(npp_3d, dims=["T", "Y", "X"], coords={"T": dates, "Y": lat, "X": lon})


# =============================================================================
# 5. HELPER FUNCTIONS
# =============================================================================


def create_config(params: dict) -> Config:
    """Create a Config object with given parameters."""
    return Config.from_dict(
        {
            "parameters": {
                "lambda_0": {"value": params["lambda_0"]},
                "gamma_lambda": {"value": params["gamma_lambda"]},
                "tau_r_0": {"value": params["tau_r_0"]},
                "gamma_tau_r": {"value": params["gamma_tau_r"]},
                "t_ref": {"value": params["t_ref"]},
                "efficiency": {"value": params["efficiency"]},
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
                "dt": dt,
                "forcing_interpolation": "linear",
                "batch_size": 1000,
            },
        }
    )


# Compile model ONCE with true parameters (structure only matters)
print("Compiling base model...")
_base_config = create_config(TRUE_PARAMS)
_base_model = compile_model(blueprint, _base_config, backend="jax")
_step_fn = build_step_fn(_base_model, params_as_argument=True)
_n_timesteps = _base_model.n_timesteps
_initial_state = _base_model.state
_forcings_stacked = {name: jnp.asarray(arr) for name, arr in _base_model.forcings.items()}


def run_simulation_with_params(params: dict) -> jnp.ndarray:
    """Run LMTL simulation with given parameters (JAX-traceable)."""
    # Merge optimized params with fixed params
    full_params = {**params, **{k: jnp.array(v) for k, v in FIXED_PARAMS.items()}}
    # Also need cohort_ages from base model
    full_params["cohort_ages"] = _base_model.parameters["cohort_ages"]

    # Run simulation with scan
    def scan_body(carry, t):
        state, p = carry
        # Slice forcings for this timestep
        forcings_t = {}
        for name, arr in _forcings_stacked.items():
            if arr.ndim > 0 and arr.shape[0] == _n_timesteps:
                forcings_t[name] = arr[t]
            else:
                forcings_t[name] = arr
        new_carry, outputs = _step_fn((state, p), forcings_t)
        new_state, _ = new_carry
        # Extract biomass (mean over grid)
        biomass = jnp.mean(outputs["biomass"])
        return (new_state, p), biomass

    init_carry = (_initial_state, full_params)
    _, biomass_history = jax.lax.scan(scan_body, init_carry, jnp.arange(_n_timesteps))
    return biomass_history


def nrmse_std(pred: jnp.ndarray, obs: jnp.ndarray, obs_std: float) -> jnp.ndarray:
    """Compute NRMSE normalized by observation std."""
    rmse = jnp.sqrt(jnp.mean((pred - obs) ** 2))
    return rmse / obs_std


# =============================================================================
# 6. GENERATE OBSERVATIONS (Twin Experiment)
# =============================================================================

print("=" * 60)
print("LMTL Optimization Comparison")
print("=" * 60)
print(f"Grid size: {grid_size}")
print(f"Simulation: {start_date} to {end_date}")
print(f"Parameters to optimize: {len(TRUE_PARAMS)}")
print()

print("Generating observations with TRUE parameters...")
t0 = time.time()
# Convert TRUE_PARAMS to JAX arrays for simulation (only optimized params)
true_params_jax = {k: jnp.array(TRUE_PARAMS[k]) for k in BOUNDS}
true_biomass = run_simulation_with_params(true_params_jax)
print(f"  Simulation completed in {time.time() - t0:.2f}s")
print(f"  Biomass shape: {true_biomass.shape}")
print(f"  Biomass range: [{float(jnp.min(true_biomass)):.4f}, {float(jnp.max(true_biomass)):.4f}]")

# Select 1% of observations
n_timesteps = len(true_biomass)
n_obs = max(1, int(0.01 * n_timesteps))
rng = np.random.default_rng(42)
obs_indices = np.sort(rng.choice(n_timesteps, size=n_obs, replace=False))
observations = true_biomass[obs_indices]
obs_std = float(jnp.std(observations))

print(f"  Selected {n_obs} observations ({100 * n_obs / n_timesteps:.1f}%)")
print(f"  Observation std: {obs_std:.6f}")
print()

# =============================================================================
# 7. LOSS FUNCTION
# =============================================================================


def loss_fn(params: dict) -> jnp.ndarray:
    """Compute NRMSE-std loss for given parameters (JAX-traceable)."""
    biomass = run_simulation_with_params(params)
    pred = biomass[obs_indices]
    return nrmse_std(pred, observations, obs_std)


# =============================================================================
# 8. INITIAL GUESS (perturbed from true)
# =============================================================================

# Start from 2x true values (within bounds)
# Only parameters to optimize (t_ref is fixed)
INITIAL_PARAMS = {
    "lambda_0": jnp.array(2.0 * TRUE_PARAMS["lambda_0"]),
    "gamma_lambda": jnp.array(2.0 * TRUE_PARAMS["gamma_lambda"]),
    "tau_r_0": jnp.array(2.0 * TRUE_PARAMS["tau_r_0"]),
    "gamma_tau_r": jnp.array(2.0 * TRUE_PARAMS["gamma_tau_r"]),
    "efficiency": jnp.array(2.0 * TRUE_PARAMS["efficiency"]),
}

print("Initial guess (2x true values):")
for k, v in INITIAL_PARAMS.items():
    print(f"  {k}: {float(v):.6g} (true: {TRUE_PARAMS[k]:.6g})")

initial_loss = loss_fn(INITIAL_PARAMS)
print(f"Initial loss (NRMSE-std): {float(initial_loss):.6f}")
print()

# =============================================================================
# 9. OPTIMIZATION
# =============================================================================

results = {}

# --- Gradient-based ---
print("Running Gradient (Adam)...")
t0 = time.time()
grad_opt = Optimizer(
    algorithm="adam",
    learning_rate=0.01,  # Slightly higher LR
    bounds=BOUNDS,
    scaling="bounds",
)
results["gradient"] = grad_opt.run(loss_fn, INITIAL_PARAMS, n_steps=300)
grad_time = time.time() - t0
print(f"  Time: {grad_time:.2f}s")
print(f"  Loss: {results['gradient'].loss:.6f}")
print()

# --- CMA-ES ---
print("Running CMA-ES...")
t0 = time.time()
evo_opt = EvolutionaryOptimizer(
    popsize=10_000,  # Larger population for 5 params
    bounds=BOUNDS,
    seed=42,
)
results["cma_es"] = evo_opt.run(loss_fn, INITIAL_PARAMS, n_generations=50)
evo_time = time.time() - t0
print(f"  Time: {evo_time:.2f}s")
print(f"  Loss: {results['cma_es'].loss:.6f}")
print()

# --- Hybrid ---
print("Running Hybrid (CMA-ES + Gradient)...")
t0 = time.time()
hybrid_opt = HybridOptimizer(
    popsize=1000,
    top_k=5,
    bounds=BOUNDS,
    gradient_steps=100,
    gradient_lr=0.005,
    seed=42,
)
results["hybrid"] = hybrid_opt.run(loss_fn, INITIAL_PARAMS, n_generations=30)
hybrid_time = time.time() - t0
print(f"  Time: {hybrid_time:.2f}s")
print(f"  Loss: {results['hybrid'].loss:.6f}")
print()

# =============================================================================
# 10. SUMMARY
# =============================================================================

print("=" * 60)
print("Summary")
print("=" * 60)

# Only show optimized parameters (not fixed ones like t_ref)
param_names = list(BOUNDS.keys())
header = f"{'Method':<12} {'Loss':>10}"
for p in param_names:
    header += f" {p[:8]:>10}"
header += f" {'Time':>8}"
print(header)
print("-" * len(header))

# True values
row = f"{'True':<12} {0.0:>10.6f}"
for p in param_names:
    row += f" {TRUE_PARAMS[p]:>10.4g}"
row += f" {'-':>8}"
print(row)

# Results
times = {"gradient": grad_time, "cma_es": evo_time, "hybrid": hybrid_time}
for method, result in results.items():
    row = f"{method:<12} {result.loss:>10.6f}"
    for p in param_names:
        row += f" {float(result.params[p]):>10.4g}"
    row += f" {times[method]:>7.1f}s"
    print(row)

print(f"\nFixed parameters: t_ref = {FIXED_PARAMS['t_ref']}")

# =============================================================================
# 11. VISUALIZATION
# =============================================================================

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

# --- Plot 2: Biomass trajectories ---
ax2 = axes[1]
timesteps = np.arange(n_timesteps)
ax2.plot(timesteps, true_biomass, "k-", label="True", linewidth=2, alpha=0.7)
ax2.scatter(obs_indices, observations, c="red", s=20, zorder=5, label="Observations")

for method, result in results.items():
    params_float = {k: float(v) for k, v in result.params.items()}
    pred_biomass = run_simulation_with_params(params_float)
    ax2.plot(timesteps, pred_biomass, "--", label=method, alpha=0.7)

ax2.set_xlabel("Timestep")
ax2.set_ylabel("Biomass (g/m²)")
ax2.set_title("Biomass Trajectories")
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- Plot 3: Parameter recovery ---
ax3 = axes[2]
x = np.arange(len(param_names))
width = 0.2

# Normalize by true values for comparison
true_vals = np.array([TRUE_PARAMS[p] for p in param_names])

ax3.bar(x - 1.5 * width, np.ones(len(param_names)), width, label="True", color="black", alpha=0.3)

for i, (method, result) in enumerate(results.items()):
    pred_vals = np.array([float(result.params[p]) for p in param_names])
    ratios = pred_vals / true_vals
    ax3.bar(x + (i - 0.5) * width, ratios, width, label=method, alpha=0.7)

ax3.set_xticks(x)
ax3.set_xticklabels([p[:8] for p in param_names], rotation=45, ha="right")
ax3.set_ylabel("Ratio to True")
ax3.set_title("Parameter Recovery")
ax3.axhline(y=1, color="k", linestyle="--", alpha=0.5)
ax3.legend()
ax3.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("optimization_comparison_lmtl.png", dpi=150)
print("\nPlot saved to optimization_comparison_lmtl.png")
