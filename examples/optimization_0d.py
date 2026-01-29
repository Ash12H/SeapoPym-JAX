"""Example: Parameter Optimization on 0D LMTL Model.

This script demonstrates gradient-based parameter optimization using
the seapopym optimization module. It:

1. Creates a simple 0D LMTL model
2. Generates synthetic "observations" with known parameters
3. Attempts to recover the parameters from the observations

This serves as a proof-of-concept for the optimization workflow.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
import xarray as xr

from seapopym.blueprint import Blueprint, Config, functional
from seapopym.compiler import compile_model
from seapopym.optimization import GradientRunner, Optimizer, SparseObservations

# Initialize Pint registry
ureg = pint.get_application_registry()

# =============================================================================
# 1. DEFINE MODEL FUNCTIONS
# =============================================================================


@functional(name="opt:growth", backend="jax", units={"biomass": "g", "rate": "1/s", "return": "g/s"})
def growth_tendency(biomass, rate):
    """Simple exponential growth with saturation."""
    carrying_capacity = 100.0
    return rate * biomass * (1.0 - biomass / carrying_capacity)


@functional(name="opt:mortality", backend="jax", units={"biomass": "g", "rate": "1/s", "return": "g/s"})
def mortality_tendency(biomass, rate):
    """Simple mortality loss."""
    return -rate * biomass


# =============================================================================
# 2. BLUEPRINT AND CONFIG
# =============================================================================

blueprint = Blueprint.from_dict(
    {
        "id": "optimization-demo",
        "version": "1.0",
        "declarations": {
            "state": {"biomass": {"units": "g", "dims": []}},
            "parameters": {
                "growth_rate": {"units": "1/s"},
                "mortality_rate": {"units": "1/s"},
            },
            "forcings": {},
        },
        "process": [
            {
                "func": "opt:growth",
                "inputs": {"biomass": "state.biomass", "rate": "parameters.growth_rate"},
                "outputs": {"return": {"target": "tendencies.biomass", "type": "tendency"}},
            },
            {
                "func": "opt:mortality",
                "inputs": {"biomass": "state.biomass", "rate": "parameters.mortality_rate"},
                "outputs": {"return": {"target": "tendencies.biomass", "type": "tendency"}},
            },
        ],
    }
)

# =============================================================================
# 3. TRUE PARAMETERS (to be recovered)
# =============================================================================

TRUE_GROWTH_RATE = 0.1 / 86400.0  # 0.1 per day in per second
TRUE_MORTALITY_RATE = 0.05 / 86400.0  # 0.05 per day in per second

# =============================================================================
# 4. GENERATE SYNTHETIC OBSERVATIONS
# =============================================================================

# Simulation setup
n_days = 50
dt_hours = 6
dt_seconds = dt_hours * 3600
n_timesteps = n_days * 24 // dt_hours

# Time coordinates
time_coords = pd.date_range("2000-01-01", periods=n_timesteps, freq=f"{dt_hours}h")

# Create config with TRUE parameters
config_true = Config.from_dict(
    {
        "parameters": {
            "growth_rate": {"value": TRUE_GROWTH_RATE},
            "mortality_rate": {"value": TRUE_MORTALITY_RATE},
        },
        "forcings": {},
        "initial_state": {"biomass": xr.DataArray(1.0)},  # Start with 1g
        "execution": {
            "time_start": "2000-01-01",
            "time_end": str((pd.Timestamp("2000-01-01") + pd.Timedelta(days=n_days)).date()),
            "dt": f"{dt_hours}h",
        },
    }
)

# Compile and run to get "true" trajectory
print("Generating synthetic observations with true parameters...")
model_true = compile_model(blueprint, config_true, backend="jax")

# Run model manually to get outputs
from seapopym.engine.step import build_step_fn  # noqa: E402

step_fn = build_step_fn(model_true, params_as_argument=False)

state = model_true.state
biomass_trajectory = [float(state["biomass"])]

for _t in range(n_timesteps):
    # Empty forcings for this simple model
    forcings_t = {}
    state, _ = step_fn(state, forcings_t)
    biomass_trajectory.append(float(state["biomass"]))

biomass_trajectory = jnp.array(biomass_trajectory[:-1])  # Remove last (one extra)

# Select sparse observations (every 5 days)
obs_interval = 5 * 24 // dt_hours  # timesteps per 5 days
obs_indices = jnp.arange(0, n_timesteps, obs_interval)
obs_values = biomass_trajectory[obs_indices]

# Add small noise
noise = 0.02 * obs_values * jnp.array(np.random.randn(len(obs_values)))
obs_values_noisy = obs_values + noise

print(f"Generated {len(obs_indices)} observations at {obs_interval}-timestep intervals")

# =============================================================================
# 5. OPTIMIZATION
# =============================================================================

# Initial guess (wrong parameters)
INITIAL_GROWTH_RATE = 0.11 / 86400.0  # 50% higher than true
INITIAL_MORTALITY_RATE = 0.04 / 86400.0  # 60% lower than true

# Create config with INITIAL (wrong) parameters
config_init = Config.from_dict(
    {
        "parameters": {
            "growth_rate": {"value": INITIAL_GROWTH_RATE},
            "mortality_rate": {"value": INITIAL_MORTALITY_RATE},
        },
        "forcings": {},
        "initial_state": {"biomass": xr.DataArray(1.0)},
        "execution": {
            "time_start": "2000-01-01",
            "time_end": str((pd.Timestamp("2000-01-01") + pd.Timedelta(days=n_days)).date()),
            "dt": f"{dt_hours}h",
        },
    }
)

model_init = compile_model(blueprint, config_init, backend="jax")

# Create GradientRunner
runner = GradientRunner(model_init)

# Create sparse observations (0D model, so y=0, x=0)
observations = SparseObservations(
    variable="biomass",
    times=obs_indices,
    y=jnp.zeros_like(obs_indices),  # No spatial dimension
    x=jnp.zeros_like(obs_indices),
    values=obs_values_noisy,
)

# Create optimizer with bounds and scaling
# Note: scaling="bounds" normalizes parameters to [0,1], enabling normal learning rates
optimizer = Optimizer(
    algorithm="adam",
    learning_rate=0.001,
    bounds={
        "growth_rate": (0.01 / 86400.0, 0.5 / 86400.0),
        "mortality_rate": (0.01 / 86400.0, 0.5 / 86400.0),
    },
    scaling="bounds",
)

print("\nStarting optimization...")
print(f"True parameters: growth={TRUE_GROWTH_RATE * 86400:.4f}/day, mortality={TRUE_MORTALITY_RATE * 86400:.4f}/day")
print(
    f"Initial parameters: growth={INITIAL_GROWTH_RATE * 86400:.4f}/day, mortality={INITIAL_MORTALITY_RATE * 86400:.4f}/day"
)

# For 0D model, we need a different approach since the indexing expects spatial dims
# Let's create a custom loss function that works with 0D


# Manual loss function for 0D case
def loss_fn_0d(params):
    """Loss function for 0D model."""
    # Run model with new params
    step_fn_opt = build_step_fn(model_init, params_as_argument=True)

    # Run simulation
    state = model_init.state
    carry = (state, params)

    biomass_list = []
    for _t in range(n_timesteps):
        carry, outputs = step_fn_opt(carry, {})
        biomass_list.append(outputs["biomass"])

    pred_trajectory = jnp.stack(biomass_list)

    # Extract at observation times
    pred_at_obs = pred_trajectory[obs_indices]

    # MSE loss
    return jnp.mean((pred_at_obs - obs_values_noisy) ** 2)


# Initial parameters
initial_params = {
    "growth_rate": jnp.array(INITIAL_GROWTH_RATE),
    "mortality_rate": jnp.array(INITIAL_MORTALITY_RATE),
}

# Run optimization using optimizer.run() - scaling is handled automatically
print("\nOptimization progress:")
result = optimizer.run(
    loss_fn=loss_fn_0d,
    initial_params=initial_params,
    n_steps=100,
    verbose=True,
)

params = result.params
loss_history = result.loss_history

print(f"\nFinal loss: {result.loss:.6f}")
print(
    f"Optimized parameters: growth={params['growth_rate'] * 86400:.4f}/day, mortality={params['mortality_rate'] * 86400:.4f}/day"
)
print(f"True parameters: growth={TRUE_GROWTH_RATE * 86400:.4f}/day, mortality={TRUE_MORTALITY_RATE * 86400:.4f}/day")
print(f"Converged: {result.converged} ({result.message})")

# =============================================================================
# 6. VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Loss history
ax1 = axes[0]
ax1.semilogy(loss_history)
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Loss (MSE)")
ax1.set_title("Optimization Convergence")
ax1.grid(True, alpha=0.3)

# Plot 2: Trajectories
ax2 = axes[1]

# Run with optimized params to get trajectory
state = model_init.state
carry = (state, params)
step_fn_opt = build_step_fn(model_init, params_as_argument=True)

opt_trajectory = []
for _t in range(n_timesteps):
    carry, outputs = step_fn_opt(carry, {})
    opt_trajectory.append(float(outputs["biomass"]))

opt_trajectory = jnp.array(opt_trajectory)

time_days = jnp.arange(n_timesteps) * dt_hours / 24

ax2.plot(time_days, biomass_trajectory, "b-", label="True", linewidth=2)
ax2.plot(time_days, opt_trajectory, "g--", label="Optimized", linewidth=2)
ax2.scatter(obs_indices * dt_hours / 24, obs_values_noisy, c="red", s=50, zorder=5, label="Observations")
ax2.set_xlabel("Time (days)")
ax2.set_ylabel("Biomass (g)")
ax2.set_title("Model Trajectories")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("optimization_0d_results.png")
print("\nPlot saved to optimization_0d_results.png")
