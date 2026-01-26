"""Simple 0D predator-prey model (box model).

Classic Lotka-Volterra dynamics:
- Prey: dN/dt = r*N - a*N*P
- Predator: dP/dt = d*N*P - m*P

No spatial dimension - single box model.
"""

# %%
# Imports
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr
from scipy.signal import find_peaks

from seapopym.blueprint import Blueprint, Config, functional
from seapopym.compiler import compile_model
from seapopym.engine import StreamingRunner

# %%
# Register biological functions


@functional(name="demo:prey_growth", backend="numpy")
def prey_growth_numpy(prey, growth_rate, predator, attack_rate):
    """Prey population change: growth - predation."""
    growth = growth_rate * prey
    predation = attack_rate * prey * predator
    return growth - predation


@functional(name="demo:prey_growth", backend="jax")
def prey_growth_jax(prey, growth_rate, predator, attack_rate):
    """Prey population change: growth - predation (JAX version)."""
    growth = growth_rate * prey
    predation = attack_rate * prey * predator
    return growth - predation


@functional(name="demo:predator_dynamics", backend="numpy")
def predator_dynamics_numpy(prey, predator, conversion_rate, mortality_rate):
    """Predator population change: conversion - mortality."""
    conversion = conversion_rate * prey * predator
    mortality = mortality_rate * predator
    return conversion - mortality


@functional(name="demo:predator_dynamics", backend="jax")
def predator_dynamics_jax(prey, predator, conversion_rate, mortality_rate):
    """Predator population change: conversion - mortality (JAX version)."""
    conversion = conversion_rate * prey * predator
    mortality = mortality_rate * predator
    return conversion - mortality


# %%
# Define the Blueprint (0D - no spatial dimensions)
blueprint = Blueprint.from_dict(
    {
        "id": "lotka-volterra-0d",
        "version": "1.0.0",
        "declarations": {
            "state": {
                "prey": {"units": "individuals"},
                "predator": {"units": "individuals"},
            },
            "parameters": {
                "prey_growth_rate": {"units": "1/d"},
                "attack_rate": {"units": "1/(ind*d)"},
                "conversion_rate": {"units": "1/(ind*d)"},
                "predator_mortality": {"units": "1/d"},
            },
            "forcings": {
                "time_index": {"dims": ["T"]},  # Just to define time dimension
            },
        },
        "process": [
            {
                "func": "demo:prey_growth",
                "inputs": {
                    "prey": "state.prey",
                    "growth_rate": "parameters.prey_growth_rate",
                    "predator": "state.predator",
                    "attack_rate": "parameters.attack_rate",
                },
                "outputs": {
                    "tendency": {"target": "tendencies.prey", "type": "tendency"},
                },
            },
            {
                "func": "demo:predator_dynamics",
                "inputs": {
                    "prey": "state.prey",
                    "predator": "state.predator",
                    "conversion_rate": "parameters.conversion_rate",
                    "mortality_rate": "parameters.predator_mortality",
                },
                "outputs": {
                    "tendency": {"target": "tendencies.predator", "type": "tendency"},
                },
            },
        ],
    }
)

print("=" * 60)
print("LOTKA-VOLTERRA 0D BOX MODEL")
print("=" * 60)
print(f"Blueprint: {blueprint.id} v{blueprint.version}")
print(f"State variables: {list(blueprint.declarations.state.keys())}")
print(f"Processes: {len(blueprint.process)}")
print()

# %%
# Configure the model
# Scenario: Sardine (Prey) vs Tuna (Predator)
# Scale: 2 years simulation (shorter for demo, but stable)
n_days = 30 * 12 * 50
n_timesteps = int(n_days)

config = Config.from_dict(
    {
        "parameters": {
            # Sardine-like dynamics
            "prey_growth_rate": {"value": 0.05 / 86400},  # 5% daily growth (fast reproduction)
            # Interaction
            "attack_rate": {"value": 0.01 / 86400},  # Probability of encounter leading to predation
            "conversion_rate": {"value": 0.001 / 86400},  # 10% efficiency (0.005 * 0.1)
            # Tuna-like dynamics
            "predator_mortality": {"value": 0.01 / 86400},  # 1% daily mortality
        },
        "forcings": {
            "time_index": xr.DataArray(
                np.arange(n_timesteps),
                dims=["T"],
            ),
        },
        "initial_state": {
            "prey": xr.DataArray(22.0),  # Slightly off-equilibrium (20.0) to start oscillations
            "predator": xr.DataArray(10.0),  # Equilibrium: P* = r/a = 0.05/0.005 = 10
        },
        "execution": {
            "dt": "0.2d",  # 0.02 day timestep (small dt to avoid negative population overshoot)
        },
    }
)

print("Configuration (Sardine/Tuna Scenario):")
print(f"  Initial prey: {config.initial_state['prey'].values:.1f} (rel. biomass)")
print(f"  Initial predator: {config.initial_state['predator'].values:.1f} (rel. biomass)")
print(f"  Timesteps: {n_timesteps} (dt = {config.execution.dt})")
print(f"  Duration: {n_days / 365:.1f} years ({n_days} days)")
print()

# %%
# Compile the model
print("Compiling model with NumPy backend...")
compiled = compile_model(blueprint, config, backend="jax")

print(f"  Backend: {compiled.backend}")
print(f"  dt (seconds): {compiled.dt}")
print(f"  n_timesteps: {compiled.n_timesteps}")
print(f"  State 'prey' shape: {compiled.state['prey'].shape}")
print(f"  Graph nodes: {len(compiled.graph.nodes)}")
print()

# %%
# Run simulation with StreamingRunner


output_dir = Path(tempfile.mkdtemp(prefix="seapopym_0d_"))
output_path = output_dir / "output.zarr"

print("Running simulation...")
runner = StreamingRunner(compiled, chunk_size=200)
runner.run(str(output_path))
print(f"  Output: {output_path}")
print()

# %%
# Load and analyze results


store = zarr.open(str(output_path), mode="r")
prey_ts = store["prey"][:]  # type: ignore[union-attr]  # Shape: (T,) for 0D model
predator_ts = store["predator"][:]  # type: ignore[union-attr]

print("Results:")
print(f"  Shape: prey={prey_ts.shape}, predator={predator_ts.shape}")
print(f"  Prey range: {prey_ts.min():.2f} - {prey_ts.max():.2f}")
print(f"  Predator range: {predator_ts.min():.2f} - {predator_ts.max():.2f}")
print()

# %%
# Compute time in days
time_days = np.linspace(0, n_days, n_timesteps)

# Plot time series
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Prey
axes[0].plot(time_days, prey_ts, "g-", linewidth=2, label="Prey")
axes[0].set_ylabel("Prey (individuals)", fontsize=12)
axes[0].legend(loc="upper right", fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_title("Predator-Prey Dynamics (0D Box Model)", fontsize=14, fontweight="bold")

# Predator
axes[1].plot(time_days, predator_ts, "r-", linewidth=2, label="Predator")
axes[1].set_ylabel("Predator (individuals)", fontsize=12)
axes[1].legend(loc="upper right", fontsize=11)
axes[1].grid(True, alpha=0.3)

# Both on same plot
axes[2].plot(time_days, prey_ts, "g-", linewidth=2, label="Prey", alpha=0.7)
axes[2].plot(time_days, predator_ts, "r-", linewidth=2, label="Predator", alpha=0.7)
axes[2].set_xlabel("Time (days)", fontsize=12)
axes[2].set_ylabel("Population", fontsize=12)
axes[2].legend(loc="upper right", fontsize=11)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "timeseries.png", dpi=150)
print(f"Saved plot: {output_dir / 'timeseries.png'}")
plt.show()

# %%
# Phase portrait
fig, ax = plt.subplots(figsize=(10, 8))

# Color by time
colors = plt.cm.viridis(np.linspace(0, 1, len(prey_ts)))
for i in range(len(prey_ts) - 1):
    ax.plot(prey_ts[i : i + 2], predator_ts[i : i + 2], color=colors[i], linewidth=2)

# Start and end points
ax.plot(prey_ts[0], predator_ts[0], "go", markersize=15, label="Start (t=0)", zorder=5)
ax.plot(prey_ts[-1], predator_ts[-1], "r*", markersize=20, label=f"End (t={n_days}d)", zorder=5)

ax.set_xlabel("Prey population", fontsize=12)
ax.set_ylabel("Predator population", fontsize=12)
ax.set_title("Phase Portrait", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=n_days))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label="Time (days)")

plt.tight_layout()
plt.savefig(output_dir / "phase_portrait.png", dpi=150)
print(f"Saved plot: {output_dir / 'phase_portrait.png'}")
plt.show()

# %%
# Compute oscillation period (approximate)


prey_peaks, _ = find_peaks(prey_ts, distance=20)
if len(prey_peaks) > 1:
    periods = np.diff(time_days[prey_peaks])
    mean_period = periods.mean()
    print(f"Oscillation period: {mean_period:.2f} ± {periods.std():.2f} days")
else:
    print("Not enough oscillations to compute period")

# %%
print()
print("=" * 60)
print("DEMO COMPLETED SUCCESSFULLY")
print("=" * 60)
print()
print("Key observations:")
print("  - Box model (0D): no spatial dimensions")
print(f"  - Classic predator-prey oscillations: {len(prey_peaks)} peaks detected")
print(f"  - Prey oscillates: {prey_ts.min():.1f} - {prey_ts.max():.1f} individuals")
print(f"  - Predator oscillates: {predator_ts.min():.1f} - {predator_ts.max():.1f} individuals")
print("  - Phase portrait shows limit cycle behavior")
print()
print(f"Output directory: {output_dir}")
