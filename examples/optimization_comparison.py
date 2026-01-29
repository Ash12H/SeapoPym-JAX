"""Comparison of optimization methods: Gradient vs CMA-ES vs Hybrid.

This script compares three optimization approaches on a simple 2-parameter
growth/mortality model:

1. Gradient-based (Adam with bounds scaling)
2. Evolutionary (CMA-ES)
3. Hybrid (CMA-ES exploration + gradient refinement)

It also visualizes the loss landscape and optimization trajectories.
"""

import time

import jax
import jax.numpy as jnp
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from seapopym.optimization import EvolutionaryOptimizer, HybridOptimizer, Optimizer

# =============================================================================
# 1. DEFINE THE OPTIMIZATION PROBLEM
# =============================================================================

# True parameters (to be recovered)
TRUE_GROWTH = 0.1  # per day
TRUE_MORTALITY = 0.05  # per day

# Simulation parameters
N_TIMESTEPS = 200
DT = 0.25  # days


def simulate_model_numpy(growth: float, mortality: float) -> np.ndarray:
    """Run the growth/mortality model using NumPy (for visualization)."""
    biomass = 1.0
    trajectory = [biomass]
    for _ in range(N_TIMESTEPS):
        # Logistic growth with mortality
        tendency = growth * biomass * (1 - biomass / 100) - mortality * biomass
        biomass = max(0.0, biomass + tendency * DT)
        trajectory.append(biomass)
    return np.array(trajectory)


def simulate_model_jax(growth, mortality):
    """Run the growth/mortality model using JAX (for optimization)."""

    def step(biomass, _):
        tendency = growth * biomass * (1 - biomass / 100) - mortality * biomass
        new_biomass = jnp.maximum(0.0, biomass + tendency * DT)
        return new_biomass, new_biomass

    init_biomass = jnp.array(1.0)
    _, trajectory = jax.lax.scan(step, init_biomass, None, length=N_TIMESTEPS)
    return trajectory


# Generate synthetic observations
true_trajectory = simulate_model_numpy(TRUE_GROWTH, TRUE_MORTALITY)
obs_indices_np = np.arange(0, N_TIMESTEPS, 20)  # Every 20 timesteps
obs_values = true_trajectory[obs_indices_np]
# Add noise
np.random.seed(42)
obs_values_noisy = obs_values + 0.02 * obs_values * np.random.randn(len(obs_values))
obs_values_noisy_jax = jnp.array(obs_values_noisy)
obs_indices_jax = jnp.array(obs_indices_np)


def loss_fn(params: dict):
    """Loss function: MSE between model and observations (JAX-compatible)."""
    growth = params["growth"]
    mortality = params["mortality"]
    trajectory = simulate_model_jax(growth, mortality)
    pred = trajectory[obs_indices_jax]
    return jnp.mean((pred - obs_values_noisy_jax) ** 2)


# Parameter bounds
BOUNDS = {
    "growth": (0.01, 0.3),
    "mortality": (0.01, 0.3),
}

# Initial guess (intentionally far from true)
INITIAL_PARAMS = {
    "growth": jnp.array(0.2),
    "mortality": jnp.array(0.02),
}

# =============================================================================
# 2. RUN OPTIMIZATIONS
# =============================================================================

print("=" * 60)
print("Optimization Comparison: Gradient vs CMA-ES vs Hybrid")
print("=" * 60)
print(f"True parameters: growth={TRUE_GROWTH}, mortality={TRUE_MORTALITY}")
print(f"Initial guess: growth={float(INITIAL_PARAMS['growth'])}, mortality={float(INITIAL_PARAMS['mortality'])}")
print(f"Initial loss: {loss_fn(INITIAL_PARAMS):.4f}")
print()

results = {}

# --- Gradient-based ---
print("Running Gradient (Adam)...")
t0 = time.time()
grad_opt = Optimizer(
    algorithm="adam",
    learning_rate=0.05,
    bounds=BOUNDS,
    scaling="bounds",
)
results["gradient"] = grad_opt.run(loss_fn, INITIAL_PARAMS, n_steps=200)
grad_time = time.time() - t0
print(f"  Time: {grad_time:.2f}s")
print(f"  Loss: {results['gradient'].loss:.6f}")
print(
    f"  Params: growth={float(results['gradient'].params['growth']):.4f}, "
    f"mortality={float(results['gradient'].params['mortality']):.4f}"
)
print()

# --- CMA-ES ---
print("Running CMA-ES...")
t0 = time.time()
evo_opt = EvolutionaryOptimizer(
    strategy="cma_es",
    popsize=32,
    bounds=BOUNDS,
    seed=42,
)
results["cma_es"] = evo_opt.run(loss_fn, INITIAL_PARAMS, n_generations=100)
cma_time = time.time() - t0
print(f"  Time: {cma_time:.2f}s")
print(f"  Loss: {results['cma_es'].loss:.6f}")
print(
    f"  Params: growth={float(results['cma_es'].params['growth']):.4f}, "
    f"mortality={float(results['cma_es'].params['mortality']):.4f}"
)
print()

# --- Hybrid ---
print("Running Hybrid (CMA-ES + Gradient)...")
t0 = time.time()
hybrid_opt = HybridOptimizer(
    popsize=32,
    top_k=5,
    bounds=BOUNDS,
    gradient_steps=50,
    gradient_lr=0.05,
    seed=42,
)
results["hybrid"] = hybrid_opt.run(loss_fn, INITIAL_PARAMS, n_generations=50)
hybrid_time = time.time() - t0
print(f"  Time: {hybrid_time:.2f}s")
print(f"  Loss: {results['hybrid'].loss:.6f}")
print(
    f"  Params: growth={float(results['hybrid'].params['growth']):.4f}, "
    f"mortality={float(results['hybrid'].params['mortality']):.4f}"
)
print()

# =============================================================================
# 3. SUMMARY
# =============================================================================

print("=" * 60)
print("Summary")
print("=" * 60)
print(f"{'Method':<15} {'Loss':>12} {'Growth':>10} {'Mortality':>10} {'Time':>8}")
print("-" * 60)
print(f"{'True':<15} {0.0:>12.6f} {TRUE_GROWTH:>10.4f} {TRUE_MORTALITY:>10.4f} {'-':>8}")
for name, result in results.items():
    t = {"gradient": grad_time, "cma_es": cma_time, "hybrid": hybrid_time}[name]
    print(
        f"{name:<15} {result.loss:>12.6f} "
        f"{float(result.params['growth']):>10.4f} "
        f"{float(result.params['mortality']):>10.4f} "
        f"{t:>7.2f}s"
    )
print()

# =============================================================================
# 4. VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# --- Plot 1: Loss landscape with trajectories ---
ax1 = axes[0]

# Create loss landscape grid
growth_range = np.linspace(0.01, 0.3, 50)
mortality_range = np.linspace(0.01, 0.3, 50)
G, M = np.meshgrid(growth_range, mortality_range)
Z = np.zeros_like(G)

for i in range(G.shape[0]):
    for j in range(G.shape[1]):
        Z[i, j] = loss_fn({"growth": jnp.array(G[i, j]), "mortality": jnp.array(M[i, j])})

# Plot contours
levels = np.logspace(-1, 3, 20)
contour = ax1.contourf(G, M, Z, levels=levels, cmap="viridis", norm=matplotlib.colors.LogNorm())
ax1.contour(G, M, Z, levels=levels, colors="white", alpha=0.3, linewidths=0.5)
plt.colorbar(contour, ax=ax1, label="Loss (log scale)")

# Mark true and initial
ax1.scatter([TRUE_GROWTH], [TRUE_MORTALITY], c="red", s=100, marker="*", label="True", zorder=5)
ax1.scatter(
    [float(INITIAL_PARAMS["growth"])],
    [float(INITIAL_PARAMS["mortality"])],
    c="white",
    s=100,
    marker="o",
    edgecolors="black",
    label="Initial",
    zorder=5,
)

# Mark final positions
colors = {"gradient": "cyan", "cma_es": "yellow", "hybrid": "magenta"}
for name, result in results.items():
    ax1.scatter(
        [float(result.params["growth"])],
        [float(result.params["mortality"])],
        c=colors[name],
        s=80,
        marker="s",
        edgecolors="black",
        label=name,
        zorder=5,
    )

ax1.set_xlabel("Growth rate (/day)")
ax1.set_ylabel("Mortality rate (/day)")
ax1.set_title("Loss Landscape")
ax1.legend(loc="upper right")

# --- Plot 2: Loss convergence ---
ax2 = axes[1]
for name, result in results.items():
    ax2.semilogy(result.loss_history, label=name, color=colors[name])
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Loss (log scale)")
ax2.set_title("Convergence")
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- Plot 3: Final trajectories ---
ax3 = axes[2]
time_days = np.arange(N_TIMESTEPS + 1) * DT

# True trajectory
ax3.plot(time_days, true_trajectory, "k-", linewidth=2, label="True")

# Observations
ax3.scatter(obs_indices_np * DT, obs_values_noisy, c="red", s=40, zorder=5, label="Observations")

# Optimized trajectories
for name, result in results.items():
    traj = simulate_model_numpy(float(result.params["growth"]), float(result.params["mortality"]))
    ax3.plot(time_days, traj, "--", color=colors[name], linewidth=1.5, label=name)

ax3.set_xlabel("Time (days)")
ax3.set_ylabel("Biomass")
ax3.set_title("Model Trajectories")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("optimization_comparison.png", dpi=150)
print("Plot saved to optimization_comparison.png")
