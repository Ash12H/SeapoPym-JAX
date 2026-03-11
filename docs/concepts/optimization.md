# Optimization

SeapoPym provides built-in parameter calibration through four optimization algorithms. Because the simulation is end-to-end JAX, gradient-based methods can differentiate through the entire model, while evolutionary strategies offer gradient-free alternatives.

## Overview

| Optimizer | Algorithm | Gradients | Library | Best For |
|-----------|-----------|-----------|---------|----------|
| `GradientOptimizer` | Adam, SGD, RMSProp | Yes (autodiff) | Optax | Smooth, local optimization |
| `CMAESOptimizer` | CMA-ES | No | evosax | Noisy, non-smooth landscapes |
| `GAOptimizer` | Genetic Algorithm | No | evosax | Discontinuous landscapes |
| `IPOPCMAESOptimizer` | IPOP-CMA-ES | No | evosax | Multimodal (finding multiple solutions) |

## Quick Example

```python
from seapopym.optimization import GAOptimizer, Objective
import xarray as xr

# Define what to compare against
objective = Objective(
    observations=observed_biomass,  # xr.DataArray or pd.DataFrame
    target="biomass",               # extract this variable from model outputs
)

# Set parameter bounds
bounds = {
    "lambda_0": (1e-8, 1e-5),
    "gamma_lambda": (0.01, 0.5),
    "tau_r_0": (1e6, 1e8),
}

# Create and run optimizer
optimizer = GAOptimizer(
    objectives=[(objective, "rmse", 1.0)],
    bounds=bounds,
    popsize=64,
)
result = optimizer.run(model, n_generations=200, patience=50)

print(f"Best loss: {result.loss:.6f}")
print(f"Converged: {result.converged} ({result.n_iterations} iterations)")
print(f"Best parameters: {result.params}")
```

## Objectives

An `Objective` defines what model outputs to compare against observations.

### Target Mode (Auto-Extraction)

Extract a variable by name from model outputs:

```python
objective = Objective(
    observations=obs_data,   # xr.DataArray with matching coords
    target="biomass",        # variable name in model outputs
)
```

The optimizer automatically handles coordinate alignment — observations don't need to cover the full model grid.

### Transform Mode (Custom Extraction)

Define a custom function to derive predictions from outputs:

```python
def extract_surface_biomass(outputs):
    """Sum biomass over functional groups, select surface layer."""
    return outputs["biomass"].sum(axis=0)  # sum over F dimension

objective = Objective(
    observations=obs_data,
    transform=extract_surface_biomass,
)
```

## Loss Functions

Three built-in metrics:

| Metric | Formula | Properties |
|--------|---------|------------|
| `"rmse"` | $\sqrt{\text{mean}((y - \hat{y})^2)}$ | Same units as data |
| `"nrmse"` | $\text{RMSE} / \sigma_y$ | Dimensionless, comparable across variables |
| `"mse"` | $\text{mean}((y - \hat{y})^2)$ | More stable gradients than RMSE |

All metrics support **sparse observations** via masking — only grid points with valid observations contribute to the loss.

### Weighted Multi-Objective

Combine multiple objectives with weights:

```python
optimizer = GAOptimizer(
    objectives=[
        (biomass_objective, "rmse", 1.0),
        (production_objective, "nrmse", 0.5),
    ],
    bounds=bounds,
)
```

Total loss: $\mathcal{L} = \sum_i w_i \cdot \text{metric}_i(y_i, \hat{y}_i) + \text{prior penalty}$

## Priors

Priors add regularization and define parameter bounds for evolutionary strategies:

```python
from seapopym.optimization import Uniform, Normal, LogNormal, PriorSet

priors = PriorSet({
    "lambda_0": LogNormal(mu=-15, sigma=1.0),     # positive, log-scale
    "gamma_lambda": Normal(loc=0.1, scale=0.05),   # centered around 0.1
    "tau_r_0": Uniform(low=1e6, high=1e8),          # flat prior
})
```

| Prior | Use Case | Bounds |
|-------|----------|--------|
| `Uniform(low, high)` | Hard bounds, no preference | `[low, high]` |
| `Normal(loc, scale)` | Soft centering around expected value | From quantiles |
| `LogNormal(mu, sigma)` | Positive parameters, log-scale | From quantiles |
| `HalfNormal(scale)` | Positive parameters, zero-centered | `[0, quantile]` |
| `TruncatedNormal(loc, scale, low, high)` | Soft shape with hard bounds | `[low, high]` |

## Gradient Optimizer

Uses Optax for gradient-based optimization. The entire simulation is differentiated via `jax.value_and_grad`.

```python
from seapopym.optimization import GradientOptimizer

optimizer = GradientOptimizer(
    objectives=[(objective, "mse", 1.0)],
    bounds=bounds,
    algorithm="adam",
    learning_rate=1e-3,
    scaling="bounds",       # normalize params to [0,1]
)
result = optimizer.run(model, n_steps=500, tolerance=1e-6)
```

**Scaling options:**

| Mode | Description |
|------|-------------|
| `"none"` | Raw parameter values |
| `"bounds"` | Normalize to [0, 1] using bounds |
| `"log"` | Log-space optimization (positive params) |

## CMA-ES Optimizer

Covariance Matrix Adaptation Evolution Strategy — a population-based optimizer that learns the search distribution.

```python
from seapopym.optimization import CMAESOptimizer

optimizer = CMAESOptimizer(
    objectives=[(objective, "rmse", 1.0)],
    bounds=bounds,
    popsize=32,
    seed=42,
)
result = optimizer.run(model, n_generations=200, patience=50)
```

!!! tip "When to use CMA-ES"
    CMA-ES excels when the loss landscape is noisy or non-smooth. It requires no gradient information and adapts its search covariance to the local geometry.

## Genetic Algorithm

Simple evolutionary strategy with crossover and mutation.

```python
from seapopym.optimization import GAOptimizer

optimizer = GAOptimizer(
    objectives=[(objective, "rmse", 1.0)],
    bounds=bounds,
    popsize=64,
    seed=42,
)
result = optimizer.run(model, n_generations=250, patience=50, progress_bar=True)
```

## IPOP-CMA-ES

**Increasing Population CMA-ES** — runs multiple CMA-ES restarts with doubling population size. Designed to find multiple local optima.

```python
from seapopym.optimization import IPOPCMAESOptimizer

optimizer = IPOPCMAESOptimizer(
    objectives=[(objective, "rmse", 1.0)],
    bounds=bounds,
    initial_popsize=8,
    n_restarts=8,
    n_generations=100,
    distance_threshold=0.1,  # min distance between distinct modes
    seed=42,
)
result = optimizer.run(model)

# result.modes contains distinct solutions sorted by loss
for i, mode in enumerate(result.modes):
    print(f"Mode {i}: loss={mode.loss:.6f}")
```

## OptimizeResult

All optimizers return an `OptimizeResult`:

| Attribute | Type | Description |
|-----------|------|-------------|
| `params` | dict[str, Array] | Best parameter values found |
| `loss` | float | Final loss value |
| `loss_history` | list[float] | Loss at each iteration/generation |
| `n_iterations` | int | Number of steps performed |
| `converged` | bool | Whether convergence criterion was met |
| `message` | str | Status description |

IPOP-CMA-ES returns an `IPOPResult` with additional:

| Attribute | Type | Description |
|-----------|------|-------------|
| `modes` | list[OptimizeResult] | Distinct solutions, sorted by loss |
| `all_results` | list[OptimizeResult] | All restart results |
| `n_restarts` | int | Total restarts performed |
