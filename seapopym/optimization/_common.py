"""Shared helpers for optimization classes.

Provides loss-building utilities, metric resolution, and parameter space
transformations used by CMAESOptimizer, GAOptimizer, IPOPCMAESOptimizer,
and GradientOptimizer.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from seapopym.engine.run import run
from seapopym.engine.step import build_step_fn
from seapopym.optimization.loss import mse, nrmse, rmse
from seapopym.optimization.objective import Objective, PreparedObjective
from seapopym.optimization.prior import PriorSet, Uniform
from seapopym.types import Array, Params

if TYPE_CHECKING:
    from seapopym.compiler.model import CompiledModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric resolution
# ---------------------------------------------------------------------------

METRICS: dict[str, Callable[[Array, Array], Array]] = {
    "mse": mse,
    "rmse": rmse,
    "nrmse": lambda p, o: nrmse(p, o, mode="std"),
    "nrmse_std": lambda p, o: nrmse(p, o, mode="std"),
    "nrmse_mean": lambda p, o: nrmse(p, o, mode="mean"),
    "nrmse_minmax": lambda p, o: nrmse(p, o, mode="minmax"),
}


def resolve_metric(metric: str | Callable[[Array, Array], Array]) -> Callable[[Array, Array], Array]:
    """Resolve a metric name to a callable."""
    if callable(metric):
        return metric
    if metric in METRICS:
        return METRICS[metric]
    available = sorted(METRICS.keys())
    msg = f"Unknown metric '{metric}'. Available: {available}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Prior / objective helpers
# ---------------------------------------------------------------------------


def build_default_priors(bounds: dict[str, tuple[float, float]]) -> PriorSet:
    """Build a PriorSet of Uniform priors from bounds."""
    return PriorSet({name: Uniform(low, high) for name, (low, high) in bounds.items()})


def setup_objectives(
    objectives: list[tuple[Objective, str | Callable, float]],
    model_coords: dict[str, Array],
) -> list[tuple[PreparedObjective, Callable, float]]:
    """Prepare objectives for evaluation."""
    prepared: list[tuple[PreparedObjective, Callable, float]] = []
    for obj, metric, weight in objectives:
        p = obj.setup(model_coords)
        prepared.append((p, resolve_metric(metric), weight))
    return prepared


def build_loss_fn(
    model: CompiledModel,
    prepared_objectives: list[tuple[PreparedObjective, Callable, float]],
    priors: PriorSet | None,
    export_variables: list[str] | None = None,
    chunk_size: int | None = None,
) -> Callable[[Params], Array]:
    """Build composite loss: sum(w_i * metric_i) + prior_penalty."""
    step_fn = build_step_fn(model, export_variables=export_variables)

    def loss_fn(free_params: Params) -> Array:
        merged = {**model.parameters, **free_params}
        state = dict(model.state)
        _, outputs = run(step_fn, model, state, merged, chunk_size=chunk_size)

        total = jnp.array(0.0)
        for p, metric_fn, weight in prepared_objectives:
            pred = p.extract_fn(outputs)
            total = total + weight * metric_fn(pred, p.obs_array)

        if priors is not None:
            penalty = -priors.log_prob(free_params)
            total = total + penalty
        return total

    return loss_fn


# ---------------------------------------------------------------------------
# Parameter-space utilities
# ---------------------------------------------------------------------------


def flatten_params(params: Params) -> tuple[list[str], Array]:
    """Flatten parameter dict to sorted keys + 1D array."""
    keys = sorted(params.keys())
    values = [jnp.atleast_1d(params[k]).flatten() for k in keys]
    flat = jnp.concatenate(values) if values else jnp.array([])
    return keys, flat


def unflatten_params(
    keys: list[str],
    flat: Array,
    shapes: dict[str, tuple],
    original_params: Params | None = None,
) -> Params:
    """Reconstruct parameter dict from flat array."""
    params: Params = {}
    idx = 0
    for k in keys:
        shape = shapes[k]
        size = math.prod(shape) if shape else 1
        values = flat[idx : idx + size]

        if original_params is not None and jnp.ndim(original_params[k]) == 0:
            params[k] = values[0]
        elif shape:
            params[k] = values.reshape(shape)
        else:
            params[k] = values[0]
        idx += size
    return params


def build_bounds_arrays(
    keys: list[str],
    params: Params,
    bounds: dict[str, tuple[float, float]],
) -> tuple[Array, Array]:
    """Build lower/upper bound arrays matching the flattened parameter vector."""
    lowers: list[float] = []
    uppers: list[float] = []
    for k in keys:
        size = jnp.atleast_1d(params[k]).size
        if k in bounds:
            low, high = bounds[k]
            if low >= high:
                msg = f"Invalid bounds for '{k}': low ({low}) >= high ({high})"
                raise ValueError(msg)
            lowers.extend([low] * size)
            uppers.extend([high] * size)
        else:
            lowers.extend([0.0] * size)
            uppers.extend([1.0] * size)
    return jnp.array(lowers), jnp.array(uppers)


def normalize(flat: Array, lower: Array, upper: Array) -> Array:
    """Transform from original space to [0,1]."""
    return (flat - lower) / (upper - lower)


def denormalize(flat_norm: Array, lower: Array, upper: Array) -> Array:
    """Transform from [0,1] back to original space."""
    return flat_norm * (upper - lower) + lower


# ---------------------------------------------------------------------------
# Shared evolution-strategy loop
# ---------------------------------------------------------------------------


def run_evolution_strategy(
    strategy,
    es_params,
    state,
    eval_population: Callable[[Array], Array],
    n_generations: int,
    tol_fun: float,
    patience: int,
    key: Array,
    norm_lower: Array,
    norm_upper: Array,
    x0_norm: Array,
    progress_bar: bool = False,
) -> tuple[Array, list[float], bool]:
    """Shared ask/tell loop for evolutionary strategies.

    Returns ``(best_flat_norm, loss_history, converged)``.
    """
    loss_history: list[float] = []
    best_loss = float("inf")
    best_flat_norm = x0_norm
    stall_count = 0
    converged = False

    for gen in range(n_generations):
        key, ask_key, tell_key = jax.random.split(key, 3)

        population, state = strategy.ask(ask_key, state, es_params)
        population = jnp.clip(population, norm_lower, norm_upper)
        fitness = eval_population(population)
        state, _metrics = strategy.tell(tell_key, population, fitness, state, es_params)

        min_fitness = float(jnp.min(fitness))
        loss_history.append(min_fitness)

        if min_fitness < best_loss - tol_fun:
            best_loss = min_fitness
            best_idx = jnp.argmin(fitness)
            best_flat_norm = population[best_idx]
            stall_count = 0
        else:
            stall_count += 1

        if stall_count >= patience:
            converged = True
            logger.info(
                "Converged at generation %d: no improvement over %d generations (best_loss=%.6e)",
                gen,
                patience,
                best_loss,
            )
            break

        if gen % 50 == 0:
            logger.info(
                "gen %d/%d: best_loss=%.6e, stall=%d/%d",
                gen,
                n_generations,
                best_loss,
                stall_count,
                patience,
            )

        if progress_bar:
            print_rate = max(1, n_generations // 20)
            if gen % print_rate == 0 or gen == n_generations - 1:
                print(f"\r  [{gen + 1}/{n_generations}] loss={best_loss:.4e}", end="", flush=True)

    if progress_bar:
        print()

    return best_flat_norm, loss_history, converged
