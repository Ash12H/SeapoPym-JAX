"""Shared helpers for optimization classes.

Provides loss-building utilities, metric resolution, and parameter space
transformations used by CMAESOptimizer, GAOptimizer, IPOPCMAESOptimizer,
and GradientOptimizer.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import jax.numpy as jnp

from seapopym.optimization.loss import mse, nrmse, rmse
from seapopym.optimization.objective import Objective, PreparedObjective
from seapopym.optimization.prior import PriorSet, Uniform
from seapopym.types import Array, Params

if TYPE_CHECKING:
    from seapopym.compiler.model import CompiledModel
    from seapopym.engine.runner import Runner

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
    runner: Runner,
    model: CompiledModel,
    prepared_objectives: list[tuple[PreparedObjective, Callable, float]],
    priors: PriorSet | None,
) -> Callable[[Params], Array]:
    """Build composite loss: sum(w_i * metric_i) + prior_penalty."""

    def loss_fn(free_params: Params) -> Array:
        outputs = runner(model, free_params)

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
