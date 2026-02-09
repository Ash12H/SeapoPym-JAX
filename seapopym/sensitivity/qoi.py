"""Quantity of Interest (QoI) functions for sensitivity analysis.

All functions are pure JAX operations, suitable for GPU execution.
They operate on time series extracted at specific grid points.

Input shape convention: (batch_size, T, n_points)
Output shape convention: (batch_size, n_points)
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

Array = Any  # jax.Array

# Registry of available QoI functions
_QOI_REGISTRY: dict[str, Any] = {}


def _register(name: str):
    """Decorator to register a QoI function."""

    def decorator(func):
        _QOI_REGISTRY[name] = func
        return func

    return decorator


@_register("mean")
def qoi_mean(time_series: Array) -> Array:
    """Temporal mean. (batch, T, n_points) -> (batch, n_points)."""
    return jnp.mean(time_series, axis=1)


@_register("var")
def qoi_var(time_series: Array) -> Array:
    """Temporal variance. (batch, T, n_points) -> (batch, n_points)."""
    return jnp.var(time_series, axis=1)


@_register("std")
def qoi_std(time_series: Array) -> Array:
    """Temporal standard deviation. (batch, T, n_points) -> (batch, n_points)."""
    return jnp.std(time_series, axis=1)


@_register("min")
def qoi_min(time_series: Array) -> Array:
    """Temporal minimum. (batch, T, n_points) -> (batch, n_points)."""
    return jnp.min(time_series, axis=1)


@_register("max")
def qoi_max(time_series: Array) -> Array:
    """Temporal maximum. (batch, T, n_points) -> (batch, n_points)."""
    return jnp.max(time_series, axis=1)


@_register("median")
def qoi_median(time_series: Array) -> Array:
    """Temporal median. (batch, T, n_points) -> (batch, n_points)."""
    return jnp.median(time_series, axis=1)


@_register("argmax")
def qoi_argmax(time_series: Array) -> Array:
    """Index of temporal maximum (e.g., day of peak). (batch, T, n_points) -> (batch, n_points).

    Returns the time index (integer) as a float for consistency with other QoIs.
    """
    return jnp.argmax(time_series, axis=1).astype(jnp.float32)


def available_qoi() -> list[str]:
    """Return list of available QoI names."""
    return list(_QOI_REGISTRY.keys())


def compute_qoi(time_series: Array, qoi_names: list[str]) -> dict[str, Array]:
    """Compute multiple QoI from time series at extraction points.

    Args:
        time_series: Array of shape (batch_size, T, n_points).
        qoi_names: List of QoI names to compute (e.g., ["mean", "var", "argmax"]).

    Returns:
        Dict mapping QoI name to array of shape (batch_size, n_points).

    Raises:
        ValueError: If a QoI name is not recognized.
    """
    results: dict[str, Array] = {}
    for name in qoi_names:
        if name not in _QOI_REGISTRY:
            raise ValueError(f"Unknown QoI '{name}'. Available: {available_qoi()}")
        results[name] = _QOI_REGISTRY[name](time_series)
    return results
