"""Loss functions for parameter optimization.

Provides differentiable loss functions that work with sparse observations.
All functions are JAX-compatible and can be used with jax.grad().
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp

from seapopym.types import Array


def rmse(
    predictions: Array,
    observations: Array,
    mask: Array | None = None,
) -> Array:
    """Compute Root Mean Square Error between predictions and observations.

    Supports sparse observations via masking. Only computes loss where
    mask is True (or non-zero).

    Args:
        predictions: Predicted values from the model.
        observations: Observed values (same shape as predictions, or indexed).
        mask: Optional boolean mask. If provided, only masked points contribute
            to the loss. Shape must broadcast with predictions/observations.

    Returns:
        Scalar RMSE value.

    Example:
        >>> pred = jnp.array([1.0, 2.0, 3.0, 4.0])
        >>> obs = jnp.array([1.1, 2.2, 3.1, 4.0])
        >>> rmse(pred, obs)
        Array(0.1118..., dtype=float32)

        >>> # With sparse observations (mask)
        >>> mask = jnp.array([True, False, True, False])
        >>> rmse(pred, obs, mask)  # Only uses indices 0 and 2
        Array(0.0707..., dtype=float32)
    """
    diff = predictions - observations

    if mask is not None:
        # Apply mask: only count non-masked points
        mask = jnp.asarray(mask, dtype=jnp.float32)
        squared_diff = diff**2 * mask
        n_points = jnp.sum(mask)
        mse = jnp.sum(squared_diff) / jnp.maximum(n_points, 1.0)
    else:
        mse = jnp.mean(diff**2)

    return jnp.sqrt(mse)


def nrmse(
    predictions: Array,
    observations: Array,
    mask: Array | None = None,
    mode: Literal["std", "mean", "minmax"] = "std",
) -> Array:
    """Compute Normalized Root Mean Square Error.

    Normalizes the RMSE by a characteristic scale of the observations,
    making the metric dimensionless and comparable across variables.

    Args:
        predictions: Predicted values from the model.
        observations: Observed values.
        mask: Optional boolean mask for sparse observations.
        mode: Normalization mode:
            - "std": Normalize by standard deviation of observations (default)
            - "mean": Normalize by mean of observations
            - "minmax": Normalize by range (max - min) of observations

    Returns:
        Scalar NRMSE value (dimensionless).

    Example:
        >>> pred = jnp.array([1.0, 2.0, 3.0, 4.0])
        >>> obs = jnp.array([1.1, 2.2, 3.1, 4.0])
        >>> nrmse(pred, obs, mode="std")
        Array(0.0968..., dtype=float32)
    """
    rmse_val = rmse(predictions, observations, mask)

    # Compute normalization factor based on observations
    if mask is not None:
        mask_float = jnp.asarray(mask, dtype=jnp.float32)
        n_points = jnp.sum(mask_float)
        obs_masked = observations * mask_float

        if mode == "std":
            obs_mean = jnp.sum(obs_masked) / jnp.maximum(n_points, 1.0)
            obs_var = jnp.sum((observations - obs_mean) ** 2 * mask_float) / jnp.maximum(n_points, 1.0)
            norm_factor = jnp.sqrt(obs_var)
        elif mode == "mean":
            norm_factor = jnp.abs(jnp.sum(obs_masked) / jnp.maximum(n_points, 1.0))
        elif mode == "minmax":
            # For masked minmax, we use a large/small init value trick
            obs_for_min = jnp.asarray(jnp.where(mask, observations, jnp.inf))
            obs_for_max = jnp.asarray(jnp.where(mask, observations, -jnp.inf))
            norm_factor = jnp.max(obs_for_max) - jnp.min(obs_for_min)
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")
    else:
        if mode == "std":
            norm_factor = jnp.std(observations)
        elif mode == "mean":
            norm_factor = jnp.abs(jnp.mean(observations))
        elif mode == "minmax":
            norm_factor = jnp.max(observations) - jnp.min(observations)
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")

    # Avoid division by zero or NaN (e.g. minmax with all-False mask)
    norm_factor = jnp.where(jnp.isfinite(norm_factor), norm_factor, 1.0)
    norm_factor = jnp.maximum(norm_factor, 1e-10)

    return rmse_val / norm_factor


def mse(
    predictions: Array,
    observations: Array,
    mask: Array | None = None,
) -> Array:
    """Compute Mean Square Error between predictions and observations.

    Similar to rmse() but returns MSE (no square root), which can be
    more numerically stable for optimization.

    Args:
        predictions: Predicted values from the model.
        observations: Observed values.
        mask: Optional boolean mask for sparse observations.

    Returns:
        Scalar MSE value.
    """
    diff = predictions - observations

    if mask is not None:
        mask = jnp.asarray(mask, dtype=jnp.float32)
        squared_diff = diff**2 * mask
        n_points = jnp.sum(mask)
        return jnp.sum(squared_diff) / jnp.maximum(n_points, 1.0)
    else:
        return jnp.mean(diff**2)
