"""Optimization module for parameter estimation with gradient-based methods.

This module provides tools for optimizing model parameters by minimizing
the difference between simulated outputs and observations.

Main components:
- Loss functions (RMSE, NRMSE) with support for sparse observations
- Optimizer wrapper for Optax algorithms
- GradientRunner for differentiable model execution
"""

from __future__ import annotations

from seapopym.optimization.gradient import GradientRunner, SparseObservations
from seapopym.optimization.loss import mse, nrmse, rmse
from seapopym.optimization.optimizer import Optimizer, OptimizeResult

__all__ = [
    "rmse",
    "nrmse",
    "mse",
    "Optimizer",
    "OptimizeResult",
    "GradientRunner",
    "SparseObservations",
]
