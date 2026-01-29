"""Optimization module for parameter estimation.

This module provides tools for optimizing model parameters by minimizing
the difference between simulated outputs and observations.

Main components:
- Loss functions (RMSE, NRMSE) with support for sparse observations
- Optimizer wrapper for Optax algorithms (gradient-based)
- GradientRunner for differentiable model execution
- EvolutionaryOptimizer for CMA-ES optimization (requires evosax)
- HybridOptimizer for combined CMA-ES + gradient optimization (requires evosax)

Evolutionary optimizers require the optional evosax dependency:
    pip install seapopym[optimization]
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

# Optional imports (require evosax)
try:
    from seapopym.optimization.evolutionary import EvolutionaryOptimizer
    from seapopym.optimization.hybrid import HybridOptimizer

    __all__ += ["EvolutionaryOptimizer", "HybridOptimizer"]
    _HAS_EVOSAX = True
except ImportError:
    _HAS_EVOSAX = False


def __getattr__(name: str):
    """Provide helpful error message for optional dependencies."""
    if name in ("EvolutionaryOptimizer", "HybridOptimizer") and not _HAS_EVOSAX:
        raise ImportError(f"{name} requires the evosax package. Install it with: pip install seapopym[optimization]")
    raise AttributeError(f"module 'seapopym.optimization' has no attribute '{name}'")
