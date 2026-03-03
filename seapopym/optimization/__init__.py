"""Optimization module for model calibration.

**High-level (orchestration)**:
- :class:`Optimizer` — assembles loss from objectives, dispatches to strategy
- :class:`Objective` — observation data + extraction method

**Low-level (building blocks)**:
- Loss functions: :func:`mse`, :func:`rmse`, :func:`nrmse`
- Prior distributions: :class:`Uniform`, :class:`Normal`, etc.
- :class:`GradientOptimizer`, :class:`EvolutionaryOptimizer` (requires evosax)
"""

from __future__ import annotations

from seapopym.optimization.gradient_optimizer import GradientOptimizer, OptimizeResult
from seapopym.optimization.loss import mse, nrmse, rmse
from seapopym.optimization.objective import Objective, PreparedObjective
from seapopym.optimization.optimizer import Optimizer
from seapopym.optimization.prior import (
    HalfNormal,
    LogNormal,
    Normal,
    PriorSet,
    TruncatedNormal,
    Uniform,
)

__all__ = [
    # High-level
    "Objective",
    "PreparedObjective",
    "Optimizer",
    # Loss functions
    "rmse",
    "nrmse",
    "mse",
    # Low-level optimizers
    "GradientOptimizer",
    "OptimizeResult",
    # Priors
    "Uniform",
    "Normal",
    "LogNormal",
    "HalfNormal",
    "TruncatedNormal",
    "PriorSet",
]

# Optional imports (require evosax)
try:
    from seapopym.optimization.evolutionary import EvolutionaryOptimizer
    from seapopym.optimization.ipop import IPOPResult, run_ipop, run_ipop_cmaes

    __all__ += ["EvolutionaryOptimizer", "IPOPResult", "run_ipop", "run_ipop_cmaes"]
    _HAS_EVOSAX = True
except (ImportError, KeyError):
    _HAS_EVOSAX = False


def __getattr__(name: str):
    """Provide helpful error message for optional dependencies."""
    if (
        name in ("EvolutionaryOptimizer", "IPOPResult", "run_ipop", "run_ipop_cmaes")
        and not _HAS_EVOSAX
    ):
        raise ImportError(
            f"{name} requires the evosax package. "
            "Install it with: pip install seapopym[optimization]"
        )
    raise AttributeError(f"module 'seapopym.optimization' has no attribute '{name}'")
