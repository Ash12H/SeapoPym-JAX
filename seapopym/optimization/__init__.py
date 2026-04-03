"""Optimization module for model calibration.

**Optimizer classes** (1 algorithm = 1 class):
- :class:`GradientOptimizer` — Optax (adam/sgd/rmsprop/adagrad)
- :class:`CMAESOptimizer` — evosax CMA-ES (requires evosax)
- :class:`GAOptimizer` — evosax SimpleGA (requires evosax)
- :class:`IPOPCMAESOptimizer` — IPOP multi-restart CMA-ES (requires evosax)

**Building blocks**:
- :class:`Objective` / :class:`PreparedObjective` — observation data + extraction
- Loss functions: :func:`mse`, :func:`rmse`, :func:`nrmse`
- Prior distributions: :class:`Uniform`, :class:`Normal`, etc.
"""

from __future__ import annotations

from seapopym.optimization._common import GenerationResult, HallOfFame
from seapopym.optimization.gradient_optimizer import GradientOptimizer, OptimizeResult
from seapopym.optimization.loss import mse, nrmse, rmse
from seapopym.optimization.multistart_gradient import MultiStartGradientOptimizer, MultiStartResult
from seapopym.optimization.objective import Objective, PreparedObjective
from seapopym.optimization.prior import (
    HalfNormal,
    LogNormal,
    Normal,
    PriorSet,
    TruncatedNormal,
    Uniform,
)

__all__ = [
    # Optimizers (always available)
    "GradientOptimizer",
    "MultiStartGradientOptimizer",
    "OptimizeResult",
    "MultiStartResult",
    # Step-based results
    "GenerationResult",
    "HallOfFame",
    # Objectives
    "Objective",
    "PreparedObjective",
    # Loss functions
    "rmse",
    "nrmse",
    "mse",
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
    from seapopym.optimization.cmaes import CMAESOptimizer
    from seapopym.optimization.ga import GAOptimizer
    from seapopym.optimization.ipop import IPOPCMAESOptimizer, IPOPResult

    __all__ += ["CMAESOptimizer", "GAOptimizer", "IPOPCMAESOptimizer", "IPOPResult"]
    _HAS_EVOSAX = True
except (ImportError, KeyError):
    _HAS_EVOSAX = False


def __getattr__(name: str):
    """Provide helpful error message for optional dependencies."""
    if name in ("CMAESOptimizer", "GAOptimizer", "IPOPCMAESOptimizer", "IPOPResult") and not _HAS_EVOSAX:
        raise ImportError(f"{name} requires the evosax package. Install it with: pip install seapopym[optimization]")
    raise AttributeError(f"module 'seapopym.optimization' has no attribute '{name}'")
