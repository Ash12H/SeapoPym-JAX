"""Optimization module for model calibration.

Provides two levels of API:

**High-level (orchestration)**:
- :class:`Optimizer` — assembles loss from objectives, dispatches to strategy
- :class:`Sampler` — builds log-posterior, runs NUTS (requires blackjax)
- :class:`Objective` — observation data + extraction method
- :class:`CalibrationRunner` — model execution for calibration

**Low-level (building blocks)**:
- Loss functions: :func:`mse`, :func:`rmse`, :func:`nrmse`
- Prior distributions: :class:`Uniform`, :class:`Normal`, etc.
- :class:`GradientOptimizer`, :class:`EvolutionaryOptimizer` (requires evosax)
- :func:`run_nuts` (requires blackjax)
"""

from __future__ import annotations

from seapopym.optimization.gradient_optimizer import GradientOptimizer, OptimizeResult
from seapopym.optimization.likelihood import (
    GaussianLikelihood,
    reparameterize_log_posterior,
)
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
from seapopym.optimization.runner import CalibrationRunner

__all__ = [
    # High-level
    "Objective",
    "PreparedObjective",
    "CalibrationRunner",
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
    # Likelihood
    "GaussianLikelihood",
    "reparameterize_log_posterior",
]

# Optional imports (require evosax)
try:
    from seapopym.optimization.evolutionary import EvolutionaryOptimizer
    from seapopym.optimization.hybrid import HybridOptimizer
    from seapopym.optimization.ipop import IPOPResult, run_ipop, run_ipop_cmaes

    __all__ += ["EvolutionaryOptimizer", "HybridOptimizer", "IPOPResult", "run_ipop", "run_ipop_cmaes"]
    _HAS_EVOSAX = True
except (ImportError, KeyError):
    _HAS_EVOSAX = False

# Optional imports (require blackjax)
try:
    from seapopym.optimization.nuts import NUTSResult, run_nuts
    from seapopym.optimization.sampler import Sampler

    __all__ += ["NUTSResult", "run_nuts", "Sampler"]
    _HAS_BLACKJAX = True
except ImportError:
    _HAS_BLACKJAX = False


def __getattr__(name: str):
    """Provide helpful error message for optional dependencies."""
    if (
        name in ("EvolutionaryOptimizer", "HybridOptimizer", "IPOPResult", "run_ipop", "run_ipop_cmaes")
        and not _HAS_EVOSAX
    ):
        raise ImportError(f"{name} requires the evosax package. Install it with: pip install seapopym[optimization]")
    if name in ("NUTSResult", "run_nuts", "Sampler") and not _HAS_BLACKJAX:
        raise ImportError(f"{name} requires the blackjax package. Install it with: pip install seapopym[optimization]")
    raise AttributeError(f"module 'seapopym.optimization' has no attribute '{name}'")
