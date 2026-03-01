"""Sensitivity analysis module for SeapoPym.

This module provides Sobol variance-based global sensitivity analysis,
leveraging JAX's vmap for batched model evaluation via JAX.

Main components:
- SobolAnalyzer: Orchestrates the full Sobol analysis workflow
- SobolResult: Structured result with first/total-order indices

Requires the optional SALib dependency:
    pip install seapopym[sensitivity]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from seapopym.sensitivity.sobol import SobolAnalyzer, SobolResult

__all__ = [
    "SobolAnalyzer",
    "SobolResult",
]

try:
    import SALib  # noqa: F401

    _HAS_SALIB = True
except ImportError:
    _HAS_SALIB = False


def __getattr__(name: str):
    """Lazy imports with helpful error messages for optional dependencies."""
    if name in ("SobolAnalyzer", "SobolResult"):
        if not _HAS_SALIB:
            raise ImportError(f"{name} requires the SALib package. Install it with: pip install seapopym[sensitivity]")
        from seapopym.sensitivity.sobol import SobolAnalyzer, SobolResult

        return {"SobolAnalyzer": SobolAnalyzer, "SobolResult": SobolResult}[name]
    raise AttributeError(f"module 'seapopym.sensitivity' has no attribute '{name}'")
