"""Tests for seapopym.sensitivity lazy imports."""

from __future__ import annotations

import pytest


def test_getattr_sobol_analyzer():
    """SobolAnalyzer is importable via the lazy __getattr__."""
    from seapopym.sensitivity import SobolAnalyzer
    from seapopym.sensitivity.sobol import SobolAnalyzer as Direct

    assert SobolAnalyzer is Direct


def test_getattr_sobol_result():
    """SobolResult is importable via the lazy __getattr__."""
    from seapopym.sensitivity import SobolResult
    from seapopym.sensitivity.sobol import SobolResult as Direct

    assert SobolResult is Direct


def test_getattr_unknown_raises():
    """Accessing an unknown attribute raises AttributeError."""
    import seapopym.sensitivity as mod

    with pytest.raises(AttributeError, match="has no attribute 'DoesNotExist'"):
        _ = mod.DoesNotExist


def test_all_exports():
    """__all__ contains exactly SobolAnalyzer and SobolResult."""
    import seapopym.sensitivity as mod

    assert set(mod.__all__) == {"SobolAnalyzer", "SobolResult"}
