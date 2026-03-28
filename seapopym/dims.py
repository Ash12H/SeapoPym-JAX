"""Canonical dimension ordering.

The engine operates on arrays whose dimensions follow a fixed canonical order.
This module defines that order and provides a helper to sort any subset of
dimension names accordingly.
"""

from __future__ import annotations

CANONICAL_DIMS: tuple[str, ...] = ("E", "T", "F", "C", "Z", "Y", "X")


def get_canonical_order(dims: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Sort dimensions: canonical dims first (in canonical order), then non-canonical dims (in original order).

    Non-canonical dimensions are preserved but never treated as broadcast
    dimensions by the engine — they must be declared as core_dims.

    Args:
        dims: Dimension names present in the data.

    Returns:
        Tuple of dimension names sorted with canonical dims first.

    Example:
        >>> get_canonical_order(["X", "Y", "T"])
        ("T", "Y", "X")
        >>> get_canonical_order(["F", "P"])
        ("F", "P")
    """
    canonical = tuple(d for d in CANONICAL_DIMS if d in dims)
    extra = tuple(d for d in dims if d not in CANONICAL_DIMS)
    return canonical + extra
