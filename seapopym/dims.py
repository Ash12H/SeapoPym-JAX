"""Canonical dimension ordering.

The engine operates on arrays whose dimensions follow a fixed canonical order.
This module defines that order and provides a helper to sort any subset of
dimension names accordingly.
"""

from __future__ import annotations

CANONICAL_DIMS: tuple[str, ...] = ("E", "T", "F", "C", "Z", "Y", "X")


def get_canonical_order(dims: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Get the canonical order for a subset of dimensions.

    Args:
        dims: Dimension names present in the data.

    Returns:
        Tuple of dimension names in canonical order.

    Example:
        >>> get_canonical_order(["X", "Y", "T"])
        ("T", "Y", "X")
    """
    return tuple(d for d in CANONICAL_DIMS if d in dims)
