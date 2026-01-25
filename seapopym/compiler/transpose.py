"""Dimension mapping and canonical transposition.

This module handles:
1. Renaming dimensions from user names to canonical names (E, T, F, C, Z, Y, X)
2. Transposing arrays to the canonical dimension order
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from .exceptions import TransposeError
from .model import CANONICAL_DIMS

# Type alias
Array = Any  # np.ndarray | jax.Array


def apply_dimension_mapping(
    ds: xr.Dataset | xr.DataArray,
    mapping: dict[str, str] | None,
) -> xr.Dataset | xr.DataArray:
    """Rename dimensions according to user-provided mapping.

    Args:
        ds: xarray Dataset or DataArray to rename.
        mapping: Dict mapping original names to canonical names.
                 Example: {"lat": "Y", "lon": "X", "time": "T"}

    Returns:
        Dataset/DataArray with renamed dimensions.
    """
    if mapping is None:
        return ds

    # Only rename dimensions that exist in the data
    rename_dict = {old: new for old, new in mapping.items() if old in ds.dims}

    if not rename_dict:
        return ds

    return ds.rename(rename_dict)


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
    # Filter canonical dims to only those present
    return tuple(d for d in CANONICAL_DIMS if d in dims)


def transpose_canonical(
    da: xr.DataArray,
    target_order: tuple[str, ...] | None = None,
) -> xr.DataArray:
    """Transpose a DataArray to canonical dimension order.

    Args:
        da: xarray DataArray to transpose.
        target_order: Optional specific order. If None, uses canonical order.

    Returns:
        Transposed DataArray.

    Raises:
        TransposeError: If transposition fails.
    """
    try:
        # Convert dims to strings
        str_dims = tuple(str(d) for d in da.dims)

        if target_order is None:
            target_order = get_canonical_order(str_dims)

        # Only include dims that exist in the data
        present_dims = tuple(d for d in target_order if d in str_dims)

        # Add any dims not in canonical order at the end (preserve them)
        extra_dims = tuple(d for d in str_dims if d not in present_dims)
        final_order = present_dims + extra_dims

        return da.transpose(*final_order)

    except Exception as e:
        name = str(da.name) if da.name is not None else "unnamed"
        raise TransposeError(name, str(e)) from e


def transpose_array(
    arr: Array,
    current_dims: tuple[str, ...] | list[str],
    target_order: tuple[str, ...] | None = None,
) -> tuple[Array, tuple[str, ...]]:
    """Transpose a NumPy/JAX array to canonical dimension order.

    Args:
        arr: Array to transpose.
        current_dims: Current dimension names.
        target_order: Target dimension order. If None, uses canonical.

    Returns:
        Tuple of (transposed array, new dimension order).
    """
    current_dims = tuple(current_dims)

    if target_order is None:
        target_order = get_canonical_order(current_dims)

    # Build axis permutation
    # Include dims in target_order first, then extras
    present_in_target = [d for d in target_order if d in current_dims]
    extras = [d for d in current_dims if d not in target_order]
    new_order = present_in_target + extras

    # Get axis indices
    axes = tuple(current_dims.index(d) for d in new_order)

    # Transpose
    transposed = np.transpose(arr, axes)

    return transposed, tuple(new_order)


def ensure_contiguous(arr: Array) -> Array:
    """Ensure array is C-contiguous in memory.

    Args:
        arr: Input array.

    Returns:
        C-contiguous array (may be a copy).
    """
    arr = np.asarray(arr)
    if arr.flags["C_CONTIGUOUS"]:
        return arr
    return np.ascontiguousarray(arr)
