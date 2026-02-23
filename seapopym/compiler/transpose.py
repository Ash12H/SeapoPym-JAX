"""Dimension mapping and canonical transposition.

This module handles:
1. Renaming dimensions from user names to canonical names (E, T, F, C, Z, Y, X)
2. Transposing arrays to the canonical dimension order
"""

from __future__ import annotations

import xarray as xr

from seapopym.dims import get_canonical_order

from .exceptions import TransposeError


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




