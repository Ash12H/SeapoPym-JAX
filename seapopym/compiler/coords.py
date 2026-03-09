"""Coordinate-to-index conversion utilities.

Provides a helper to convert physical coordinates (lat, lon, date, etc.)
into integer indices suitable for JAX array indexing.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from seapopym.types import Array


def coords_to_indices(
    grid: dict[str, Array],
    sel_kwargs: dict | None = None,
    **dims: Array,
) -> tuple[Array, ...]:
    """Convert physical coordinates to integer indices on a grid.

    Uses xarray's ``.sel()`` for coordinate matching, delegating all matching
    logic (nearest-neighbor, exact, tolerance, etc.) to xarray.

    Args:
        grid: Coordinate arrays for each dimension, typically ``model.coords``.
        sel_kwargs: Keyword arguments forwarded to ``xr.DataArray.sel()``.
            Defaults to ``{"method": "nearest"}``.
        **dims: Coordinate values to convert, keyed by dimension name.
            Each value should be an array-like of physical coordinates.

    Returns:
        Tuple of integer index arrays, one per dimension in ``dims``,
        in the same order as the ``**dims`` arguments.

    Raises:
        ValueError: If a dimension name is not found in ``grid``.

    Example:
        >>> indices = coords_to_indices(
        ...     grid=model.coords,
        ...     T=np.array(["2000-01-05", "2000-01-15"], dtype="datetime64[ns]"),
        ...     Y=np.array([45.0, 46.5]),
        ...     X=np.array([-3.0, -2.5]),
        ... )
        >>> t_idx, y_idx, x_idx = indices
    """
    if sel_kwargs is None:
        sel_kwargs = {"method": "nearest"}

    result = []
    for dim_name, coord_values in dims.items():
        if dim_name not in grid:
            available = list(grid.keys())
            msg = f"Dimension '{dim_name}' not found in grid. Available: {available}"
            raise ValueError(msg)

        coord_array = grid[dim_name]
        da = xr.DataArray(np.arange(len(coord_array)), dims=[dim_name], coords={dim_name: coord_array})
        indices = da.sel({dim_name: coord_values}, **sel_kwargs).values
        result.append(indices)

    return tuple(result)
