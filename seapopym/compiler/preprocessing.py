"""Data preprocessing utilities.

Provides coordinate extraction from xr.DataArray sources.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from seapopym.types import Array


def extract_coords(
    source: xr.DataArray,
    dimension_mapping: dict[str, str] | None = None,
) -> dict[str, Array]:
    """Extract coordinate arrays from a DataArray.

    Args:
        source: DataArray with coordinates.
        dimension_mapping: Optional dimension name mapping.

    Returns:
        Dict mapping dimension names to coordinate arrays.
    """
    ds = source.to_dataset(name="data")

    if dimension_mapping:
        rename_dict = {old: new for old, new in dimension_mapping.items() if old in ds.dims}
        if rename_dict:
            ds = ds.rename(rename_dict)

    coords: dict[str, Array] = {}
    for name, coord in ds.coords.items():
        coords[str(name)] = np.asarray(coord.values)

    return coords
