"""Data preprocessing: xarray stripping, NaN handling, and mask generation.

This module handles:
1. Converting xarray DataArrays to JAX arrays
2. Replacing NaN values with a fill value
3. Generating binary masks from NaN patterns
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import xarray as xr

from .transpose import apply_dimension_mapping, transpose_canonical

from seapopym.types import Array


def load_data(
    source: str | Path | xr.DataArray | xr.Dataset | np.ndarray | Any,
    variable_name: str | None = None,
) -> xr.DataArray | np.ndarray:
    """Load data from various sources.

    Args:
        source: File path, xarray object, or array.
        variable_name: Variable name to extract from Dataset (if applicable).

    Returns:
        xarray DataArray or NumPy array.
    """
    if isinstance(source, str | Path):
        path = Path(source)
        ds = xr.open_zarr(path) if path.suffix == ".zarr" or path.is_dir() else xr.open_dataset(path)

        if variable_name and variable_name in ds:
            return ds[variable_name]
        if len(ds.data_vars) == 1:
            return ds[list(ds.data_vars)[0]]
        raise ValueError(
            f"Dataset from '{path}' contains {len(ds.data_vars)} variables "
            f"({list(ds.data_vars)}). Specify variable_name to select one."
        )

    if isinstance(source, xr.Dataset):
        if variable_name and variable_name in source:
            return source[variable_name]
        if len(source.data_vars) == 1:
            return source[list(source.data_vars)[0]]
        raise ValueError(
            f"Dataset contains {len(source.data_vars)} variables "
            f"({list(source.data_vars)}). Specify variable_name to select one."
        )

    if isinstance(source, xr.DataArray):
        return source

    # Assume it's an array-like
    return np.asarray(source)


def strip_xarray(
    da: xr.DataArray,
) -> Array:
    """Convert xarray DataArray to JAX array.

    Args:
        da: xarray DataArray.

    Returns:
        JAX array.
    """
    # Get values (triggers compute if dask-backed)
    return jnp.asarray(da.values)


def preprocess_nan(
    data: Array,
    fill_value: float = 0.0,
) -> tuple[Array, Array]:
    """Replace NaN values and generate a mask.

    Args:
        data: Input array (may contain NaN).
        fill_value: Value to replace NaN with.

    Returns:
        Tuple of (cleaned data, mask). Mask is True where data is valid.
    """
    mask = ~jnp.isnan(data)
    data_clean = jnp.where(mask, data, fill_value)

    return data_clean, mask


def prepare_array(
    source: str | Path | xr.DataArray | xr.Dataset | np.ndarray | Any,
    dimension_mapping: dict[str, str] | None = None,
    fill_nan: float | None = 0.0,
    variable_name: str | None = None,
) -> tuple[Array, tuple[str, ...], Array | None]:
    """Full preprocessing pipeline for a single data source.

    Args:
        source: Data source (file path, xarray, or array).
        dimension_mapping: Optional dimension name mapping.
        fill_nan: Value to replace NaN. If None, NaN are preserved.
        variable_name: Variable name to extract from Dataset.

    Returns:
        Tuple of (prepared array, dimension names, mask or None).
    """
    # Load data
    data = load_data(source, variable_name)

    mask = None
    dims: tuple[str, ...] = ()

    if isinstance(data, xr.DataArray):
        # Apply dimension mapping (preserves DataArray type)
        mapped_data = apply_dimension_mapping(data, dimension_mapping)
        if not isinstance(mapped_data, xr.DataArray):
            msg = "apply_dimension_mapping returned unexpected type"
            raise TypeError(msg)

        # Transpose to canonical order
        data = transpose_canonical(mapped_data)

        # Remember dims before stripping
        dims = tuple(str(d) for d in data.dims)

        # Strip xarray
        arr = strip_xarray(data)

    else:
        # Raw array - no dims info
        arr = jnp.asarray(np.asarray(data))

    # Handle NaN
    if fill_nan is not None:
        arr, mask = preprocess_nan(arr, fill_nan)

    return arr, dims, mask


def extract_coords(
    source: str | Path | xr.DataArray | xr.Dataset,
    dimension_mapping: dict[str, str] | None = None,
) -> dict[str, Array]:
    """Extract coordinate arrays from a data source.

    Args:
        source: Data source with coordinates.
        dimension_mapping: Optional dimension name mapping.

    Returns:
        Dict mapping dimension names to coordinate arrays.
    """
    # Load as xarray
    opened: xr.Dataset | None = None
    if isinstance(source, str | Path):
        path = Path(source)
        opened = xr.open_zarr(path) if path.suffix == ".zarr" or path.is_dir() else xr.open_dataset(path)
        ds = opened
    elif isinstance(source, xr.DataArray):
        ds = source.to_dataset(name="data")
    else:
        ds = source

    try:
        # Apply mapping
        if dimension_mapping:
            rename_dict = {old: new for old, new in dimension_mapping.items() if old in ds.dims}
            if rename_dict:
                ds = ds.rename(rename_dict)

        coords: dict[str, Array] = {}
        for name, coord in ds.coords.items():
            coords[str(name)] = np.asarray(coord.values)

        return coords
    finally:
        if opened is not None:
            opened.close()
