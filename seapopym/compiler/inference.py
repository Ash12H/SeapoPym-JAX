"""Shape inference from data metadata.

This module reads dimension sizes from xarray datasets without loading
the actual data into memory (lazy loading via chunks={}).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from seapopym.blueprint import Config

from .exceptions import GridAlignmentError, ShapeInferenceError


def infer_shapes_from_file(path: str | Path) -> dict[str, int]:
    """Read dimension sizes from a single file.

    Args:
        path: Path to NetCDF/Zarr file.

    Returns:
        Dict mapping dimension names to sizes.

    Raises:
        ShapeInferenceError: If file cannot be opened or read.
    """
    path = Path(path)

    try:
        # Use chunks={} for lazy loading (metadata only)
        ds = xr.open_zarr(path) if path.suffix == ".zarr" or path.is_dir() else xr.open_dataset(path, chunks={})

        shapes = {str(k): v for k, v in ds.sizes.items()}
        ds.close()
        return shapes

    except Exception as e:
        raise ShapeInferenceError(str(path), str(e)) from e


def infer_shapes_from_array(arr: np.ndarray | Any, dims: list[str] | None) -> dict[str, int]:
    """Infer shapes from an in-memory array.

    Args:
        arr: NumPy array or array-like.
        dims: Dimension names corresponding to array axes.

    Returns:
        Dict mapping dimension names to sizes.
    """
    if dims is None:
        return {}

    arr = np.asarray(arr)
    if len(dims) != arr.ndim:
        return {}

    return dict(zip(dims, arr.shape, strict=True))


def infer_shapes(config: Config, blueprint_dims: dict[str, list[str] | None] | None = None) -> dict[str, int]:
    """Infer all dimension sizes from config data sources.

    Reads metadata from files (lazy) and in-memory arrays to build
    a complete mapping of dimension names to sizes.

    Args:
        config: Configuration with forcings and initial_state.
        blueprint_dims: Optional mapping of variable names to their declared dims.

    Returns:
        Dict mapping dimension names to sizes.

    Raises:
        GridAlignmentError: If same dimension has different sizes in different files.
        ShapeInferenceError: If a file cannot be read.
    """
    shapes: dict[str, int] = {}
    size_sources: dict[str, dict[str, int]] = {}  # dim -> {source: size}

    def record_shape(source_name: str, dim: str, size: int) -> None:
        """Record a dimension size and check for conflicts."""
        if dim not in size_sources:
            size_sources[dim] = {}
        size_sources[dim][source_name] = size

        if dim in shapes and shapes[dim] != size:
            raise GridAlignmentError(dim, size_sources[dim])
        shapes[dim] = size

    # Infer from forcings
    for name, value in config.forcings.items():
        if isinstance(value, str | Path):
            # File path - read metadata
            file_shapes = infer_shapes_from_file(value)
            for dim, size in file_shapes.items():
                record_shape(f"forcings.{name}", dim, size)

        elif isinstance(value, xr.DataArray):
            # xarray DataArray - use dims directly
            for dim, size in zip(value.dims, value.shape, strict=True):
                record_shape(f"forcings.{name}", str(dim), size)

        elif isinstance(value, xr.Dataset):
            # xarray Dataset - use sizes
            for dim, size in value.sizes.items():
                record_shape(f"forcings.{name}", str(dim), size)

        elif hasattr(value, "shape"):
            # NumPy array or similar - need dims from blueprint
            if blueprint_dims and f"forcings.{name}" in blueprint_dims:
                dims = blueprint_dims[f"forcings.{name}"]
                arr_shapes = infer_shapes_from_array(value, dims)
                for dim, size in arr_shapes.items():
                    record_shape(f"forcings.{name}", dim, size)

    # Infer from initial_state
    _infer_from_nested(config.initial_state, "initial_state", shapes, size_sources, blueprint_dims, record_shape)

    return shapes


def _infer_from_nested(
    data: dict[str, Any],
    prefix: str,
    shapes: dict[str, int],
    size_sources: dict[str, dict[str, int]],
    blueprint_dims: dict[str, list[str] | None] | None,
    record_shape: Any,
) -> None:
    """Recursively infer shapes from nested dict structure."""
    for name, value in data.items():
        full_name = f"{prefix}.{name}"

        if isinstance(value, dict) and not hasattr(value, "shape"):
            # Nested group - recurse
            _infer_from_nested(value, full_name, shapes, size_sources, blueprint_dims, record_shape)

        elif isinstance(value, str | Path):
            # File path
            file_shapes = infer_shapes_from_file(value)
            for dim, size in file_shapes.items():
                record_shape(full_name, dim, size)

        elif isinstance(value, xr.DataArray):
            for dim, size in zip(value.dims, value.shape, strict=True):
                record_shape(full_name, str(dim), size)

        elif hasattr(value, "shape"):
            # Array - need dims from blueprint
            # Map initial_state.X to state.X for blueprint lookup
            blueprint_key = full_name.replace("initial_state.", "state.")
            if blueprint_dims and blueprint_key in blueprint_dims:
                dims = blueprint_dims[blueprint_key]
                arr_shapes = infer_shapes_from_array(value, dims)
                for dim, size in arr_shapes.items():
                    record_shape(full_name, dim, size)
