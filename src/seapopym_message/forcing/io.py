"""I/O utilities for loading forcing datasets.

These are convenience functions to help users load forcing data from files.
ForcingManager itself does not handle I/O - users have full control over
how datasets are loaded (chunking, caching, parallel loading, etc.).
"""

from typing import Any

import xarray as xr


def load_forcing_from_path(path: str, **kwargs: Any) -> xr.Dataset:
    """Load a forcing dataset from file path.

    This is a convenience function that automatically detects Zarr stores
    vs NetCDF files based on the path extension.

    Args:
        path: Path to Zarr store or NetCDF file.
        **kwargs: Additional arguments passed to xr.open_zarr() or xr.open_dataset().
                 Examples: chunks={'time': 10}, decode_times=False, etc.

    Returns:
        Loaded xarray Dataset.

    Example:
        >>> # Load with custom chunking
        >>> ds = load_forcing_from_path(
        ...     "data/temperature.zarr",
        ...     chunks={'time': 10, 'depth': 5}
        ... )
        >>>
        >>> # Add metadata
        >>> ds.attrs['units'] = '°C'
        >>> ds.attrs['interpolation_method'] = 'linear'
        >>>
        >>> # Use in ForcingManager
        >>> from seapopym_message.forcing import ForcingManager
        >>> manager = ForcingManager(datasets={'temperature': ds})
    """
    if path.endswith(".zarr") or ".zarr/" in path or ".zarr" in path:
        # Zarr store
        return xr.open_zarr(path, **kwargs)
    else:
        # NetCDF file (or other xarray-supported format)
        return xr.open_dataset(path, **kwargs)


def load_forcings_from_paths(paths: dict[str, str], **kwargs: Any) -> dict[str, xr.Dataset]:
    """Load multiple forcing datasets from paths.

    Args:
        paths: Dictionary mapping forcing names to file paths.
        **kwargs: Additional arguments passed to load functions.

    Returns:
        Dictionary mapping forcing names to loaded Datasets.

    Example:
        >>> paths = {
        ...     'temperature': 'data/temp.zarr',
        ...     'currents': 'data/currents.nc',
        ...     'primary_production': 'data/pp.zarr',
        ... }
        >>> datasets = load_forcings_from_paths(paths, chunks={'time': 10})
        >>>
        >>> # Add metadata to each dataset
        >>> datasets['temperature'].attrs['units'] = '°C'
        >>> datasets['temperature'].attrs['interpolation_method'] = 'linear'
        >>>
        >>> # Create manager
        >>> from seapopym_message.forcing import ForcingManager
        >>> manager = ForcingManager(datasets=datasets)
    """
    datasets = {}
    for name, path in paths.items():
        datasets[name] = load_forcing_from_path(path, **kwargs)
    return datasets
