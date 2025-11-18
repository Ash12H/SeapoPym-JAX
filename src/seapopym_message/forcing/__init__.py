"""Forcing data management: interpolation and distribution.

This module provides:
- ForcingManager: Central orchestrator for environmental forcings
- @derived_forcing: Decorator to create derived forcings
- DerivedForcing: Metadata for derived forcings
- I/O helpers: load_forcing_from_path, load_forcings_from_paths

Example:
    >>> import xarray as xr
    >>> from seapopym_message.forcing import (
    ...     ForcingManager,
    ...     derived_forcing,
    ...     load_forcing_from_path,
    ... )
    >>>
    >>> # Option 1: User controls I/O directly
    >>> temp_ds = xr.open_zarr("data/temp.zarr", chunks={'time': 10})
    >>> temp_ds.attrs['units'] = '°C'
    >>> temp_ds.attrs['interpolation_method'] = 'linear'
    >>>
    >>> # Option 2: Use helper function
    >>> temp_ds = load_forcing_from_path("data/temp.zarr", chunks={'time': 10})
    >>> temp_ds.attrs['units'] = '°C'
    >>> temp_ds.attrs['interpolation_method'] = 'linear'
    >>>
    >>> # Create manager with pre-loaded datasets
    >>> manager = ForcingManager(datasets={'temperature': temp_ds})
    >>>
    >>> # Add derived forcing
    >>> @derived_forcing(
    ...     name="recruitment",
    ...     inputs=["primary_production"],
    ...     params=["transfer_coefficient"],
    ... )
    >>> def compute_recruitment(primary_production, transfer_coefficient):
    ...     return primary_production * transfer_coefficient
    >>>
    >>> manager.register_derived(compute_recruitment)
"""

from seapopym_message.forcing.derived import DerivedForcing, derived_forcing
from seapopym_message.forcing.io import load_forcing_from_path, load_forcings_from_paths
from seapopym_message.forcing.manager import ForcingManager

__all__ = [
    "ForcingManager",
    "DerivedForcing",
    "derived_forcing",
    "load_forcing_from_path",
    "load_forcings_from_paths",
]
