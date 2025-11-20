"""Forcing data management: interpolation and distribution.

This module provides:
- ForcingManager: Central orchestrator for environmental forcings
- @derived_forcing: Decorator to create derived forcings
- DerivedForcing: Metadata for derived forcings

Example:
    >>> import xarray as xr
    >>> from seapopym_message.forcing import (
    ...     ForcingManager,
    ...     derived_forcing,
    ... )
    >>>
    >>> # Load forcing data using xarray directly
    >>> temp_ds = xr.open_zarr("data/temp.zarr", chunks={'time': 10})
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
from seapopym_message.forcing.manager import ForcingManager

__all__ = [
    "ForcingManager",
    "DerivedForcing",
    "derived_forcing",
]
