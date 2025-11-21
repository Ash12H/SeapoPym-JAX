"""Forcing data management: interpolation and distribution.

This module provides:
- ForcingManager: Central orchestrator for environmental forcings
- ForcingSource: Wrapper for environmental data sources
- @derived_forcing: Decorator to create derived forcings
- DerivedForcing: Metadata for derived forcings

Example:
    >>> import xarray as xr
    >>> from seapopym_message.forcing import (
    ...     ForcingManager,
    ...     ForcingSource,
    ...     derived_forcing,
    ... )
    >>>
    >>> # Load forcing data using xarray directly
    >>> temp_ds = xr.open_zarr("data/temp.zarr", chunks={'time': 10})
    >>>
    >>> # Wrap in ForcingSource
    >>> temp_source = ForcingSource(temp_ds, name="temperature")
    >>>
    >>> # Create manager
    >>> manager = ForcingManager(forcings=[temp_source])
"""

from seapopym_message.forcing.derived import DerivedForcing, derived_forcing
from seapopym_message.forcing.manager import ForcingManager
from seapopym_message.forcing.source import ForcingSource

__all__ = [
    "ForcingManager",
    "ForcingSource",
    "DerivedForcing",
    "derived_forcing",
]
