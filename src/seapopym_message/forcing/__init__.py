"""Forcing data management: NetCDF/Zarr readers, interpolation.

This module provides:
- ForcingManager: Central orchestrator for environmental forcings
- ForcingConfig: Configuration for base forcings (from files)
- @derived_forcing: Decorator to create derived forcings
- DerivedForcing: Metadata for derived forcings

Example:
    >>> from seapopym_message.forcing import (
    ...     ForcingManager,
    ...     ForcingConfig,
    ...     derived_forcing,
    ... )
    >>>
    >>> # Configure base forcings
    >>> config = {
    ...     "temperature": ForcingConfig(
    ...         source="data/temp.zarr",
    ...         dims=["time", "depth", "lat", "lon"],
    ...     ),
    ... }
    >>>
    >>> # Create manager
    >>> manager = ForcingManager(config)
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
from seapopym_message.forcing.manager import (
    ForcingConfig,
    ForcingManager,
    ForcingManagerConfig,
)

__all__ = [
    "ForcingManager",
    "ForcingManagerConfig",
    "ForcingConfig",
    "DerivedForcing",
    "derived_forcing",
]
