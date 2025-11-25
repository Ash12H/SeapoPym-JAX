"""Forcing Manager module for handling external forcings with temporal interpolation."""

from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

from seapopym.standard import Coordinates


class ForcingManager:
    """Manages external forcings (e.g., temperature, currents) for the simulation.

    Handles temporal interpolation of input data to match simulation timesteps.
    """

    def __init__(self, forcings: xr.Dataset):
        """Initialize the ForcingManager.

        Args:
            forcings: xarray Dataset containing the forcing variables.
                     Must have a time dimension (Coordinates.T).

        Raises:
            ValueError: If time dimension is missing, empty, or not monotonically increasing.
        """
        if Coordinates.T not in forcings.dims:
            raise ValueError(f"Forcing dataset must have a '{Coordinates.T}' dimension.")

        if forcings.sizes[Coordinates.T] == 0:
            raise ValueError(f"Forcing dataset '{Coordinates.T}' dimension cannot be empty.")

        # Ensure time is monotonically increasing for correct interpolation/selection
        time_values = forcings[Coordinates.T].values
        # Check if sorted (monotonically increasing)
        # We use numpy for efficient check. Handling potential dask arrays:
        # If it's dask, computing just the time coordinate is usually cheap and safe.
        if hasattr(time_values, "compute"):
            time_values = time_values.compute()

        if not np.all(time_values[:-1] <= time_values[1:]):
            raise ValueError(
                f"Forcing dataset '{Coordinates.T}' dimension must be sorted (monotonically increasing)."
            )

        self.forcings = forcings

    def get_forcings(self, current_time: datetime) -> xr.Dataset:
        """Get forcings interpolated at the specific current_time.

        Args:
            current_time: The datetime for which to retrieve forcings.

        Returns:
            xarray Dataset with forcings at current_time. The time dimension is dropped.

        Raises:
            ValueError: If current_time is outside the forcing time range.
        """
        # Check bounds
        t = pd.Timestamp(current_time)
        min_time = pd.Timestamp(self.forcings[Coordinates.T].min().values)
        max_time = pd.Timestamp(self.forcings[Coordinates.T].max().values)

        if t < min_time or t > max_time:
            raise ValueError(
                f"Requested time {t} is outside forcing range [{min_time}, {max_time}]."
            )

        # Interpolation
        return self.forcings.interp({Coordinates.T: t}, method="linear", assume_sorted=True)
