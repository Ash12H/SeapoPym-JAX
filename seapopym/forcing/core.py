"""Forcing Manager module for handling external forcings with temporal interpolation."""

import logging
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from seapopym.standard import Coordinates

logger = logging.getLogger(__name__)


class ForcingManager:
    """Manages external forcings (e.g., temperature, currents) for the simulation.

    Handles temporal selection or interpolation of input data to match simulation timesteps.
    """

    def __init__(
        self, forcings: xr.Dataset, method: Literal["interp", "ffill", "nearest"] = "interp"
    ):
        """Initialize the ForcingManager.

        Args:
            forcings: xarray Dataset containing the forcing variables.
                     Must have a time dimension (Coordinates.T).
            method: Method for temporal alignment:
                   - "interp": Linear interpolation between time steps (default, most accurate but creates tasks)
                   - "ffill": Forward-fill from previous time step (fast, avoids rechunking)
                   - "nearest": Select nearest time step (fast, avoids rechunking)

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
        self.method = method
        logger.info(f"ForcingManager initialized with method='{method}'")

    def get_forcings(self, current_time: datetime) -> xr.Dataset:
        """Get forcings at the specific current_time using the configured method.

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

        if self.method == "interp":
            # Linear interpolation between time steps
            # Note: interp() can alter chunking (e.g. fracturing spatial chunks).
            # We must restore the original chunking for preserving performance.
            logger.debug(f"Interpolating forcings at {t}")
            result = self.forcings.interp({Coordinates.T: t}, method="linear", assume_sorted=True)

            # Restore chunks for remaining dimensions
            chunks_to_restore = {}
            for dim in result.dims:
                if dim in self.forcings.chunksizes:
                    chunks_to_restore[dim] = self.forcings.chunksizes[dim]

            if chunks_to_restore:
                result = result.chunk(chunks_to_restore)

        elif self.method == "ffill":
            # Forward-fill: select the last time <= current_time
            # This avoids interpolation and rechunking issues
            logger.debug(f"Forward-filling forcings at {t}")
            result = self.forcings.sel({Coordinates.T: t}, method="ffill")

        elif self.method == "nearest":
            # Select nearest time step
            logger.debug(f"Selecting nearest forcings to {t}")
            result = self.forcings.sel({Coordinates.T: t}, method="nearest")

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return result
