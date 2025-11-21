"""ForcingSource: Wrapper for environmental data sources.

This module defines the ForcingSource class, which standardizes how environmental
data (temperature, currents, etc.) is loaded, validated, and accessed.
"""

from typing import Literal

import xarray as xr


class ForcingSource:
    """A standardized source of environmental forcing data.

    Wraps an xarray.DataArray and ensures it has the required dimensions and metadata
    for use in the simulation.

    Attributes:
        data: The underlying xarray DataArray.
        name: Name of the forcing variable (e.g., "temperature").
        interpolation_method: Method for temporal interpolation ("linear" or "nearest").
    """

    def __init__(
        self,
        data: xr.DataArray,
        name: str | None = None,
        interpolation_method: Literal["linear", "nearest"] = "linear",
    ) -> None:
        """Initialize a ForcingSource.

        Args:
            data: Input xarray DataArray. Must have 'time', 'lat', and 'lon' dimensions.
            name: Optional name for the forcing. If None, uses data.name.
            interpolation_method: Temporal interpolation method. Defaults to "linear".

        Raises:
            ValueError: If required dimensions are missing or if name is not provided/found.
        """
        self._validate_data(data)
        self.data = data
        self.name = name or data.name
        self.interpolation_method = interpolation_method

        if self.name is None:
            msg = "Forcing name must be provided either via 'name' argument or data.name"
            raise ValueError(msg)

    def _validate_data(self, data: xr.DataArray) -> None:
        """Validate that data has required dimensions.

        Args:
            data: DataArray to validate.

        Raises:
            ValueError: If 'time', 'lat', or 'lon' dimensions are missing.
        """
        required_dims = {"time", "lat", "lon"}
        missing_dims = required_dims - set(data.dims)
        if missing_dims:
            msg = f"Forcing data missing required dimensions: {missing_dims}"
            raise ValueError(msg)

    def interpolate(self, time: float) -> xr.DataArray:
        """Interpolate data to a specific time.

        Args:
            time: Target time.

        Returns:
            Interpolated DataArray (time dimension removed).
        """
        return self.data.interp(time=time, method=self.interpolation_method)

    def __repr__(self) -> str:
        """String representation."""
        dims = ", ".join(str(d) for d in self.data.dims)
        return f"ForcingSource(name='{self.name}', dims=[{dims}], method='{self.interpolation_method}')"
