"""Temporal grid configuration for simulations.

Provides TimeGrid dataclass and timestep parsing via pint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pint

_ureg = pint.UnitRegistry()


def _parse_dt(dt_str: str) -> float:
    """Parse timestep string to seconds using pint.

    Args:
        dt_str: Timestep string like "1d", "6h", "30min", "1.5h".

    Returns:
        Timestep in seconds.
    """
    try:
        return _ureg(dt_str).to("seconds").magnitude
    except (pint.UndefinedUnitError, pint.DimensionalityError):
        raise ValueError(
            f"Invalid dt format: '{dt_str}'. Expected a time duration (e.g. '1d', '6h', '30min', '1.5h')."
        ) from None


@dataclass
class TimeGrid:
    """Temporal grid configuration computed from user parameters.

    This class represents the temporal discretization of a simulation,
    computed from explicit start/end times and timestep duration.

    Attributes:
        start: Simulation start time (datetime64).
        end: Simulation end time (datetime64).
        dt_seconds: Timestep duration in seconds.
        n_timesteps: Number of timesteps in the simulation.
        coords: Temporal coordinates array (datetime64, shape: (n_timesteps,)).

    Example:
        >>> grid = TimeGrid.from_config("2000-01-01", "2000-01-10", "1d")
        >>> grid.n_timesteps
        9
        >>> grid.coords[0]
        numpy.datetime64('2000-01-01')
    """

    start: np.datetime64
    end: np.datetime64
    dt_seconds: float
    n_timesteps: int
    coords: np.ndarray  # dtype=datetime64[ns]

    @classmethod
    def from_config(cls, time_start: str, time_end: str, dt_str: str) -> TimeGrid:
        """Compute temporal grid from configuration strings.

        Args:
            time_start: Start time (ISO format, e.g., "2000-01-01").
            time_end: End time (ISO format, e.g., "2020-12-31").
            dt_str: Timestep duration (e.g., "1d", "0.05d", "6h").

        Returns:
            TimeGrid instance with computed n_timesteps and coordinates.

        Raises:
            ValueError: If time_end <= time_start, or if time range is not
                evenly divisible by dt (remainder > 1 second).

        Example:
            >>> grid = TimeGrid.from_config("2000-01-01", "2000-01-10", "1d")
            >>> grid.n_timesteps
            9
        """
        # 1. Parse dates
        start_pd = pd.to_datetime(time_start)
        end_pd = pd.to_datetime(time_end)

        if end_pd <= start_pd:
            raise ValueError(f"time_end ({time_end}) must be after time_start ({time_start})")

        start_dt64 = start_pd.to_datetime64()
        end_dt64 = end_pd.to_datetime64()

        # 2. Parse timestep
        dt_seconds = _parse_dt(dt_str)
        dt_td = pd.Timedelta(seconds=dt_seconds)

        # 3. Compute number of timesteps
        duration = end_pd - start_pd
        n_timesteps_float = duration / dt_td

        # Round to nearest integer
        n_timesteps = int(np.round(n_timesteps_float))

        # 4. Validate: remainder should be negligible (< 1 second)
        expected_duration = n_timesteps * dt_td
        remainder = abs(duration - expected_duration)

        if remainder > pd.Timedelta(seconds=1):
            raise ValueError(
                f"Time range [{time_start}, {time_end}] is not evenly divisible by dt={dt_str}. "
                f"Duration: {duration}, n_timesteps*dt: {expected_duration}, remainder: {remainder}. "
                f"Adjust time_end or dt to ensure exact alignment."
            )

        # 5. Generate temporal coordinates
        # Use freq instead of periods to avoid inclusive end (want [start, end))
        # Generate exactly n_timesteps starting from start with spacing dt
        coords = pd.date_range(start=start_pd, periods=n_timesteps, freq=dt_td).to_numpy()

        return cls(
            start=start_dt64,
            end=end_dt64,
            dt_seconds=dt_seconds,
            n_timesteps=n_timesteps,
            coords=coords,
        )
