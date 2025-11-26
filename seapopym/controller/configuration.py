"""Configuration structures for the simulation."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration parameters for the simulation run."""

    start_date: Any
    end_date: Any
    timestep: timedelta = timedelta(days=1)

    def __post_init__(self) -> None:
        """Validate and convert configuration parameters."""
        # Convert dates to pandas Timestamp for consistency with xarray
        object.__setattr__(self, "start_date", pd.to_datetime(self.start_date))
        object.__setattr__(self, "end_date", pd.to_datetime(self.end_date))

        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        if self.timestep.total_seconds() <= 0:
            raise ValueError("timestep must be positive")
