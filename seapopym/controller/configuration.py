"""Configuration structures for the simulation."""

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration parameters for the simulation run."""

    start_date: datetime
    end_date: datetime
    timestep: timedelta = timedelta(days=1)

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        if self.timestep.total_seconds() <= 0:
            raise ValueError("timestep must be positive")
