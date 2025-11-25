"""Base interface for execution backends."""

from abc import ABC, abstractmethod
from typing import Any

import xarray as xr

from seapopym.blueprint.nodes import ComputeNode


class ComputeBackend(ABC):
    """Abstract base class for execution backends."""

    @abstractmethod
    def execute(
        self,
        task_groups: list[tuple[str, list[ComputeNode]]],
        state: xr.Dataset,
    ) -> dict[str, Any]:
        """Execute a sequence of task groups on the given state.

        Args:
            task_groups: List of (group_name, list_of_tasks) tuples.
            state: The current state of the simulation.

        Returns:
            A dictionary containing the results of the execution {variable_name: value}.
        """
        pass
