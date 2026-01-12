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

    def prepare_data(self, data: xr.Dataset) -> xr.Dataset:
        """Prepare data for efficient computation with this backend.

        This hook is called once at setup time to optimize data layout and storage
        according to the backend's execution strategy. It allows the backend to:
        - Persist lazy arrays to distributed memory (DataParallelBackend)
        - Compute arrays eagerly to avoid overhead (TaskParallelBackend)
        - Perform any other backend-specific data preparation

        This is typically called on:
        - Initial state before starting the simulation
        - Forcings before creating the ForcingManager
        - Any other large datasets that will be used repeatedly

        Args:
            data: The dataset to prepare (state, forcings, etc.)

        Returns:
            The prepared dataset, optimized for this backend's execution.

        Note:
            Default implementation returns data unchanged (no-op).
        """
        return data

    def stabilize_state(self, state: xr.Dataset) -> xr.Dataset:
        """Stabilize the state representation between timesteps.

        This hook allows the backend to perform necessary lifecycle management on the state data
        to ensure stability for the next simulation step.

        Examples of usage:
        - Cutting dependency graph lineages (Dask persist) to prevent graph explosion.
        - Synchronizing memory between devices (CPU/GPU).
        - Materializing lazy buffers.

        Args:
            state: The state at the end of the current timestep.

        Returns:
            The stabilized state ready for the next timestep.
        """
        return state

    def process_io_task(self, task: Any) -> None:
        """Execute an IO task (typically writing results).

        This allows the backend to decide HOW to execute the task:
        - Synchronously (blocking) for SequentialBackend.
        - Asynchronously (background) for DistributedBackend.

        Args:
            task: The task to execute. Usually a dask.delayed object or a callable.
        """
        # Default implementation: execute immediately if callable or compute if dask object
        if hasattr(task, "compute"):
            task.compute()
        elif callable(task):
            task()
