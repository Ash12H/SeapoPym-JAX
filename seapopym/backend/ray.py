"""Ray execution backend."""

import logging
from typing import Any

import xarray as xr

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    ray = None
    RAY_AVAILABLE = False

from seapopym.backend.base import ComputeBackend
from seapopym.blueprint.nodes import ComputeNode

# Define remote function only if ray is available to avoid ImportErrors
if RAY_AVAILABLE:

    @ray.remote
    def run_group_remote(
        tasks: list[ComputeNode],
        state: xr.Dataset,
        external_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a group of tasks remotely.

        Args:
            tasks: Tasks to execute.
            state: Global state (read-only).
            external_context: Results from previous groups (already resolved by Ray).

        Returns:
            Dictionary of results produced by this group.
        """
        # Import here to ensure it's available in Ray worker
        from seapopym.backend.core import execute_task_sequence

        return execute_task_sequence(tasks, state, external_context)

else:
    run_group_remote = None


class RayBackend(ComputeBackend):
    """Distributed execution backend using Ray.

    Important: Ray must be initialized before creating a RayBackend instance.
    Call ray.init() in your application before using this backend.
    """

    def __init__(self) -> None:
        """Initialize RayBackend.

        Raises:
            ImportError: If Ray is not installed.
            RuntimeError: If Ray is not initialized.
        """
        if not RAY_AVAILABLE:
            raise ImportError("Ray is not installed. Install with: pip install 'ray[default]'")

        if not ray.is_initialized():
            raise RuntimeError(
                "Ray is not initialized. Call ray.init() before creating RayBackend."
            )

    def execute(
        self,
        task_groups: list[tuple[str, list[ComputeNode]]],
        state: xr.Dataset,
    ) -> dict[str, Any]:
        """Execute task groups using Ray.

        Groups are executed sequentially to respect dependencies between groups.
        Within each group, tasks are executed sequentially (future optimization possible).

        Args:
            task_groups: List of (group_name, tasks) tuples.
            state: Current simulation state.

        Returns:
            Dictionary of all results from all groups.
        """
        if not task_groups:
            return {}

        # 1. Put state in object store (shared by all workers)
        state_ref = ray.put(state)

        # 2. Accumulate results from all groups
        all_results: dict[str, Any] = {}

        # 3. Execute groups sequentially to respect inter-group dependencies
        for group_name, tasks in task_groups:
            # Put accumulated results in object store
            context_ref = ray.put(all_results)

            # Launch remote task
            result_ref = run_group_remote.remote(tasks, state_ref, context_ref)

            # Wait for completion and merge results
            group_results = ray.get(result_ref)
            all_results.update(group_results)

            logging.debug(
                f"Group '{group_name}' completed. Produced variables: {list(group_results.keys())}"
            )

        return all_results
