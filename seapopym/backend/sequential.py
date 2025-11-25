"""Sequential execution backend."""

from typing import Any

import xarray as xr

from seapopym.backend.base import ComputeBackend
from seapopym.backend.core import execute_task_sequence
from seapopym.blueprint.nodes import ComputeNode


class SequentialBackend(ComputeBackend):
    """Default backend that executes tasks sequentially in the current process."""

    def execute(
        self,
        task_groups: list[tuple[str, list[ComputeNode]]],
        state: xr.Dataset,
    ) -> dict[str, Any]:
        """Execute tasks sequentially.

        Implements the logic previously found in FunctionalGroup.compute.
        """
        all_results: dict[str, Any] = {}

        for _group_name, tasks in task_groups:
            # Execute tasks for this group
            # We pass all_results as external_context because in sequential mode,
            # all previous results are available.
            group_results = execute_task_sequence(tasks, state, all_results)
            all_results.update(group_results)

        return all_results
