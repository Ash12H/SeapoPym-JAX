"""Distributed backend using dask.distributed for optimal execution.

This backend leverages Xarray's integration with Dask to handle distributed execution.
Instead of manually managing Futures, it relies on Xarray to build a lazy task graph,
which is then executed and persisted on the cluster at each timestep.

Key Concepts:
- **Lazy Construction**: The task graph is built using standard Xarray operations.
- **Persist & Wait**: At the end of each timestep, data is persisted to distributed RAM
  to cut the task graph and prevent memory explosion (Distributed Time-Stepping).
- **Data Parallelism**: Handled natively by Dask Array chunks.
"""

import logging
from typing import Any

import xarray as xr
from dask.distributed import fire_and_forget, get_client, wait

from seapopym.backend.base import ComputeBackend
from seapopym.backend.core import execute_task_sequence
from seapopym.backend.validation import validate_has_chunks
from seapopym.blueprint.nodes import ComputeNode

logger = logging.getLogger(__name__)


class DistributedBackend(ComputeBackend):
    """Distributed execution backend using dask.distributed.

    It implements the "Distributed Time-Stepping" pattern:
    1. Build a lazy graph for the current timestep (Flux computation).
    2. Persist the result to distributed memory (triggers computation).
    3. Wait for completion (backpressure) and release old states.
    """

    def __init__(self) -> None:
        """Initialize DistributedBackend.

        Raises:
            ValueError: If no dask.distributed client is available.
        """
        try:
            self.client = get_client()
            logger.info(
                f"DistributedBackend initialized with {len(self.client.scheduler_info()['workers'])} workers"
            )
        except ValueError as e:
            raise ValueError(
                "DistributedBackend requires an active dask.distributed client. "
                "Please create a Client before initializing the controller."
            ) from e

    def process_io_task(self, task: Any) -> None:
        """Execute IO task.

        - Dask objects (ZarrWriter task) are executed asynchronously on the cluster (fire-and-forget).
        - Callables (MemoryWriter task) are executed synchronously locally to preserve state.
        """
        if hasattr(task, "compute"):
            # It's a Dask object (Delayed), submit it to the cluster
            logger.debug("Submitting IO task to background...")
            future = self.client.compute(task)
            fire_and_forget(future)
        elif callable(task):
            # It's a function (e.g. MemoryWriter), execute it locally and synchronously
            # Submitting it to the cluster would serialize the Writer and lose the result.
            logger.info("DEBUG: Executing IO task locally")  # Uncomment for debug
            task()
        else:
            logger.warning(f"Unknown IO task type: {type(task)}. Executing synchronously.")
            super().process_io_task(task)

    def prepare_data(self, data: xr.Dataset) -> xr.Dataset:
        """Prepare data for distributed execution.

        Ensures that the dataset is chunked (lazy Dask arrays) and persists it
        to the cluster to start with "hot" data.

        Args:
            data: Dataset to prepare (state, forcings, etc.)

        Returns:
            Dataset persisted in distributed memory.
        """
        logger.info("Preparing dataset for distributed execution...")

        # 1. Ensure data is chunked (convert numpy to dask if needed)
        # If chunks were provided in setup(), xarray usually handles this before.
        # But we double-check here.
        if not any(hasattr(var.data, "__dask_graph__") for var in data.data_vars.values()):
            # Fallback: if no chunks are present, we apply a default chunking or warn
            # Ideally, the user should provide chunks in controller.setup(chunks={...})
            logger.warning(
                "Data does not seem to be chunked. "
                "DistributedBackend works best with dask-backed datasets."
            )

        # 2. Persist initial data to workers
        # This uploads local data to the cluster or triggers loading if lazy.
        data = data.persist()
        wait(data)
        logger.debug("Dataset persisted and ready in distributed memory.")
        return data

    def execute(
        self,
        task_groups: list[tuple[str, list[ComputeNode]]],
        state: xr.Dataset,
    ) -> dict[str, Any]:
        """Execute tasks by building a lazy Dask graph.

        Unlike the previous implementation, this does NOT trigger computation.
        It returns lazy Dask arrays that represent the result of the functions.

        Args:
            task_groups: List of (group_name, list_of_tasks) tuples
            state: Current simulation state (Lazy Dask Dataset)

        Returns:
            Dictionary of lazy results {variable_name: DataArray}
        """
        validate_has_chunks(state, "DistributedBackend")

        # We reuse the core logic. Since inputs (state) are Dask arrays,
        # outputs will automatically be Dask arrays (Lazy).
        # We pass previous results as context to handle dependencies between groups.
        all_results: dict[str, Any] = {}

        for _group_name, tasks in task_groups:
            # execute_task_sequence builds the graph by calling functions on dask arrays.
            # It updates all_results with new lazy arrays.
            group_results = execute_task_sequence(tasks, state, all_results)
            all_results.update(group_results)

        return all_results

    def stabilize_state(self, state: xr.Dataset) -> xr.Dataset:
        """Stabilize state by persisting it to the cluster.

        This is the CRITICAL step for iterative simulations with Dask.
        It "cuts" the lineage of the dask graph:
        1. Calls persist() -> triggers computation of the next state.
        2. Returns a new Dask object pointing to the results in RAM.
        3. Calls wait() -> ensures the driver doesn't run too far ahead (Backpressure).

        Args:
            state: State at t+dt (Lazy Dask object with deep history)

        Returns:
            State at t+dt (Persisted Dask object, fresh history)
        """
        # 1. Persist: Trigger computation and store in Distributed RAM
        # This returns a new dataset with a minimal graph (just "get from worker")
        logger.debug("Stabilizing state (Persist)...")
        new_state = state.persist()

        # 2. Backpressure: Wait for computation to finish
        # This prevents the Driver from generating thousands of future tasks
        # while the Cluster is still processing step 1.
        wait(new_state)

        # 3. Explicitly release memory of the old graph/futures is handled
        # by Python's GC when the old 'state' variable goes out of scope
        # in the Controller loop (self.state = new_state).

        return new_state
