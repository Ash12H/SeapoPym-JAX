"""Data parallelism backend using Dask array chunking."""

import logging
from typing import Any

import xarray as xr

from seapopym.backend.base import ComputeBackend
from seapopym.backend.core import execute_task_sequence
from seapopym.backend.validation import validate_has_chunks
from seapopym.blueprint.nodes import ComputeNode

logger = logging.getLogger(__name__)


class DataParallelBackend(ComputeBackend):
    """Data parallelism via Dask array chunking (intra-task parallelism).

    This backend is optimized for parallel execution within individual tasks by operating
    on chunked Dask arrays. It preserves lazy evaluation throughout the computation,
    allowing Dask to automatically parallelize operations across chunks.

    Key characteristics:
    - Executes tasks sequentially but processes data chunks in parallel
    - Requires chunked Dask arrays in the state
    - No materialization of intermediate results (unless persist enabled)
    - Uses `xr.apply_ufunc(..., dask="parallelized")` for parallel chunk processing

    Typical use cases:
    - Transport operations with large spatial grids
    - Processing many independent cohorts/layers/ensembles
    - Out-of-core computation (data larger than RAM)

    Performance:
    - Speedup scales with number of chunks (up to available workers)
    - Example: 50 cohortes with chunks={"cohort": 1} → up to 50× speedup
    - Best when individual chunks fit in memory but full dataset doesn't

    Args:
        None

    Example:
        >>> backend = DataParallelBackend()
        >>> controller = SimulationController(config, backend=backend)
        >>> controller.setup(
        ...     configure_model,
        ...     initial_state=state,
        ...     chunks={"cohort": 1}  # Chunk by cohort for parallel transport
        ... )
    """

    def __init__(self) -> None:
        """Initialize DataParallelBackend."""
        logger.info("DataParallelBackend initialized (auto-persist enabled)")

    def prepare_data(self, data: xr.Dataset) -> xr.Dataset:
        """Prepare data by persisting to distributed memory.

        This prevents large numpy arrays or lazy Dask graphs from being embedded
        into task graphs, which would cause huge graph sizes and slow serialization.

        Args:
            data: Dataset to prepare (state, forcings, etc.)

        Returns:
            Dataset with arrays persisted to distributed memory.
        """
        if hasattr(data, "persist"):
            logger.info("Persisting dataset to distributed memory (cutting graph lineage)")
            persisted = data.persist()

            # CRITICAL: Wait for persist to complete before returning
            # Without this, the data might still be lazy when first used,
            # causing it to be embedded in the task graph
            from dask.distributed import wait

            # Collect all dask arrays from the dataset (data_vars AND coords)
            futures = []
            for var in persisted.data_vars.values():
                if hasattr(var.data, "__dask_graph__"):
                    futures.append(var.data)
            for coord in persisted.coords.values():
                if hasattr(coord.data, "__dask_graph__"):
                    futures.append(coord.data)

            if futures:
                logger.debug(f"Waiting for {len(futures)} arrays to be persisted...")
                wait(futures)
                logger.debug("Persist complete")

            return persisted
        return data

    def execute(
        self,
        task_groups: list[tuple[str, list[ComputeNode]]],
        state: xr.Dataset,
    ) -> dict[str, Any]:
        """Execute tasks sequentially while preserving chunked arrays for parallel processing.

        This method validates that the state contains chunked data, then executes tasks
        sequentially. However, the actual computation within each task is parallelized
        automatically by Dask when operations are applied to chunked arrays.

        Crucially, this backend applies .persist() after each task group. This forces
        the computation of intermediate results into distributed memory, cutting the
        task graph lineage. This is essential for iterative simulations to prevent
        exponential graph growth.

        Args:
            task_groups: List of (group_name, list_of_tasks) tuples
            state: Current simulation state (should contain chunked Dask arrays)

        Returns:
            Dictionary of results {variable_name: DataArray} (persisted)

        Warnings:
            Issues a warning if no chunked data is detected, suggesting alternative backends.
        """
        # Validation: warn if no chunked data found
        validate_has_chunks(state, "DataParallelBackend")

        all_results: dict[str, Any] = {}

        for group_name, tasks in task_groups:
            logger.debug(f"Executing group '{group_name}' with {len(tasks)} tasks")

            # Execute tasks sequentially but preserve lazy evaluation
            # Dask will parallelize operations within each task automatically
            group_results = execute_task_sequence(tasks, state, all_results)

            # Always persist intermediate results to prevent graph explosion
            # This keeps data in distributed memory (workers) and cuts the lineage
            logger.debug(f"Persisting {len(group_results)} results from group '{group_name}'")
            group_results = self._persist_results(group_results)

            all_results.update(group_results)

        logger.info(
            f"DataParallelBackend execution complete. {len(all_results)} variables produced."
        )
        return all_results

    def stabilize_state(self, state: xr.Dataset) -> xr.Dataset:
        """Stabilize the state by persisting it to distributed memory.

        This cuts the Dask dependency graph lineage, which is critical for iterative
        simulations to prevent linear growth of the task graph size.

        Args:
            state: The current simulation state (lazy Dask arrays).

        Returns:
            The stabilized (persisted) state.
        """
        # We persist the whole dataset. Dask is smart enough to only compute
        # what hasn't been computed yet (e.g., static variables won't be recomputed).
        logger.debug("Stabilizing state: Persisting full state to cut graph lineage.")
        return state.persist()

    def _persist_results(self, results: dict[str, Any]) -> dict[str, Any]:
        """Persist lazy Dask arrays to distributed memory.

        This forces computation of the current results and stores them in the
        distributed memory of Dask workers, preventing graph accumulation but
        increasing memory usage.

        Args:
            results: Dictionary of results to persist

        Returns:
            Dictionary with persisted results
        """
        persisted = {}
        for key, value in results.items():
            if hasattr(value, "persist"):
                # Persist Dask arrays/datasets
                persisted[key] = value.persist()
                logger.debug(f"Persisted variable '{key}'")
            else:
                # Keep non-Dask objects as-is
                persisted[key] = value

        return persisted
