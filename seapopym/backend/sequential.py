"""Sequential execution backend."""

import logging
from typing import Any

import xarray as xr

from seapopym.backend.base import ComputeBackend
from seapopym.backend.core import execute_task_sequence
from seapopym.blueprint.nodes import ComputeNode

logger = logging.getLogger(__name__)


class SequentialBackend(ComputeBackend):
    """Pure sequential execution with eager computation (no parallelism).

    This backend executes tasks strictly sequentially and materializes all lazy
    data (Dask arrays) immediately. It provides fully deterministic, single-threaded
    execution useful for debugging and testing.

    Key characteristics:
    - No parallelism (neither task nor data)
    - Eager computation (materializes all Dask arrays)
    - Fully deterministic execution
    - Minimal memory overhead (no graph accumulation)

    Typical use cases:
    - Debugging and development
    - Small-scale simulations that fit in memory
    - Unit testing with deterministic results
    - Baseline performance measurements

    Example:
        >>> backend = SequentialBackend()
        >>> controller = SimulationController(config, backend=backend)
        >>> controller.setup(
        ...     configure_model,
        ...     initial_state=state  # Will be materialized immediately
        ... )
    """

    def __init__(self) -> None:
        """Initialize SequentialBackend."""
        logger.info("SequentialBackend initialized (eager computation, no parallelism)")

    def prepare_data(self, data: xr.Dataset) -> xr.Dataset:
        """Prepare data by materializing all lazy arrays.

        Consistent with SequentialBackend's philosophy of eager computation,
        this ensures all data is materialized into memory before use.

        Args:
            data: Dataset to prepare (state, forcings, etc.)

        Returns:
            Dataset with all arrays materialized (Numpy arrays).
        """
        return self._materialize_dataset(data)

    def execute(
        self,
        task_groups: list[tuple[str, list[ComputeNode]]],
        state: xr.Dataset,
    ) -> dict[str, Any]:
        """Execute tasks sequentially with eager computation.

        Materializes the state and all intermediate results to ensure
        pure sequential execution without any lazy evaluation.

        Args:
            task_groups: List of (group_name, list_of_tasks) tuples
            state: Current simulation state

        Returns:
            Dictionary of materialized results {variable_name: value}
        """
        # Materialize state before execution to ensure eager computation
        state = self._materialize_dataset(state)

        all_results: dict[str, Any] = {}

        for group_name, tasks in task_groups:
            logger.debug(f"Executing group '{group_name}' with {len(tasks)} tasks")

            # Execute tasks for this group
            # We pass all_results as external_context because in sequential mode,
            # all previous results are available.
            group_results = execute_task_sequence(tasks, state, all_results)

            # Materialize intermediate results to prevent lazy evaluation
            group_results = self._materialize_results(group_results)
            all_results.update(group_results)

        logger.debug(
            f"SequentialBackend execution complete. {len(all_results)} variables produced."
        )
        return all_results

    def _materialize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Force computation of all lazy arrays in dataset.

        Args:
            ds: Dataset potentially containing Dask arrays

        Returns:
            Dataset with all arrays materialized (Numpy arrays)
        """
        if hasattr(ds, "compute"):
            logger.debug("Materializing dataset (calling .compute())")
            return ds.compute()
        return ds

    def _materialize_results(self, results: dict[str, Any]) -> dict[str, Any]:
        """Force computation of all lazy arrays in results dictionary.

        Args:
            results: Dictionary of results potentially containing Dask arrays

        Returns:
            Dictionary with all arrays materialized
        """
        materialized = {}
        for key, value in results.items():
            if hasattr(value, "compute"):
                logger.debug(f"Materializing variable '{key}'")
                materialized[key] = value.compute()
            else:
                materialized[key] = value
        return materialized
