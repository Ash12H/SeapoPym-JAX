"""Task parallelism backend using dask.delayed."""

import logging
from collections.abc import Callable
from typing import Any

import dask
import xarray as xr

from seapopym.backend.base import ComputeBackend
from seapopym.backend.exceptions import ExecutionError
from seapopym.backend.validation import validate_no_chunks
from seapopym.blueprint.nodes import ComputeNode

logger = logging.getLogger(__name__)


def _validate_and_execute(
    func: Callable, node_name: str, output_mapping: dict, **kwargs: Any
) -> dict:
    """Execute function and validate output.

    Args:
        func: Function to execute
        node_name: Name of the compute node (for error messages)
        output_mapping: Expected output keys
        **kwargs: Arguments to pass to func

    Returns:
        Dictionary of outputs from func

    Raises:
        ExecutionError: If function execution fails or output is invalid
    """
    try:
        result = func(**kwargs)
    except Exception as e:
        raise ExecutionError(f"Error executing unit '{node_name}': {str(e)}") from e

    if not isinstance(result, dict):
        raise TypeError(f"Function '{node_name}' must return a dictionary, got {type(result)}.")

    for key in output_mapping:
        if key not in result:
            raise KeyError(f"Function '{node_name}' did not return expected key '{key}'.")

        if not isinstance(result[key], xr.DataArray):
            raise TypeError(f"Output '{key}' from unit '{node_name}' must be a DataArray.")

    return result


class TaskParallelBackend(ComputeBackend):
    """Task parallelism via dask.delayed (inter-task parallelism).

    This backend parallelizes independent tasks in the computation DAG using dask.delayed.
    It builds a task dependency graph where each task depends only on its specific inputs,
    then executes all tasks in parallel using Dask's scheduler.

    Key characteristics:
    - Parallelizes independent tasks across the DAG
    - Materializes all inputs (incompatible with chunked Dask arrays)
    - Builds a fine-grained task graph with explicit dependencies
    - Suitable for models with multiple independent functional groups

    Limitations:
    - INCOMPATIBLE with chunked Dask arrays (data parallelism)
    - All data must fit in memory (no out-of-core computation)
    - Speedup limited by number of independent tasks in DAG

    Performance:
    - Speedup scales with number of independent tasks
    - Example: Model with 12 independent functional groups → up to 12× speedup
    - Limited by longest sequential chain in DAG (Amdahl's Law)

    Typical use cases:
    - Multi-species models with independent groups
    - Models where biological processes dominate (not transport)
    - Workstations with multiple cores but limited RAM

    Example:
        >>> backend = TaskParallelBackend()
        >>> controller = SimulationController(config, backend=backend)
        >>> controller.setup(
        ...     configure_model,
        ...     initial_state=state  # Must NOT contain chunked arrays
        ... )

    Raises:
        BackendConfigurationError: If chunked Dask arrays are detected in state
    """

    def __init__(self) -> None:
        """Initialize TaskParallelBackend."""
        logger.info("TaskParallelBackend initialized")

    def execute(
        self,
        task_groups: list[tuple[str, list[ComputeNode]]],
        state: xr.Dataset,
    ) -> dict[str, Any]:
        """Execute tasks using dask.delayed building a full dependency graph.

        This implementation ignores the grouping structure for execution purposes,
        treating all tasks as a flat list ordered topologically. It builds a
        fine-grained Dask graph where each task depends only on its specific inputs.

        Args:
            task_groups: List of (group_name, list_of_tasks) tuples
            state: Current simulation state (must NOT contain chunked arrays)

        Returns:
            Dictionary of computed results {variable_name: value}

        Raises:
            BackendConfigurationError: If state contains chunked Dask arrays
            ExecutionError: If task execution or graph building fails
        """
        # Validation: reject chunked data
        validate_no_chunks(state, "TaskParallelBackend")

        # Registry of available variables (Delayed objects)
        # Key: graph_variable_name, Value: dask delayed object
        registry: dict[str, Any] = {}

        # Flatten the task list (topological order is preserved)
        all_tasks = [task for _, tasks in task_groups for task in tasks]
        logger.info(f"Building task graph with {len(all_tasks)} tasks")

        for task in all_tasks:
            try:
                # 1. Input resolution
                inputs = {}
                for arg_name, graph_var_name in task.input_mapping.items():
                    if graph_var_name in registry:
                        # Use the delayed object from previous computation
                        inputs[arg_name] = registry[graph_var_name]
                    elif graph_var_name in state:
                        # Use variable from state
                        inputs[arg_name] = state[graph_var_name]
                    else:
                        raise KeyError(
                            f"Variable '{graph_var_name}' not found in state or previous results "
                            f"(required by unit '{task.name}')."
                        )

                # 2. Create delayed task with validation
                # We wrap the execution to ensure output validity
                task_output = dask.delayed(_validate_and_execute)(
                    func=task.func,
                    node_name=task.name,
                    output_mapping=task.output_mapping,
                    **inputs,
                )

                # 3. Register outputs
                for key_retour, graph_var_name in task.output_mapping.items():
                    # Extract specific variable from the result dict (lazy operation)
                    # This creates a new delayed node that depends on task_output
                    var_delayed = task_output[key_retour]
                    registry[graph_var_name] = var_delayed
                    logger.debug(
                        f"Registered delayed output '{graph_var_name}' from task '{task.name}'"
                    )

            except Exception as e:
                raise ExecutionError(
                    f"Error building graph for unit '{task.name}': {str(e)}"
                ) from e

        # 4. Compute all results
        if not registry:
            logger.warning("No tasks to execute (empty registry)")
            return {}

        # We compute all registered variables
        # Note: dask.compute optimizes the graph and runs tasks in parallel
        keys = list(registry.keys())
        values = list(registry.values())

        logger.info(f"Computing {len(values)} delayed variables in parallel...")
        computed_values = dask.compute(*values)

        # Reconstruct the result dictionary
        result = dict(zip(keys, computed_values, strict=False))
        logger.info(f"TaskParallelBackend execution complete. {len(result)} variables computed.")
        return result
