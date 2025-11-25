"""Dask execution backend."""

from collections.abc import Callable
from typing import Any

import dask
import xarray as xr

from seapopym.backend.base import ComputeBackend
from seapopym.backend.exceptions import ExecutionError
from seapopym.blueprint.nodes import ComputeNode


def _validate_and_execute(
    func: Callable, node_name: str, output_mapping: dict, **kwargs: Any
) -> dict:
    """Execute function and validate output."""
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


class DaskBackend(ComputeBackend):
    """Distributed execution backend using Dask with fine-grained task graph."""

    def execute(
        self,
        task_groups: list[tuple[str, list[ComputeNode]]],
        state: xr.Dataset,
    ) -> dict[str, Any]:
        """Execute tasks using dask.delayed building a full dependency graph.

        This implementation ignores the grouping structure for execution purposes,
        treating all tasks as a flat list ordered topologically.
        It builds a fine-grained Dask graph where each task depends only on its specific inputs.
        """
        # Registry of available variables (Delayed objects)
        # Key: graph_variable_name, Value: dask delayed object
        registry: dict[str, Any] = {}

        # Flatten the task list (topological order is preserved)
        all_tasks = [task for _, tasks in task_groups for task in tasks]

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

            except Exception as e:
                raise ExecutionError(
                    f"Error building graph for unit '{task.name}': {str(e)}"
                ) from e

        # 4. Compute all results
        if not registry:
            return {}

        # We compute all registered variables
        # Note: dask.compute optimizes the graph and runs tasks in parallel
        keys = list(registry.keys())
        values = list(registry.values())

        computed_values = dask.compute(*values)

        # Reconstruct the result dictionary
        return dict(zip(keys, computed_values, strict=False))
