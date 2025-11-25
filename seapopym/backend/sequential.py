"""Sequential execution backend."""

from collections.abc import Hashable
from typing import Any

import xarray as xr

from seapopym.backend.base import ComputeBackend
from seapopym.backend.exceptions import ExecutionError
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
            # Local context for the group (intermediate results)
            local_context: dict[Hashable, xr.DataArray] = {}

            for node in tasks:
                try:
                    # 1. Input resolution (Local > All Results > State)
                    inputs = {}
                    for arg_name, graph_var_name in node.input_mapping.items():
                        if graph_var_name in local_context:
                            inputs[arg_name] = local_context[graph_var_name]
                        elif graph_var_name in all_results:
                            inputs[arg_name] = all_results[graph_var_name]
                        elif graph_var_name in state:
                            inputs[arg_name] = state[graph_var_name]
                        else:
                            raise KeyError(
                                f"Variable '{graph_var_name}' not found in state, local context, or previous results "
                                f"(required by unit '{node.name}')."
                            )

                    # 2. Function execution
                    unit_output = node.func(**inputs)

                    if not isinstance(unit_output, dict):
                        raise TypeError(
                            f"Function '{node.name}' must return a dictionary, got {type(unit_output)}."
                        )

                    # 3. Output mapping
                    for key_retour, graph_var_name in node.output_mapping.items():
                        if key_retour not in unit_output:
                            raise KeyError(
                                f"Function '{node.name}' did not return expected key '{key_retour}'."
                            )

                        data = unit_output[key_retour]

                        if not isinstance(data, xr.DataArray):
                            raise TypeError(
                                f"Output '{key_retour}' from unit '{node.name}' must be a DataArray, "
                                f"got {type(data)}."
                            )

                        local_context[graph_var_name] = data
                        all_results[str(graph_var_name)] = data

                except Exception as e:
                    raise ExecutionError(f"Error executing unit '{node.name}': {str(e)}") from e

        return all_results
