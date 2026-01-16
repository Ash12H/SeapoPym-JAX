"""Monitoring backend for profiling execution time and memory usage."""

import logging
import time
from collections import defaultdict
from typing import Any

import xarray as xr

from seapopym.backend.base import ComputeBackend
from seapopym.backend.exceptions import ExecutionError
from seapopym.blueprint.nodes import ComputeNode

logger = logging.getLogger(__name__)


class MonitoringBackend(ComputeBackend):
    """Backend for monitoring execution time of tasks and groups.

    This backend extends SequentialBackend to collect detailed timing information
    about each compute node and group execution. It provides insights into:
    - Time spent in each function/node
    - Time spent in each group
    - Cumulative statistics across multiple executions

    Key characteristics:
    - Sequential execution (no parallelism)
    - Eager computation (materializes all Dask arrays)
    - Detailed timing collection per node and group
    - Minimal memory overhead

    Typical use cases:
    - Profiling and performance analysis
    - Identifying bottlenecks in computation
    - Comparing execution time across different functions
    - Understanding time distribution across model components

    Example:
        >>> backend = MonitoringBackend()
        >>> controller = SimulationController(config, backend=backend)
        >>> controller.setup(configure_model, initial_state=state)
        >>> controller.run()
        >>> stats = backend.get_statistics()
        >>> print(stats['by_group'])
    """

    def __init__(self) -> None:
        """Initialize MonitoringBackend with empty statistics."""
        logger.info("MonitoringBackend initialized (sequential execution with profiling)")
        self._reset_statistics()

    def _reset_statistics(self) -> None:
        """Reset all collected statistics."""
        # Per-node statistics: {node_name: {'count': int, 'total_time': float, 'times': list[float]}}
        self._node_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "total_time": 0.0, "times": []}
        )

        # Per-group statistics: {group_name: {'count': int, 'total_time': float, 'times': list[float]}}
        self._group_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "total_time": 0.0, "times": []}
        )

        # Timestep statistics: list of {'timestep': int, 'total_time': float, 'groups': dict}
        self._timestep_stats: list[dict[str, Any]] = []

        self._current_timestep = 0

    def prepare_data(self, data: xr.Dataset) -> xr.Dataset:
        """Prepare data by materializing all lazy arrays.

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
        """Execute tasks sequentially while collecting timing statistics.

        Args:
            task_groups: List of (group_name, list_of_tasks) tuples
            state: Current simulation state

        Returns:
            Dictionary of materialized results {variable_name: value}
        """
        # Materialize state before execution
        state = self._materialize_dataset(state)

        all_results: dict[str, Any] = {}
        timestep_start = time.perf_counter()
        timestep_groups: dict[str, float] = {}

        for group_name, tasks in task_groups:
            logger.debug(f"Executing group '{group_name}' with {len(tasks)} tasks")

            group_start = time.perf_counter()

            # Execute with node-level monitoring
            group_results = self._execute_monitored_sequence(tasks, state, all_results, group_name)

            group_time = time.perf_counter() - group_start

            # Update group statistics
            self._group_stats[group_name]["count"] += 1
            self._group_stats[group_name]["total_time"] += group_time
            self._group_stats[group_name]["times"].append(group_time)

            timestep_groups[group_name] = group_time

            # Materialize intermediate results
            group_results = self._materialize_results(group_results)
            all_results.update(group_results)

        timestep_time = time.perf_counter() - timestep_start

        # Record timestep statistics
        self._timestep_stats.append(
            {
                "timestep": self._current_timestep,
                "total_time": timestep_time,
                "groups": timestep_groups,
            }
        )
        self._current_timestep += 1

        logger.debug(
            f"MonitoringBackend execution complete. {len(all_results)} variables produced. "
            f"Total time: {timestep_time:.6f}s"
        )

        return all_results

    def _execute_monitored_sequence(
        self,
        tasks: list[ComputeNode],
        state: xr.Dataset,
        external_context: dict[str, Any],
        group_name: str,
    ) -> dict[str, Any]:
        """Execute a sequence of tasks with per-node timing.

        This is a modified version of execute_task_sequence that collects
        timing information for each node.

        Args:
            tasks: List of tasks to execute
            state: Global state (read-only)
            external_context: Results from other groups/tasks (dependencies)
            group_name: Name of the current group (for logging)

        Returns:
            Dictionary of results produced by this sequence
        """
        results: dict[str, Any] = {}
        local_context: dict[Any, xr.DataArray] = {}

        for node in tasks:
            try:
                # 1. Input resolution (Local > External > State)
                inputs = {}
                for arg_name, graph_var_name in node.input_mapping.items():
                    if graph_var_name in local_context:
                        inputs[arg_name] = local_context[graph_var_name]
                    elif graph_var_name in external_context:
                        inputs[arg_name] = external_context[graph_var_name]
                    elif graph_var_name in state:
                        inputs[arg_name] = state[graph_var_name]
                    else:
                        raise KeyError(
                            f"Variable '{graph_var_name}' not found in state, local context, or previous results "
                            f"(required by unit '{node.name}')."
                        )

                # 2. Function execution with timing
                node_start = time.perf_counter()
                unit_output = node.func(**inputs)
                node_time = time.perf_counter() - node_start

                # Update node statistics
                self._node_stats[node.name]["count"] += 1
                self._node_stats[node.name]["total_time"] += node_time
                self._node_stats[node.name]["times"].append(node_time)
                self._node_stats[node.name]["group"] = group_name

                logger.debug(f"  Node '{node.name}' executed in {node_time:.6f}s")

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
                    results[str(graph_var_name)] = data

            except Exception as e:
                raise ExecutionError(f"Error executing unit '{node.name}': {str(e)}") from e

        return results

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

    def get_statistics(self) -> dict[str, Any]:
        """Get collected monitoring statistics.

        Returns:
            Dictionary containing:
            - 'by_node': Per-node statistics with mean, std, min, max times
            - 'by_group': Per-group statistics with mean, std, min, max times
            - 'by_timestep': Per-timestep breakdown
            - 'summary': Overall summary statistics
        """
        import numpy as np

        # Compute per-node statistics
        node_summary = {}
        for node_name, stats in self._node_stats.items():
            times = np.array(stats["times"])
            node_summary[node_name] = {
                "group": stats.get("group", "Unknown"),
                "count": stats["count"],
                "total_time": stats["total_time"],
                "mean_time": np.mean(times) if len(times) > 0 else 0.0,
                "std_time": np.std(times) if len(times) > 0 else 0.0,
                "min_time": np.min(times) if len(times) > 0 else 0.0,
                "max_time": np.max(times) if len(times) > 0 else 0.0,
            }

        # Compute per-group statistics
        group_summary = {}
        for group_name, stats in self._group_stats.items():
            times = np.array(stats["times"])
            group_summary[group_name] = {
                "count": stats["count"],
                "total_time": stats["total_time"],
                "mean_time": np.mean(times) if len(times) > 0 else 0.0,
                "std_time": np.std(times) if len(times) > 0 else 0.0,
                "min_time": np.min(times) if len(times) > 0 else 0.0,
                "max_time": np.max(times) if len(times) > 0 else 0.0,
            }

        # Compute overall summary
        total_time = sum(stats["total_time"] for stats in self._group_stats.values())
        summary = {
            "total_execution_time": total_time,
            "num_timesteps": self._current_timestep,
            "num_nodes": len(self._node_stats),
            "num_groups": len(self._group_stats),
            "mean_timestep_time": total_time / self._current_timestep
            if self._current_timestep > 0
            else 0.0,
        }

        return {
            "by_node": node_summary,
            "by_group": group_summary,
            "by_timestep": self._timestep_stats,
            "summary": summary,
        }

    def reset_statistics(self) -> None:
        """Reset all collected statistics.

        Useful when running multiple benchmarks with the same backend instance.
        """
        logger.info("Resetting monitoring statistics")
        self._reset_statistics()

    def print_summary(self, top_n: int = 10) -> None:
        """Print a formatted summary of collected statistics.

        Args:
            top_n: Number of top time-consuming nodes to display
        """
        stats = self.get_statistics()

        print("\n" + "=" * 80)
        print("MONITORING SUMMARY")
        print("=" * 80)

        # Overall summary
        summary = stats["summary"]
        print(f"\nTotal execution time: {summary['total_execution_time']:.3f}s")
        print(f"Number of timesteps: {summary['num_timesteps']}")
        print(f"Number of nodes: {summary['num_nodes']}")
        print(f"Number of groups: {summary['num_groups']}")
        print(f"Mean timestep time: {summary['mean_timestep_time']:.3f}s")

        # Group summary
        print("\n" + "-" * 80)
        print("TIME BY GROUP")
        print("-" * 80)
        print(f"{'Group':<30} {'Total Time':<15} {'Mean Time':<15} {'Percentage':<15}")
        print("-" * 80)

        total_time = summary["total_execution_time"]
        for group_name, group_stats in sorted(
            stats["by_group"].items(), key=lambda x: x[1]["total_time"], reverse=True
        ):
            percentage = (group_stats["total_time"] / total_time * 100) if total_time > 0 else 0
            print(
                f"{group_name:<30} {group_stats['total_time']:<15.3f} "
                f"{group_stats['mean_time']:<15.6f} {percentage:<15.1f}%"
            )

        # Top nodes
        print("\n" + "-" * 80)
        print(f"TOP {top_n} TIME-CONSUMING NODES")
        print("-" * 80)
        print(f"{'Node':<40} {'Group':<20} {'Total Time':<15} {'Mean Time':<15} {'Percentage':<15}")
        print("-" * 80)

        sorted_nodes = sorted(
            stats["by_node"].items(), key=lambda x: x[1]["total_time"], reverse=True
        )[:top_n]

        for node_name, node_stats in sorted_nodes:
            percentage = (node_stats["total_time"] / total_time * 100) if total_time > 0 else 0
            short_name = node_name.split("/")[-1][:39]  # Truncate long names
            print(
                f"{short_name:<40} {node_stats['group']:<20} {node_stats['total_time']:<15.3f} "
                f"{node_stats['mean_time']:<15.6f} {percentage:<15.1f}%"
            )

        print("=" * 80 + "\n")
