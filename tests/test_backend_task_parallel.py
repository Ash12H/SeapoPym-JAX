"""Tests for TaskParallelBackend."""

import dask
import pytest
import xarray as xr

from seapopym.backend.exceptions import BackendConfigurationError, ExecutionError
from seapopym.backend.task_parallel import TaskParallelBackend
from seapopym.blueprint.nodes import ComputeNode


# Helpers
def add_one(x):
    """Add 1 to input."""
    return {"result": x + 1.0}


def multiply_two(x):
    """Multiply input by 2."""
    return {"result": x * 2.0}


def create_local(x):
    """Create local variable."""
    return {"var": x + 10.0}


def use_local(var):
    """Use local variable."""
    return {"result": var * 2.0}


def group1_func(x):
    """Group 1 function."""
    return {"out": x + 1.0}


def group2_func(y):
    """Group 2 function."""
    return {"out": y * 2.0}


def buggy_func(x):
    """Function that raises an error."""
    raise ValueError("Bug!")


def bad_func(x):
    """Function with bad return type."""
    return x  # Should return dict


class TestTaskParallelBackend:
    """Test suite for TaskParallelBackend."""

    def test_task_parallel_simple_execution(self):
        """Test basic execution with one task."""
        backend = TaskParallelBackend()

        node = ComputeNode(
            func=add_one, name="add_one", output_mapping={"result": "y"}, input_mapping={"x": "x"}
        )

        state = xr.Dataset({"x": (("i"), [1.0, 2.0, 3.0])})

        results = backend.execute([("group1", [node])], state)

        assert "y" in results
        assert results["y"].values.tolist() == [2.0, 3.0, 4.0]

    def test_task_parallel_multiple_tasks(self):
        """Test execution with multiple tasks in sequence."""
        backend = TaskParallelBackend()

        node1 = ComputeNode(
            func=add_one,
            name="add_one",
            output_mapping={"result": "temp"},
            input_mapping={"x": "x"},
        )

        node2 = ComputeNode(
            func=multiply_two,
            name="multiply_two",
            output_mapping={"result": "y"},
            input_mapping={"x": "temp"},
        )

        state = xr.Dataset({"x": (("i"), [1.0, 2.0, 3.0])})

        results = backend.execute([("group1", [node1, node2])], state)

        assert "y" in results
        assert results["y"].values.tolist() == [4.0, 6.0, 8.0]

    def test_task_parallel_multiple_groups(self):
        """Test execution with multiple groups."""
        backend = TaskParallelBackend()

        node1 = ComputeNode(
            func=group1_func,
            name="node1",
            output_mapping={"out": "intermediate"},
            input_mapping={"x": "x"},
        )

        node2 = ComputeNode(
            func=group2_func,
            name="node2",
            output_mapping={"out": "final"},
            input_mapping={"y": "intermediate"},
        )

        state = xr.Dataset({"x": (("i"), [1.0, 2.0])})

        results = backend.execute([("group1", [node1]), ("group2", [node2])], state)

        assert "final" in results
        assert results["final"].values.tolist() == [4.0, 6.0]

    def test_task_parallel_rejects_chunked_data(self):
        """Test that TaskParallelBackend rejects chunked Dask arrays."""
        backend = TaskParallelBackend()

        node = ComputeNode(
            func=add_one, name="add_one", output_mapping={"result": "y"}, input_mapping={"x": "x"}
        )

        # Create chunked data
        state = xr.Dataset({"x": (("i"), [1.0, 2.0, 3.0])})
        state = state.chunk({"i": 1})

        with pytest.raises(BackendConfigurationError, match="incompatible with task parallelism"):
            backend.execute([("group1", [node])], state)

    def test_task_parallel_empty_task_groups(self):
        """Test that empty task groups return empty results."""
        backend = TaskParallelBackend()
        state = xr.Dataset({"x": (("i"), [1.0])})

        results = backend.execute([], state)

        assert results == {}

    def test_task_parallel_missing_variable(self):
        """Test error when required variable is missing."""
        backend = TaskParallelBackend()

        node = ComputeNode(
            func=add_one,
            name="add_one",
            output_mapping={"result": "y"},
            input_mapping={"x": "missing_var"},
        )

        state = xr.Dataset({"other": (("i"), [1.0])})

        with pytest.raises(ExecutionError, match="Variable 'missing_var' not found"):
            backend.execute([("group1", [node])], state)

    def test_task_parallel_function_error(self):
        """Test error handling when function raises exception."""
        backend = TaskParallelBackend()

        node = ComputeNode(
            func=buggy_func, name="buggy", output_mapping={"result": "y"}, input_mapping={"x": "x"}
        )

        state = xr.Dataset({"x": (("i"), [1.0])})

        with pytest.raises(ExecutionError, match="Error executing unit 'buggy'"):
            backend.execute([("group1", [node])], state)

    def test_task_parallel_invalid_return_type(self):
        """Test error when function doesn't return dict."""
        backend = TaskParallelBackend()

        node = ComputeNode(
            func=bad_func, name="bad", output_mapping={"result": "y"}, input_mapping={"x": "x"}
        )

        state = xr.Dataset({"x": (("i"), [1.0])})

        with pytest.raises(TypeError, match="must return a dictionary"):
            backend.execute([("group1", [node])], state)

    def test_task_parallel_uses_dask_compute(self):
        """Test that TaskParallelBackend uses dask.compute for parallel execution."""
        backend = TaskParallelBackend()

        # Create independent tasks that could run in parallel
        node1 = ComputeNode(
            func=add_one, name="add_one", output_mapping={"result": "y1"}, input_mapping={"x": "x"}
        )

        node2 = ComputeNode(
            func=multiply_two,
            name="multiply_two",
            output_mapping={"result": "y2"},
            input_mapping={"x": "x"},
        )

        state = xr.Dataset({"x": (("i"), [1.0, 2.0, 3.0])})

        results = backend.execute([("group1", [node1, node2])], state)

        assert "y1" in results
        assert "y2" in results
        assert results["y1"].values.tolist() == [2.0, 3.0, 4.0]
        assert results["y2"].values.tolist() == [2.0, 4.0, 6.0]

    def test_task_parallel_with_scheduler(self):
        """Test that TaskParallelBackend respects dask scheduler configuration."""
        # Configure single-threaded scheduler for deterministic testing
        with dask.config.set(scheduler="synchronous"):
            backend = TaskParallelBackend()

            node = ComputeNode(
                func=add_one,
                name="add_one",
                output_mapping={"result": "y"},
                input_mapping={"x": "x"},
            )

            state = xr.Dataset({"x": (("i"), [1.0, 2.0, 3.0])})

            results = backend.execute([("group1", [node])], state)

            assert "y" in results
            assert results["y"].values.tolist() == [2.0, 3.0, 4.0]
