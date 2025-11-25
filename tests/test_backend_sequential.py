"""Tests for SequentialBackend."""

import pytest
import xarray as xr

from seapopym.backend import SequentialBackend
from seapopym.backend.exceptions import ExecutionError
from seapopym.blueprint.nodes import ComputeNode


def test_sequential_backend_simple_execution():
    """Test basic execution with one task."""
    backend = SequentialBackend()

    def add_one(x):
        return {"result": x + 1.0}

    node = ComputeNode(
        func=add_one, name="add_one", output_mapping={"result": "y"}, input_mapping={"x": "x"}
    )

    state = xr.Dataset({"x": (("i"), [1.0, 2.0, 3.0])})

    results = backend.execute([("group1", [node])], state)

    assert "y" in results
    assert results["y"].values.tolist() == [2.0, 3.0, 4.0]


def test_sequential_backend_multiple_tasks():
    """Test execution with multiple tasks in sequence."""
    backend = SequentialBackend()

    def add_one(x):
        return {"result": x + 1.0}

    def multiply_two(x):
        return {"result": x * 2.0}

    node1 = ComputeNode(
        func=add_one, name="add_one", output_mapping={"result": "temp"}, input_mapping={"x": "x"}
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


def test_sequential_backend_local_context():
    """Test that local context is used before global state."""
    backend = SequentialBackend()

    def create_local(x):
        return {"var": x + 10.0}

    def use_local(var):
        return {"result": var * 2.0}

    node1 = ComputeNode(
        func=create_local,
        name="create_local",
        output_mapping={"var": "var"},
        input_mapping={"x": "x"},
    )

    node2 = ComputeNode(
        func=use_local,
        name="use_local",
        output_mapping={"result": "y"},
        input_mapping={"var": "var"},
    )

    # State has var=0, but local context should override
    state = xr.Dataset({"x": (("i"), [1.0]), "var": (("i"), [0.0])})

    results = backend.execute([("group1", [node1, node2])], state)

    # Should use local var (11.0) not state var (0.0)
    assert results["y"].values[0] == 22.0  # (1.0 + 10.0) * 2.0


def test_sequential_backend_missing_variable():
    """Test error when required variable is missing."""
    backend = SequentialBackend()

    def dummy(x):
        return {"result": x}

    node = ComputeNode(
        func=dummy, name="dummy", output_mapping={"result": "y"}, input_mapping={"x": "missing_var"}
    )

    state = xr.Dataset({"other": (("i"), [1.0])})

    with pytest.raises(ExecutionError, match="Variable 'missing_var' not found"):
        backend.execute([("group1", [node])], state)


def test_sequential_backend_function_error():
    """Test error handling when function raises exception."""
    backend = SequentialBackend()

    def buggy_func(x):
        raise ValueError("Intentional error")

    node = ComputeNode(
        func=buggy_func, name="buggy", output_mapping={"result": "y"}, input_mapping={"x": "x"}
    )

    state = xr.Dataset({"x": (("i"), [1.0])})

    with pytest.raises(ExecutionError, match="Error executing unit 'buggy'"):
        backend.execute([("group1", [node])], state)


def test_sequential_backend_invalid_return_type():
    """Test error when function doesn't return dict."""
    backend = SequentialBackend()

    def bad_func(x):
        return x  # Should return dict

    node = ComputeNode(
        func=bad_func, name="bad", output_mapping={"result": "y"}, input_mapping={"x": "x"}
    )

    state = xr.Dataset({"x": (("i"), [1.0])})

    with pytest.raises(ExecutionError, match="must return a dictionary"):
        backend.execute([("group1", [node])], state)


def test_sequential_backend_empty_task_groups():
    """Test that empty task groups return empty results."""
    backend = SequentialBackend()
    state = xr.Dataset({"x": (("i"), [1.0])})

    results = backend.execute([], state)

    assert results == {}


def test_sequential_backend_multiple_groups():
    """Test execution with multiple groups."""
    backend = SequentialBackend()

    def group1_func(x):
        return {"out": x + 1.0}

    def group2_func(y):
        return {"out": y * 2.0}

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
