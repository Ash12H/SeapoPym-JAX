"""Tests for DaskBackend."""

import pytest
import xarray as xr

from seapopym.backend.dask import DaskBackend
from seapopym.blueprint.nodes import ComputeNode


# Helpers
def add_one(x):
    return {"result": x + 1.0}


def multiply_two(x):
    return {"result": x * 2.0}


def group1_func(x):
    return {"out": x + 1.0}


def group2_func(y):
    return {"out": y * 2.0}


class TestDaskBackend:
    def test_dask_backend_simple_execution(self):
        """Test basic execution with one task."""
        backend = DaskBackend()

        node = ComputeNode(
            func=add_one, name="add_one", output_mapping={"result": "y"}, input_mapping={"x": "x"}
        )

        state = xr.Dataset({"x": (("i"), [1.0, 2.0, 3.0])})

        results = backend.execute([("group1", [node])], state)

        assert "y" in results
        assert results["y"].values.tolist() == [2.0, 3.0, 4.0]

    def test_dask_backend_multiple_tasks(self):
        """Test execution with multiple tasks in sequence."""
        backend = DaskBackend()

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

    def test_dask_backend_multiple_groups(self):
        """Test execution with multiple groups (dependency)."""
        backend = DaskBackend()

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

    def test_dask_backend_empty_task_groups(self):
        """Test that empty task groups return empty results."""
        backend = DaskBackend()
        state = xr.Dataset({"x": (("i"), [1.0])})

        results = backend.execute([], state)

        assert results == {}

    def test_dask_backend_invalid_return_type(self):
        """Test error when function doesn't return dict."""
        backend = DaskBackend()

        def bad_func(x):
            return x  # Should return dict

        node = ComputeNode(
            func=bad_func, name="bad", output_mapping={"result": "y"}, input_mapping={"x": "x"}
        )

        state = xr.Dataset({"x": (("i"), [1.0])})

        with pytest.raises(TypeError, match="must return a dictionary"):
            backend.execute([("group1", [node])], state)
