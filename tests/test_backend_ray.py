"""Tests for RayBackend."""

import pytest
import xarray as xr

try:
    import ray
except ImportError:
    ray = None

# Import helper functions from separate module to ensure Ray can serialize them
from backend_test_helpers import add_one, group1_func, group2_func, multiply_two
from seapopym.backend.ray import RayBackend
from seapopym.blueprint.nodes import ComputeNode


@pytest.mark.skipif(ray is None, reason="Ray is not installed")
class TestRayBackend:
    @classmethod
    def setup_class(cls):
        if not ray.is_initialized():
            ray.init()

    @classmethod
    def teardown_class(cls):
        ray.shutdown()

    def test_ray_backend_simple_execution(self):
        """Test basic execution with one task."""
        backend = RayBackend()

        node = ComputeNode(
            func=add_one, name="add_one", output_mapping={"result": "y"}, input_mapping={"x": "x"}
        )

        state = xr.Dataset({"x": (("i"), [1.0, 2.0, 3.0])})

        results = backend.execute([("group1", [node])], state)

        assert "y" in results
        assert results["y"].values.tolist() == [2.0, 3.0, 4.0]

    def test_ray_backend_multiple_tasks(self):
        """Test execution with multiple tasks in sequence."""
        backend = RayBackend()

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

    def test_ray_backend_multiple_groups(self):
        """Test execution with multiple groups (dependency)."""
        backend = RayBackend()

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

    def test_ray_backend_empty_task_groups(self):
        """Test that empty task groups return empty results."""
        backend = RayBackend()
        state = xr.Dataset({"x": (("i"), [1.0])})

        results = backend.execute([], state)

        assert results == {}
