"""Tests for DistributedBackend."""

import pytest
import xarray as xr
from dask.distributed import Client as DaskClient

from seapopym.backend.distributed import DistributedBackend
from seapopym.backend.exceptions import ExecutionError
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


@pytest.fixture(scope="module")
def dask_client():
    client = DaskClient(processes=False)  # Use processes=False for lighter tests
    yield client
    client.close()


class TestDistributedBackend:
    def test_distributed_backend_simple_execution(self, dask_client):
        """Test basic execution with one task."""
        backend = DistributedBackend()

        node = ComputeNode(
            func=add_one, name="add_one", output_mapping={"result": "y"}, input_mapping={"x": "x"}
        )

        # State must be chunked for valid execution in DistributedBackend
        state = xr.Dataset({"x": (("i"), [1.0, 2.0, 3.0])}).chunk({"i": 1})

        results = backend.execute([("group1", [node])], state)

        assert "y" in results
        # Result is a dask array, compute to check value
        assert results["y"].compute().values.tolist() == [2.0, 3.0, 4.0]

    def test_distributed_backend_multiple_tasks(self, dask_client):
        """Test execution with multiple tasks in sequence."""
        backend = DistributedBackend()

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

        state = xr.Dataset({"x": (("i"), [1.0, 2.0, 3.0])}).chunk({"i": 1})

        results = backend.execute([("group1", [node1, node2])], state)

        assert "y" in results
        assert results["y"].compute().values.tolist() == [4.0, 6.0, 8.0]

    def test_distributed_backend_multiple_groups(self, dask_client):
        """Test execution with multiple groups (dependency)."""
        backend = DistributedBackend()

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

        state = xr.Dataset({"x": (("i"), [1.0, 2.0])}).chunk({"i": 1})

        results = backend.execute([("group1", [node1]), ("group2", [node2])], state)

        assert "final" in results
        assert results["final"].compute().values.tolist() == [4.0, 6.0]

    def test_distributed_backend_empty_task_groups(self, dask_client):
        """Test that empty task groups return empty results."""
        backend = DistributedBackend()
        state = xr.Dataset({"x": (("i"), [1.0])}).chunk({"i": 1})

        results = backend.execute([], state)

        assert results == {}

    def test_distributed_backend_invalid_return_type(self, dask_client):
        """Test error when function doesn't return dict."""
        backend = DistributedBackend()

        def bad_func(x):
            return x  # Should return dict

        node = ComputeNode(
            func=bad_func, name="bad", output_mapping={"result": "y"}, input_mapping={"x": "x"}
        )

        state = xr.Dataset({"x": (("i"), [1.0])}).chunk({"i": 1})

        with pytest.raises(ExecutionError, match="must return a dictionary"):
            backend.execute([("group1", [node])], state)

    def test_distributed_backend_init_no_client(self):
        """Test initialization failure when no client is available."""
        from unittest.mock import patch

        # dask get_client always checks active client. We mock it to raise.
        with (
            patch("seapopym.backend.distributed.get_client", side_effect=ValueError),
            pytest.raises(ValueError, match="DistributedBackend requires an active"),
        ):
            DistributedBackend()

    def test_process_io_task(self, dask_client, caplog):
        """Test process_io_task with different task types."""
        import logging

        import dask

        caplog.set_level(logging.DEBUG)

        backend = DistributedBackend()

        # 1. Dask object (has .compute)
        task_dask = dask.delayed(lambda: 1)()

        backend.process_io_task(task_dask)
        # It triggers the "Submitting IO task to background..." log
        assert "Submitting IO task to background..." in caplog.text
        caplog.clear()

        # 2. Callable
        was_called = False

        def my_task():
            nonlocal was_called
            was_called = True

        backend.process_io_task(my_task)
        assert was_called
        assert "DEBUG: Executing IO task locally" in caplog.text
        caplog.clear()

        # 3. Unknown type
        # The default process_io_task in BaseBackend (which super calls) does nothing
        # but DistributedBackend logs a warning before calling it.
        backend.process_io_task("invalid_task")
        assert "Unknown IO task type" in caplog.text

    def test_prepare_data_no_chunks(self, dask_client, caplog):
        """Test prepare_data warns when data is not chunked."""
        import logging

        caplog.set_level(logging.WARNING)

        backend = DistributedBackend()
        data = xr.Dataset({"x": (("i"), [1, 2, 3])})

        backend.prepare_data(data)

        assert "Data does not seem to be chunked" in caplog.text

    def test_prepare_data_with_chunks(self, dask_client):
        """Test prepare_data with chunked data."""
        backend = DistributedBackend()
        data = xr.Dataset({"x": (("i"), [1, 2, 3])}).chunk({"i": 1})

        res = backend.prepare_data(data)
        assert hasattr(res["x"].data, "dask")

    def test_stabilize_state(self, dask_client):
        """Test stabilize_state returns persisted result."""
        backend = DistributedBackend()
        state = xr.Dataset({"x": (("i"), [1, 2, 3])}).chunk({"i": 1})
        state = state + 1

        new_state = backend.stabilize_state(state)

        assert hasattr(new_state["x"].data, "dask")
        assert new_state["x"].compute().values.tolist() == [2, 3, 4]
