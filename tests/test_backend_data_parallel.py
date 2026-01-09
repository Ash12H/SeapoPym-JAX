"""Tests for DataParallelBackend."""

import dask.array as da
import pytest
import xarray as xr

from seapopym.backend.data_parallel import DataParallelBackend
from seapopym.blueprint.nodes import ComputeNode


# Helpers
def add_one(x):
    """Add 1 to input."""
    return {"result": x + 1.0}


def multiply_two(x):
    """Multiply input by 2."""
    return {"result": x * 2.0}


def group1_func(x):
    """Group 1 function."""
    return {"out": x + 1.0}


def group2_func(y):
    """Group 2 function."""
    return {"out": y * 2.0}


def sum_reduction(x):
    """Reduction operation (sum)."""
    return {"result": x.sum()}


class TestDataParallelBackend:
    """Test suite for DataParallelBackend."""

    def test_data_parallel_simple_execution_with_chunks(self):
        """Test basic execution with chunked data."""
        backend = DataParallelBackend(persist_intermediates=False)

        node = ComputeNode(
            func=add_one, name="add_one", output_mapping={"result": "y"}, input_mapping={"x": "x"}
        )

        # Create chunked data
        state = xr.Dataset({"x": (("i"), [1.0, 2.0, 3.0])})
        state = state.chunk({"i": 1})

        results = backend.execute([("group1", [node])], state)

        assert "y" in results
        # Force computation to verify results
        assert results["y"].compute().values.tolist() == [2.0, 3.0, 4.0]

    def test_data_parallel_multiple_tasks(self):
        """Test execution with multiple tasks in sequence."""
        backend = DataParallelBackend(persist_intermediates=False)

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
        state = state.chunk({"i": 1})

        results = backend.execute([("group1", [node1, node2])], state)

        assert "y" in results
        assert results["y"].compute().values.tolist() == [4.0, 6.0, 8.0]

    def test_data_parallel_multiple_groups(self):
        """Test execution with multiple groups."""
        backend = DataParallelBackend(persist_intermediates=False)

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
        state = state.chunk({"i": 1})

        results = backend.execute([("group1", [node1]), ("group2", [node2])], state)

        assert "final" in results
        assert results["final"].compute().values.tolist() == [4.0, 6.0]

    def test_data_parallel_with_persist_intermediates(self):
        """Test that persist_intermediates materializes results."""
        backend = DataParallelBackend(persist_intermediates=True)

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
        state = state.chunk({"i": 1})

        results = backend.execute([("group1", [node1]), ("group2", [node2])], state)

        # With persist_intermediates, results should be persisted (not just lazy)
        assert "y" in results
        # The result should still be a dask array but persisted
        assert isinstance(results["y"].data, da.Array)
        assert results["y"].compute().values.tolist() == [4.0, 6.0, 8.0]

    def test_data_parallel_warns_without_chunks(self):
        """Test that DataParallelBackend warns when no chunked data is detected."""
        backend = DataParallelBackend(persist_intermediates=False)

        node = ComputeNode(
            func=add_one, name="add_one", output_mapping={"result": "y"}, input_mapping={"x": "x"}
        )

        # Create non-chunked data
        state = xr.Dataset({"x": (("i"), [1.0, 2.0, 3.0])})

        # Should execute but warn
        with pytest.warns(UserWarning, match="no chunked Dask arrays found"):
            results = backend.execute([("group1", [node])], state)

        assert "y" in results
        assert results["y"].values.tolist() == [2.0, 3.0, 4.0]

    def test_data_parallel_empty_task_groups(self):
        """Test that empty task groups return empty results."""
        backend = DataParallelBackend(persist_intermediates=False)
        state = xr.Dataset({"x": (("i"), [1.0])})

        results = backend.execute([], state)

        assert results == {}

    def test_data_parallel_preserves_lazy_evaluation(self):
        """Test that DataParallelBackend preserves lazy evaluation (without persist)."""
        backend = DataParallelBackend(persist_intermediates=False)

        node = ComputeNode(
            func=add_one, name="add_one", output_mapping={"result": "y"}, input_mapping={"x": "x"}
        )

        state = xr.Dataset({"x": (("i"), [1.0, 2.0, 3.0])})
        state = state.chunk({"i": 1})

        results = backend.execute([("group1", [node])], state)

        # Result should be a dask array (lazy)
        assert isinstance(results["y"].data, da.Array)

        # But computation should work
        assert results["y"].compute().values.tolist() == [2.0, 3.0, 4.0]

    def test_data_parallel_with_2d_chunks(self):
        """Test execution with 2D chunked data."""
        backend = DataParallelBackend(persist_intermediates=False)

        node = ComputeNode(
            func=add_one, name="add_one", output_mapping={"result": "y"}, input_mapping={"x": "x"}
        )

        # Create 2D chunked data
        state = xr.Dataset({"x": (("i", "j"), [[1.0, 2.0], [3.0, 4.0]])})
        state = state.chunk({"i": 1, "j": 1})

        results = backend.execute([("group1", [node])], state)

        assert "y" in results
        assert results["y"].compute().values.tolist() == [[2.0, 3.0], [4.0, 5.0]]

    def test_data_parallel_respects_chunk_strategy(self):
        """Test that DataParallelBackend respects the chunking strategy."""
        backend = DataParallelBackend(persist_intermediates=False)

        node = ComputeNode(
            func=add_one, name="add_one", output_mapping={"result": "y"}, input_mapping={"x": "x"}
        )

        # Create data with specific chunking
        state = xr.Dataset({"x": (("i"), list(range(10)))})
        state = state.chunk({"i": 2})  # 5 chunks of size 2

        results = backend.execute([("group1", [node])], state)

        # Verify chunking is preserved
        assert isinstance(results["y"].data, da.Array)
        assert results["y"].data.npartitions == 5  # Should have 5 chunks

    def test_data_parallel_mixed_chunked_and_unchunked(self):
        """Test execution with both chunked and unchunked variables."""
        backend = DataParallelBackend(persist_intermediates=False)

        def add_arrays(x, y):
            return {"result": x + y}

        node = ComputeNode(
            func=add_arrays,
            name="add_arrays",
            output_mapping={"result": "z"},
            input_mapping={"x": "x", "y": "y"},
        )

        # x is chunked, y is not
        state = xr.Dataset(
            {
                "x": (("i"), [1.0, 2.0, 3.0]),
                "y": (("i"), [10.0, 20.0, 30.0]),
            }
        )
        state["x"] = state["x"].chunk({"i": 1})

        results = backend.execute([("group1", [node])], state)

        assert "z" in results
        assert results["z"].compute().values.tolist() == [11.0, 22.0, 33.0]
