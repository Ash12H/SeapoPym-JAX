from unittest.mock import MagicMock

import networkx as nx
import numpy as np
import pytest
import xarray as xr

from seapopym.blueprint.nodes import DataNode
from seapopym.compiler import CompiledModel
from seapopym.engine.io import MemoryWriter


@pytest.fixture
def mock_model():
    # Create a mock CompiledModel
    model = MagicMock(spec=CompiledModel)

    # Mock coords
    model.coords = {"time": np.arange(10), "lat": np.arange(5), "lon": np.arange(5)}

    # Mock graph
    graph = nx.DiGraph()
    # Add nodes with dimensions
    node_bio = DataNode(name="biomass", dims=("time", "lat", "lon"))
    node_prod = DataNode(name="production", dims=("time", "lat", "lon"))
    node_hidden = DataNode(name="hidden", dims=("time", "lat", "lon"))  # Not requested

    graph.add_node(node_bio)
    graph.add_node(node_prod)
    graph.add_node(node_hidden)

    model.graph = graph
    return model


def test_memory_writer_lifecycle(mock_model):
    writer = MemoryWriter(mock_model)

    # Initialize requesting only biomass
    variables = ["biomass"]
    writer.initialize({}, variables)

    # Simuler 2 chunks
    # Chunk 1: time 0-5
    chunk1 = {
        "biomass": np.ones((5, 5, 5)),
        "production": np.zeros((5, 5, 5)),  # Should be ignored because not in variables
        "hidden": np.ones((5, 5, 5)),  # Should be ignored
    }
    writer.append(chunk1, 0)

    # Chunk 2: time 5-10
    chunk2 = {
        "biomass": np.ones((5, 5, 5)) * 2,
    }
    writer.append(chunk2, 1)

    # Finalize
    ds = writer.finalize()

    assert isinstance(ds, xr.Dataset)
    assert "biomass" in ds
    assert "production" not in ds
    assert "hidden" not in ds

    # Check dimensions
    assert ds["biomass"].dims == ("time", "lat", "lon")
    assert ds["biomass"].shape == (10, 5, 5)  # 5 + 5

    # Check values
    assert np.all(ds["biomass"].isel(time=slice(0, 5)) == 1)
    assert np.all(ds["biomass"].isel(time=slice(5, 10)) == 2)

    # Check coords
    assert "time" in ds.coords
    assert "lat" in ds.coords
