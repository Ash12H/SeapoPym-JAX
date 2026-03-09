"""Tests for MemoryWriter."""

import numpy as np
import pytest
import xarray as xr

from seapopym.engine.io import MemoryWriter


def test_memory_writer_lifecycle():
    writer = MemoryWriter()

    # Initialize requesting only biomass
    variables = ["biomass"]
    coords = {"T": np.arange(10), "Y": np.arange(5), "X": np.arange(5)}
    var_dims = {
        "biomass": ("Y", "X"),
        "production": ("Y", "X"),
        "hidden": ("Y", "X"),
    }
    writer.initialize({}, variables, coords=coords, var_dims=var_dims)

    # Simulate 2 chunks
    # Chunk 1: time 0-5
    chunk1 = {
        "biomass": np.ones((5, 5, 5)),
        "production": np.zeros((5, 5, 5)),  # Should be ignored because not in variables
        "hidden": np.ones((5, 5, 5)),  # Should be ignored
    }
    writer.append(chunk1)

    # Chunk 2: time 5-10
    chunk2 = {
        "biomass": np.ones((5, 5, 5)) * 2,
    }
    writer.append(chunk2)

    # Finalize
    ds = writer.finalize()

    assert isinstance(ds, xr.Dataset)
    assert "biomass" in ds
    assert "production" not in ds
    assert "hidden" not in ds

    # Check dimensions — canonical names
    assert ds["biomass"].dims == ("T", "Y", "X")
    assert ds["biomass"].shape == (10, 5, 5)  # 5 + 5

    # Check values
    assert np.all(ds["biomass"].isel(T=slice(0, 5)) == 1)
    assert np.all(ds["biomass"].isel(T=slice(5, 10)) == 2)

    # Check coords
    assert "T" in ds.coords
    assert "Y" in ds.coords
