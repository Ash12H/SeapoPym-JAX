"""Tests for preprocessing module."""

import numpy as np
import xarray as xr

from seapopym.compiler.preprocessing import extract_coords


class TestExtractCoords:
    """Tests for extract_coords function."""

    def test_from_dataarray(self):
        """Test extracting coords from a DataArray."""
        da = xr.DataArray(
            np.zeros((3, 4)),
            dims=["Y", "X"],
            coords={"Y": [10.0, 20.0, 30.0], "X": [100.0, 101.0, 102.0, 103.0]},
        )
        coords = extract_coords(da)
        assert "Y" in coords
        assert "X" in coords
        np.testing.assert_array_equal(coords["Y"], [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(coords["X"], [100.0, 101.0, 102.0, 103.0])

    def test_with_dimension_mapping(self):
        """Test extracting coords with dimension renaming."""
        da = xr.DataArray(
            np.zeros((3, 4)),
            dims=["lat", "lon"],
            coords={"lat": [10.0, 20.0, 30.0], "lon": [100.0, 101.0, 102.0, 103.0]},
        )
        coords = extract_coords(da, dimension_mapping={"lat": "Y", "lon": "X"})
        assert "Y" in coords
        assert "X" in coords

    def test_returns_numpy_arrays(self):
        """Test that coords are numpy arrays, not JAX arrays."""
        da = xr.DataArray(
            np.zeros((3,)),
            dims=["Y"],
            coords={"Y": [1.0, 2.0, 3.0]},
        )
        coords = extract_coords(da)
        assert type(coords["Y"]) is np.ndarray
