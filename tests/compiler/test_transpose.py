"""Tests for transpose module."""

import numpy as np
import xarray as xr

from seapopym.compiler.transpose import (
    apply_dimension_mapping,
    get_canonical_order,
    transpose_array,
    transpose_canonical,
)


class TestGetCanonicalOrder:
    """Tests for get_canonical_order function."""

    def test_full_order(self):
        """Test with all canonical dimensions."""
        dims = ["X", "Y", "T", "C", "F", "Z", "E"]
        result = get_canonical_order(dims)
        assert result == ("E", "T", "F", "C", "Z", "Y", "X")

    def test_partial_order(self):
        """Test with subset of dimensions."""
        dims = ["X", "Y", "T"]
        result = get_canonical_order(dims)
        assert result == ("T", "Y", "X")

    def test_spatial_only(self):
        """Test with spatial dimensions only."""
        dims = ["Y", "X"]
        result = get_canonical_order(dims)
        assert result == ("Y", "X")

    def test_empty(self):
        """Test with empty dimensions."""
        result = get_canonical_order([])
        assert result == ()

    def test_non_canonical_dims_ignored(self):
        """Test that non-canonical dims are not included."""
        dims = ["X", "Y", "custom_dim"]
        result = get_canonical_order(dims)
        assert result == ("Y", "X")


class TestApplyDimensionMapping:
    """Tests for apply_dimension_mapping function."""

    def test_basic_mapping(self):
        """Test basic dimension renaming."""
        da = xr.DataArray(
            np.random.rand(3, 4, 5),
            dims=["time", "lat", "lon"],
        )
        mapping = {"time": "T", "lat": "Y", "lon": "X"}
        result = apply_dimension_mapping(da, mapping)
        assert list(result.dims) == ["T", "Y", "X"]

    def test_partial_mapping(self):
        """Test mapping with only some dimensions."""
        da = xr.DataArray(
            np.random.rand(3, 4),
            dims=["time", "lat"],
        )
        mapping = {"time": "T"}
        result = apply_dimension_mapping(da, mapping)
        assert list(result.dims) == ["T", "lat"]

    def test_no_mapping(self):
        """Test with None mapping."""
        da = xr.DataArray(
            np.random.rand(3, 4),
            dims=["time", "lat"],
        )
        result = apply_dimension_mapping(da, None)
        assert list(result.dims) == ["time", "lat"]

    def test_mapping_nonexistent_dim(self):
        """Test mapping with non-existent dimension (should be ignored)."""
        da = xr.DataArray(
            np.random.rand(3, 4),
            dims=["time", "lat"],
        )
        mapping = {"lon": "X"}  # lon doesn't exist
        result = apply_dimension_mapping(da, mapping)
        assert list(result.dims) == ["time", "lat"]

    def test_dataset_mapping(self):
        """Test mapping on Dataset."""
        ds = xr.Dataset(
            {
                "temp": (["time", "lat", "lon"], np.random.rand(3, 4, 5)),
            }
        )
        mapping = {"time": "T", "lat": "Y", "lon": "X"}
        result = apply_dimension_mapping(ds, mapping)
        assert list(result.dims) == ["T", "Y", "X"]


class TestTransposeCanonical:
    """Tests for transpose_canonical function."""

    def test_basic_transpose(self):
        """Test transposing to canonical order."""
        da = xr.DataArray(
            np.random.rand(5, 4, 3),
            dims=["X", "Y", "T"],
        )
        result = transpose_canonical(da)
        assert list(result.dims) == ["T", "Y", "X"]
        assert result.shape == (3, 4, 5)

    def test_already_canonical(self):
        """Test array already in canonical order."""
        da = xr.DataArray(
            np.random.rand(3, 4, 5),
            dims=["T", "Y", "X"],
        )
        result = transpose_canonical(da)
        assert list(result.dims) == ["T", "Y", "X"]
        assert result.shape == (3, 4, 5)

    def test_extra_dims_preserved(self):
        """Test that non-canonical dims are preserved at the end."""
        da = xr.DataArray(
            np.random.rand(5, 4, 3, 2),
            dims=["X", "Y", "T", "custom"],
        )
        result = transpose_canonical(da)
        # custom should be at the end
        assert list(result.dims) == ["T", "Y", "X", "custom"]

    def test_custom_target_order(self):
        """Test with custom target order."""
        da = xr.DataArray(
            np.random.rand(5, 4, 3),
            dims=["X", "Y", "T"],
        )
        result = transpose_canonical(da, target_order=("X", "Y", "T"))
        assert list(result.dims) == ["X", "Y", "T"]


class TestTransposeArray:
    """Tests for transpose_array function."""

    def test_basic_transpose(self):
        """Test transposing numpy array."""
        arr = np.random.rand(5, 4, 3)
        current_dims = ("X", "Y", "T")
        result, new_dims = transpose_array(arr, current_dims)
        assert new_dims == ("T", "Y", "X")
        assert result.shape == (3, 4, 5)

    def test_preserves_data(self):
        """Test that data values are preserved."""
        arr = np.arange(24).reshape(4, 3, 2)
        current_dims = ("X", "Y", "T")
        result, new_dims = transpose_array(arr, current_dims)
        # Check a specific value
        # Original: arr[x=2, y=1, t=0] = arr[2,1,0] = 2*6 + 1*2 + 0 = 14
        # After: result[t=0, y=1, x=2] should be same
        assert result[0, 1, 2] == arr[2, 1, 0]

    def test_extra_dims(self):
        """Test with extra non-canonical dimensions."""
        arr = np.random.rand(5, 4, 3, 2)
        current_dims = ("X", "Y", "T", "custom")
        result, new_dims = transpose_array(arr, current_dims)
        # custom goes at the end
        assert new_dims == ("T", "Y", "X", "custom")
        assert result.shape == (3, 4, 5, 2)
