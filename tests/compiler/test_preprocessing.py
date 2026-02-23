"""Tests for preprocessing module."""

import numpy as np
import pytest
import xarray as xr

from seapopym.compiler.preprocessing import (
    extract_coords,
    load_data,
    prepare_array,
    preprocess_nan,
    strip_xarray,
)


class TestPreprocessNan:
    """Tests for preprocess_nan function."""

    def test_replaces_nan(self):
        """Test that NaN values are replaced."""
        data = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        cleaned, mask = preprocess_nan(data, fill_value=0.0)
        np.testing.assert_array_equal(np.asarray(cleaned), [1.0, 0.0, 3.0, 0.0, 5.0])

    def test_mask_correct(self):
        """Test that mask correctly identifies valid values."""
        data = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        cleaned, mask = preprocess_nan(data, fill_value=0.0)
        np.testing.assert_array_equal(np.asarray(mask), [True, False, True, False, True])

    def test_custom_fill_value(self):
        """Test with custom fill value."""
        data = np.array([1.0, np.nan, 3.0])
        cleaned, mask = preprocess_nan(data, fill_value=-999.0)
        np.testing.assert_array_equal(np.asarray(cleaned), [1.0, -999.0, 3.0])

    def test_no_nan(self):
        """Test with data that has no NaN."""
        data = np.array([1.0, 2.0, 3.0])
        cleaned, mask = preprocess_nan(data, fill_value=0.0)
        np.testing.assert_array_equal(np.asarray(cleaned), data)
        np.testing.assert_array_equal(np.asarray(mask), [True, True, True])

    def test_all_nan(self):
        """Test with all NaN values."""
        data = np.array([np.nan, np.nan, np.nan])
        cleaned, mask = preprocess_nan(data, fill_value=0.0)
        np.testing.assert_array_equal(np.asarray(cleaned), [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(np.asarray(mask), [False, False, False])

    def test_multidimensional(self):
        """Test with multidimensional array."""
        data = np.array([[1.0, np.nan], [np.nan, 4.0]])
        cleaned, mask = preprocess_nan(data, fill_value=0.0)
        expected_clean = np.array([[1.0, 0.0], [0.0, 4.0]])
        expected_mask = np.array([[True, False], [False, True]])
        np.testing.assert_array_equal(np.asarray(cleaned), expected_clean)
        np.testing.assert_array_equal(np.asarray(mask), expected_mask)


class TestStripXarray:
    """Tests for strip_xarray function."""

    def test_basic_strip(self):
        """Test stripping xarray to JAX array."""
        da = xr.DataArray(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            dims=["Y", "X"],
        )
        result = strip_xarray(da)
        # Result is a JAX array
        assert hasattr(result, "device")
        np.testing.assert_array_equal(np.asarray(result), [[1.0, 2.0], [3.0, 4.0]])

    def test_strip_preserves_values(self):
        """Test that values are preserved after stripping."""
        original = np.random.rand(10, 20)
        da = xr.DataArray(original, dims=["Y", "X"])
        result = strip_xarray(da)
        np.testing.assert_allclose(np.asarray(result), original, rtol=1e-6)


class TestLoadData:
    """Tests for load_data function."""

    def test_load_dataarray(self):
        """Test loading from DataArray."""
        da = xr.DataArray(np.array([1, 2, 3]), dims=["X"])
        result = load_data(da)
        assert isinstance(result, xr.DataArray)

    def test_load_numpy(self):
        """Test loading from numpy array."""
        arr = np.array([1, 2, 3])
        result = load_data(arr)
        np.testing.assert_array_equal(result, arr)

    def test_load_list(self):
        """Test loading from list."""
        data = [1, 2, 3]
        result = load_data(data)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_load_dataset_single_var(self):
        """Test loading from Dataset with single variable."""
        ds = xr.Dataset({"temp": (["X"], np.array([1, 2, 3]))})
        result = load_data(ds)
        assert isinstance(result, xr.DataArray)
        np.testing.assert_array_equal(result.values, [1, 2, 3])

    def test_load_dataset_with_var_name(self):
        """Test loading specific variable from Dataset."""
        ds = xr.Dataset(
            {
                "temp": (["X"], np.array([1, 2, 3])),
                "salt": (["X"], np.array([4, 5, 6])),
            }
        )
        result = load_data(ds, variable_name="salt")
        assert isinstance(result, xr.DataArray)
        np.testing.assert_array_equal(result.values, [4, 5, 6])

    def test_load_dataset_multi_var_no_name_error(self):
        """Test error when Dataset has multiple variables and no variable_name."""
        ds = xr.Dataset(
            {
                "temp": (["X"], np.array([1, 2, 3])),
                "salt": (["X"], np.array([4, 5, 6])),
            }
        )
        with pytest.raises(ValueError, match="contains 2 variables"):
            load_data(ds)


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

    def test_from_dataset(self):
        """Test extracting coords from a Dataset."""
        ds = xr.Dataset(
            {"temp": (["Y", "X"], np.zeros((3, 4)))},
            coords={"Y": [10.0, 20.0, 30.0], "X": [100.0, 101.0, 102.0, 103.0]},
        )
        coords = extract_coords(ds)
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


class TestPrepareArray:
    """Tests for prepare_array function."""

    def test_from_dataarray(self):
        """Test full pipeline from DataArray."""
        da = xr.DataArray(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            dims=["Y", "X"],
        )
        arr, dims, mask = prepare_array(da)
        assert dims == ("Y", "X")
        assert hasattr(arr, "device")  # JAX array
        np.testing.assert_array_equal(np.asarray(arr), [[1.0, 2.0], [3.0, 4.0]])

    def test_nan_handling(self):
        """Test NaN replacement and mask generation."""
        da = xr.DataArray(
            np.array([[1.0, np.nan], [np.nan, 4.0]]),
            dims=["Y", "X"],
        )
        arr, dims, mask = prepare_array(da, fill_nan=0.0)
        np.testing.assert_array_equal(np.asarray(arr), [[1.0, 0.0], [0.0, 4.0]])
        assert mask is not None
        np.testing.assert_array_equal(np.asarray(mask), [[True, False], [False, True]])

    def test_no_nan_handling(self):
        """Test with fill_nan=None preserves NaN."""
        da = xr.DataArray(
            np.array([1.0, np.nan, 3.0]),
            dims=["X"],
        )
        arr, dims, mask = prepare_array(da, fill_nan=None)
        assert mask is None
        assert np.isnan(np.asarray(arr)[1])

    def test_from_numpy(self):
        """Test pipeline from raw numpy array."""
        data = np.array([1.0, 2.0, 3.0])
        arr, dims, mask = prepare_array(data)
        assert dims == ()  # no dim info from raw array
        assert hasattr(arr, "device")

    def test_dimension_mapping(self):
        """Test dimension mapping is applied."""
        da = xr.DataArray(
            np.zeros((3, 4)),
            dims=["lat", "lon"],
        )
        arr, dims, mask = prepare_array(da, dimension_mapping={"lat": "Y", "lon": "X"})
        assert dims == ("Y", "X")

    def test_canonical_transpose(self):
        """Test that dims are transposed to canonical order."""
        da = xr.DataArray(
            np.zeros((4, 3)),
            dims=["X", "Y"],
        )
        arr, dims, mask = prepare_array(da)
        assert dims == ("Y", "X")
        assert np.asarray(arr).shape == (3, 4)
