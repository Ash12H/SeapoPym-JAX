"""Tests for preprocessing module."""

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("jax")

from seapopym.compiler.preprocessing import (
    generate_mask_from_data,
    load_data,
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


class TestGenerateMaskFromData:
    """Tests for generate_mask_from_data function."""

    def test_basic_mask(self):
        """Test basic mask generation."""
        data = np.array([1.0, np.nan, 3.0])
        mask = generate_mask_from_data(data)
        np.testing.assert_array_equal(np.asarray(mask), [True, False, True])

    def test_2d_mask(self):
        """Test 2D mask generation."""
        data = np.array([[1.0, np.nan], [3.0, 4.0]])
        mask = generate_mask_from_data(data)
        expected = np.array([[True, False], [True, True]])
        np.testing.assert_array_equal(np.asarray(mask), expected)


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
