"""Tests for preprocessing module."""

import numpy as np
import pytest
import xarray as xr

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
        cleaned, mask = preprocess_nan(data, fill_value=0.0, backend="numpy")
        np.testing.assert_array_equal(cleaned, [1.0, 0.0, 3.0, 0.0, 5.0])

    def test_mask_correct(self):
        """Test that mask correctly identifies valid values."""
        data = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        cleaned, mask = preprocess_nan(data, fill_value=0.0, backend="numpy")
        np.testing.assert_array_equal(mask, [True, False, True, False, True])

    def test_custom_fill_value(self):
        """Test with custom fill value."""
        data = np.array([1.0, np.nan, 3.0])
        cleaned, mask = preprocess_nan(data, fill_value=-999.0, backend="numpy")
        np.testing.assert_array_equal(cleaned, [1.0, -999.0, 3.0])

    def test_no_nan(self):
        """Test with data that has no NaN."""
        data = np.array([1.0, 2.0, 3.0])
        cleaned, mask = preprocess_nan(data, fill_value=0.0, backend="numpy")
        np.testing.assert_array_equal(cleaned, data)
        np.testing.assert_array_equal(mask, [True, True, True])

    def test_all_nan(self):
        """Test with all NaN values."""
        data = np.array([np.nan, np.nan, np.nan])
        cleaned, mask = preprocess_nan(data, fill_value=0.0, backend="numpy")
        np.testing.assert_array_equal(cleaned, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(mask, [False, False, False])

    def test_multidimensional(self):
        """Test with multidimensional array."""
        data = np.array([[1.0, np.nan], [np.nan, 4.0]])
        cleaned, mask = preprocess_nan(data, fill_value=0.0, backend="numpy")
        expected_clean = np.array([[1.0, 0.0], [0.0, 4.0]])
        expected_mask = np.array([[True, False], [False, True]])
        np.testing.assert_array_equal(cleaned, expected_clean)
        np.testing.assert_array_equal(mask, expected_mask)

    def test_jax_backend(self):
        """Test with JAX backend."""
        pytest.importorskip("jax")
        import jax.numpy as jnp

        data = np.array([1.0, np.nan, 3.0])
        cleaned, mask = preprocess_nan(jnp.asarray(data), fill_value=0.0, backend="jax")
        np.testing.assert_array_equal(np.asarray(cleaned), [1.0, 0.0, 3.0])
        np.testing.assert_array_equal(np.asarray(mask), [True, False, True])


class TestGenerateMaskFromData:
    """Tests for generate_mask_from_data function."""

    def test_basic_mask(self):
        """Test basic mask generation."""
        data = np.array([1.0, np.nan, 3.0])
        mask = generate_mask_from_data(data, backend="numpy")
        np.testing.assert_array_equal(mask, [True, False, True])

    def test_2d_mask(self):
        """Test 2D mask generation."""
        data = np.array([[1.0, np.nan], [3.0, 4.0]])
        mask = generate_mask_from_data(data, backend="numpy")
        expected = np.array([[True, False], [True, True]])
        np.testing.assert_array_equal(mask, expected)


class TestStripXarray:
    """Tests for strip_xarray function."""

    def test_basic_strip(self):
        """Test stripping xarray to numpy."""
        da = xr.DataArray(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            dims=["Y", "X"],
        )
        result = strip_xarray(da, backend="numpy")
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [[1.0, 2.0], [3.0, 4.0]])

    def test_strip_preserves_values(self):
        """Test that values are preserved after stripping."""
        original = np.random.rand(10, 20)
        da = xr.DataArray(original, dims=["Y", "X"])
        result = strip_xarray(da, backend="numpy")
        np.testing.assert_array_equal(result, original)

    def test_strip_to_jax(self):
        """Test stripping to JAX array."""
        pytest.importorskip("jax")

        da = xr.DataArray(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            dims=["Y", "X"],
        )
        result = strip_xarray(da, backend="jax")
        assert hasattr(result, "device")  # JAX array attribute
        np.testing.assert_array_equal(np.asarray(result), [[1.0, 2.0], [3.0, 4.0]])

    def test_contiguous(self):
        """Test that result is C-contiguous."""
        # Create non-contiguous array
        original = np.random.rand(10, 20).T  # Transpose makes it non-contiguous
        da = xr.DataArray(original, dims=["Y", "X"])
        result = strip_xarray(da, backend="numpy")
        assert result.flags["C_CONTIGUOUS"]


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
