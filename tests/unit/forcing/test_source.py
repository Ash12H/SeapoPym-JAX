"""Unit tests for ForcingSource."""

import numpy as np
import pytest
import xarray as xr

from seapopym_message.forcing.source import ForcingSource


@pytest.fixture
def sample_data_2d():
    """Create a sample 2D (time, lat, lon) DataArray."""
    time = np.arange(0, 10)
    lat = np.arange(0, 5)
    lon = np.arange(0, 5)
    data = np.random.rand(len(time), len(lat), len(lon))
    return xr.DataArray(
        data,
        coords={"time": time, "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
        name="temp",
    )


@pytest.fixture
def sample_data_3d():
    """Create a sample 3D (time, depth, lat, lon) DataArray."""
    time = np.arange(0, 10)
    depth = np.arange(0, 3)
    lat = np.arange(0, 5)
    lon = np.arange(0, 5)
    data = np.random.rand(len(time), len(depth), len(lat), len(lon))
    return xr.DataArray(
        data,
        coords={"time": time, "depth": depth, "lat": lat, "lon": lon},
        dims=("time", "depth", "lat", "lon"),
        name="temp_3d",
    )


def test_init_valid(sample_data_2d):
    """Test valid initialization."""
    source = ForcingSource(sample_data_2d)
    assert source.name == "temp"
    assert source.interpolation_method == "linear"


def test_init_with_name(sample_data_2d):
    """Test initialization with explicit name."""
    source = ForcingSource(sample_data_2d, name="my_temp")
    assert source.name == "my_temp"


def test_init_missing_dims():
    """Test initialization fails with missing dimensions."""
    data = xr.DataArray(np.zeros((5, 5)), dims=("lat", "lon"))
    with pytest.raises(ValueError, match="missing required dimensions"):
        ForcingSource(data, name="bad")


def test_init_no_name():
    """Test initialization fails without name."""
    data = xr.DataArray(
        np.zeros((2, 2, 2)),
        coords={"time": [0, 1], "lat": [0, 1], "lon": [0, 1]},
        dims=("time", "lat", "lon"),
    )
    data.name = None
    with pytest.raises(ValueError, match="Forcing name must be provided"):
        ForcingSource(data)


def test_interpolate_2d(sample_data_2d):
    """Test interpolation of 2D data."""
    source = ForcingSource(sample_data_2d)
    # Interpolate at t=0.5
    result = source.interpolate(0.5)
    assert "time" not in result.dims
    assert result.shape == (5, 5)
    # Check value is roughly average of t=0 and t=1
    expected = (sample_data_2d.sel(time=0) + sample_data_2d.sel(time=1)) / 2
    np.testing.assert_allclose(result.values, expected.values)


def test_interpolate_3d(sample_data_3d):
    """Test interpolation of 3D data (N-dim support)."""
    source = ForcingSource(sample_data_3d)
    result = source.interpolate(0.5)
    assert "time" not in result.dims
    assert result.shape == (3, 5, 5)  # (depth, lat, lon)
