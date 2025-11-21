"""Unit tests for ForcingManager with N-dimensional data."""

import jax.numpy as jnp
import numpy as np
import pytest
import ray
import xarray as xr

from seapopym_message.forcing.manager import ForcingManager
from seapopym_message.forcing.source import ForcingSource


@pytest.fixture
def manager_ndim():
    """Create a ForcingManager with 3D (depth, lat, lon) data."""
    # Create 3D data
    time = np.arange(0, 10)
    depth = np.arange(0, 3)
    lat = np.arange(0, 5)
    lon = np.arange(0, 5)
    data = np.random.rand(len(time), len(depth), len(lat), len(lon))

    da = xr.DataArray(
        data,
        coords={"time": time, "depth": depth, "lat": lat, "lon": lon},
        dims=("time", "depth", "lat", "lon"),
        name="temp_3d",
    )

    source = ForcingSource(da)
    return ForcingManager(forcings=[source])


def test_prepare_timestep_ndim(manager_ndim):
    """Test prepare_timestep returns correct N-dim JAX arrays."""
    forcings = manager_ndim.prepare_timestep(time=0.5)

    assert "temp_3d" in forcings
    assert isinstance(forcings["temp_3d"], jnp.ndarray)
    # Shape should be (depth, lat, lon) -> (3, 5, 5)
    assert forcings["temp_3d"].shape == (3, 5, 5)


def test_prepare_timestep_distributed_ndim(manager_ndim):
    """Test distributed preparation with N-dim data."""
    if not ray.is_initialized():
        ray.init()

    ref = manager_ndim.prepare_timestep_distributed(time=0.5)
    forcings = ray.get(ref)

    assert "temp_3d" in forcings
    assert forcings["temp_3d"].shape == (3, 5, 5)


def test_mixed_dims():
    """Test manager with mixed 2D and 3D forcings."""
    # 2D data
    da_2d = xr.DataArray(
        np.zeros((2, 5, 5)),
        coords={"time": [0, 1], "lat": range(5), "lon": range(5)},
        dims=("time", "lat", "lon"),
        name="npp",
    )

    # 3D data
    da_3d = xr.DataArray(
        np.zeros((2, 3, 5, 5)),
        coords={"time": [0, 1], "depth": range(3), "lat": range(5), "lon": range(5)},
        dims=("time", "depth", "lat", "lon"),
        name="temp",
    )

    manager = ForcingManager(forcings=[ForcingSource(da_2d), ForcingSource(da_3d)])

    forcings = manager.prepare_timestep(0.0)
    assert forcings["npp"].shape == (5, 5)
    assert forcings["temp"].shape == (3, 5, 5)
