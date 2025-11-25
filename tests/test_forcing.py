from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from seapopym.forcing import ForcingManager
from seapopym.standard import Coordinates


@pytest.fixture
def simple_forcing():
    """Create a simple forcing dataset with temperature."""
    times = pd.date_range("2020-01-01", "2020-01-05", freq="D")
    # Temperature increases by 1 degree per day: 10, 11, 12, 13, 14
    temp = np.linspace(10, 14, len(times))

    ds = xr.Dataset(
        {"temperature": ((Coordinates.T, "x"), temp[:, None] * np.ones((len(times), 2)))},
        coords={Coordinates.T: times, "x": [0, 1]},
    )
    return ds


def test_init_validation():
    """Test that init raises error if time dimension is missing."""
    ds = xr.Dataset({"temp": (("x"), [1, 2])})
    with pytest.raises(ValueError, match="must have a .* dimension"):
        ForcingManager(ds)

    ds_empty = xr.Dataset({"temp": ((Coordinates.T), [])}, coords={Coordinates.T: []})
    with pytest.raises(ValueError, match="cannot be empty"):
        ForcingManager(ds_empty)


def test_exact_time_selection(simple_forcing):
    """Test retrieving forcings at an exact existing timestamp."""
    manager = ForcingManager(simple_forcing)
    target_time = datetime(2020, 1, 2)  # Should be 11.0

    result = manager.get_forcings(target_time)

    assert Coordinates.T not in result.dims  # Should be dropped or scalar
    assert result["temperature"].isel(x=0).item() == 11.0


def test_interpolation(simple_forcing):
    """Test linear interpolation between timesteps."""
    manager = ForcingManager(simple_forcing)
    # Halfway between Jan 1 (10.0) and Jan 2 (11.0) -> 10.5
    target_time = datetime(2020, 1, 1, 12, 0)

    result = manager.get_forcings(target_time)

    assert result["temperature"].isel(x=0).item() == 10.5


def test_out_of_bounds(simple_forcing):
    """Test that requesting time outside range raises ValueError."""
    manager = ForcingManager(simple_forcing)

    # Before
    with pytest.raises(ValueError, match="outside forcing range"):
        manager.get_forcings(datetime(2019, 12, 31))

    # After
    with pytest.raises(ValueError, match="outside forcing range"):
        manager.get_forcings(datetime(2020, 1, 6))


def test_dask_lazy_evaluation():
    """Test that ForcingManager preserves dask lazy evaluation."""
    # Create dask-backed dataset
    times = pd.date_range("2020-01-01", "2020-01-05", freq="D")
    data = np.random.rand(len(times), 10, 10)
    ds = xr.Dataset(
        {"temp": ((Coordinates.T, "x", "y"), data)}, coords={Coordinates.T: times}
    ).chunk({Coordinates.T: 1})

    manager = ForcingManager(ds)
    target_time = datetime(2020, 1, 2, 12, 0)

    result = manager.get_forcings(target_time)

    # Check that data is still a dask array (not computed)
    import dask.array as da

    assert isinstance(result["temp"].data, da.Array)

    # Check that we can compute it
    val = result["temp"].compute()
    assert val.shape == (10, 10)
