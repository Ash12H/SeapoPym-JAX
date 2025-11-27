"""Tests for LMTL model functions."""

import numpy as np
import pytest
import xarray as xr

from seapopym.lmtl.core import (
    compute_aging_tendency,
    compute_day_length,
    compute_mean_temperature,
    compute_mortality_tendency,
    compute_production_initialization,
    compute_recruitment_age,
    compute_recruitment_tendency,
)
from seapopym.standard.coordinates import Coordinates


@pytest.fixture
def dummy_data():
    """Create dummy xarray data."""
    # Dimensions: time=2, lat=2, lon=2, depth=2, cohort=3
    times = np.array(["2020-01-01", "2020-07-01"], dtype="datetime64[ns]")
    lats = np.array([0.0, 45.0])
    lons = np.array([0.0, 10.0])
    depths = np.array([0.0, 100.0])
    cohorts = np.array([0, 1, 2])

    temp_data = np.ones((2, 2, 2, 2)) * 20.0  # 20 degrees everywhere
    # Make depth 100 cooler
    temp_data[:, :, :, 1] = 10.0

    temperature = xr.DataArray(
        temp_data,
        coords={
            Coordinates.T: times,
            Coordinates.Y: lats,
            Coordinates.X: lons,
            Coordinates.Z: depths,
        },
        dims=(Coordinates.T, Coordinates.Y, Coordinates.X, Coordinates.Z),
    )

    production = xr.DataArray(
        np.ones((2, 2, 2, 3)),
        coords={Coordinates.T: times, Coordinates.Y: lats, Coordinates.X: lons, "cohort": cohorts},
        dims=(Coordinates.T, Coordinates.Y, Coordinates.X, "cohort"),
    )

    return {
        "temperature": temperature,
        "production": production,
        "times": times,
        "lats": lats,
    }


def test_compute_day_length(dummy_data):
    """Test day length calculation."""
    # At equator (lat=0), day length should be 0.5 (12h)
    lat = xr.DataArray([0.0], dims=Coordinates.Y)
    time = xr.DataArray(
        dummy_data["times"], dims=Coordinates.T, coords={Coordinates.T: dummy_data["times"]}
    )

    res = compute_day_length(lat, time)
    dl = res["output"]
    assert np.allclose(dl.values, 0.5, atol=0.01)

    # At 45 deg north
    lat_45 = xr.DataArray([45.0], dims=Coordinates.Y)
    res_45 = compute_day_length(lat_45, time)
    dl_45 = res_45["output"]

    # Winter (Jan) should be < 0.5, Summer (Jul) > 0.5
    assert dl_45.sel({Coordinates.T: "2020-01-01"}) < 0.5
    assert dl_45.sel({Coordinates.T: "2020-07-01"}) > 0.5


def test_compute_mean_temperature(dummy_data):
    """Test mean temperature calculation."""
    temp = dummy_data["temperature"]
    # Day length 0.5
    dl = xr.DataArray(0.5)

    # Day at 0m (20C), Night at 100m (10C)
    res = compute_mean_temperature(temp, dl, day_layer=0, night_layer=100)
    t_mean = res["output"]

    # Should be (20 * 0.5) + (10 * 0.5) = 15
    assert np.allclose(t_mean, 15.0)


def test_compute_recruitment_age():
    """Test recruitment age."""
    t_mean = xr.DataArray([20.0])
    tau_r_0 = 10.0
    gamma = 0.1
    t_ref = 20.0

    # At T_ref, should be tau_r_0
    res = compute_recruitment_age(t_mean, tau_r_0, gamma, t_ref)
    age = res["output"]
    assert age == 10.0

    # At higher temp, should be lower
    t_hot = xr.DataArray([30.0])
    res_hot = compute_recruitment_age(t_hot, tau_r_0, gamma, t_ref)
    age_hot = res_hot["output"]
    assert age_hot < 10.0


def test_compute_production_initialization(dummy_data):
    """Test production init."""
    npp = dummy_data["temperature"].isel({Coordinates.Z: 0})  # Use temp as dummy NPP
    E = 0.1
    npp = dummy_data["temperature"].isel({Coordinates.Z: 0})  # Use temp as dummy NPP
    E = 0.1

    res = compute_production_initialization(npp, dummy_data["production"].cohort, E)
    assert "output" in res
    tendency = res["output"]

    # Should have cohort dim with size matching production (3)
    assert "cohort" in tendency.dims
    assert tendency.sizes["cohort"] == 3

    # Value check
    assert np.allclose(tendency.isel(cohort=0), npp * E)
    # Other cohorts should be 0
    assert np.allclose(tendency.isel(cohort=1), 0.0)


def test_compute_aging_tendency(dummy_data):
    """Test aging tendency."""
    prod = dummy_data["production"]  # All ones
    dt = 1.0

    res = compute_aging_tendency(prod, dt)
    tendency = res["aging_flux"]

    # p[c] = 1. p[c-1] = 1 (except c=0 where p[-1]=0)
    # Tendency = (p[c-1] - p[c]) / dt
    # c=0: (0 - 1) = -1
    # c=1: (1 - 1) = 0
    # c=2: (1 - 1) = 0

    assert np.allclose(tendency.isel(cohort=0), -1.0)
    assert np.allclose(tendency.isel(cohort=1), 0.0)
    assert np.allclose(tendency.isel(cohort=2), 0.0)


def test_compute_recruitment_tendency(dummy_data):
    """Test recruitment tendency."""
    prod = dummy_data["production"]  # All ones
    dt = 1.0
    cohort_ages = xr.DataArray([0, 10, 20], dims="cohort")

    # Case 1: Recruitment age = 15. Cohort 2 (20) is recruited.
    rec_age = xr.DataArray(15.0)

    res = compute_recruitment_tendency(prod, rec_age, cohort_ages, dt)

    sink = res["recruitment_sink"]
    source = res["recruitment_source"]

    # Sink: Should be -1/dt for cohort 2, 0 elsewhere
    assert np.allclose(sink.isel(cohort=0), 0.0)
    assert np.allclose(sink.isel(cohort=1), 0.0)
    assert np.allclose(sink.isel(cohort=2), -1.0)

    # Source: Should be sum of recruited (1) / dt
    # Source has no cohort dim
    assert "cohort" not in source.dims
    assert np.allclose(source, 1.0)


def test_compute_mortality_tendency():
    """Test mortality."""
    biomass = xr.DataArray([100.0])
    t_mean = xr.DataArray([20.0])
    lambda_0 = 0.1
    gamma = 0.0
    t_ref = 0.0

    res = compute_mortality_tendency(biomass, t_mean, lambda_0, gamma, t_ref)
    loss = res["mortality_loss"]

    # Rate = 0.1 * exp(0) = 0.1
    # Loss = -0.1 * 100 = -10
    assert np.allclose(loss, -10.0)
