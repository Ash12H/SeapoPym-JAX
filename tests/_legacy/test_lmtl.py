"""Tests for LMTL model functions."""

import numpy as np
import pytest
import xarray as xr

from seapopym.lmtl.core import (
    compute_day_length,
    compute_layer_weighted_mean,
    compute_mortality_tendency,
    compute_production_dynamics,
    compute_production_initialization,
    compute_recruitment_age,
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
            Coordinates.T.value: times,
            Coordinates.Y.value: lats,
            Coordinates.X.value: lons,
            Coordinates.Z.value: depths,
        },
        dims=(Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value, Coordinates.Z.value),
    )

    production = xr.DataArray(
        np.ones((2, 2, 2, 3)),
        coords={
            Coordinates.T.value: times,
            Coordinates.Y.value: lats,
            Coordinates.X.value: lons,
            "cohort": cohorts,
        },
        dims=(Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value, "cohort"),
    )

    # Need simpler coords for specific tests
    return {
        "temperature": temperature,
        "production": production,
        "times": times,
        "lats": lats,
    }


def test_compute_day_length(dummy_data):
    """Test day length calculation."""
    # At equator (lat=0), day length should be 0.5 (12h)
    lat = xr.DataArray([0.0], dims=Coordinates.Y.value)
    time = xr.DataArray(
        dummy_data["times"],
        dims=Coordinates.T.value,
        coords={Coordinates.T.value: dummy_data["times"]},
    )

    res = compute_day_length(lat, time)
    dl = res["output"]
    assert np.allclose(dl.values, 0.5, atol=0.01)

    # At 45 deg north
    lat_45 = xr.DataArray([45.0], dims=Coordinates.Y.value)
    res_45 = compute_day_length(lat_45, time)
    dl_45 = res_45["output"]

    # Winter (Jan) should be < 0.5, Summer (Jul) > 0.5
    assert dl_45.sel({Coordinates.T.value: "2020-01-01"}) < 0.5
    assert dl_45.sel({Coordinates.T.value: "2020-07-01"}) > 0.5


def test_compute_layer_weighted_mean(dummy_data):
    """Test mean temperature calculation using layer weighted mean."""
    temp = dummy_data["temperature"]
    # Day length 0.5
    dl = xr.DataArray(0.5)

    # Day at 0m (20C), Night at 100m (10C)
    # Note: method="nearest" is handled inside the function via sel
    res = compute_layer_weighted_mean(temp, dl, day_layer=0, night_layer=100)
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
    npp = dummy_data["temperature"].isel({Coordinates.Z.value: 0})  # Use temp as dummy NPP
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


def test_compute_production_dynamics(dummy_data):
    """Test production dynamics (aging + recruitment)."""
    # Define consistent cohorts
    cohort_coords = [0, 10, 20]
    cohort_ages = xr.DataArray(cohort_coords, dims="cohort", coords={"cohort": cohort_coords})

    # Create production array matching these cohorts
    # Dimensions: (T, Y, X, cohort) matching dummy_data shape but with new coords
    # shape (2, 2, 2, 3)
    prod = xr.DataArray(
        np.ones((2, 2, 2, 3)),
        coords={
            Coordinates.T.value: dummy_data["times"],
            Coordinates.Y.value: dummy_data["lats"],
            Coordinates.X.value: [0.0, 10.0],  # lons is numpy array in dummy_data
            "cohort": cohort_coords,
        },
        dims=(Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value, "cohort"),
    )

    dt = 1.0

    # Case 1: Recruitment age = 25. No recruitment.
    # Outflow = 1.0 * 0.1 = 0.1
    # Influx to C0 = 0 (assumed)
    # Influx to C1 = Outflow from C0 * 1 = 0.1
    # Influx to C2 = Outflow from C1 * 1 = 0.1

    # Tendency = Influx - Outflow
    # T0 = 0 - 0.1 = -0.1
    # T1 = 0.1 - 0.1 = 0
    # T2 = 0.1 - 0.1 = 0

    rec_age_high = xr.DataArray(25.0)
    res = compute_production_dynamics(prod, rec_age_high, cohort_ages, dt)

    tend = res["production_tendency"]
    src = res["recruitment_source"]

    assert np.allclose(src, 0.0)
    assert np.allclose(tend.isel(cohort=0), -0.1)
    assert np.allclose(tend.isel(cohort=1), 0.0)
    assert np.allclose(tend.isel(cohort=2), 0.0)  # Actually -0.1 outflow + 0.1 inflow = 0

    # Case 2: Recruitment age = 15. Cohort 2 (20) recruited.
    # C0, C1 not recruited. C2 recruited.
    # Outflow C0 = 0.1 -> Goes to C1
    # Outflow C1 = 0.1 -> Goes to C2
    # Outflow C2 = 0.1 -> Goes to Biomass

    # Influx C0 = 0
    # Influx C1 = 0.1 (from C0, not recruited)
    # Influx C2 = 0.1 (from C1, not recruited)

    # T0 = -0.1
    # T1 = 0
    # T2 = Influx(0.1) - Outflow(0.1) = 0

    # Wait, check recruitment source.
    # Source = Outflow C2 = 0.1

    rec_age_mid = xr.DataArray(15.0)
    res_mid = compute_production_dynamics(prod, rec_age_mid, cohort_ages, dt)

    src_mid = res_mid["recruitment_source"]
    tend_mid = res_mid["production_tendency"]

    assert np.allclose(src_mid, 0.1)  # Sum of recruited outflows (only C2)
    assert np.allclose(tend_mid.isel(cohort=0), -0.1)
    assert np.allclose(tend_mid.isel(cohort=1), 0.0)
    assert np.allclose(tend_mid.isel(cohort=2), 0.0), f"Access C2: {tend_mid.isel(cohort=2).values}"


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
