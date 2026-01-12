"""Tests for correct behavior of LMTL functions with chunked data (Dask).

Objective: Ensure that chunking strategies (cohort, spatial, mixed) do not alter numerical results.
"""

import numpy as np
import pytest
import xarray as xr

from seapopym.lmtl.core import (
    compute_mortality_tendency,
    compute_production_dynamics,
    compute_production_initialization,
    compute_recruitment_age,
)
from seapopym.standard.coordinates import Coordinates


@pytest.fixture
def lmtl_data():
    """Create realistic LMTL data for testing."""
    # Dimensions
    n_times = 5
    n_lats = 10
    n_lons = 10
    n_cohorts = 20  # Enough to be useful

    # Coordinates
    times = np.arange(n_times)
    lats = np.linspace(-40, 40, n_lats)
    lons = np.linspace(140, 220, n_lons)
    cohorts = np.arange(n_cohorts, dtype=float)  # daily cohorts

    # Variables
    # Random production data
    production = xr.DataArray(
        np.random.rand(n_times, n_lats, n_lons, n_cohorts) + 1.0,
        coords={
            Coordinates.T: times,
            Coordinates.Y: lats,
            Coordinates.X: lons,
            "cohort": cohorts,
        },
        dims=[Coordinates.T, Coordinates.Y, Coordinates.X, "cohort"],
        name="production",
    )

    # Random biomass data
    biomass = xr.DataArray(
        np.random.rand(n_times, n_lats, n_lons) + 1.0,
        coords={Coordinates.T: times, Coordinates.Y: lats, Coordinates.X: lons},
        dims=[Coordinates.T, Coordinates.Y, Coordinates.X],
        name="biomass",
    )

    # Random temperature data (10-30 degC)
    temperature = xr.DataArray(
        np.random.rand(n_times, n_lats, n_lons) * 20 + 10,
        coords={Coordinates.T: times, Coordinates.Y: lats, Coordinates.X: lons},
        dims=[Coordinates.T, Coordinates.Y, Coordinates.X],
        name="temperature",
    )

    # Recruitment age (spatially varying for realism)
    recruitment_age = xr.DataArray(
        np.random.rand(n_times, n_lats, n_lons) * 5 + 5,  # 5-10 days
        coords={Coordinates.T: times, Coordinates.Y: lats, Coordinates.X: lons},
        dims=[Coordinates.T, Coordinates.Y, Coordinates.X],
        name="recruitment_age",
    )

    # Primary production for initialization
    primary_production = xr.DataArray(
        np.random.rand(n_times, n_lats, n_lons) * 10,
        coords={Coordinates.T: times, Coordinates.Y: lats, Coordinates.X: lons},
        dims=[Coordinates.T, Coordinates.Y, Coordinates.X],
        name="primary_production",
    )

    return {
        "production": production,
        "biomass": biomass,
        "temperature": temperature,
        "recruitment_age": recruitment_age,
        "primary_production": primary_production,
        "cohorts": xr.DataArray(cohorts, coords={"cohort": cohorts}, dims="cohort", name="cohort"),
    }


def test_production_dynamics_cohort_chunks(lmtl_data):
    """Test production dynamics with chunking along cohort dimension."""
    production = lmtl_data["production"]
    recruitment_age = lmtl_data["recruitment_age"]
    cohorts = lmtl_data["cohorts"]
    dt = 1.0

    # Reference calculation (compute eagerly)
    ref_results = compute_production_dynamics(production, recruitment_age, cohorts, dt)
    ref_tendency = ref_results["production_tendency"]
    ref_recruitment = ref_results["recruitment_source"]

    # Chunked calculation
    # Chunk cohort dimension
    prod_chunked = production.chunk({"cohort": 5})
    # Coordinates usually don't need chunking, but let's be consistent if they are large arrays
    # recruitment_age and cohorts are small/dense or broadcasted.
    # recruitment_age matches T, Y, X.

    chunked_results = compute_production_dynamics(prod_chunked, recruitment_age, cohorts, dt)
    chunked_tendency = chunked_results["production_tendency"]
    chunked_recruitment = chunked_results["recruitment_source"]

    # Verify return types are lazy
    assert hasattr(chunked_tendency.data, "dask"), "Result should be a dask array"
    assert hasattr(chunked_recruitment.data, "dask"), "Result should be a dask array"

    # Compute and compare
    xr.testing.assert_allclose(chunked_tendency.compute(), ref_tendency)
    xr.testing.assert_allclose(chunked_recruitment.compute(), ref_recruitment)


def test_production_dynamics_spatial_chunks(lmtl_data):
    """Test production dynamics with chunking along spatial dimensions."""
    production = lmtl_data["production"]
    recruitment_age = lmtl_data["recruitment_age"]
    cohorts = lmtl_data["cohorts"]
    dt = 1.0

    # Reference
    ref_results = compute_production_dynamics(production, recruitment_age, cohorts, dt)

    # Chunked: spatial (X, Y)
    prod_chunked = production.chunk({Coordinates.X: 5, Coordinates.Y: 5})
    # recruitment_age must also be chunked to match or compatible
    rec_chunked = recruitment_age.chunk({Coordinates.X: 5, Coordinates.Y: 5})

    chunked_results = compute_production_dynamics(prod_chunked, rec_chunked, cohorts, dt)

    # Check dask
    assert hasattr(chunked_results["production_tendency"].data, "dask")

    # Compare
    xr.testing.assert_allclose(
        chunked_results["production_tendency"].compute(), ref_results["production_tendency"]
    )
    xr.testing.assert_allclose(
        chunked_results["recruitment_source"].compute(), ref_results["recruitment_source"]
    )


def test_production_dynamics_mixed_chunks(lmtl_data):
    """Test production dynamics with mixed chunking (cohort + spatial)."""
    production = lmtl_data["production"]
    recruitment_age = lmtl_data["recruitment_age"]
    cohorts = lmtl_data["cohorts"]
    dt = 1.0

    ref_results = compute_production_dynamics(production, recruitment_age, cohorts, dt)

    # Mixed chunks
    prod_chunked = production.chunk({Coordinates.X: 5, Coordinates.Y: 5, "cohort": 5})
    rec_chunked = recruitment_age.chunk({Coordinates.X: 5, Coordinates.Y: 5})

    chunked_results = compute_production_dynamics(prod_chunked, rec_chunked, cohorts, dt)

    assert hasattr(chunked_results["production_tendency"].data, "dask")

    xr.testing.assert_allclose(
        chunked_results["production_tendency"].compute(), ref_results["production_tendency"]
    )
    xr.testing.assert_allclose(
        chunked_results["recruitment_source"].compute(), ref_results["recruitment_source"]
    )


def test_mortality_tendency_chunking(lmtl_data):
    """Test mortality tendency with spatial chunks."""
    biomass = lmtl_data["biomass"]
    temperature = lmtl_data["temperature"]
    lambda_0 = 0.1
    gamma_lambda = 0.05
    T_ref = 25.0

    ref_results = compute_mortality_tendency(biomass, temperature, lambda_0, gamma_lambda, T_ref)

    # Chunking
    b_chunked = biomass.chunk({Coordinates.X: 5, Coordinates.Y: 5})
    t_chunked = temperature.chunk({Coordinates.X: 5, Coordinates.Y: 5})

    chunked_results = compute_mortality_tendency(
        b_chunked, t_chunked, lambda_0, gamma_lambda, T_ref
    )

    assert hasattr(chunked_results["mortality_loss"].data, "dask")

    xr.testing.assert_allclose(
        chunked_results["mortality_loss"].compute(), ref_results["mortality_loss"]
    )


def test_recruitment_age_chunking(lmtl_data):
    """Test recruitment age computation with chunks."""
    temperature = lmtl_data["temperature"]
    tau_r_0 = 10.0
    gamma_tau_r = 0.1
    T_ref = 25.0

    ref_res = compute_recruitment_age(temperature, tau_r_0, gamma_tau_r, T_ref)

    # Chunking
    t_chunked = temperature.chunk({Coordinates.T: 2, Coordinates.X: 5})

    chunked_res = compute_recruitment_age(t_chunked, tau_r_0, gamma_tau_r, T_ref)

    assert hasattr(chunked_res["output"].data, "dask")
    xr.testing.assert_allclose(chunked_res["output"].compute(), ref_res["output"])


def test_production_initialization_chunking(lmtl_data):
    """Test production initialization with chunks."""
    pp = lmtl_data["primary_production"]
    cohorts = lmtl_data["cohorts"]
    E = 0.1

    ref_res = compute_production_initialization(pp, cohorts, E)

    # Chunking PP (spatial + time)
    pp_chunked = pp.chunk({Coordinates.T: 2, Coordinates.X: 5, Coordinates.Y: 5})

    chunked_res = compute_production_initialization(pp_chunked, cohorts, E)

    assert hasattr(chunked_res["output"].data, "dask")

    xr.testing.assert_allclose(chunked_res["output"].compute(), ref_res["output"])
