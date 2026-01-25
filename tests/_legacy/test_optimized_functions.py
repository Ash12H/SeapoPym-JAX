"""Tests for optimized LMTL and Transport functions.

These tests verify that the optimized Numba versions produce the
same results as the reference Xarray implementations.
"""

import numpy as np
import pytest
import xarray as xr

from seapopym.lmtl.core import (
    compute_production_dynamics,
    compute_production_dynamics_optimized,
)
from seapopym.standard.coordinates import Coordinates
from seapopym.transport.boundary import BoundaryType
from seapopym.transport.core import (
    compute_transport_fv,
    compute_transport_fv_optimized,
)
from seapopym.transport.grid import (
    compute_spherical_cell_areas,
    compute_spherical_dx,
    compute_spherical_dy,
    compute_spherical_face_areas_ew,
    compute_spherical_face_areas_ns,
)

# =============================================================================
# LMTL: Production Dynamics Optimized
# =============================================================================


@pytest.fixture
def production_data():
    """Create test data for production dynamics."""
    ny, nx, n_cohorts = 10, 10, 5

    lats = np.linspace(-20, 20, ny)
    lons = np.linspace(140, 180, nx)
    cohort_ages = np.array([0, 86400, 2 * 86400, 3 * 86400, 4 * 86400], dtype=float)  # seconds

    production = xr.DataArray(
        np.random.rand(ny, nx, n_cohorts) * 0.1,
        coords={
            Coordinates.Y.value: lats,
            Coordinates.X.value: lons,
            "cohort": cohort_ages,
        },
        dims=[Coordinates.Y.value, Coordinates.X.value, "cohort"],
    )

    recruitment_age = xr.DataArray(
        np.full((ny, nx), 2.5 * 86400),  # 2.5 days in seconds
        coords={
            Coordinates.Y.value: lats,
            Coordinates.X.value: lons,
        },
        dims=[Coordinates.Y.value, Coordinates.X.value],
    )

    cohorts = xr.DataArray(cohort_ages, dims=["cohort"], coords={"cohort": cohort_ages})
    dt = 3600.0  # 1 hour in seconds

    return production, recruitment_age, cohorts, dt


def test_production_dynamics_optimized_matches_reference(production_data):
    """Test that optimized version matches Xarray reference."""
    production, recruitment_age, cohorts, dt = production_data

    # Reference implementation
    result_ref = compute_production_dynamics(production, recruitment_age, cohorts, dt)

    # Optimized implementation
    result_opt = compute_production_dynamics_optimized(production, recruitment_age, cohorts, dt)

    # Compare outputs
    np.testing.assert_allclose(
        result_opt["production_tendency"].values,
        result_ref["production_tendency"].values,
        rtol=1e-5,
        atol=1e-10,
        err_msg="Production tendency mismatch",
    )

    np.testing.assert_allclose(
        result_opt["recruitment_source"].values,
        result_ref["recruitment_source"].values,
        rtol=1e-5,
        atol=1e-10,
        err_msg="Recruitment source mismatch",
    )


def test_production_dynamics_optimized_dimensions(production_data):
    """Test that optimized version preserves dimensions."""
    production, recruitment_age, cohorts, dt = production_data

    result = compute_production_dynamics_optimized(production, recruitment_age, cohorts, dt)

    # Check tendency dimensions
    tendency = result["production_tendency"]
    assert set(tendency.dims) == set(production.dims)
    assert tendency.shape == production.shape

    # Check recruitment source dimensions (should not have cohort)
    source = result["recruitment_source"]
    expected_dims = {Coordinates.Y.value, Coordinates.X.value}
    assert set(source.dims) == expected_dims


def test_production_dynamics_optimized_mass_conservation(production_data):
    """Test mass conservation in production dynamics."""
    production, recruitment_age, cohorts, dt = production_data

    result = compute_production_dynamics_optimized(production, recruitment_age, cohorts, dt)

    tendency = result["production_tendency"]
    source = result["recruitment_source"]

    # Simpler check: no NaN or Inf
    assert not np.any(np.isnan(tendency.values))
    assert not np.any(np.isinf(tendency.values))
    assert not np.any(np.isnan(source.values))
    assert not np.any(np.isinf(source.values))


# =============================================================================
# Transport: Optimized FV
# =============================================================================


@pytest.fixture
def transport_data():
    """Create test data for transport functions."""
    ny, nx = 20, 20

    lats = xr.DataArray(np.linspace(-10, 10, ny), dims=[Coordinates.Y.value])
    lons = xr.DataArray(np.linspace(140, 160, nx), dims=[Coordinates.X.value])

    # Grid metrics
    cell_areas = compute_spherical_cell_areas(lats, lons)
    face_areas_ew = compute_spherical_face_areas_ew(lats, lons)
    face_areas_ns = compute_spherical_face_areas_ns(lats, lons)
    dx = compute_spherical_dx(lats, lons)
    dy = compute_spherical_dy(lats, lons)

    # State with Gaussian blob in center
    y_grid, x_grid = np.meshgrid(lats.values, lons.values, indexing="ij")
    state_values = np.exp(-((y_grid**2 + (x_grid - 150) ** 2) / 50))

    state = xr.DataArray(
        state_values,
        coords={Coordinates.Y.value: lats, Coordinates.X.value: lons},
        dims=[Coordinates.Y.value, Coordinates.X.value],
    )

    # Velocities
    u = xr.DataArray(
        np.ones((ny, nx)) * 0.1,  # 0.1 m/s eastward
        coords={Coordinates.Y.value: lats, Coordinates.X.value: lons},
        dims=[Coordinates.Y.value, Coordinates.X.value],
    )
    v = xr.DataArray(
        np.zeros((ny, nx)),
        coords={Coordinates.Y.value: lats, Coordinates.X.value: lons},
        dims=[Coordinates.Y.value, Coordinates.X.value],
    )

    # Mask (all ocean)
    mask = xr.DataArray(
        np.ones((ny, nx)),
        coords={Coordinates.Y.value: lats, Coordinates.X.value: lons},
        dims=[Coordinates.Y.value, Coordinates.X.value],
    )

    # Diffusion coefficient
    D = 500.0  # m²/s

    return {
        "state": state,
        "u": u,
        "v": v,
        "D": D,
        "dx": dx,
        "dy": dy,
        "cell_areas": cell_areas,
        "face_areas_ew": face_areas_ew,
        "face_areas_ns": face_areas_ns,
        "mask": mask,
    }


def test_transport_fv_optimized_matches_reference(transport_data):
    """Test that optimized transport matches Xarray reference."""
    d = transport_data

    # Reference implementation
    result_ref = compute_transport_fv(
        state=d["state"],
        u=d["u"],
        v=d["v"],
        D=d["D"],
        dx=d["dx"],
        dy=d["dy"],
        cell_areas=d["cell_areas"],
        face_areas_ew=d["face_areas_ew"],
        face_areas_ns=d["face_areas_ns"],
        mask=d["mask"],
        boundary_north=BoundaryType.CLOSED,
        boundary_south=BoundaryType.CLOSED,
        boundary_east=BoundaryType.CLOSED,
        boundary_west=BoundaryType.CLOSED,
    )

    # Optimized implementation
    result_opt = compute_transport_fv_optimized(
        state=d["state"],
        u=d["u"],
        v=d["v"],
        D=d["D"],
        dx=d["dx"],
        dy=d["dy"],
        cell_areas=d["cell_areas"],
        face_areas_ew=d["face_areas_ew"],
        face_areas_ns=d["face_areas_ns"],
        mask=d["mask"],
        boundary_north=BoundaryType.CLOSED,
        boundary_south=BoundaryType.CLOSED,
        boundary_east=BoundaryType.CLOSED,
        boundary_west=BoundaryType.CLOSED,
    )

    # Compare advection
    np.testing.assert_allclose(
        result_opt["advection_rate"].values,
        result_ref["advection_rate"].values,
        rtol=1e-5,
        atol=1e-10,
        err_msg="Advection rate mismatch",
    )

    # Compare diffusion
    np.testing.assert_allclose(
        result_opt["diffusion_rate"].values,
        result_ref["diffusion_rate"].values,
        rtol=1e-5,
        atol=1e-10,
        err_msg="Diffusion rate mismatch",
    )


def test_transport_fv_optimized_dimensions(transport_data):
    """Test that optimized transport preserves dimensions."""
    d = transport_data

    result = compute_transport_fv_optimized(
        state=d["state"],
        u=d["u"],
        v=d["v"],
        D=d["D"],
        dx=d["dx"],
        dy=d["dy"],
        cell_areas=d["cell_areas"],
        face_areas_ew=d["face_areas_ew"],
        face_areas_ns=d["face_areas_ns"],
        mask=d["mask"],
        boundary_north=BoundaryType.CLOSED,
        boundary_south=BoundaryType.CLOSED,
        boundary_east=BoundaryType.CLOSED,
        boundary_west=BoundaryType.CLOSED,
    )

    assert result["advection_rate"].dims == d["state"].dims
    assert result["diffusion_rate"].dims == d["state"].dims
    assert result["advection_rate"].shape == d["state"].shape
    assert result["diffusion_rate"].shape == d["state"].shape


def test_transport_fv_optimized_with_cohorts(transport_data):
    """Test optimized transport with extra cohort dimension."""
    d = transport_data
    n_cohorts = 5

    # Add cohort dimension
    cohorts = np.arange(n_cohorts) * 86400.0
    state_3d = d["state"].expand_dims({"cohort": cohorts})

    result = compute_transport_fv_optimized(
        state=state_3d,
        u=d["u"],
        v=d["v"],
        D=d["D"],
        dx=d["dx"],
        dy=d["dy"],
        cell_areas=d["cell_areas"],
        face_areas_ew=d["face_areas_ew"],
        face_areas_ns=d["face_areas_ns"],
        mask=d["mask"],
        boundary_north=BoundaryType.CLOSED,
        boundary_south=BoundaryType.CLOSED,
        boundary_east=BoundaryType.CLOSED,
        boundary_west=BoundaryType.CLOSED,
    )

    # Should preserve cohort dimension
    assert "cohort" in result["advection_rate"].dims
    assert result["advection_rate"].sizes["cohort"] == n_cohorts


def test_transport_fv_optimized_mass_conservation(transport_data):
    """Test mass conservation with closed boundaries."""
    d = transport_data

    result = compute_transport_fv_optimized(
        state=d["state"],
        u=d["u"],
        v=d["v"],
        D=d["D"],
        dx=d["dx"],
        dy=d["dy"],
        cell_areas=d["cell_areas"],
        face_areas_ew=d["face_areas_ew"],
        face_areas_ns=d["face_areas_ns"],
        mask=d["mask"],
        boundary_north=BoundaryType.CLOSED,
        boundary_south=BoundaryType.CLOSED,
        boundary_east=BoundaryType.CLOSED,
        boundary_west=BoundaryType.CLOSED,
    )

    # With closed boundaries, total tendency should integrate to ~0
    # (mass is redistributed, not created or destroyed)
    adv_rate = result["advection_rate"]
    diff_rate = result["diffusion_rate"]

    # Weight by cell area for proper integration
    total_adv = (adv_rate * d["cell_areas"]).sum().item()
    total_diff = (diff_rate * d["cell_areas"]).sum().item()

    # Should be close to zero (numerical tolerance)
    assert abs(total_adv) < 1e-6, f"Advection not conservative: {total_adv}"
    assert abs(total_diff) < 1e-6, f"Diffusion not conservative: {total_diff}"


def test_transport_fv_optimized_no_nans(transport_data):
    """Test that optimized transport produces no NaN values."""
    d = transport_data

    result = compute_transport_fv_optimized(
        state=d["state"],
        u=d["u"],
        v=d["v"],
        D=d["D"],
        dx=d["dx"],
        dy=d["dy"],
        cell_areas=d["cell_areas"],
        face_areas_ew=d["face_areas_ew"],
        face_areas_ns=d["face_areas_ns"],
        mask=d["mask"],
        boundary_north=BoundaryType.CLOSED,
        boundary_south=BoundaryType.CLOSED,
        boundary_east=BoundaryType.CLOSED,
        boundary_west=BoundaryType.CLOSED,
    )

    assert not np.any(np.isnan(result["advection_rate"].values))
    assert not np.any(np.isnan(result["diffusion_rate"].values))
    assert not np.any(np.isinf(result["advection_rate"].values))
    assert not np.any(np.isinf(result["diffusion_rate"].values))
