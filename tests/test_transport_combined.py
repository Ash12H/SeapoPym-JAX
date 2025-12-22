import numpy as np
import pytest
import xarray as xr

from seapopym.standard.coordinates import Coordinates
from seapopym.transport import (
    compute_spherical_cell_areas,
    compute_spherical_dx,
    compute_spherical_dy,
    compute_spherical_face_areas_ew,
    compute_spherical_face_areas_ns,
    compute_transport_numba,
    compute_transport_xarray,
)


@pytest.fixture
def simple_grid():
    """Create a simple 10x10 grid for testing."""
    lats = np.linspace(-10, 10, 21)
    lons = np.linspace(0, 20, 21)

    lats_da = xr.DataArray(lats, dims=[Coordinates.Y.value])
    lons_da = xr.DataArray(lons, dims=[Coordinates.X.value])

    cell_areas = compute_spherical_cell_areas(lats_da, lons_da)
    face_areas_ew = compute_spherical_face_areas_ew(lats_da, lons_da)
    face_areas_ns = compute_spherical_face_areas_ns(lats_da, lons_da)
    dx = compute_spherical_dx(lats_da, lons_da)
    dy = compute_spherical_dy(lats_da, lons_da)

    return {
        "lats": lats_da,
        "lons": lons_da,
        "cell_areas": cell_areas,
        "face_areas_ew": face_areas_ew,
        "face_areas_ns": face_areas_ns,
        "dx": dx,
        "dy": dy,
    }


@pytest.fixture
def boundary_all_closed():
    """Boundary conditions: all edges closed (as integers)."""
    return {
        "boundary_north": 0,  # 0 = CLOSED
        "boundary_south": 0,
        "boundary_east": 0,
        "boundary_west": 0,
    }


def create_blob(grid, center=(10, 10), radius=2.0, amplitude=100.0):
    """Create a Gaussian blob."""
    lons_vals = grid["lons"].values
    lats_vals = grid["lats"].values

    X, Y = np.meshgrid(lons_vals, lats_vals)
    center_lon = lons_vals[center[1]]
    center_lat = lats_vals[center[0]]

    blob = amplitude * np.exp(-((X - center_lon) ** 2 + (Y - center_lat) ** 2) / (2 * radius**2))

    return xr.DataArray(
        blob,
        dims=[Coordinates.Y.value, Coordinates.X.value],
        coords={
            Coordinates.Y.value: grid["lats"],
            Coordinates.X.value: grid["lons"],
        },
    )


def test_combined_transport_numba_vs_xarray(simple_grid, boundary_all_closed):
    """Verify that Numba and Xarray implementations produce identical results."""
    biomass = create_blob(simple_grid)

    # Advection and Diffusion parameters
    u = xr.full_like(biomass, 0.1)
    v = xr.full_like(biomass, 0.05)
    D = 100.0

    mask = xr.ones_like(biomass)

    # Xarray version
    res_xr = compute_transport_xarray(
        state=biomass,
        u=u,
        v=v,
        D=D,
        dx=simple_grid["dx"],
        dy=simple_grid["dy"],
        cell_areas=simple_grid["cell_areas"],
        face_areas_ew=simple_grid["face_areas_ew"],
        face_areas_ns=simple_grid["face_areas_ns"],
        mask=mask,
        **boundary_all_closed,
    )

    # Numba version
    res_nb = compute_transport_numba(
        state=biomass,
        u=u,
        v=v,
        D=D,
        dx=simple_grid["dx"],
        dy=simple_grid["dy"],
        cell_areas=simple_grid["cell_areas"],
        face_areas_ew=simple_grid["face_areas_ew"],
        face_areas_ns=simple_grid["face_areas_ns"],
        mask=mask,
        **boundary_all_closed,
    )

    # Check advection
    adv_xr = res_xr["advection_rate"]
    adv_nb = res_nb["advection_rate"]

    diff_adv = np.abs(adv_xr - adv_nb)
    max_diff_adv = diff_adv.max().item()

    assert max_diff_adv < 1e-10, f"Advection mismatch: {max_diff_adv}"

    # Check diffusion
    diff_xr = res_xr["diffusion_rate"]
    diff_nb = res_nb["diffusion_rate"]

    diff_diff = np.abs(diff_xr - diff_nb)
    max_diff_diff = diff_diff.max().item()

    assert max_diff_diff < 1e-10, f"Diffusion mismatch: {max_diff_diff}"


def test_combined_transport_mass_conservation(simple_grid, boundary_all_closed):
    """Verify global mass conservation for the combined solver."""
    biomass = create_blob(simple_grid)

    u = xr.full_like(biomass, 1.0)
    v = xr.full_like(biomass, -0.5)
    D = 500.0

    mask = xr.ones_like(biomass)

    res = compute_transport_numba(
        state=biomass,
        u=u,
        v=v,
        D=D,
        dx=simple_grid["dx"],
        dy=simple_grid["dy"],
        cell_areas=simple_grid["cell_areas"],
        face_areas_ew=simple_grid["face_areas_ew"],
        face_areas_ns=simple_grid["face_areas_ns"],
        mask=mask,
        **boundary_all_closed,
    )

    # Total tendency
    tendency = res["advection_rate"] + res["diffusion_rate"]

    # Integrate tendency over the domain Area
    # dM/dt = Integral(dC/dt * dA)
    total_tendency_mass = (tendency * simple_grid["cell_areas"]).sum().item()

    # Should be nearly 0.0 for finite volume with closed boundaries
    assert abs(total_tendency_mass) < 1e-12, f"Mass not conserved! Residual: {total_tendency_mass}"


def test_combined_transport_with_mask(simple_grid, boundary_all_closed):
    """Verify behavior with land mask (island)."""
    biomass = create_blob(simple_grid, center=(5, 5))

    mask = xr.ones_like(biomass, dtype=np.float32)
    # Create island
    mask[10:12, 10:12] = 0.0

    # Ensure biomass is zero on land initially
    biomass = biomass * mask

    u = xr.full_like(biomass, 1.0)
    v = xr.full_like(biomass, 0.0)
    D = 200.0

    res = compute_transport_numba(
        state=biomass,
        u=u,
        v=v,
        D=D,
        dx=simple_grid["dx"],
        dy=simple_grid["dy"],
        cell_areas=simple_grid["cell_areas"],
        face_areas_ew=simple_grid["face_areas_ew"],
        face_areas_ns=simple_grid["face_areas_ns"],
        mask=mask,
        **boundary_all_closed,
    )

    # Total tendency
    tendency_adv = res["advection_rate"]
    tendency_diff = res["diffusion_rate"]

    # Check 1: Tendencies on land are perfectly zero
    tendency_adv_on_land = tendency_adv.where(mask == 0, drop=True)
    tendency_diff_on_land = tendency_diff.where(mask == 0, drop=True)

    assert np.all(tendency_adv_on_land == 0.0), "Advection calculated on land!"
    assert np.all(tendency_diff_on_land == 0.0), "Diffusion calculated on land!"

    # Check 2: Global mass conservation (excluding land)
    tendency_total = tendency_adv + tendency_diff
    total_tendency_mass = (tendency_total * simple_grid["cell_areas"] * mask).sum().item()

    assert (
        abs(total_tendency_mass) < 1e-9
    ), f"Mass not conserved with mask! Residual: {total_tendency_mass}"


def test_separate_components(simple_grid, boundary_all_closed):
    """Test that advection and diffusion components are returned separately."""
    biomass = create_blob(simple_grid)

    u = xr.full_like(biomass, 0.5)
    v = xr.full_like(biomass, 0.0)
    D = 1000.0

    res = compute_transport_xarray(
        state=biomass,
        u=u,
        v=v,
        D=D,
        dx=simple_grid["dx"],
        dy=simple_grid["dy"],
        cell_areas=simple_grid["cell_areas"],
        face_areas_ew=simple_grid["face_areas_ew"],
        face_areas_ns=simple_grid["face_areas_ns"],
        **boundary_all_closed,
    )

    # Check that both components are returned
    assert "advection_rate" in res, "Missing advection_rate"
    assert "diffusion_rate" in res, "Missing diffusion_rate"

    # Check shapes
    assert res["advection_rate"].shape == biomass.shape
    assert res["diffusion_rate"].shape == biomass.shape

    # Check that they are different (not both zero)
    assert not np.allclose(res["advection_rate"].values, 0.0)
    assert not np.allclose(res["diffusion_rate"].values, 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
