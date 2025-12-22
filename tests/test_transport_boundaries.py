import numpy as np
import pytest
import xarray as xr

from seapopym.standard.coordinates import Coordinates
from seapopym.transport import (
    BoundaryType,
    compute_spherical_cell_areas,
    compute_spherical_dx,
    compute_spherical_dy,
    compute_spherical_face_areas_ew,
    compute_spherical_face_areas_ns,
    compute_transport_numba,
    compute_transport_xarray,
)


@pytest.fixture
def periodic_grid():
    """Create a periodic grid in X (longitude)."""
    # 20 cells in X from 0 to 40 (each 2 units)
    # 10 cells in Y from -10 to 10
    lats = np.linspace(-10, 10, 11)
    lons = np.linspace(0, 40, 21)

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
def boundary_periodic_x():
    """Periodic in X, closed in Y."""
    return {
        "boundary_north": BoundaryType.CLOSED,
        "boundary_south": BoundaryType.CLOSED,
        "boundary_east": BoundaryType.PERIODIC,
        "boundary_west": BoundaryType.PERIODIC,
    }


@pytest.fixture
def boundary_open_x():
    """Open in X, closed in Y."""
    return {
        "boundary_north": BoundaryType.CLOSED,
        "boundary_south": BoundaryType.CLOSED,
        "boundary_east": BoundaryType.OPEN,
        "boundary_west": BoundaryType.OPEN,
    }


def test_periodic_transport_numba_vs_xarray(periodic_grid, boundary_periodic_x):
    """Verify Numba vs Xarray with periodic boundaries."""
    # Place a blob near the right boundary to test wrap-around
    # Grid is 11 (lat) x 21 (lon)
    # Center at index (5, 19) is near the east edge
    amplitude = 100.0
    lons_vals = periodic_grid["lons"].values
    lats_vals = periodic_grid["lats"].values
    X, Y = np.meshgrid(lons_vals, lats_vals)

    # Gaussian blob at the east edge
    center_lon = lons_vals[19]
    center_lat = lats_vals[5]
    radius = 2.0
    blob = amplitude * np.exp(-((X - center_lon) ** 2 + (Y - center_lat) ** 2) / (2 * radius**2))

    biomass = xr.DataArray(
        blob,
        dims=[Coordinates.Y.value, Coordinates.X.value],
        coords={
            Coordinates.Y.value: periodic_grid["lats"],
            Coordinates.X.value: periodic_grid["lons"],
        },
    )

    u = xr.full_like(biomass, 1.0)  # Flow towards East
    v = xr.full_like(biomass, 0.0)
    D = 100.0

    # Xarray version
    res_xr = compute_transport_xarray(
        biomass,
        u,
        v,
        D,
        periodic_grid["dx"],
        periodic_grid["dy"],
        periodic_grid["cell_areas"],
        periodic_grid["face_areas_ew"],
        periodic_grid["face_areas_ns"],
        **boundary_periodic_x,
    )

    # Numba version
    res_nb = compute_transport_numba(
        biomass,
        u,
        v,
        D,
        periodic_grid["dx"],
        periodic_grid["dy"],
        periodic_grid["cell_areas"],
        periodic_grid["face_areas_ew"],
        periodic_grid["face_areas_ns"],
        **boundary_periodic_x,
    )

    # Check advection consistency
    assert np.allclose(res_xr["advection_rate"], res_nb["advection_rate"], atol=1e-10)

    # Check diffusion consistency
    assert np.allclose(res_xr["diffusion_rate"], res_nb["diffusion_rate"], atol=1e-10)


def test_open_boundary_flow(periodic_grid, boundary_open_x):
    """Verify that OPEN boundaries allow biomass to leave the domain."""
    # Place a blob near the East boundary
    amplitude = 100.0
    lons_vals = periodic_grid["lons"].values
    lats_vals = periodic_grid["lats"].values
    X, Y = np.meshgrid(lons_vals, lats_vals)
    center_lon = lons_vals[19]  # Almost at the edge
    center_lat = lats_vals[5]
    radius = 2.0
    blob = amplitude * np.exp(-((X - center_lon) ** 2 + (Y - center_lat) ** 2) / (2 * radius**2))

    biomass = xr.DataArray(
        blob,
        dims=[Coordinates.Y.value, Coordinates.X.value],
    )

    u = xr.full_like(biomass, 10.0)  # Strong flow towards East
    v = xr.full_like(biomass, 0.0)
    D = 0.0  # No diffusion for simpler check

    res = compute_transport_numba(
        biomass,
        u,
        v,
        D,
        periodic_grid["dx"],
        periodic_grid["dy"],
        periodic_grid["cell_areas"],
        periodic_grid["face_areas_ew"],
        periodic_grid["face_areas_ns"],
        **boundary_open_x,
    )

    # Total mass change should be NEGATIVE (biomass leaving through East)
    total_tendency = res["advection_rate"]
    total_mass_change = (total_tendency * periodic_grid["cell_areas"]).sum().item()

    assert total_mass_change < -1.0  # Significant amount leaves

    # Compare with CLOSED version
    boundary_closed = {
        "boundary_north": BoundaryType.CLOSED,
        "boundary_south": BoundaryType.CLOSED,
        "boundary_east": BoundaryType.CLOSED,
        "boundary_west": BoundaryType.CLOSED,
    }
    res_closed = compute_transport_numba(
        biomass,
        u,
        v,
        D,
        periodic_grid["dx"],
        periodic_grid["dy"],
        periodic_grid["cell_areas"],
        periodic_grid["face_areas_ew"],
        periodic_grid["face_areas_ns"],
        **boundary_closed,
    )
    total_mass_change_closed = (
        (res_closed["advection_rate"] * periodic_grid["cell_areas"]).sum().item()
    )
    assert abs(total_mass_change_closed) < 1e-7  # Conserved in CLOSED


def test_periodic_mass_conservation(periodic_grid, boundary_periodic_x):
    """Verify mass conservation with periodic boundaries."""
    # Complex state with values everywhere
    state = xr.DataArray(
        np.random.rand(*periodic_grid["cell_areas"].shape),
        dims=[Coordinates.Y.value, Coordinates.X.value],
    )

    u = xr.DataArray(np.random.rand(*state.shape) - 0.5, dims=state.dims)
    v = xr.DataArray(np.random.rand(*state.shape) - 0.5, dims=state.dims)
    D = 200.0

    res = compute_transport_numba(
        state,
        u,
        v,
        D,
        periodic_grid["dx"],
        periodic_grid["dy"],
        periodic_grid["cell_areas"],
        periodic_grid["face_areas_ew"],
        periodic_grid["face_areas_ns"],
        **boundary_periodic_x,
    )

    total_tendency = res["advection_rate"] + res["diffusion_rate"]
    total_mass_change = (total_tendency * periodic_grid["cell_areas"]).sum().item()

    # Mass should be conserved (sum of divergences = 0 for periodic/closed)
    assert abs(total_mass_change) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
