import numpy as np
import pytest
import xarray as xr

from seapopym.standard.coordinates import Coordinates
from seapopym.transport import (
    BoundaryConditions,
    BoundaryType,
    compute_spherical_cell_areas,
    compute_spherical_dx,
    compute_spherical_dy,
    compute_spherical_face_areas_ew,
    compute_spherical_face_areas_ns,
    compute_transport_fv,
    get_neighbors_with_bc,
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

    res = compute_transport_fv(
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
    res_closed = compute_transport_fv(
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

    res = compute_transport_fv(
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


def test_get_neighbors_with_bc_direct():
    """Test the get_neighbors_with_bc helper function directly.

    Grid Layout (3x3):
    (2,0)=7  (2,1)=8  (2,2)=9  <- North (Max Y)
    (1,0)=4  (1,1)=5  (1,2)=6
    (0,0)=1  (0,1)=2  (0,2)=3  <- South (Min Y)
      ^        ^        ^
    West      Mid      East
    (Min X)           (Max X)
    """
    # Create a simple 2D DataArray (3x3)
    data = xr.DataArray(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        dims=[Coordinates.Y.value, Coordinates.X.value],
        coords={
            Coordinates.Y.value: [0, 1, 2],
            Coordinates.X.value: [0, 1, 2],
        },
    )

    # 1. Test CLOSED boundaries (Zero Gradient / Neumann)
    # The neighbor value should be valid (not NaN) and equal to the edge cell itself.
    bc_closed = BoundaryConditions(
        north=BoundaryType.CLOSED,
        south=BoundaryType.CLOSED,
        east=BoundaryType.CLOSED,
        west=BoundaryType.CLOSED,
    )
    west, east, south, north = get_neighbors_with_bc(data, bc_closed)

    # SOUTH-WEST CORNER (0,0) = 1
    # West neighbor of (0,0) -> Ghost cell outside West boundary. Should repeat 1.
    assert west.isel({Coordinates.Y.value: 0, Coordinates.X.value: 0}) == 1
    # South neighbor of (0,0) -> Ghost cell outside South boundary. Should repeat 1.
    assert south.isel({Coordinates.Y.value: 0, Coordinates.X.value: 0}) == 1

    # NORTH-EAST CORNER (2,2) = 9
    # East neighbor of (2,2) -> Ghost cell outside East boundary. Should repeat 9.
    assert east.isel({Coordinates.Y.value: 2, Coordinates.X.value: 2}) == 9
    # North neighbor of (2,2) -> Ghost cell outside North boundary. Should repeat 9.
    assert north.isel({Coordinates.Y.value: 2, Coordinates.X.value: 2}) == 9

    # INTERNAL CHECK
    # West neighbor of (0,1) [Value 2] is (0,0) [Value 1]
    assert west.isel({Coordinates.Y.value: 0, Coordinates.X.value: 1}) == 1

    # 2. Test PERIODIC boundaries
    # The neighbor value should wrap around the domain.
    bc_periodic = BoundaryConditions(
        north=BoundaryType.PERIODIC,
        south=BoundaryType.PERIODIC,
        east=BoundaryType.PERIODIC,
        west=BoundaryType.PERIODIC,
    )
    west_p, east_p, south_p, north_p = get_neighbors_with_bc(data, bc_periodic)

    # Longitudinal Wrap (X)
    # West neighbor of (0,0) [Value 1] -> wraps to East edge (0,2) [Value 3]
    assert west_p.isel({Coordinates.Y.value: 0, Coordinates.X.value: 0}) == 3
    # East neighbor of (0,2) [Value 3] -> wraps to West edge (0,0) [Value 1]
    assert east_p.isel({Coordinates.Y.value: 0, Coordinates.X.value: 2}) == 1

    # Latitudinal Wrap (Y)
    # South neighbor of (0,0) [Value 1] -> wraps to North edge (2,0) [Value 7]
    assert south_p.isel({Coordinates.Y.value: 0, Coordinates.X.value: 0}) == 7
    # North neighbor of (2,0) [Value 7] -> wraps to South edge (0,0) [Value 1]
    assert north_p.isel({Coordinates.Y.value: 2, Coordinates.X.value: 0}) == 1

    # 3. Test OPEN boundaries
    # Should behave like CLOSED here (ghost cell = edge value)
    bc_open = BoundaryConditions(
        north=BoundaryType.OPEN,
        south=BoundaryType.OPEN,
        east=BoundaryType.OPEN,
        west=BoundaryType.OPEN,
    )
    west_o, east_o, south_o, north_o = get_neighbors_with_bc(data, bc_open)

    # South-West (0,0) -> Neighbors should be itself (1)
    assert west_o.isel({Coordinates.Y.value: 0, Coordinates.X.value: 0}) == 1
    assert south_o.isel({Coordinates.Y.value: 0, Coordinates.X.value: 0}) == 1

    # North-East (2,2) -> Neighbors should be itself (9)
    assert east_o.isel({Coordinates.Y.value: 2, Coordinates.X.value: 2}) == 9
    assert north_o.isel({Coordinates.Y.value: 2, Coordinates.X.value: 2}) == 9


def test_mixed_boundaries_coverage():
    """Test mixed boundaries (e.g. West=PERIODIC, East=CLOSED) for full coverage.

    This covers the branch where we are NOT fully periodic (so we use shift),
    but one side is technically PERIODIC, so it skips the 'fillna' step for that side.
    """
    # Use floats to ensure NaN handling works as expected
    data = xr.DataArray(
        np.array([[1.0, 2.0, 3.0]]),
        dims=[Coordinates.Y.value, Coordinates.X.value],
        coords={Coordinates.Y.value: [0], Coordinates.X.value: [0, 1, 2]},
    )

    # Hybrid X: West=PERIODIC (skip fill), East=CLOSED (do fill)
    bc_mixed_x = BoundaryConditions(
        north=BoundaryType.CLOSED,
        south=BoundaryType.CLOSED,
        east=BoundaryType.CLOSED,
        west=BoundaryType.PERIODIC,
    )
    from seapopym.transport import get_neighbors_with_bc

    west, east, _, _ = get_neighbors_with_bc(data, bc_mixed_x)

    # West neighbor of (0,0): should be NaN because we skipped fillna
    # data is (1, 3). Index (0,0) west neighbor is ghost cell.
    # Check directly on numpy array to avoid Xarray scalar ambiguity
    assert np.isnan(west.values[0, 0])

    # East neighbor of (0,2): should be filled with self (3.0)
    # Index (0,2) east neighbor is ghost cell.
    assert east.values[0, 2] == 3.0

    # Hybrid Y: South=PERIODIC (skip fill), North=CLOSED (do fill)
    bc_mixed_y = BoundaryConditions(
        north=BoundaryType.CLOSED,
        south=BoundaryType.PERIODIC,
        east=BoundaryType.CLOSED,
        west=BoundaryType.CLOSED,
    )
    _, _, south, north = get_neighbors_with_bc(data, bc_mixed_y)

    # South neighbor (should be NaN) of (0,0)
    # south array has shape (1, 3) like input.
    # Shift +1 along Y puts NaN in row 0.
    assert np.isnan(south.values[0, 0])

    # -------------------------------------------------------------------------
    # Symmetric Mixed X: West=CLOSED, East=PERIODIC
    # -------------------------------------------------------------------------
    bc_mixed_x_sym = BoundaryConditions(
        north=BoundaryType.CLOSED,
        south=BoundaryType.CLOSED,
        east=BoundaryType.PERIODIC,
        west=BoundaryType.CLOSED,
    )
    west_sym, east_sym, _, _ = get_neighbors_with_bc(data, bc_mixed_x_sym)

    # West (CLOSED) -> Should be filled
    assert west_sym.values[0, 0] == 1.0

    # East (PERIODIC) -> Should be NaN (skipped fill)
    assert np.isnan(east_sym.values[0, 2])

    # -------------------------------------------------------------------------
    # Symmetric Mixed Y: South=CLOSED, North=PERIODIC
    # -------------------------------------------------------------------------
    bc_mixed_y_sym = BoundaryConditions(
        north=BoundaryType.PERIODIC,
        south=BoundaryType.CLOSED,
        east=BoundaryType.CLOSED,
        west=BoundaryType.CLOSED,
    )
    _, _, south_sym, north_sym = get_neighbors_with_bc(data, bc_mixed_y_sym)

    # South (CLOSED) -> Should be filled
    assert south_sym.values[0, 0] == 1.0

    # North (PERIODIC) -> Should be NaN (skipped fill)
    assert np.isnan(north_sym.values[0, 0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
