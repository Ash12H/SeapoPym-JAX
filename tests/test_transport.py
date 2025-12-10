"""Tests for transport module (advection and diffusion).

These tests verify:
1. Advection moves biomass in the direction of flow
2. Mass conservation for both advection and diffusion
3. Diffusion spreads biomass and reduces peaks
4. Boundary conditions are properly enforced
"""

import numpy as np
import pytest
import xarray as xr

from seapopym.standard.coordinates import Coordinates
from seapopym.transport import (
    BoundaryConditions,
    BoundaryType,
    check_diffusion_stability,
    compute_advection_cfl,
    compute_advection_tendency,
    compute_diffusion_tendency,
    compute_spherical_cell_areas,
    compute_spherical_dx,
    compute_spherical_dy,
    compute_spherical_face_areas_ew,
    compute_spherical_face_areas_ns,
)


@pytest.fixture
def simple_grid():
    """Create a simple 10x10 grid for testing."""
    lats = np.linspace(0, 10, 10)
    lons = np.linspace(0, 10, 10)

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
def boundary_closed_periodic():
    """Boundary conditions: closed poles, periodic longitude."""
    return BoundaryConditions(
        north=BoundaryType.CLOSED,
        south=BoundaryType.CLOSED,
        east=BoundaryType.PERIODIC,
        west=BoundaryType.PERIODIC,
    )


@pytest.fixture
def boundary_all_closed():
    """Boundary conditions: all edges closed."""
    return BoundaryConditions(
        north=BoundaryType.CLOSED,
        south=BoundaryType.CLOSED,
        east=BoundaryType.CLOSED,
        west=BoundaryType.CLOSED,
    )


def create_blob(grid, center=(5, 5), radius=2, amplitude=100.0):
    """Create a Gaussian blob centered at (center_y, center_x)."""
    ny, nx = len(grid["lats"]), len(grid["lons"])
    y_indices, x_indices = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")

    center_y, center_x = center
    distance = np.sqrt((y_indices - center_y) ** 2 + (x_indices - center_x) ** 2)

    blob = amplitude * np.exp(-(distance**2) / (2 * radius**2))

    return xr.DataArray(
        blob,
        dims=[Coordinates.Y.value, Coordinates.X.value],
        coords={
            Coordinates.Y.value: grid["lats"],
            Coordinates.X.value: grid["lons"],
        },
    )


def test_advection_eastward_movement(simple_grid, boundary_closed_periodic):
    """Test that advection with eastward flow moves biomass to the east."""
    # Create initial blob at center
    biomass = create_blob(simple_grid, center=(5, 5), radius=1.5)

    # Uniform eastward velocity: u = 1.0 m/s, v = 0
    u = xr.full_like(biomass, 1.0)
    v = xr.full_like(biomass, 0.0)

    # Compute advection tendency
    result = compute_advection_tendency(
        state=biomass,
        u=u,
        v=v,
        cell_areas=simple_grid["cell_areas"],
        face_areas_ew=simple_grid["face_areas_ew"],
        face_areas_ns=simple_grid["face_areas_ns"],
        boundary_conditions=boundary_closed_periodic,
        mask=None,
    )

    tendency = result["advection_rate"]

    # Check that tendency is not all zeros
    assert not np.allclose(tendency.values, 0.0), "Advection tendency is zero!"

    # Integrate tendency for one timestep
    dt = 3600.0  # 1 hour
    biomass_new = biomass + tendency * dt

    # Compute center of mass before and after
    total_before = (biomass * simple_grid["cell_areas"]).sum().values
    total_after = (biomass_new * simple_grid["cell_areas"]).sum().values

    # With periodic boundaries, mass should be conserved
    mass_conservation_error = abs(total_after - total_before) / total_before
    assert (
        mass_conservation_error < 0.01
    ), f"Mass not conserved: {mass_conservation_error:.2%} error"

    # Check that center of mass moved eastward (increased x)
    def center_of_mass(field, areas):
        total = (field * areas).sum()
        x_idx = np.arange(len(field.coords[Coordinates.X.value]))
        y_idx = np.arange(len(field.coords[Coordinates.Y.value]))
        X, Y = np.meshgrid(x_idx, y_idx)

        x_com = ((field * areas * X).sum() / total).values
        y_com = ((field * areas * Y).sum() / total).values
        return x_com, y_com

    x_before, y_before = center_of_mass(biomass, simple_grid["cell_areas"])
    x_after, y_after = center_of_mass(biomass_new, simple_grid["cell_areas"])

    print(f"\nCenter of mass before: ({x_before:.2f}, {y_before:.2f})")
    print(f"Center of mass after:  ({x_after:.2f}, {y_after:.2f})")
    print(f"Displacement: Δx = {x_after - x_before:.2f}, Δy = {y_after - y_before:.2f}")

    # With eastward flow, x should increase
    assert x_after > x_before, f"Blob did not move east! Δx = {x_after - x_before:.4f}"


def test_advection_mass_conservation_closed(simple_grid, boundary_all_closed):
    """Test that advection conserves mass with closed boundaries."""
    # Create initial blob
    biomass = create_blob(simple_grid, center=(5, 5), radius=2)

    # Uniform eastward velocity
    u = xr.full_like(biomass, 2.0)
    v = xr.full_like(biomass, 0.0)

    # Compute tendency
    result = compute_advection_tendency(
        state=biomass,
        u=u,
        v=v,
        cell_areas=simple_grid["cell_areas"],
        face_areas_ew=simple_grid["face_areas_ew"],
        face_areas_ns=simple_grid["face_areas_ns"],
        boundary_conditions=boundary_all_closed,
        mask=None,
    )

    # Integrate for multiple timesteps
    biomass_current = biomass.copy()
    dt = 1800.0  # 30 minutes

    total_initial = (biomass * simple_grid["cell_areas"]).sum().values

    for _ in range(10):
        result = compute_advection_tendency(
            state=biomass_current,
            u=u,
            v=v,
            cell_areas=simple_grid["cell_areas"],
            face_areas_ew=simple_grid["face_areas_ew"],
            face_areas_ns=simple_grid["face_areas_ns"],
            boundary_conditions=boundary_all_closed,
            mask=None,
        )
        biomass_current = biomass_current + result["advection_rate"] * dt

    total_final = (biomass_current * simple_grid["cell_areas"]).sum().values

    mass_error = abs(total_final - total_initial) / total_initial
    print(f"\nMass conservation error: {mass_error:.4%}")
    print(f"Total mass initial: {total_initial:.2e}")
    print(f"Total mass final:   {total_final:.2e}")

    assert mass_error < 0.05, f"Mass conservation error too large: {mass_error:.2%}"


def test_diffusion_spreading(simple_grid, boundary_closed_periodic):
    """Test that diffusion spreads biomass and reduces peak."""
    # Create sharp blob
    biomass = create_blob(simple_grid, center=(5, 5), radius=0.8, amplitude=100.0)

    # Diffusion coefficient
    D = 5000.0  # m²/s

    # Check stability
    dt = 1800.0  # 30 minutes
    stability = check_diffusion_stability(
        D=D,
        dx=simple_grid["dx"],
        dy=simple_grid["dy"],
        dt=dt,
    )
    assert stability["is_stable"], f"Diffusion is unstable! CFL = {stability['cfl_diffusion']:.3f}"

    # Compute tendency
    result = compute_diffusion_tendency(
        state=biomass,
        D=D,
        dx=simple_grid["dx"],
        dy=simple_grid["dy"],
        boundary_conditions=boundary_closed_periodic,
        mask=None,
    )

    tendency = result["diffusion_rate"]

    # Integrate
    biomass_new = biomass + tendency * dt

    # Check that peak decreased (diffusion smooths)
    peak_before = biomass.max().values
    peak_after = biomass_new.max().values

    print(f"\nPeak before: {peak_before:.2f}")
    print(f"Peak after:  {peak_after:.2f}")
    print(f"Peak reduction: {(peak_before - peak_after) / peak_before:.1%}")

    assert peak_after < peak_before, "Diffusion did not reduce peak!"

    # Check mass conservation
    total_before = (biomass * simple_grid["cell_areas"]).sum().values
    total_after = (biomass_new * simple_grid["cell_areas"]).sum().values

    mass_error = abs(total_after - total_before) / total_before
    print(f"Mass conservation error: {mass_error:.4%}")

    assert mass_error < 0.01, f"Diffusion does not conserve mass: {mass_error:.2%} error"


def test_diffusion_stability_check(simple_grid):
    """Test diffusion stability checking utility."""
    D = 10000.0  # Diffusion coefficient

    # Calculate theoretical dt_max from stability criterion: dt_max = min(dx², dy²) / (4·D)
    dx_min = simple_grid["dx"].min().values
    dy_min = simple_grid["dy"].min().values
    dt_max_theory = min(dx_min**2, dy_min**2) / (4 * D)

    # Use dt values relative to dt_max to ensure we test both stable and unstable regimes
    dt_stable = 0.1 * dt_max_theory  # 10% of limit - should be stable (CFL = 0.025)
    dt_unstable = 2.0 * dt_max_theory  # 200% of limit - should be unstable (CFL = 0.5)

    stability_stable = check_diffusion_stability(
        D=D,
        dx=simple_grid["dx"],
        dy=simple_grid["dy"],
        dt=dt_stable,
    )

    stability_unstable = check_diffusion_stability(
        D=D,
        dx=simple_grid["dx"],
        dy=simple_grid["dy"],
        dt=dt_unstable,
    )

    print(f"\ndt_max (theory): {dt_max_theory:.1f}s ({dt_max_theory / 3600:.1f} hours)")
    print(
        f"Stable dt: {dt_stable:.1f}s ({dt_stable / 3600:.2f}h), CFL = {stability_stable['cfl_diffusion']:.3f}"
    )
    print(
        f"Unstable dt: {dt_unstable:.1f}s ({dt_unstable / 3600:.2f}h), CFL = {stability_unstable['cfl_diffusion']:.3f}"
    )

    assert stability_stable["is_stable"], "Small timestep should be stable"
    assert not stability_unstable["is_stable"], "Large timestep should be unstable"
    assert stability_stable["dt_max"] == pytest.approx(
        dt_max_theory, rel=0.01
    ), "dt_max should match theory"


def test_combined_advection_diffusion(simple_grid, boundary_closed_periodic):
    """Test combined advection and diffusion."""
    # Initial blob
    biomass = create_blob(simple_grid, center=(3, 5), radius=1.5, amplitude=100.0)

    # Eastward velocity
    u = xr.full_like(biomass, 1.5)
    v = xr.full_like(biomass, 0.0)

    # Diffusion
    D = 3000.0
    dt = 1800.0  # 30 minutes

    # Check stability
    stability = check_diffusion_stability(
        D=D,
        dx=simple_grid["dx"],
        dy=simple_grid["dy"],
        dt=dt,
    )
    assert stability["is_stable"], "Simulation unstable!"

    # Run for multiple steps
    biomass_current = biomass.copy()

    for _ in range(5):
        # Advection
        adv_result = compute_advection_tendency(
            state=biomass_current,
            u=u,
            v=v,
            cell_areas=simple_grid["cell_areas"],
            face_areas_ew=simple_grid["face_areas_ew"],
            face_areas_ns=simple_grid["face_areas_ns"],
            boundary_conditions=boundary_closed_periodic,
            mask=None,
        )

        # Diffusion
        diff_result = compute_diffusion_tendency(
            state=biomass_current,
            D=D,
            dx=simple_grid["dx"],
            dy=simple_grid["dy"],
            boundary_conditions=boundary_closed_periodic,
            mask=None,
        )

        # Combined tendency
        total_tendency = adv_result["advection_rate"] + diff_result["diffusion_rate"]

        # Update
        biomass_current = biomass_current + total_tendency * dt

    # Check that blob moved and spread
    # Center of mass should have moved east
    def center_of_mass_x(field, areas):
        total = (field * areas).sum()
        x_idx = np.arange(len(field.coords[Coordinates.X.value]))
        X = x_idx[None, :]  # Broadcast to 2D
        x_com = ((field * areas * X).sum() / total).values
        return x_com

    x_initial = center_of_mass_x(biomass, simple_grid["cell_areas"])
    x_final = center_of_mass_x(biomass_current, simple_grid["cell_areas"])

    print(f"\nX position: initial = {x_initial:.2f}, final = {x_final:.2f}")
    print(f"Displacement: {x_final - x_initial:.2f} grid cells")

    assert x_final > x_initial, "Blob did not move eastward!"

    # Peak should have decreased (diffusion)
    peak_initial = biomass.max().values
    peak_final = biomass_current.max().values

    print(f"Peak: initial = {peak_initial:.2f}, final = {peak_final:.2f}")

    assert peak_final < peak_initial, "Diffusion did not reduce peak!"


def test_advection_with_mask(simple_grid, boundary_closed_periodic):
    """Test advection with land masking (island)."""
    # Create an island in the middle (mask=0)
    mask = xr.ones_like(simple_grid["lats"] * simple_grid["lons"])
    # Set center 2x2 cells as land
    mask[4:6, 4:6] = 0.0

    # Initialize biomass west of the island
    biomass = create_blob(simple_grid, center=(5, 2), radius=1.0)
    biomass = biomass * mask  # Ensure start is clean

    # Uniform eastward velocity
    u = xr.full_like(biomass, 1.0)
    v = xr.full_like(biomass, 0.0)

    # Compute tendency
    result = compute_advection_tendency(
        state=biomass,
        u=u,
        v=v,
        cell_areas=simple_grid["cell_areas"],
        face_areas_ew=simple_grid["face_areas_ew"],
        face_areas_ns=simple_grid["face_areas_ns"],
        boundary_conditions=boundary_closed_periodic,
        mask=mask,
    )

    tendency = result["advection_rate"]

    # Check that tendency is zero on land
    assert np.all(tendency.where(mask == 0, drop=True) == 0.0), "Advection calculated on land!"

    # Integrate for a bit
    dt = 3600.0
    biomass_new = biomass + tendency * dt

    # Biomass shouldn't enter land
    assert np.all(biomass_new.where(mask == 0, drop=True) == 0.0), "Biomass moved onto land!"

    # Check mass conservation (masked domain)
    total_before = (biomass * simple_grid["cell_areas"] * mask).sum().values
    total_after = (biomass_new * simple_grid["cell_areas"] * mask).sum().values

    # Mass should pile up against the island or flow around (if flow allowed around)
    # But strictly speaking, standard upwind into a wall stops flux.
    # The divergence at the face entering land will be:
    # Flux_in (from west) - Flux_out (to east, which is 0 because face is closed)
    # So mass accumulates in the cell just west of the island.

    # Total mass should still be conserved in the system (what enters the cell stays there)
    mass_error = abs(total_after - total_before) / total_before
    assert mass_error < 0.01, f"Mass not conserved with mask: {mass_error:.2%} error"


def test_diffusion_with_mask(simple_grid, boundary_closed_periodic):
    """Test diffusion with land masking (zero-gradient BC)."""
    # Create an island
    mask = xr.ones_like(simple_grid["lats"] * simple_grid["lons"])
    mask[4:6, 4:6] = 0.0

    # Initialize biomass near the island
    biomass = create_blob(simple_grid, center=(5, 3), radius=1.0)
    # Ensure zero on land initially
    biomass = biomass * mask

    D = 2000.0

    result = compute_diffusion_tendency(
        state=biomass,
        D=D,
        dx=simple_grid["dx"],
        dy=simple_grid["dy"],
        boundary_conditions=boundary_closed_periodic,
        mask=mask,
    )

    tendency = result["diffusion_rate"]

    # Check tendencies on land are zero
    assert np.all(tendency.where(mask == 0, drop=True) == 0.0), "Diffusion calculated on land!"

    dt = 1800.0
    biomass_new = biomass + tendency * dt

    # Check mass conservation
    total_before = (biomass * simple_grid["cell_areas"] * mask).sum().values
    total_after = (biomass_new * simple_grid["cell_areas"] * mask).sum().values

    mass_error = abs(total_after - total_before) / total_before
    assert mass_error < 0.01, f"Mass not conserved with mask: {mass_error:.2%} error"


def test_advection_stability_check(simple_grid):
    """Test advection CFL condition check."""
    u = 1.0  # m/s
    v = 1.0  # m/s

    dx_min = simple_grid["dx"].min().values
    dy_min = simple_grid["dy"].min().values

    # Calculate dt that exactly hits CFL=1 for the worst case (smallest cell)
    # CFL = |u|dt/dx + |v|dt/dy = dt * (|u|/dx + |v|/dy)
    # dt_limit = 1 / (|u|/dx + |v|/dy)
    metric = (abs(u) / dx_min) + (abs(v) / dy_min)
    dt_limit = 1.0 / metric

    # Test stable case (CFL = 0.5)
    dt_stable = 0.5 * dt_limit
    stability_stable = compute_advection_cfl(
        u=u, v=v, dx=simple_grid["dx"], dy=simple_grid["dy"], dt=dt_stable
    )
    assert stability_stable["is_stable"], "Should be stable"
    assert stability_stable["cfl_max"] < 0.6

    # Test unstable case (CFL = 1.5)
    dt_unstable = 1.5 * dt_limit
    stability_unstable = compute_advection_cfl(
        u=u, v=v, dx=simple_grid["dx"], dy=simple_grid["dy"], dt=dt_unstable
    )
    assert not stability_unstable["is_stable"], "Should be unstable"
    assert stability_unstable["cfl_max"] > 1.0


def test_high_latitude_stability(simple_grid):
    """Test stability check at high latitudes where dx is small."""
    # Create high latitude grid (85°N to 89°N)
    lats = np.linspace(85, 89, 5)
    lons = np.linspace(0, 10, 5)

    lats_da = xr.DataArray(lats, dims=[Coordinates.Y.value])
    lons_da = xr.DataArray(lons, dims=[Coordinates.X.value])

    dx = compute_spherical_dx(lats_da, lons_da)
    dy = compute_spherical_dy(lats_da, lons_da)

    D = 1000.0

    # Check stability at equator for comparison
    # Need at least 2 points to compute dlats/dlons if using compute functions
    lats_eq = np.array([0.0, 1.0])
    lats_eq_da = xr.DataArray(lats_eq, dims=[Coordinates.Y.value])

    dx_eq = compute_spherical_dx(lats_eq_da, lons_da)
    dy_eq = compute_spherical_dy(lats_eq_da, lons_da)

    stability_eq = check_diffusion_stability(D, dx_eq, dy_eq, dt=1.0)
    stability_high = check_diffusion_stability(D, dx, dy, dt=1.0)

    print(f"\ndt_max (Equator): {stability_eq['dt_max']:.1f} s")
    print(f"dt_max (85°N):    {stability_high['dt_max']:.1f} s")
    print(f"Ratio: {stability_eq['dt_max'] / stability_high['dt_max']:.1f}")

    # Verify significant reduction in time step
    assert (
        stability_high["dt_max"] < stability_eq["dt_max"] / 50.0
    ), "dt_max should be much smaller at high latitude"


def test_fully_periodic_domain(simple_grid):
    """Test advection across North/South boundaries in a fully periodic domain."""
    # For N/S periodicity to conserve mass, we need a Torus geometry (constant areas).
    # Spherical geometry has varying face areas (A_north != A_south), so raw periodicity
    # would create/destroy mass at the boundary.

    # We override the grid geometry to be constant (Cylindrical/Torus)
    grid = simple_grid.copy()
    grid["cell_areas"] = xr.full_like(simple_grid["cell_areas"], 1000.0)
    grid["face_areas_ew"] = xr.full_like(simple_grid["face_areas_ew"], 10.0)
    grid["face_areas_ns"] = xr.full_like(simple_grid["face_areas_ns"], 10.0)

    # Custom boundary conditions: Periodic everywhere
    bc_torus = BoundaryConditions(
        north=BoundaryType.PERIODIC,
        south=BoundaryType.PERIODIC,
        east=BoundaryType.PERIODIC,
        west=BoundaryType.PERIODIC,
    )

    # Create blob at North edge (y=9)
    biomass = create_blob(grid, center=(9, 5), radius=1.0)

    # Velocity strictly North
    u = xr.full_like(biomass, 0.0)
    v = xr.full_like(biomass, 1.0)

    # Compute tendency
    result = compute_advection_tendency(
        state=biomass,
        u=u,
        v=v,
        cell_areas=grid["cell_areas"],
        face_areas_ew=grid["face_areas_ew"],
        face_areas_ns=grid["face_areas_ns"],
        boundary_conditions=bc_torus,
        mask=None,
    )

    tendency = result["advection_rate"]

    # Check mass conservation
    total_tendency = (tendency * grid["cell_areas"]).sum().values

    # The sum of tendencies * area should be zero (what leaves one side enters the other)
    assert abs(total_tendency) < 1e-10, "Mass not conserved in periodic torus"

    # Check flux crossover
    # Mass leaving top (North) should result in positive tendency at bottom (South)
    tendency_south = tendency.isel({Coordinates.Y.value: 0}).sum()

    assert tendency_south > 0, "Mass did not wrap around to South"


def test_closed_boundary_zero_flux():
    """Test that advection produces zero flux at closed boundaries.

    This test verifies Bug #1: face masking at closed boundaries.
    At closed boundaries, the face mask should be 0, resulting in zero flux.
    """
    # Create simple 5x5 grid
    lats = np.linspace(0, 4, 5)
    lons = np.linspace(0, 4, 5)

    lats_da = xr.DataArray(lats, dims=[Coordinates.Y.value])
    lons_da = xr.DataArray(lons, dims=[Coordinates.X.value])

    cell_areas = compute_spherical_cell_areas(lats_da, lons_da)
    face_areas_ew = compute_spherical_face_areas_ew(lats_da, lons_da)
    face_areas_ns = compute_spherical_face_areas_ns(lats_da, lons_da)

    # Uniform biomass
    biomass = xr.DataArray(
        np.ones((5, 5)) * 10.0,
        dims=[Coordinates.Y.value, Coordinates.X.value],
        coords={Coordinates.Y.value: lats, Coordinates.X.value: lons},
    )

    # Strong eastward velocity everywhere
    u = xr.full_like(biomass, 2.0)
    v = xr.full_like(biomass, 0.0)

    # ALL boundaries closed - no flux should cross
    bc_all_closed = BoundaryConditions(
        north=BoundaryType.CLOSED,
        south=BoundaryType.CLOSED,
        east=BoundaryType.CLOSED,
        west=BoundaryType.CLOSED,
    )

    result = compute_advection_tendency(
        state=biomass,
        u=u,
        v=v,
        cell_areas=cell_areas,
        face_areas_ew=face_areas_ew,
        face_areas_ns=face_areas_ns,
        boundary_conditions=bc_all_closed,
        mask=None,
    )

    tendency = result["advection_rate"]

    # At closed boundaries with uniform concentration, interior cells should have zero tendency
    # (flux in = flux out). But boundary cells should also have limited tendency.

    # More importantly: total integrated tendency should be exactly zero
    # (what leaves one cell enters another, no net change in closed domain)
    total_tendency = (tendency * cell_areas).sum().values

    print(f"\nTotal integrated tendency: {total_tendency:.10e}")
    print("Should be zero for closed domain with no sources/sinks")

    # Check that total tendency is zero (conservation in closed domain)
    assert (
        abs(total_tendency) < 1e-10
    ), f"Closed boundary violates conservation! Total tendency = {total_tendency:.3e}"


def test_closed_boundary_no_mass_escape():
    """Test that mass cannot escape through closed boundaries.

    This test integrates the advection over time and verifies that:
    1. Mass is conserved in the closed domain
    2. No flux crosses the boundaries
    """
    # Simple grid
    lats = np.linspace(0, 9, 10)
    lons = np.linspace(0, 9, 10)

    lats_da = xr.DataArray(lats, dims=[Coordinates.Y.value])
    lons_da = xr.DataArray(lons, dims=[Coordinates.X.value])

    cell_areas = compute_spherical_cell_areas(lats_da, lons_da)
    face_areas_ew = compute_spherical_face_areas_ew(lats_da, lons_da)
    face_areas_ns = compute_spherical_face_areas_ns(lats_da, lons_da)

    # Blob in center
    biomass = xr.DataArray(
        np.zeros((10, 10)),
        dims=[Coordinates.Y.value, Coordinates.X.value],
        coords={Coordinates.Y.value: lats, Coordinates.X.value: lons},
    )
    biomass[4:6, 4:6] = 100.0  # Concentrated mass in center

    # Strong velocity pushing toward boundaries
    u = xr.full_like(biomass, 3.0)  # Eastward
    v = xr.full_like(biomass, 2.0)  # Northward

    # Closed boundaries
    bc_closed = BoundaryConditions(
        north=BoundaryType.CLOSED,
        south=BoundaryType.CLOSED,
        east=BoundaryType.CLOSED,
        west=BoundaryType.CLOSED,
    )

    # Initial mass
    mass_initial = (biomass * cell_areas).sum().values

    # Integrate for several timesteps
    biomass_current = biomass.copy()
    dt = 1800.0  # 30 minutes

    for _ in range(20):
        result = compute_advection_tendency(
            state=biomass_current,
            u=u,
            v=v,
            cell_areas=cell_areas,
            face_areas_ew=face_areas_ew,
            face_areas_ns=face_areas_ns,
            boundary_conditions=bc_closed,
            mask=None,
        )
        biomass_current = biomass_current + result["advection_rate"] * dt

    mass_final = (biomass_current * cell_areas).sum().values

    print(f"\nMass initial: {mass_initial:.6e}")
    print(f"Mass final:   {mass_final:.6e}")
    print(f"Difference:   {mass_final - mass_initial:.6e}")
    print(f"Relative error: {abs(mass_final - mass_initial) / mass_initial:.2%}")

    # In a closed domain, mass must be exactly conserved
    mass_error = abs(mass_final - mass_initial) / mass_initial
    assert mass_error < 1e-10, f"Mass escaped through closed boundaries! Error = {mass_error:.2%}"


def test_boundary_flux_computation():
    """Test that fluxes are computed correctly at domain boundaries.

    This test specifically checks the velocity interpolation at boundaries (Bug #2).
    """
    # Minimal 3x3 grid
    lats = np.array([0.0, 1.0, 2.0])
    lons = np.array([0.0, 1.0, 2.0])

    lats_da = xr.DataArray(lats, dims=[Coordinates.Y.value])
    lons_da = xr.DataArray(lons, dims=[Coordinates.X.value])

    cell_areas = compute_spherical_cell_areas(lats_da, lons_da)
    face_areas_ew = compute_spherical_face_areas_ew(lats_da, lons_da)
    face_areas_ns = compute_spherical_face_areas_ns(lats_da, lons_da)

    # Concentration gradient: high on west, low on east
    biomass = xr.DataArray(
        [[10.0, 5.0, 1.0], [10.0, 5.0, 1.0], [10.0, 5.0, 1.0]],
        dims=[Coordinates.Y.value, Coordinates.X.value],
        coords={Coordinates.Y.value: lats, Coordinates.X.value: lons},
    )

    # Uniform eastward velocity
    u = xr.full_like(biomass, 1.0)
    v = xr.full_like(biomass, 0.0)

    # Periodic E-W, closed N-S
    bc = BoundaryConditions(
        north=BoundaryType.CLOSED,
        south=BoundaryType.CLOSED,
        east=BoundaryType.PERIODIC,
        west=BoundaryType.PERIODIC,
    )

    result = compute_advection_tendency(
        state=biomass,
        u=u,
        v=v,
        cell_areas=cell_areas,
        face_areas_ew=face_areas_ew,
        face_areas_ns=face_areas_ns,
        boundary_conditions=bc,
        mask=None,
    )

    tendency = result["advection_rate"]

    # With periodic BC, check mass conservation
    total_tendency = (tendency * cell_areas).sum().values
    print(f"\nTotal tendency (periodic): {total_tendency:.6e}")

    assert abs(total_tendency) < 1e-10, "Periodic BC should conserve mass exactly"

    # Now test with closed boundaries
    bc_closed = BoundaryConditions(
        north=BoundaryType.CLOSED,
        south=BoundaryType.CLOSED,
        east=BoundaryType.CLOSED,
        west=BoundaryType.CLOSED,
    )

    result_closed = compute_advection_tendency(
        state=biomass,
        u=u,
        v=v,
        cell_areas=cell_areas,
        face_areas_ew=face_areas_ew,
        face_areas_ns=face_areas_ns,
        boundary_conditions=bc_closed,
        mask=None,
    )

    tendency_closed = result_closed["advection_rate"]
    total_tendency_closed = (tendency_closed * cell_areas).sum().values

    print(f"Total tendency (closed):   {total_tendency_closed:.6e}")

    assert abs(total_tendency_closed) < 1e-10, "Closed BC should conserve mass exactly"


def test_numba_xarray_equivalence(simple_grid, boundary_closed_periodic):
    """Test that Numba and Xarray implementations produce identical results."""
    pytest.importorskip("numba", reason="Numba not available")

    try:
        from seapopym.transport.core import compute_advection_numba
    except ImportError:
        pytest.skip("Numba implementation not available")

    # Create test data with multiple dimensions to test broadcasting
    ny, nx = len(simple_grid["lats"]), len(simple_grid["lons"])
    ncohort = 3
    ntime = 2

    # State with extra dimensions (time, cohort, y, x)
    state = xr.DataArray(
        np.random.rand(ntime, ncohort, ny, nx),
        dims=["time", "cohort", Coordinates.Y.value, Coordinates.X.value],
        coords={
            "time": np.arange(ntime),
            "cohort": np.arange(ncohort),
            Coordinates.Y.value: simple_grid["lats"],
            Coordinates.X.value: simple_grid["lons"],
        },
    )

    # Velocity without cohort dimension (tests broadcasting)
    u = xr.DataArray(
        np.random.rand(ntime, ny, nx) * 0.1,
        dims=["time", Coordinates.Y.value, Coordinates.X.value],
        coords={
            "time": np.arange(ntime),
            Coordinates.Y.value: simple_grid["lats"],
            Coordinates.X.value: simple_grid["lons"],
        },
    )
    v = xr.DataArray(
        np.random.rand(ntime, ny, nx) * 0.1,
        dims=["time", Coordinates.Y.value, Coordinates.X.value],
        coords={
            "time": np.arange(ntime),
            Coordinates.Y.value: simple_grid["lats"],
            Coordinates.X.value: simple_grid["lons"],
        },
    )

    # Compute with both implementations
    result_xarray = compute_advection_tendency(
        state=state,
        u=u,
        v=v,
        cell_areas=simple_grid["cell_areas"],
        face_areas_ew=simple_grid["face_areas_ew"],
        face_areas_ns=simple_grid["face_areas_ns"],
        boundary_conditions=boundary_closed_periodic,
    )

    result_numba = compute_advection_numba(
        state=state,
        u=u,
        v=v,
        cell_areas=simple_grid["cell_areas"],
        face_areas_ew=simple_grid["face_areas_ew"],
        face_areas_ns=simple_grid["face_areas_ns"],
        boundary_conditions=boundary_closed_periodic,
    )

    # Compare results
    xarray_vals = result_xarray["advection_rate"].values
    numba_vals = result_numba["advection_rate"].values

    # Check shapes match
    assert xarray_vals.shape == numba_vals.shape, "Shape mismatch between implementations"

    # Check dimension order is preserved
    assert (
        result_xarray["advection_rate"].dims == state.dims
    ), "Xarray: dimension order not preserved"
    assert result_numba["advection_rate"].dims == state.dims, "Numba: dimension order not preserved"

    # Check numerical equivalence
    rtol, atol = 1e-5, 1e-8
    assert np.allclose(
        xarray_vals, numba_vals, rtol=rtol, atol=atol
    ), f"Results differ beyond tolerance (rtol={rtol}, atol={atol})"

    # Report statistics
    abs_diff = np.abs(xarray_vals - numba_vals)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()

    print("\nNumba vs Xarray comparison:")
    print(f"  Mean absolute difference: {mean_diff:.6e}")
    print(f"  Max absolute difference:  {max_diff:.6e}")
    print(f"  Xarray mean: {xarray_vals.mean():.6e}")
    print(f"  Numba mean:  {numba_vals.mean():.6e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
