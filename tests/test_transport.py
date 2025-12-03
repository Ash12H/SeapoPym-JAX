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

    print(f"\ndt_max (theory): {dt_max_theory:.1f}s ({dt_max_theory/3600:.1f} hours)")
    print(
        f"Stable dt: {dt_stable:.1f}s ({dt_stable/3600:.2f}h), CFL = {stability_stable['cfl_diffusion']:.3f}"
    )
    print(
        f"Unstable dt: {dt_unstable:.1f}s ({dt_unstable/3600:.2f}h), CFL = {stability_unstable['cfl_diffusion']:.3f}"
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
