"""Unit tests for advection transport scheme.

Tests verify:
- Mass conservation
- Zero velocity produces no change
- Correct transport direction
- Upwind choice
- Land masking (flux blocking)
- Boundary conditions
"""

import jax.numpy as jnp

from seapopym_message.transport.advection import (
    advection_upwind_flux,
    compute_advection_diagnostics,
)
from seapopym_message.transport.boundary import BoundaryConditions, BoundaryType
from seapopym_message.transport.grid import PlaneGrid, SphericalGrid
from seapopym_message.utils.grid import PlaneGridInfo, SphericalGridInfo


class TestAdvectionBasics:
    """Basic advection tests."""

    def test_advection_zero_velocity_no_change(self):
        """Test that zero velocity produces no change in biomass."""
        grid = PlaneGrid(grid_info=PlaneGridInfo(dx=10e3, dy=10e3, nlat=10, nlon=10))
        biomass = jnp.ones((10, 10), dtype=jnp.float32) * 5.0
        u = jnp.zeros((10, 10), dtype=jnp.float32)
        v = jnp.zeros((10, 10), dtype=jnp.float32)
        dt = 3600.0

        bc = BoundaryConditions(
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
        )

        biomass_new = advection_upwind_flux(biomass, u, v, dt, grid, bc)

        # Should be unchanged
        assert jnp.allclose(biomass_new, biomass, rtol=1e-5)

    def test_advection_conserves_mass_uniform_flow(self):
        """Test mass conservation with uniform eastward flow."""
        grid = PlaneGrid(grid_info=PlaneGridInfo(dx=10e3, dy=10e3, nlat=10, nlon=20))
        biomass = jnp.ones((10, 20), dtype=jnp.float32) * 2.0
        u = jnp.ones((10, 20), dtype=jnp.float32) * 0.5  # 0.5 m/s eastward
        v = jnp.zeros((10, 20), dtype=jnp.float32)
        dt = 3600.0

        # Periodic in E/W for conservation test
        bc = BoundaryConditions(
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.PERIODIC,
            BoundaryType.PERIODIC,
        )

        biomass_new = advection_upwind_flux(biomass, u, v, dt, grid, bc)

        # Compute total mass before and after
        cell_areas = grid.cell_areas()
        mass_before = jnp.sum(biomass * cell_areas)
        mass_after = jnp.sum(biomass_new * cell_areas)

        # Should conserve mass perfectly with periodic BC
        assert jnp.isclose(mass_after, mass_before, rtol=1e-6)

    def test_advection_eastward_flow_shifts_field(self):
        """Test that eastward flow shifts concentration field to the east."""
        # Create plane grid
        grid = PlaneGrid(grid_info=PlaneGridInfo(dx=10e3, dy=10e3, nlat=5, nlon=10))

        # Create initial field with blob in western half
        biomass = jnp.zeros((5, 10), dtype=jnp.float32)
        biomass = biomass.at[:, 2:4].set(10.0)  # Blob at columns 2-3

        # Eastward velocity
        u = jnp.ones((5, 10), dtype=jnp.float32) * 1.0  # 1 m/s
        v = jnp.zeros((5, 10), dtype=jnp.float32)

        # Use small timestep to see shift
        dt = 5000.0  # 5000 s = ~0.5 dx at 1 m/s

        bc = BoundaryConditions(
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.PERIODIC,
            BoundaryType.PERIODIC,
        )

        biomass_new = advection_upwind_flux(biomass, u, v, dt, grid, bc)

        # Check that concentration has moved eastward
        # Original peak at column 2-3, should shift somewhat east
        peak_before = jnp.argmax(jnp.sum(biomass, axis=0))
        peak_after = jnp.argmax(jnp.sum(biomass_new, axis=0))

        # Peak should shift east (index increases)
        assert peak_after >= peak_before

        # Total mass should be conserved (periodic BC)
        cell_areas = grid.cell_areas()
        mass_before = jnp.sum(biomass * cell_areas)
        mass_after = jnp.sum(biomass_new * cell_areas)
        assert jnp.isclose(mass_after, mass_before, rtol=1e-5)


class TestAdvectionUpwindChoice:
    """Tests for upwind scheme logic."""

    def test_upwind_choice_positive_velocity(self):
        """Test upwind choice when velocity is positive (eastward)."""
        grid = PlaneGrid(grid_info=PlaneGridInfo(dx=10e3, dy=10e3, nlat=3, nlon=5))

        # Create field with gradient: low on left, high on right
        biomass = jnp.array(
            [
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
            ],
            dtype=jnp.float32,
        )

        # Positive (eastward) velocity
        u = jnp.ones((3, 5), dtype=jnp.float32) * 1.0
        v = jnp.zeros((3, 5), dtype=jnp.float32)
        dt = 1000.0

        bc = BoundaryConditions(
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
        )

        biomass_new = advection_upwind_flux(biomass, u, v, dt, grid, bc)

        # With positive u, upwind scheme uses left cell value
        # So flux at east face of cell uses cell's own value (not neighbor)
        # This should produce stable, non-oscillatory transport

        # Check that field doesn't develop negative values or overshoots
        assert jnp.all(biomass_new >= 0)
        assert jnp.all(biomass_new <= jnp.max(biomass) * 1.1)  # Small overshoot allowed

    def test_upwind_choice_negative_velocity(self):
        """Test upwind choice when velocity is negative (westward)."""
        grid = PlaneGrid(grid_info=PlaneGridInfo(dx=10e3, dy=10e3, nlat=3, nlon=5))

        # Create field with gradient
        biomass = jnp.array(
            [
                [5, 4, 3, 2, 1],
                [5, 4, 3, 2, 1],
                [5, 4, 3, 2, 1],
            ],
            dtype=jnp.float32,
        )

        # Negative (westward) velocity
        u = jnp.ones((3, 5), dtype=jnp.float32) * (-1.0)
        v = jnp.zeros((3, 5), dtype=jnp.float32)
        dt = 1000.0

        bc = BoundaryConditions(
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
        )

        biomass_new = advection_upwind_flux(biomass, u, v, dt, grid, bc)

        # Check stability
        assert jnp.all(biomass_new >= 0)
        assert jnp.all(biomass_new <= jnp.max(biomass) * 1.1)


class TestAdvectionMasking:
    """Tests for land masking and flux blocking."""

    def test_advection_with_mask_blocks_flux(self):
        """Test that land mask sets velocity to zero and blocks flux."""
        grid = PlaneGrid(grid_info=PlaneGridInfo(dx=10e3, dy=10e3, nlat=5, nlon=10))

        # Create biomass field
        biomass = jnp.ones((5, 10), dtype=jnp.float32) * 5.0

        # Create velocities (eastward)
        u = jnp.ones((5, 10), dtype=jnp.float32) * 1.0
        v = jnp.zeros((5, 10), dtype=jnp.float32)

        # Create mask: land in middle column (column 5)
        mask = jnp.ones((5, 10), dtype=jnp.float32)
        mask = mask.at[:, 5].set(0.0)  # Land at column 5

        dt = 3600.0

        bc = BoundaryConditions(
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.PERIODIC,
            BoundaryType.PERIODIC,
        )

        biomass_new = advection_upwind_flux(biomass, u, v, dt, grid, bc, mask=mask)

        # Land cells should remain zero (or close to zero)
        assert jnp.all(biomass_new[:, 5] < 0.1)

        # Ocean cells should still have biomass
        assert jnp.all(biomass_new[:, 0] > 0)

    def test_advection_mask_with_nan(self):
        """Test that NaN in mask is treated as land."""
        grid = PlaneGrid(grid_info=PlaneGridInfo(dx=10e3, dy=10e3, nlat=3, nlon=5))

        biomass = jnp.ones((3, 5), dtype=jnp.float32) * 2.0
        u = jnp.ones((3, 5), dtype=jnp.float32) * 0.5
        v = jnp.zeros((3, 5), dtype=jnp.float32)

        # Mask with NaN for land
        mask = jnp.ones((3, 5), dtype=jnp.float32)
        mask = mask.at[1, 2].set(jnp.nan)  # NaN = land

        dt = 1000.0

        bc = BoundaryConditions(
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
        )

        biomass_new = advection_upwind_flux(biomass, u, v, dt, grid, bc, mask=mask)

        # NaN cell should be zero
        assert biomass_new[1, 2] == 0.0


class TestAdvectionBoundaryConditions:
    """Tests for different boundary conditions."""

    def test_advection_periodic_vs_closed(self):
        """Test difference between periodic and closed boundaries."""
        grid = PlaneGrid(grid_info=PlaneGridInfo(dx=10e3, dy=10e3, nlat=5, nlon=10))

        # Create field concentrated on western edge
        biomass = jnp.zeros((5, 10), dtype=jnp.float32)
        biomass = biomass.at[:, 0].set(10.0)

        # Westward velocity (should push biomass off western edge)
        u = jnp.ones((5, 10), dtype=jnp.float32) * (-1.0)
        v = jnp.zeros((5, 10), dtype=jnp.float32)
        dt = 5000.0

        # Test with CLOSED boundaries
        bc_closed = BoundaryConditions(
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
        )

        biomass_closed = advection_upwind_flux(biomass, u, v, dt, grid, bc_closed)

        # With CLOSED, mass accumulates at western boundary (wall blocks flux)
        cell_areas = grid.cell_areas()
        mass_closed = jnp.sum(biomass_closed * cell_areas)

        # Test with PERIODIC boundaries
        bc_periodic = BoundaryConditions(
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.PERIODIC,
            BoundaryType.PERIODIC,
        )

        biomass_periodic = advection_upwind_flux(biomass, u, v, dt, grid, bc_periodic)

        # With PERIODIC, mass wraps to eastern edge
        mass_periodic = jnp.sum(biomass_periodic * cell_areas)

        mass_original = jnp.sum(biomass * cell_areas)

        # Both CLOSED and PERIODIC should conserve mass perfectly
        # CLOSED = wall (no flux in or out)
        # PERIODIC = wraparound (mass exits one side, enters other side)
        assert jnp.isclose(
            mass_closed, mass_original, rtol=1e-5
        ), f"CLOSED should conserve mass: {mass_closed} vs {mass_original}"
        assert jnp.isclose(
            mass_periodic, mass_original, rtol=1e-5
        ), f"PERIODIC should conserve mass: {mass_periodic} vs {mass_original}"

        # The difference is in DISTRIBUTION, not total mass:
        # With PERIODIC, mass wraps around to eastern edge
        assert jnp.sum(biomass_periodic[:, -1]) > 0, "PERIODIC should wrap mass to eastern edge"

        # With CLOSED, mass stays at western edge (blocked by wall)
        # The western column should still have significant mass (not transported away)
        assert jnp.sum(biomass_closed[:, 0]) > 0, "CLOSED should keep mass at western wall"


class TestAdvectionDiagnostics:
    """Tests for advection diagnostics."""

    def test_diagnostics_computation(self):
        """Test that diagnostics are computed correctly."""
        grid = PlaneGrid(grid_info=PlaneGridInfo(dx=10e3, dy=10e3, nlat=10, nlon=10))
        biomass = jnp.ones((10, 10), dtype=jnp.float32) * 5.0
        u = jnp.ones((10, 10), dtype=jnp.float32) * 0.5
        v = jnp.ones((10, 10), dtype=jnp.float32) * 0.3
        dt = 3600.0

        bc = BoundaryConditions(
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.PERIODIC,
            BoundaryType.PERIODIC,
        )

        biomass_new = advection_upwind_flux(biomass, u, v, dt, grid, bc)

        diag = compute_advection_diagnostics(biomass, biomass_new, u, v, dt, grid)

        # Check that diagnostics are present
        assert "total_mass_before" in diag
        assert "total_mass_after" in diag
        assert "conservation_fraction" in diag
        assert "max_velocity" in diag
        assert "cfl_number" in diag

        # Check values are reasonable
        assert diag["total_mass_before"] > 0
        assert diag["conservation_fraction"] > 0.9  # Should be close to 1
        assert diag["max_velocity"] > 0
        assert diag["cfl_number"] > 0


class TestAdvectionSphericalGrid:
    """Tests for advection on spherical grid."""

    def test_advection_spherical_grid(self):
        """Test advection on spherical grid with varying dx(lat)."""
        grid = SphericalGrid(
            grid_info=SphericalGridInfo(
                lat_min=-30.0,
                lat_max=30.0,
                lon_min=0.0,
                lon_max=360.0,
                nlat=60,
                nlon=120,
            )
        )

        # Uniform biomass
        biomass = jnp.ones((60, 120), dtype=jnp.float32) * 3.0

        # Uniform eastward velocity
        u = jnp.ones((60, 120), dtype=jnp.float32) * 0.1
        v = jnp.zeros((60, 120), dtype=jnp.float32)
        dt = 3600.0

        bc = BoundaryConditions(
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.PERIODIC,
            BoundaryType.PERIODIC,
        )

        biomass_new = advection_upwind_flux(biomass, u, v, dt, grid, bc)

        # Should conserve mass (periodic in longitude)
        cell_areas = grid.cell_areas()
        mass_before = jnp.sum(biomass * cell_areas)
        mass_after = jnp.sum(biomass_new * cell_areas)

        assert jnp.isclose(mass_after, mass_before, rtol=1e-5)

        # Biomass should remain relatively uniform
        assert jnp.std(biomass_new) < 0.5  # Low variance
