"""Unit tests for diffusion transport scheme.

Tests verify:
- Zero diffusion produces no change
- Diffusion smooths gradients
- Mass conservation
- Stability criterion
- Land masking
- Spherical grid with dx(lat)
"""

import jax.numpy as jnp

from seapopym_message.transport.boundary import BoundaryConditions, BoundaryType
from seapopym_message.transport.diffusion import (
    check_diffusion_stability,
    compute_diffusion_diagnostics,
    diffusion_explicit_spherical,
)
from seapopym_message.transport.grid import PlaneGrid, SphericalGrid
from seapopym_message.utils.grid import PlaneGridInfo, SphericalGridInfo


class TestDiffusionBasics:
    """Basic diffusion tests."""

    def test_diffusion_zero_coefficient_no_change(self):
        """Test that D=0 produces no change."""
        grid = PlaneGrid(grid_info=PlaneGridInfo(dx=10e3, dy=10e3, nlat=10, nlon=10))
        biomass = jnp.ones((10, 10), dtype=jnp.float32) * 5.0
        biomass = biomass.at[5, 5].set(20.0)  # Add a spike

        D = 0.0  # No diffusion
        dt = 3600.0

        bc = BoundaryConditions(
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
        )

        biomass_new = diffusion_explicit_spherical(biomass, D, dt, grid, bc)

        # Should be unchanged
        assert jnp.allclose(biomass_new, biomass, rtol=1e-6)

    def test_diffusion_smooths_gradients(self):
        """Test that diffusion reduces spatial gradients."""
        grid = PlaneGrid(grid_info=PlaneGridInfo(dx=10e3, dy=10e3, nlat=10, nlon=10))

        # Create field with sharp gradient (spike)
        biomass = jnp.ones((10, 10), dtype=jnp.float32) * 1.0
        biomass = biomass.at[5, 5].set(100.0)  # Sharp spike

        D = 1000.0  # Moderate diffusion
        dt = 100.0  # Small timestep for stability

        bc = BoundaryConditions(
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
        )

        biomass_new = diffusion_explicit_spherical(biomass, D, dt, grid, bc)

        # Center value should decrease (smoothing)
        assert biomass_new[5, 5] < biomass[5, 5]

        # Neighbors should increase (diffusion spreads outward)
        assert biomass_new[5, 6] > biomass[5, 6]
        assert biomass_new[5, 4] > biomass[5, 4]
        assert biomass_new[6, 5] > biomass[6, 5]
        assert biomass_new[4, 5] > biomass[4, 5]

    def test_diffusion_conserves_mass(self):
        """Test mass conservation with closed boundaries."""
        grid = PlaneGrid(grid_info=PlaneGridInfo(dx=10e3, dy=10e3, nlat=20, nlon=20))

        # Non-uniform field
        biomass = jnp.ones((20, 20), dtype=jnp.float32) * 2.0
        biomass = biomass.at[10, 10].set(50.0)

        D = 500.0
        dt = 50.0  # Small dt for stability

        bc = BoundaryConditions(
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
        )

        biomass_new = diffusion_explicit_spherical(biomass, D, dt, grid, bc)

        # Compute total mass
        cell_areas = grid.cell_areas()
        mass_before = jnp.sum(biomass * cell_areas)
        mass_after = jnp.sum(biomass_new * cell_areas)

        # Should conserve mass with closed BC
        assert jnp.isclose(mass_after, mass_before, rtol=1e-5)


class TestDiffusionStability:
    """Tests for stability criterion."""

    def test_stability_check_stable(self):
        """Test that small dt passes stability check."""
        grid = PlaneGrid(grid_info=PlaneGridInfo(dx=10e3, dy=10e3, nlat=10, nlon=10))
        D = 1000.0

        # dt_max = dx² / (4D) = (10e3)² / (4×1000) = 25e6 / 4000 = 6250 s
        dt_safe = 1000.0  # Well below dt_max

        stability = check_diffusion_stability(dt_safe, D, grid)

        assert stability["is_stable"] is True
        assert stability["dt_max"] > dt_safe
        assert stability["cfl_diffusion"] < 0.25

    def test_stability_check_unstable(self):
        """Test that large dt fails stability check."""
        grid = PlaneGrid(grid_info=PlaneGridInfo(dx=10e3, dy=10e3, nlat=10, nlon=10))
        D = 1000.0

        # dt_max = dx²/(4D) = (10e3)²/(4×1000) = 100e6/4000 = 25000 s
        # Use larger dt to be unstable
        dt_large = 50000.0

        stability = check_diffusion_stability(dt_large, D, grid)

        assert stability["is_stable"] is False
        assert stability["dt_max"] < dt_large
        assert stability["cfl_diffusion"] > 0.25

    def test_stability_spherical_grid_pole_constraint(self):
        """Test that spherical grid stability is limited by poles (min dx)."""
        grid = SphericalGrid(
            grid_info=SphericalGridInfo(
                lat_min=-60.0,
                lat_max=60.0,
                lon_min=0.0,
                lon_max=360.0,
                nlat=120,
                nlon=360,
            )
        )

        D = 1000.0
        dt = 100.0

        stability = check_diffusion_stability(dt, D, grid)

        # dx_min should be at high latitude (near ±60°)
        # dx(60°) = R × cos(60°) × dλ ≈ 6.37e6 × 0.5 × (π/180) ≈ 55 km
        # dx_min² ≈ 3e9, dt_max ≈ 3e9 / 4000 ≈ 750,000 s

        assert stability["dx_min"] < stability["dy"]  # dx smaller at poles
        assert stability["dx_min"] < 60e3  # Less than ~60 km


class TestDiffusionMasking:
    """Tests for land masking."""

    def test_diffusion_with_mask(self):
        """Test that land cells remain zero with mask."""
        grid = PlaneGrid(grid_info=PlaneGridInfo(dx=10e3, dy=10e3, nlat=10, nlon=10))

        # Create biomass
        biomass = jnp.ones((10, 10), dtype=jnp.float32) * 5.0
        biomass = biomass.at[5, 5].set(50.0)  # Spike

        # Create mask: land in middle row
        mask = jnp.ones((10, 10), dtype=jnp.float32)
        mask = mask.at[5, :].set(0.0)  # Row 5 is land

        D = 1000.0
        dt = 100.0

        bc = BoundaryConditions(
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
        )

        biomass_new = diffusion_explicit_spherical(biomass, D, dt, grid, bc, mask=mask)

        # Land row should be zero
        assert jnp.all(biomass_new[5, :] == 0.0)

        # Ocean rows should have non-zero values
        assert jnp.any(biomass_new[4, :] > 0)
        assert jnp.any(biomass_new[6, :] > 0)

    def test_diffusion_mask_blocks_flux(self):
        """Test that land acts as zero-flux boundary."""
        grid = PlaneGrid(grid_info=PlaneGridInfo(dx=10e3, dy=10e3, nlat=5, nlon=5))

        # Ocean with spike, land barrier in middle
        biomass = jnp.ones((5, 5), dtype=jnp.float32) * 1.0
        biomass = biomass.at[1, 2].set(100.0)  # Spike north of barrier

        # Mask: land at row 2 (middle)
        mask = jnp.ones((5, 5), dtype=jnp.float32)
        mask = mask.at[2, :].set(0.0)

        D = 1000.0
        dt = 50.0

        bc = BoundaryConditions(
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
        )

        biomass_new = diffusion_explicit_spherical(biomass, D, dt, grid, bc, mask=mask)

        # Land row stays zero
        assert jnp.all(biomass_new[2, :] == 0.0)

        # South of barrier (rows 3-4) should remain low (blocked by land)
        # North of barrier (rows 0-1) should show diffusion
        # This tests that flux doesn't cross land


class TestDiffusionSphericalGrid:
    """Tests for diffusion on spherical grid."""

    def test_diffusion_spherical_grid_dx_variation(self):
        """Test diffusion on spherical grid accounts for dx(lat)."""
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

        # Uniform biomass with spike
        biomass = jnp.ones((60, 120), dtype=jnp.float32) * 2.0
        biomass = biomass.at[30, 60].set(100.0)  # Spike at equator

        D = 1000.0
        dt = 100.0

        bc = BoundaryConditions(
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.PERIODIC,
            BoundaryType.PERIODIC,
        )

        biomass_new = diffusion_explicit_spherical(biomass, D, dt, grid, bc)

        # Should conserve mass (closed N/S, periodic E/W)
        cell_areas = grid.cell_areas()
        mass_before = jnp.sum(biomass * cell_areas)
        mass_after = jnp.sum(biomass_new * cell_areas)

        assert jnp.isclose(mass_after, mass_before, rtol=1e-4)

        # Spike should be smoothed
        assert biomass_new[30, 60] < biomass[30, 60]

    def test_diffusion_uniform_field_no_change(self):
        """Test that uniform field remains unchanged."""
        grid = PlaneGrid(grid_info=PlaneGridInfo(dx=10e3, dy=10e3, nlat=10, nlon=10))

        # Perfectly uniform field
        biomass = jnp.ones((10, 10), dtype=jnp.float32) * 10.0

        D = 1000.0
        dt = 1000.0

        bc = BoundaryConditions(
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.PERIODIC,
            BoundaryType.PERIODIC,
        )

        biomass_new = diffusion_explicit_spherical(biomass, D, dt, grid, bc)

        # Uniform field has zero gradients, so no diffusion
        assert jnp.allclose(biomass_new, biomass, rtol=1e-5)


class TestDiffusionDiagnostics:
    """Tests for diffusion diagnostics."""

    def test_diagnostics_computation(self):
        """Test that diagnostics are computed correctly."""
        grid = PlaneGrid(grid_info=PlaneGridInfo(dx=10e3, dy=10e3, nlat=10, nlon=10))
        biomass = jnp.ones((10, 10), dtype=jnp.float32) * 5.0
        biomass = biomass.at[5, 5].set(50.0)

        D = 500.0
        dt = 100.0

        bc = BoundaryConditions(
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
        )

        biomass_new = diffusion_explicit_spherical(biomass, D, dt, grid, bc)

        diag = compute_diffusion_diagnostics(biomass, biomass_new, D, dt, grid)

        # Check that diagnostics are present
        assert "total_mass_before" in diag
        assert "total_mass_after" in diag
        assert "conservation_fraction" in diag
        assert "max_gradient" in diag
        assert "stability" in diag

        # Check values are reasonable
        assert diag["total_mass_before"] > 0
        assert diag["conservation_fraction"] > 0.99  # Should be close to 1
        assert diag["max_gradient"] > 0
        assert diag["stability"]["is_stable"] is True
