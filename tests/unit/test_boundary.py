"""Unit tests for transport boundary conditions.

Tests verify:
- Boundary type enum
- Boundary conditions validation
- Ghost cell creation for CLOSED/OPEN boundaries
- Ghost cell creation for PERIODIC boundaries
- Neighbor extraction with BC
"""

import jax.numpy as jnp
import pytest

from seapopym_message.transport.boundary import (
    BoundaryConditions,
    BoundaryType,
    apply_boundary_conditions,
    get_neighbors_with_bc,
)


class TestBoundaryType:
    """Tests for BoundaryType enum."""

    def test_boundary_type_values(self):
        """Test that boundary types have correct string values."""
        assert BoundaryType.CLOSED.value == "closed"
        assert BoundaryType.PERIODIC.value == "periodic"
        assert BoundaryType.OPEN.value == "open"

    def test_boundary_type_enum_membership(self):
        """Test that all expected types are in the enum."""
        types = {t.value for t in BoundaryType}
        assert types == {"closed", "periodic", "open"}


class TestBoundaryConditions:
    """Tests for BoundaryConditions dataclass."""

    def test_boundary_conditions_creation(self):
        """Test creating boundary conditions."""
        bc = BoundaryConditions(
            north=BoundaryType.CLOSED,
            south=BoundaryType.CLOSED,
            east=BoundaryType.PERIODIC,
            west=BoundaryType.PERIODIC,
        )

        assert bc.north == BoundaryType.CLOSED
        assert bc.south == BoundaryType.CLOSED
        assert bc.east == BoundaryType.PERIODIC
        assert bc.west == BoundaryType.PERIODIC

    def test_periodic_validation_passes(self):
        """Test that matching periodic east/west is valid."""
        # Should not raise
        bc = BoundaryConditions(
            north=BoundaryType.CLOSED,
            south=BoundaryType.CLOSED,
            east=BoundaryType.PERIODIC,
            west=BoundaryType.PERIODIC,
        )
        assert bc.east == BoundaryType.PERIODIC

    def test_periodic_validation_fails(self):
        """Test that mismatched periodic east/west raises error."""
        with pytest.raises(ValueError, match="East and West boundaries must both be PERIODIC"):
            BoundaryConditions(
                north=BoundaryType.CLOSED,
                south=BoundaryType.CLOSED,
                east=BoundaryType.PERIODIC,
                west=BoundaryType.CLOSED,  # Mismatch!
            )

    def test_non_periodic_boundaries_allowed(self):
        """Test that non-periodic east/west can differ."""
        # CLOSED and OPEN can be mixed (though not typical)
        bc = BoundaryConditions(
            north=BoundaryType.CLOSED,
            south=BoundaryType.OPEN,
            east=BoundaryType.CLOSED,
            west=BoundaryType.OPEN,
        )
        assert bc.east != bc.west  # Different but valid


class TestApplyBoundaryClosed:
    """Tests for apply_boundary_conditions with CLOSED boundaries."""

    def test_closed_boundaries_all_sides(self):
        """Test CLOSED BC creates ghost cells equal to edge cells."""
        # Create simple 3x4 field
        field = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
        # [[0, 1, 2, 3],
        #  [4, 5, 6, 7],
        #  [8, 9, 10, 11]]

        bc = BoundaryConditions(
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
        )

        result = apply_boundary_conditions(field, bc)

        # Check shape (3+2, 4+2) = (5, 6)
        assert result.shape == (5, 6)

        # Check interior is unchanged
        assert jnp.array_equal(result[1:-1, 1:-1], field)

        # Check ghost cells
        # South (row 0): should equal first row of field
        assert jnp.array_equal(result[0, 1:-1], field[0, :])  # [0, 1, 2, 3]

        # North (row -1): should equal last row of field
        assert jnp.array_equal(result[-1, 1:-1], field[-1, :])  # [8, 9, 10, 11]

        # West (col 0): should equal first column of field
        assert jnp.array_equal(result[1:-1, 0], field[:, 0])  # [0, 4, 8]

        # East (col -1): should equal last column of field
        assert jnp.array_equal(result[1:-1, -1], field[:, -1])  # [3, 7, 11]

    def test_closed_preserves_field_values(self):
        """Test that CLOSED BC doesn't modify interior values."""
        field = jnp.ones((10, 20), dtype=jnp.float32) * 42.0

        bc = BoundaryConditions(
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
        )

        result = apply_boundary_conditions(field, bc)

        # All ghost cells should also be 42 (copied from edges)
        assert jnp.all(result == 42.0)


class TestApplyBoundaryPeriodic:
    """Tests for apply_boundary_conditions with PERIODIC boundaries."""

    def test_periodic_east_west(self):
        """Test PERIODIC BC wraps around in longitude direction."""
        # Create field where first and last columns are distinct
        field = jnp.array(
            [
                [10, 20, 30, 40],
                [15, 25, 35, 45],
                [11, 21, 31, 41],
            ],
            dtype=jnp.float32,
        )

        bc = BoundaryConditions(
            north=BoundaryType.CLOSED,
            south=BoundaryType.CLOSED,
            east=BoundaryType.PERIODIC,
            west=BoundaryType.PERIODIC,
        )

        result = apply_boundary_conditions(field, bc)

        # West ghost cells (col 0) should equal last interior column (col -1 of field)
        # field[:, -1] = [40, 45, 41]
        assert jnp.array_equal(result[1:-1, 0], field[:, -1])

        # East ghost cells (col -1) should equal first interior column (col 0 of field)
        # field[:, 0] = [10, 15, 11]
        assert jnp.array_equal(result[1:-1, -1], field[:, 0])

        # North/South should still be CLOSED (ghost = edge)
        assert jnp.array_equal(result[0, 1:-1], field[0, :])
        assert jnp.array_equal(result[-1, 1:-1], field[-1, :])

    def test_periodic_wrapping_continuity(self):
        """Test that periodic BC creates continuous field across wrap."""
        # Create field with gradient in longitude
        field = jnp.tile(jnp.arange(10, dtype=jnp.float32), (5, 1))
        # Each row: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        bc = BoundaryConditions(
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.PERIODIC,
            BoundaryType.PERIODIC,
        )

        result = apply_boundary_conditions(field, bc)

        # West ghost should be 9 (last value)
        assert jnp.all(result[1:-1, 0] == 9)

        # East ghost should be 0 (first value)
        assert jnp.all(result[1:-1, -1] == 0)


class TestGetNeighbors:
    """Tests for get_neighbors_with_bc function."""

    def test_get_neighbors_shape(self):
        """Test that neighbor arrays have correct shape."""
        field = jnp.ones((10, 20), dtype=jnp.float32)
        bc = BoundaryConditions(
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
        )

        west, east, south, north = get_neighbors_with_bc(field, bc)

        # All should have same shape as input
        assert west.shape == (10, 20)
        assert east.shape == (10, 20)
        assert south.shape == (10, 20)
        assert north.shape == (10, 20)

    def test_get_neighbors_interior(self):
        """Test neighbor values for interior cells."""
        # Create field with unique values
        field = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
        # [[0, 1, 2, 3],
        #  [4, 5, 6, 7],
        #  [8, 9, 10, 11]]

        bc = BoundaryConditions(
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
        )

        west, east, south, north = get_neighbors_with_bc(field, bc)

        # Check center cell (1, 1) = 5
        # West neighbor should be (1, 0) = 4
        assert west[1, 1] == 4
        # East neighbor should be (1, 2) = 6
        assert east[1, 1] == 6
        # South neighbor should be (0, 1) = 1
        assert south[1, 1] == 1
        # North neighbor should be (2, 1) = 9
        assert north[1, 1] == 9

    def test_get_neighbors_with_periodic(self):
        """Test neighbor extraction with periodic boundaries."""
        field = jnp.array(
            [
                [10, 20, 30],
                [40, 50, 60],
            ],
            dtype=jnp.float32,
        )

        bc = BoundaryConditions(
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.PERIODIC,
            BoundaryType.PERIODIC,
        )

        west, east, south, north = get_neighbors_with_bc(field, bc)

        # For cell (0, 0) = 10:
        # West neighbor should wrap to (0, -1) = 30 (periodic)
        assert west[0, 0] == 30

        # For cell (0, -1) = 30:
        # East neighbor should wrap to (0, 0) = 10 (periodic)
        assert east[0, -1] == 10

    def test_get_neighbors_boundary_cells(self):
        """Test that boundary cells get ghost neighbors correctly."""
        field = jnp.ones((5, 5), dtype=jnp.float32)
        field = field.at[0, :].set(10)  # South edge = 10
        field = field.at[-1, :].set(20)  # North edge = 20
        field = field.at[:, 0].set(30)  # West edge = 30
        field = field.at[:, -1].set(40)  # East edge = 40

        bc = BoundaryConditions(
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
        )

        west, east, south, north = get_neighbors_with_bc(field, bc)

        # South edge cells (row 0) should have south neighbor = themselves (CLOSED)
        assert jnp.all(south[0, :] == field[0, :])

        # North edge cells (row -1) should have north neighbor = themselves (CLOSED)
        assert jnp.all(north[-1, :] == field[-1, :])

        # West edge cells (col 0) should have west neighbor = themselves (CLOSED)
        assert jnp.all(west[:, 0] == field[:, 0])

        # East edge cells (col -1) should have east neighbor = themselves (CLOSED)
        assert jnp.all(east[:, -1] == field[:, -1])
