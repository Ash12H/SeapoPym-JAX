"""Unit tests for transport grid infrastructure.

Tests verify:
- Spherical grid geometry (areas decrease toward poles)
- Plane grid geometry (constant areas)
- Realistic SEAPOPYM grid parameters
"""

import jax.numpy as jnp

from seapopym_message.transport.grid import PlaneGrid, SphericalGrid


class TestSphericalGrid:
    """Tests for spherical lat/lon grid."""

    def test_spherical_grid_initialization(self):
        """Test basic initialization and array shapes."""
        grid = SphericalGrid(
            lat_min=-60.0,
            lat_max=60.0,
            lon_min=0.0,
            lon_max=360.0,
            nlat=120,
            nlon=360,
        )

        # Check dimensions
        assert grid.nlat == 120
        assert grid.nlon == 360

        # Check coordinate arrays
        assert grid.lat.shape == (120,)
        assert grid.lon.shape == (360,)

        # Check coordinate ranges
        assert jnp.isclose(grid.lat[0], -60.0 + grid.dlat / 2)
        assert jnp.isclose(grid.lat[-1], 60.0 - grid.dlat / 2)
        assert jnp.isclose(grid.lon[0], 0.0 + grid.dlon / 2)
        assert jnp.isclose(grid.lon[-1], 360.0 - grid.dlon / 2)

        # Check area array shapes
        assert grid.cell_areas().shape == (120, 360)
        assert grid.face_areas_ns().shape == (121, 360)  # nlat+1
        assert grid.face_areas_ew().shape == (120, 361)  # nlon+1

        # Check spacing
        assert grid.dx().shape == (120,)  # varies with lat
        assert isinstance(grid.dy(), float)  # scalar

    def test_cell_areas_decrease_toward_poles(self):
        """Test that cell areas decrease toward poles due to cos(lat).

        At higher latitudes, cos(lat) is smaller, so cells are narrower
        in longitude direction. This test verifies the geometric relationship:
        A(lat) = R² × cos(lat) × dλ × dφ
        """
        grid = SphericalGrid(
            lat_min=-60.0,
            lat_max=60.0,
            lon_min=0.0,
            lon_max=360.0,
            nlat=120,
            nlon=360,
        )

        areas = grid.cell_areas()

        # Find equator and pole indices
        equator_idx = jnp.argmin(jnp.abs(grid.lat))  # Closest to 0°
        pole_idx = jnp.argmax(jnp.abs(grid.lat))  # Furthest from 0° (±60°)

        # Areas at equator should be larger than at poles
        area_equator = areas[equator_idx, 0]
        area_pole = areas[pole_idx, 0]
        assert area_equator > area_pole

        # Verify the ratio matches cos(lat) relationship
        # ratio should be ~ cos(60°) / cos(0°) ≈ 0.5
        lat_pole_rad = jnp.radians(grid.lat[pole_idx])
        expected_ratio = jnp.cos(lat_pole_rad)
        actual_ratio = area_pole / area_equator
        # Use rtol=1e-3 for float32 precision
        assert jnp.isclose(actual_ratio, expected_ratio, rtol=1e-3)

    def test_face_areas_ns_vary_with_latitude(self):
        """Test that N/S face areas vary with latitude.

        North/South faces (horizontal faces) have area: A_ns = R × cos(lat_face) × dλ
        These should decrease toward poles just like cell areas.
        """
        grid = SphericalGrid(
            lat_min=-60.0,
            lat_max=60.0,
            lon_min=0.0,
            lon_max=360.0,
            nlat=120,
            nlon=360,
        )

        face_areas_ns = grid.face_areas_ns()

        # Shape is (nlat+1, nlon) - faces at cell boundaries
        assert face_areas_ns.shape == (121, 360)

        # Areas should vary with latitude (not constant along lat direction)
        area_at_different_lats = face_areas_ns[:, 0]  # First column
        assert not jnp.allclose(area_at_different_lats, area_at_different_lats[0])

        # Southern faces should be larger than northern faces (if asymmetric)
        # For symmetric domain (-60, 60), check extremes vs center
        center_idx = len(area_at_different_lats) // 2
        edge_idx = 0
        assert area_at_different_lats[center_idx] > area_at_different_lats[edge_idx]

    def test_face_areas_ew_constant(self):
        """Test that E/W face areas are constant.

        East/West faces (vertical faces) span full latitude height:
        A_ew = R × dφ (constant, independent of longitude or latitude)
        """
        grid = SphericalGrid(
            lat_min=-60.0,
            lat_max=60.0,
            lon_min=0.0,
            lon_max=360.0,
            nlat=120,
            nlon=360,
        )

        face_areas_ew = grid.face_areas_ew()

        # Shape is (nlat, nlon+1)
        assert face_areas_ew.shape == (120, 361)

        # All E/W faces should have the same area
        expected_area = grid.R * jnp.radians(grid.dlat)
        assert jnp.allclose(face_areas_ew, expected_area)

    def test_realistic_seapopym_grid(self):
        """Test realistic SEAPOPYM grid configuration.

        SEAPOPYM typically uses:
        - Domain: 60°S to 60°N, 0° to 360°E
        - Resolution: ~1° (~120 km at equator)
        - Grid: ~120 lat × 160 lon (for lower resolution runs)

        This test verifies physically reasonable values.
        """
        grid = SphericalGrid(
            lat_min=-60.0,
            lat_max=60.0,
            lon_min=0.0,
            lon_max=360.0,
            nlat=120,
            nlon=160,
            R=6371e3,  # Earth radius in meters
        )

        # Check grid spacing
        assert grid.dlat == 1.0  # 1° latitude
        assert grid.dlon == 2.25  # 360/160 = 2.25° longitude

        # Check physical dimensions at equator
        # dy should be ~111 km (1° latitude ≈ 111 km)
        expected_dy = grid.R * jnp.radians(1.0)
        assert jnp.isclose(grid.dy(), expected_dy)
        assert jnp.isclose(grid.dy(), 111e3, rtol=0.01)  # Within 1% of 111 km

        # dx at equator should be ~250 km (2.25° × 111 km/°)
        equator_idx = jnp.argmin(jnp.abs(grid.lat))
        dx_equator = grid.dx()[equator_idx]
        expected_dx_equator = grid.R * jnp.radians(2.25)
        # Use rtol=1e-3 for float32 precision
        assert jnp.isclose(dx_equator, expected_dx_equator, rtol=1e-3)
        assert jnp.isclose(dx_equator, 250e3, rtol=0.01)  # Within 1% of 250 km

        # Cell areas should be reasonable
        areas = grid.cell_areas()
        area_equator = areas[equator_idx, 0]
        # Expected: ~111 km × 250 km ≈ 27,750 km² = 2.775e10 m²
        assert jnp.isclose(area_equator, 2.78e10, rtol=0.01)

        # Check that dx decreases toward poles
        pole_idx = jnp.argmax(jnp.abs(grid.lat))
        dx_pole = grid.dx()[pole_idx]
        assert dx_pole < dx_equator

        # At pole latitude (59.5° for this grid), ratio should match cos(lat_pole)
        lat_pole_rad = jnp.radians(grid.lat[pole_idx])
        expected_ratio = jnp.cos(lat_pole_rad)
        actual_ratio = dx_pole / dx_equator
        assert jnp.isclose(actual_ratio, expected_ratio, rtol=1e-3)


class TestPlaneGrid:
    """Tests for uniform Cartesian grid."""

    def test_plane_grid_constant_areas(self):
        """Test that plane grid has constant cell and face areas."""
        grid = PlaneGrid(
            dx=10e3,  # 10 km
            dy=10e3,  # 10 km
            nlat=100,
            nlon=100,
        )

        # Check dimensions
        assert grid.nlat == 100
        assert grid.nlon == 100

        # Check spacing
        assert grid.dx() == 10e3
        assert grid.dy() == 10e3

        # Cell areas should all be dx × dy
        areas = grid.cell_areas()
        assert areas.shape == (100, 100)
        assert jnp.allclose(areas, 100e6)  # 10 km × 10 km = 100 km²

        # E/W face areas should all be dy
        face_areas_ew = grid.face_areas_ew()
        assert face_areas_ew.shape == (100, 101)
        assert jnp.allclose(face_areas_ew, 10e3)

        # N/S face areas should all be dx
        face_areas_ns = grid.face_areas_ns()
        assert face_areas_ns.shape == (101, 100)
        assert jnp.allclose(face_areas_ns, 10e3)

    def test_plane_grid_different_dx_dy(self):
        """Test plane grid with non-uniform dx and dy."""
        grid = PlaneGrid(
            dx=20e3,  # 20 km
            dy=10e3,  # 10 km
            nlat=50,
            nlon=80,
        )

        # Cell areas should be dx × dy
        areas = grid.cell_areas()
        assert jnp.allclose(areas, 200e6)  # 20 km × 10 km = 200 km²

        # E/W faces (height = dy)
        assert jnp.allclose(grid.face_areas_ew(), 10e3)

        # N/S faces (width = dx)
        assert jnp.allclose(grid.face_areas_ns(), 20e3)
