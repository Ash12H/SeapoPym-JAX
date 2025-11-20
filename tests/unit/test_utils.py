"""Tests for utility functions (grid, domain splitting)."""

import jax.numpy as jnp
import pytest

from seapopym_message.utils.domain import split_domain_2d, split_domain_2d_periodic_lon
from seapopym_message.utils.grid import SphericalGridInfo


@pytest.mark.unit
class TestGridInfo:
    """Test GridInfo dataclass."""

    def test_gridinfo_creation(self) -> None:
        """Test creating a SphericalGridInfo instance."""
        grid = SphericalGridInfo(
            lat_min=-10.0, lat_max=10.0, lon_min=140.0, lon_max=180.0, nlat=20, nlon=40
        )

        assert grid.lat_min == -10.0
        assert grid.lat_max == 10.0
        assert grid.lon_min == 140.0
        assert grid.lon_max == 180.0
        assert grid.nlat == 20
        assert grid.nlon == 40

    def test_lat_lon_coords(self) -> None:
        """Test latitude and longitude coordinate arrays."""
        grid = SphericalGridInfo(
            lat_min=0.0, lat_max=10.0, lon_min=0.0, lon_max=20.0, nlat=11, nlon=21
        )

        assert grid.lat_coords.shape == (11,)
        assert grid.lon_coords.shape == (21,)
        assert jnp.allclose(grid.lat_coords[0], 0.0)
        assert jnp.allclose(grid.lat_coords[-1], 10.0)
        assert jnp.allclose(grid.lon_coords[0], 0.0)
        assert jnp.allclose(grid.lon_coords[-1], 20.0)

    def test_dlat_dlon(self) -> None:
        """Test grid spacing in degrees."""
        grid = SphericalGridInfo(
            lat_min=0.0, lat_max=10.0, lon_min=0.0, lon_max=20.0, nlat=11, nlon=21
        )

        assert jnp.allclose(grid.dlat, 1.0)
        assert jnp.allclose(grid.dlon, 1.0)

    def test_dy_meters(self) -> None:
        """Test meridional spacing in meters."""
        grid = SphericalGridInfo(lat_min=0.0, lat_max=1.0, lon_min=0.0, lon_max=1.0, nlat=2, nlon=2)

        # 1 degree latitude ≈ 111,320 meters
        assert jnp.allclose(grid.dy, 111320.0, rtol=0.01)

    def test_dx_meters_at_equator(self) -> None:
        """Test zonal spacing in meters at equator."""
        grid = SphericalGridInfo(
            lat_min=-1.0, lat_max=1.0, lon_min=0.0, lon_max=1.0, nlat=3, nlon=2
        )

        # At equator (mean_lat=0): 1 degree longitude ≈ 111,320 meters
        # mean_lat = 0, cos(0) = 1
        assert jnp.allclose(grid.dx, 111320.0, rtol=0.01)

    def test_dx_meters_at_high_latitude(self) -> None:
        """Test zonal spacing decreases at high latitudes."""
        grid_equator = SphericalGridInfo(
            lat_min=-1.0, lat_max=1.0, lon_min=0.0, lon_max=1.0, nlat=3, nlon=2
        )
        grid_high_lat = SphericalGridInfo(
            lat_min=59.0, lat_max=61.0, lon_min=0.0, lon_max=1.0, nlat=3, nlon=2
        )

        # At 60°, cos(60°) ≈ 0.5, so dx should be ~half
        assert grid_high_lat.dx < grid_equator.dx
        assert jnp.allclose(grid_high_lat.dx / grid_equator.dx, 0.5, atol=0.05)

    def test_get_meshgrid(self) -> None:
        """Test meshgrid generation."""
        grid = SphericalGridInfo(
            lat_min=0.0, lat_max=10.0, lon_min=0.0, lon_max=20.0, nlat=5, nlon=10
        )

        LAT, LON = grid.get_meshgrid()

        assert LAT.shape == (5, 10)
        assert LON.shape == (5, 10)
        # Check corners
        assert jnp.allclose(LAT[0, 0], 0.0)
        assert jnp.allclose(LAT[-1, 0], 10.0)
        assert jnp.allclose(LON[0, 0], 0.0)
        assert jnp.allclose(LON[0, -1], 20.0)

    def test_repr(self) -> None:
        """Test string representation."""
        grid = SphericalGridInfo(
            lat_min=0.0, lat_max=10.0, lon_min=0.0, lon_max=20.0, nlat=5, nlon=10
        )

        repr_str = repr(grid)
        assert "SphericalGridInfo" in repr_str
        assert "0.00" in repr_str
        assert "(5, 10)" in repr_str


@pytest.mark.unit
class TestDomainSplitting:
    """Test domain splitting functions."""

    def test_split_domain_2d_2x2(self) -> None:
        """Test splitting into 2×2 workers."""
        patches = split_domain_2d(
            nlat_global=20, nlon_global=40, num_workers_lat=2, num_workers_lon=2
        )

        # Should have 4 patches
        assert len(patches) == 4

        # Check dimensions
        for patch in patches:
            assert patch["nlat"] == 10
            assert patch["nlon"] == 20

        # Check worker IDs
        worker_ids = [p["worker_id"] for p in patches]
        assert worker_ids == [0, 1, 2, 3]

        # Check first patch (top-left)
        assert patches[0]["lat_start"] == 0
        assert patches[0]["lat_end"] == 10
        assert patches[0]["lon_start"] == 0
        assert patches[0]["lon_end"] == 20
        assert patches[0]["neighbors"]["north"] is None
        assert patches[0]["neighbors"]["west"] is None
        assert patches[0]["neighbors"]["south"] == 2
        assert patches[0]["neighbors"]["east"] == 1

    def test_split_domain_2d_3x4(self) -> None:
        """Test splitting into 3×4 workers."""
        patches = split_domain_2d(
            nlat_global=30, nlon_global=40, num_workers_lat=3, num_workers_lon=4
        )

        assert len(patches) == 12
        assert patches[0]["nlat"] == 10
        assert patches[0]["nlon"] == 10

        # Check middle worker (should have all neighbors)
        middle = patches[5]  # i_lat=1, i_lon=1
        assert middle["neighbors"]["north"] == 1
        assert middle["neighbors"]["south"] == 9
        assert middle["neighbors"]["west"] == 4
        assert middle["neighbors"]["east"] == 6

    def test_split_domain_2d_single_worker(self) -> None:
        """Test with single worker (no splitting)."""
        patches = split_domain_2d(
            nlat_global=20, nlon_global=40, num_workers_lat=1, num_workers_lon=1
        )

        assert len(patches) == 1
        assert patches[0]["worker_id"] == 0
        assert patches[0]["nlat"] == 20
        assert patches[0]["nlon"] == 40
        # All neighbors should be None
        assert all(v is None for v in patches[0]["neighbors"].values())

    def test_split_domain_2d_invalid_division(self) -> None:
        """Test that invalid division raises error."""
        with pytest.raises(ValueError, match="must be divisible"):
            split_domain_2d(nlat_global=21, nlon_global=40, num_workers_lat=2, num_workers_lon=2)

        with pytest.raises(ValueError, match="must be divisible"):
            split_domain_2d(nlat_global=20, nlon_global=41, num_workers_lat=2, num_workers_lon=2)

    def test_split_domain_coverage(self) -> None:
        """Test that patches cover entire domain without overlap."""
        patches = split_domain_2d(
            nlat_global=20, nlon_global=40, num_workers_lat=2, num_workers_lon=2
        )

        # Collect all cells
        covered_cells = set()
        for patch in patches:
            for i in range(patch["lat_start"], patch["lat_end"]):
                for j in range(patch["lon_start"], patch["lon_end"]):
                    assert (i, j) not in covered_cells  # No overlap
                    covered_cells.add((i, j))

        # Check complete coverage
        assert len(covered_cells) == 20 * 40


@pytest.mark.unit
class TestDomainSplittingPeriodic:
    """Test periodic domain splitting."""

    def test_periodic_lon_2x4(self) -> None:
        """Test periodic splitting in longitude."""
        patches = split_domain_2d_periodic_lon(
            nlat_global=20, nlon_global=40, num_workers_lat=2, num_workers_lon=4
        )

        assert len(patches) == 8

        # Worker 0 (top-left): west neighbor wraps to worker 3
        assert patches[0]["neighbors"]["west"] == 3
        # Worker 3 (top-right): east neighbor wraps to worker 0
        assert patches[3]["neighbors"]["east"] == 0

        # Check bottom row as well
        # Worker 4 (bottom-left): west neighbor wraps to worker 7
        assert patches[4]["neighbors"]["west"] == 7
        # Worker 7 (bottom-right): east neighbor wraps to worker 4
        assert patches[7]["neighbors"]["east"] == 4

    def test_periodic_vs_non_periodic(self) -> None:
        """Test difference between periodic and non-periodic."""
        patches_regular = split_domain_2d(
            nlat_global=20, nlon_global=40, num_workers_lat=2, num_workers_lon=4
        )
        patches_periodic = split_domain_2d_periodic_lon(
            nlat_global=20, nlon_global=40, num_workers_lat=2, num_workers_lon=4
        )

        # Non-periodic: leftmost workers have west=None
        assert patches_regular[0]["neighbors"]["west"] is None
        assert patches_regular[4]["neighbors"]["west"] is None

        # Periodic: leftmost workers have west neighbor (rightmost)
        assert patches_periodic[0]["neighbors"]["west"] == 3
        assert patches_periodic[4]["neighbors"]["west"] == 7

        # Non-periodic: rightmost workers have east=None
        assert patches_regular[3]["neighbors"]["east"] is None
        assert patches_regular[7]["neighbors"]["east"] is None

        # Periodic: rightmost workers have east neighbor (leftmost)
        assert patches_periodic[3]["neighbors"]["east"] == 0
        assert patches_periodic[7]["neighbors"]["east"] == 4

    def test_periodic_north_south_still_wall(self) -> None:
        """Test that north/south boundaries are still walls (not periodic)."""
        patches = split_domain_2d_periodic_lon(
            nlat_global=20, nlon_global=40, num_workers_lat=2, num_workers_lon=2
        )

        # Top row: north should be None
        assert patches[0]["neighbors"]["north"] is None
        assert patches[1]["neighbors"]["north"] is None

        # Bottom row: south should be None
        assert patches[2]["neighbors"]["south"] is None
        assert patches[3]["neighbors"]["south"] is None
