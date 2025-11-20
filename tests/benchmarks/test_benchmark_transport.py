"""Benchmark tests for transport functions.

This module benchmarks the core transport operations:
- advection_upwind_flux: Flux-based upwind advection
- diffusion_explicit_spherical: Explicit Euler diffusion
- TransportWorker.transport_step: Full transport step (Ray actor)

Key considerations:
1. JAX synchronization with .block_until_ready()
2. JIT compilation warm-up phase
3. Grid geometry (spherical vs plane)
4. Different problem sizes for scaling analysis
"""

import jax.numpy as jnp
import pytest

from seapopym_message.transport.advection import advection_upwind_flux
from seapopym_message.transport.boundary import BoundaryConditions, BoundaryType
from seapopym_message.transport.diffusion import diffusion_explicit_spherical
from seapopym_message.transport.grid import PlaneGrid, SphericalGrid
from seapopym_message.utils.grid import PlaneGridInfo, SphericalGridInfo


@pytest.fixture
def boundary_periodic():
    """Periodic boundary conditions (typical for global ocean)."""
    return BoundaryConditions(
        north=BoundaryType.CLOSED,
        south=BoundaryType.CLOSED,
        east=BoundaryType.PERIODIC,
        west=BoundaryType.PERIODIC,
    )


@pytest.fixture
def spherical_grid_small():
    """Small spherical grid (30x60) for quick tests."""
    grid_info = SphericalGridInfo(
        lat_min=-60.0,
        lat_max=60.0,
        lon_min=0.0,
        lon_max=360.0,
        nlat=30,
        nlon=60,
    )
    return SphericalGrid(grid_info=grid_info, R=6371e3)


@pytest.fixture
def spherical_grid_medium():
    """Medium spherical grid (120x360) for realistic tests."""
    grid_info = SphericalGridInfo(
        lat_min=-60.0,
        lat_max=60.0,
        lon_min=0.0,
        lon_max=360.0,
        nlat=120,
        nlon=360,
    )
    return SphericalGrid(grid_info=grid_info, R=6371e3)


@pytest.fixture
def spherical_grid_large():
    """Large spherical grid (360x720) for performance tests."""
    grid_info = SphericalGridInfo(
        lat_min=-60.0,
        lat_max=60.0,
        lon_min=0.0,
        lon_max=360.0,
        nlat=360,
        nlon=720,
    )
    return SphericalGrid(grid_info=grid_info, R=6371e3)


@pytest.fixture
def plane_grid_small():
    """Small plane grid (30x60) for quick tests."""
    grid_info = PlaneGridInfo(dx=100e3, dy=100e3, nlat=30, nlon=60)
    return PlaneGrid(grid_info=grid_info)


@pytest.fixture
def plane_grid_medium():
    """Medium plane grid (120x360) for realistic tests."""
    grid_info = PlaneGridInfo(dx=100e3, dy=100e3, nlat=120, nlon=360)
    return PlaneGrid(grid_info=grid_info)


class TestAdvectionBenchmarks:
    """Benchmark suite for advection functions."""

    def test_benchmark_advection_spherical_small(
        self, benchmark, spherical_grid_small, boundary_periodic
    ):
        """Benchmark advection on small spherical grid (30x60)."""
        nlat, nlon = 30, 60
        biomass = jnp.ones((nlat, nlon)) * 100.0
        u = jnp.ones((nlat, nlon)) * 0.1  # 0.1 m/s eastward
        v = jnp.zeros((nlat, nlon))
        dt = 3600.0  # 1 hour

        # Warm-up
        _ = advection_upwind_flux(biomass, u, v, dt, spherical_grid_small, boundary_periodic)

        def run():
            result = advection_upwind_flux(
                biomass, u, v, dt, spherical_grid_small, boundary_periodic
            )
            result.block_until_ready()
            return result

        result = benchmark(run)
        assert result.shape == (nlat, nlon)

    def test_benchmark_advection_spherical_medium(
        self, benchmark, spherical_grid_medium, boundary_periodic
    ):
        """Benchmark advection on medium spherical grid (120x360)."""
        nlat, nlon = 120, 360
        biomass = jnp.ones((nlat, nlon)) * 100.0
        u = jnp.ones((nlat, nlon)) * 0.1
        v = jnp.zeros((nlat, nlon))
        dt = 3600.0

        # Warm-up
        _ = advection_upwind_flux(biomass, u, v, dt, spherical_grid_medium, boundary_periodic)

        def run():
            result = advection_upwind_flux(
                biomass, u, v, dt, spherical_grid_medium, boundary_periodic
            )
            result.block_until_ready()
            return result

        result = benchmark(run)
        assert result.shape == (nlat, nlon)

    def test_benchmark_advection_spherical_large(
        self, benchmark, spherical_grid_large, boundary_periodic
    ):
        """Benchmark advection on large spherical grid (360x720)."""
        nlat, nlon = 360, 720
        biomass = jnp.ones((nlat, nlon)) * 100.0
        u = jnp.ones((nlat, nlon)) * 0.1
        v = jnp.zeros((nlat, nlon))
        dt = 3600.0

        # Warm-up
        _ = advection_upwind_flux(biomass, u, v, dt, spherical_grid_large, boundary_periodic)

        def run():
            result = advection_upwind_flux(
                biomass, u, v, dt, spherical_grid_large, boundary_periodic
            )
            result.block_until_ready()
            return result

        result = benchmark(run)
        assert result.shape == (nlat, nlon)

    def test_benchmark_advection_plane_small(self, benchmark, plane_grid_small, boundary_periodic):
        """Benchmark advection on small plane grid (30x60)."""
        nlat, nlon = 30, 60
        biomass = jnp.ones((nlat, nlon)) * 100.0
        u = jnp.ones((nlat, nlon)) * 0.1
        v = jnp.zeros((nlat, nlon))
        dt = 3600.0

        # Warm-up
        _ = advection_upwind_flux(biomass, u, v, dt, plane_grid_small, boundary_periodic)

        def run():
            result = advection_upwind_flux(biomass, u, v, dt, plane_grid_small, boundary_periodic)
            result.block_until_ready()
            return result

        result = benchmark(run)
        assert result.shape == (nlat, nlon)


class TestDiffusionBenchmarks:
    """Benchmark suite for diffusion functions."""

    def test_benchmark_diffusion_spherical_small(
        self, benchmark, spherical_grid_small, boundary_periodic
    ):
        """Benchmark diffusion on small spherical grid (30x60)."""
        nlat, nlon = 30, 60
        biomass = jnp.ones((nlat, nlon)) * 100.0
        D = 1000.0  # 1000 m²/s
        dt = 3600.0

        # Warm-up
        _ = diffusion_explicit_spherical(biomass, D, dt, spherical_grid_small, boundary_periodic)

        def run():
            result = diffusion_explicit_spherical(
                biomass, D, dt, spherical_grid_small, boundary_periodic
            )
            result.block_until_ready()
            return result

        result = benchmark(run)
        assert result.shape == (nlat, nlon)

    def test_benchmark_diffusion_spherical_medium(
        self, benchmark, spherical_grid_medium, boundary_periodic
    ):
        """Benchmark diffusion on medium spherical grid (120x360)."""
        nlat, nlon = 120, 360
        biomass = jnp.ones((nlat, nlon)) * 100.0
        D = 1000.0
        dt = 3600.0

        # Warm-up
        _ = diffusion_explicit_spherical(biomass, D, dt, spherical_grid_medium, boundary_periodic)

        def run():
            result = diffusion_explicit_spherical(
                biomass, D, dt, spherical_grid_medium, boundary_periodic
            )
            result.block_until_ready()
            return result

        result = benchmark(run)
        assert result.shape == (nlat, nlon)

    def test_benchmark_diffusion_spherical_large(
        self, benchmark, spherical_grid_large, boundary_periodic
    ):
        """Benchmark diffusion on large spherical grid (360x720)."""
        nlat, nlon = 360, 720
        biomass = jnp.ones((nlat, nlon)) * 100.0
        D = 1000.0
        dt = 3600.0

        # Warm-up
        _ = diffusion_explicit_spherical(biomass, D, dt, spherical_grid_large, boundary_periodic)

        def run():
            result = diffusion_explicit_spherical(
                biomass, D, dt, spherical_grid_large, boundary_periodic
            )
            result.block_until_ready()
            return result

        result = benchmark(run)
        assert result.shape == (nlat, nlon)

    def test_benchmark_diffusion_plane_small(self, benchmark, plane_grid_small, boundary_periodic):
        """Benchmark diffusion on small plane grid (30x60)."""
        nlat, nlon = 30, 60
        biomass = jnp.ones((nlat, nlon)) * 100.0
        D = 1000.0
        dt = 3600.0

        # Warm-up
        _ = diffusion_explicit_spherical(biomass, D, dt, plane_grid_small, boundary_periodic)

        def run():
            result = diffusion_explicit_spherical(
                biomass, D, dt, plane_grid_small, boundary_periodic
            )
            result.block_until_ready()
            return result

        result = benchmark(run)
        assert result.shape == (nlat, nlon)


class TestCombinedTransportBenchmarks:
    """Benchmark suite for combined advection + diffusion."""

    def test_benchmark_full_transport_spherical_small(
        self, benchmark, spherical_grid_small, boundary_periodic
    ):
        """Benchmark full transport step on small grid (30x60)."""
        nlat, nlon = 30, 60
        biomass = jnp.ones((nlat, nlon)) * 100.0
        u = jnp.ones((nlat, nlon)) * 0.1
        v = jnp.zeros((nlat, nlon))
        D = 1000.0
        dt = 3600.0

        # Warm-up
        biomass_adv = advection_upwind_flux(
            biomass, u, v, dt, spherical_grid_small, boundary_periodic
        )
        _ = diffusion_explicit_spherical(
            biomass_adv, D, dt, spherical_grid_small, boundary_periodic
        )

        def run():
            # Advection step
            biomass_adv = advection_upwind_flux(
                biomass, u, v, dt, spherical_grid_small, boundary_periodic
            )
            biomass_adv.block_until_ready()

            # Diffusion step
            biomass_final = diffusion_explicit_spherical(
                biomass_adv, D, dt, spherical_grid_small, boundary_periodic
            )
            biomass_final.block_until_ready()
            return biomass_final

        result = benchmark(run)
        assert result.shape == (nlat, nlon)

    def test_benchmark_full_transport_spherical_medium(
        self, benchmark, spherical_grid_medium, boundary_periodic
    ):
        """Benchmark full transport step on medium grid (120x360)."""
        nlat, nlon = 120, 360
        biomass = jnp.ones((nlat, nlon)) * 100.0
        u = jnp.ones((nlat, nlon)) * 0.1
        v = jnp.zeros((nlat, nlon))
        D = 1000.0
        dt = 3600.0

        # Warm-up
        biomass_adv = advection_upwind_flux(
            biomass, u, v, dt, spherical_grid_medium, boundary_periodic
        )
        _ = diffusion_explicit_spherical(
            biomass_adv, D, dt, spherical_grid_medium, boundary_periodic
        )

        def run():
            biomass_adv = advection_upwind_flux(
                biomass, u, v, dt, spherical_grid_medium, boundary_periodic
            )
            biomass_adv.block_until_ready()

            biomass_final = diffusion_explicit_spherical(
                biomass_adv, D, dt, spherical_grid_medium, boundary_periodic
            )
            biomass_final.block_until_ready()
            return biomass_final

        result = benchmark(run)
        assert result.shape == (nlat, nlon)
