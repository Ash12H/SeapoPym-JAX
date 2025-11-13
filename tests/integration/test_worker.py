"""Integration tests for CellWorker2D."""

import jax.numpy as jnp
import pytest
import ray

from seapopym_message.core.kernel import Kernel
from seapopym_message.distributed.worker import CellWorker2D
from seapopym_message.kernels.biology import (
    compute_growth,
    compute_mortality,
    compute_recruitment_2d,
)
from seapopym_message.kernels.transport import compute_diffusion_2d
from seapopym_message.utils.grid import GridInfo


@pytest.fixture(scope="module")
def ray_context():
    """Initialize Ray for tests."""
    if not ray.is_initialized():
        ray.init(num_cpus=2, ignore_reinit_error=True)
    yield
    # ray.shutdown()  # Don't shutdown to avoid issues with other tests


@pytest.mark.integration
class TestCellWorker2DBasic:
    """Basic tests for CellWorker2D."""

    def test_worker_creation(self, ray_context) -> None:
        """Test creating a worker."""
        grid = GridInfo(lat_min=0.0, lat_max=10.0, lon_min=0.0, lon_max=20.0, nlat=10, nlon=20)
        kernel = Kernel([compute_recruitment_2d, compute_mortality, compute_growth])
        params = {"R": 5.0, "lambda": 0.1}

        worker = CellWorker2D.remote(
            worker_id=0,
            grid_info=grid,
            lat_start=0,
            lat_end=5,
            lon_start=0,
            lon_end=10,
            kernel=kernel,
            params=params,
        )

        # Should create successfully
        assert worker is not None

    def test_worker_set_initial_state(self, ray_context) -> None:
        """Test setting initial state."""
        grid = GridInfo(lat_min=0.0, lat_max=10.0, lon_min=0.0, lon_max=20.0, nlat=10, nlon=20)
        kernel = Kernel([compute_recruitment_2d, compute_mortality, compute_growth])
        params = {"R": 5.0, "lambda": 0.1}

        worker = CellWorker2D.remote(
            worker_id=0,
            grid_info=grid,
            lat_start=0,
            lat_end=5,
            lon_start=0,
            lon_end=10,
            kernel=kernel,
            params=params,
        )

        # Set initial state
        initial_state = {"biomass": jnp.ones((5, 10)) * 10.0}
        ray.get(worker.set_initial_state.remote(initial_state))

        # Get state back
        state = ray.get(worker.get_state.remote())

        assert "biomass" in state
        assert state["biomass"].shape == (5, 10)
        assert jnp.allclose(state["biomass"], 10.0)

    def test_worker_step_local_only(self, ray_context) -> None:
        """Test single step with local-only kernel."""
        grid = GridInfo(lat_min=0.0, lat_max=10.0, lon_min=0.0, lon_max=20.0, nlat=10, nlon=20)
        kernel = Kernel([compute_recruitment_2d, compute_mortality, compute_growth])
        params = {"R": 5.0, "lambda": 0.1}

        worker = CellWorker2D.remote(
            worker_id=0,
            grid_info=grid,
            lat_start=0,
            lat_end=5,
            lon_start=0,
            lon_end=10,
            kernel=kernel,
            params=params,
        )

        # Initialize
        initial_state = {"biomass": jnp.ones((5, 10)) * 10.0}
        ray.get(worker.set_initial_state.remote(initial_state))

        # Execute one step
        diagnostics = ray.get(worker.step.remote(dt=0.1))

        # Check diagnostics
        assert diagnostics["worker_id"] == 0
        assert diagnostics["t"] == 0.1
        assert "biomass_mean" in diagnostics

        # Get updated state
        state = ray.get(worker.get_state.remote())
        # Biomass should have changed (growth)
        assert not jnp.allclose(state["biomass"], 10.0)


@pytest.mark.integration
class TestCellWorker2DBoundaries:
    """Test boundary exchange between workers."""

    def test_get_boundaries(self, ray_context) -> None:
        """Test getting boundary data."""
        grid = GridInfo(lat_min=0.0, lat_max=10.0, lon_min=0.0, lon_max=20.0, nlat=10, nlon=20)
        kernel = Kernel([compute_recruitment_2d, compute_mortality, compute_growth])
        params = {"R": 5.0, "lambda": 0.1}

        worker = CellWorker2D.remote(
            worker_id=0,
            grid_info=grid,
            lat_start=0,
            lat_end=5,
            lon_start=0,
            lon_end=10,
            kernel=kernel,
            params=params,
        )

        # Set state with specific values
        state = {"biomass": jnp.arange(50, dtype=float).reshape(5, 10)}
        ray.get(worker.set_initial_state.remote(state))

        # Get boundaries
        north = ray.get(worker.get_boundary_north.remote())
        south = ray.get(worker.get_boundary_south.remote())
        east = ray.get(worker.get_boundary_east.remote())
        west = ray.get(worker.get_boundary_west.remote())

        # Check shapes
        assert north["biomass"].shape == (10,)  # nlon
        assert south["biomass"].shape == (10,)
        assert east["biomass"].shape == (5,)  # nlat
        assert west["biomass"].shape == (5,)

        # Check values
        assert jnp.allclose(north["biomass"], jnp.arange(10))  # First row
        assert jnp.allclose(south["biomass"], jnp.arange(40, 50))  # Last row
        assert jnp.allclose(west["biomass"], jnp.array([0, 10, 20, 30, 40]))  # First column
        assert jnp.allclose(east["biomass"], jnp.array([9, 19, 29, 39, 49]))  # Last column

    def test_two_workers_halo_exchange(self, ray_context) -> None:
        """Test halo exchange between two workers."""
        grid = GridInfo(lat_min=0.0, lat_max=10.0, lon_min=0.0, lon_max=20.0, nlat=10, nlon=20)

        # Kernel with global unit (diffusion)
        kernel = Kernel([compute_diffusion_2d])
        params = {"D": 1.0, "dx": 1000.0}

        # Create two workers (north and south)
        worker_north = CellWorker2D.remote(
            worker_id=0,
            grid_info=grid,
            lat_start=0,
            lat_end=5,
            lon_start=0,
            lon_end=20,
            kernel=kernel,
            params=params,
        )

        worker_south = CellWorker2D.remote(
            worker_id=1,
            grid_info=grid,
            lat_start=5,
            lat_end=10,
            lon_start=0,
            lon_end=20,
            kernel=kernel,
            params=params,
        )

        # Set neighbors
        worker_north.set_neighbors.remote(
            {"north": None, "south": worker_south, "east": None, "west": None}
        )
        worker_south.set_neighbors.remote(
            {"north": worker_north, "south": None, "east": None, "west": None}
        )

        # Initialize with different values
        state_north = {"biomass": jnp.ones((5, 20)) * 10.0}
        state_south = {"biomass": jnp.ones((5, 20)) * 50.0}

        ray.get(worker_north.set_initial_state.remote(state_north))
        ray.get(worker_south.set_initial_state.remote(state_south))

        # Execute one step (should trigger halo exchange)
        diag_north = ray.get(worker_north.step.remote(dt=0.01))
        diag_south = ray.get(worker_south.step.remote(dt=0.01))

        # Both should complete
        assert diag_north["t"] == 0.01
        assert diag_south["t"] == 0.01


@pytest.mark.integration
class TestCellWorker2DMultiStep:
    """Test multiple timesteps."""

    def test_multiple_steps(self, ray_context) -> None:
        """Test running multiple timesteps."""
        grid = GridInfo(lat_min=0.0, lat_max=10.0, lon_min=0.0, lon_max=20.0, nlat=10, nlon=20)
        kernel = Kernel([compute_recruitment_2d, compute_mortality, compute_growth])
        params = {"R": 10.0, "lambda": 0.1}

        worker = CellWorker2D.remote(
            worker_id=0,
            grid_info=grid,
            lat_start=0,
            lat_end=10,
            lon_start=0,
            lon_end=20,
            kernel=kernel,
            params=params,
        )

        # Initialize from zero
        initial_state = {"biomass": jnp.zeros((10, 20))}
        ray.get(worker.set_initial_state.remote(initial_state))

        # Run 10 steps
        for i in range(10):
            diagnostics = ray.get(worker.step.remote(dt=0.1))
            assert diagnostics["t"] == pytest.approx((i + 1) * 0.1)

        # Final state should have positive biomass
        final_state = ray.get(worker.get_state.remote())
        assert jnp.all(final_state["biomass"] > 0)

    def test_convergence_to_equilibrium(self, ray_context) -> None:
        """Test that worker converges to equilibrium."""
        grid = GridInfo(lat_min=0.0, lat_max=10.0, lon_min=0.0, lon_max=20.0, nlat=10, nlon=20)
        kernel = Kernel([compute_recruitment_2d, compute_mortality, compute_growth])
        params = {"R": 10.0, "lambda": 0.1}

        worker = CellWorker2D.remote(
            worker_id=0,
            grid_info=grid,
            lat_start=0,
            lat_end=10,
            lon_start=0,
            lon_end=20,
            kernel=kernel,
            params=params,
        )

        # Initialize
        initial_state = {"biomass": jnp.zeros((10, 20))}
        ray.get(worker.set_initial_state.remote(initial_state))

        # Run many steps
        for _ in range(500):
            ray.get(worker.step.remote(dt=0.1))

        # Should converge to B_eq = R/lambda = 100
        final_state = ray.get(worker.get_state.remote())
        assert jnp.allclose(final_state["biomass"], 100.0, rtol=0.01)
