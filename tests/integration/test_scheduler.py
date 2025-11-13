"""Integration tests for EventScheduler with CellWorker2D."""

import jax.numpy as jnp
import pytest
import ray

from seapopym_message.core.kernel import Kernel
from seapopym_message.distributed.scheduler import EventScheduler
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
        ray.init(num_cpus=4, ignore_reinit_error=True)
    yield
    # ray.shutdown()  # Don't shutdown to avoid issues with other tests


@pytest.mark.integration
class TestEventSchedulerBasic:
    """Basic tests for EventScheduler."""

    def test_scheduler_with_single_worker(self, ray_context) -> None:
        """Test scheduler with single worker."""
        grid = GridInfo(lat_min=0.0, lat_max=10.0, lon_min=0.0, lon_max=20.0, nlat=10, nlon=20)
        kernel = Kernel([compute_recruitment_2d, compute_mortality, compute_growth])
        params = {"R": 10.0, "lambda": 0.1}

        # Create single worker
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

        # Create scheduler
        scheduler = EventScheduler(workers=[worker], dt=0.1, t_max=1.0)

        # Run simulation
        diagnostics = scheduler.run()

        # Check results
        assert len(diagnostics) == 10  # 1.0 / 0.1 = 10 steps
        assert pytest.approx(diagnostics[0]["t"], abs=1e-10) == 0.1
        assert pytest.approx(diagnostics[-1]["t"], abs=1e-10) == 1.0
        assert "biomass_global_mean" in diagnostics[-1]
        assert diagnostics[-1]["biomass_global_mean"] > 0  # Should have positive biomass

    def test_scheduler_single_step(self, ray_context) -> None:
        """Test executing single step."""
        grid = GridInfo(lat_min=0.0, lat_max=10.0, lon_min=0.0, lon_max=20.0, nlat=10, nlon=20)
        kernel = Kernel([compute_recruitment_2d, compute_mortality, compute_growth])
        params = {"R": 5.0, "lambda": 0.1}

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

        initial_state = {"biomass": jnp.ones((10, 20)) * 10.0}
        ray.get(worker.set_initial_state.remote(initial_state))

        scheduler = EventScheduler(workers=[worker], dt=0.1, t_max=10.0)

        # Execute one step
        diagnostics = scheduler.step()

        assert pytest.approx(diagnostics["t"], abs=1e-10) == 0.1
        assert diagnostics["num_workers"] == 1
        assert "biomass_global_mean" in diagnostics
        assert pytest.approx(scheduler.get_current_time(), abs=1e-10) == 0.1


@pytest.mark.integration
class TestEventSchedulerMultiWorker:
    """Test scheduler with multiple workers."""

    def test_scheduler_two_workers_local_kernel(self, ray_context) -> None:
        """Test scheduler with 2 workers (local kernel, no halo exchange)."""
        grid = GridInfo(lat_min=0.0, lat_max=10.0, lon_min=0.0, lon_max=20.0, nlat=10, nlon=20)
        kernel = Kernel([compute_recruitment_2d, compute_mortality, compute_growth])
        params = {"R": 10.0, "lambda": 0.1}

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

        # Initialize both
        state_north = {"biomass": jnp.zeros((5, 20))}
        state_south = {"biomass": jnp.zeros((5, 20))}
        ray.get(worker_north.set_initial_state.remote(state_north))
        ray.get(worker_south.set_initial_state.remote(state_south))

        # Create scheduler
        scheduler = EventScheduler(workers=[worker_north, worker_south], dt=0.1, t_max=1.0)

        # Run simulation
        diagnostics = scheduler.run()

        assert len(diagnostics) == 10
        assert diagnostics[-1]["num_workers"] == 2
        assert diagnostics[-1]["biomass_global_mean"] > 0

    def test_scheduler_four_workers_with_diffusion(self, ray_context) -> None:
        """Test scheduler with 4 workers and diffusion (global kernel)."""
        grid = GridInfo(lat_min=0.0, lat_max=10.0, lon_min=0.0, lon_max=20.0, nlat=10, nlon=20)
        kernel = Kernel([compute_diffusion_2d])
        params = {"D": 1.0, "dx": 1000.0}

        # Create 4 workers (2×2 grid)
        workers = []
        for i_lat in range(2):
            for i_lon in range(2):
                worker = CellWorker2D.remote(
                    worker_id=i_lat * 2 + i_lon,
                    grid_info=grid,
                    lat_start=i_lat * 5,
                    lat_end=(i_lat + 1) * 5,
                    lon_start=i_lon * 10,
                    lon_end=(i_lon + 1) * 10,
                    kernel=kernel,
                    params=params,
                )
                workers.append(worker)

        # Set neighbors (simplified: only internal boundaries)
        workers[0].set_neighbors.remote(
            {"north": None, "south": workers[2], "east": workers[1], "west": None}
        )
        workers[1].set_neighbors.remote(
            {"north": None, "south": workers[3], "east": None, "west": workers[0]}
        )
        workers[2].set_neighbors.remote(
            {"north": workers[0], "south": None, "east": workers[3], "west": None}
        )
        workers[3].set_neighbors.remote(
            {"north": workers[1], "south": None, "east": None, "west": workers[2]}
        )

        # Initialize with different values in each patch
        states = [
            {"biomass": jnp.ones((5, 10)) * 10.0},  # NW
            {"biomass": jnp.ones((5, 10)) * 20.0},  # NE
            {"biomass": jnp.ones((5, 10)) * 30.0},  # SW
            {"biomass": jnp.ones((5, 10)) * 40.0},  # SE
        ]
        for worker, state in zip(workers, states, strict=False):
            ray.get(worker.set_initial_state.remote(state))

        # Create scheduler
        scheduler = EventScheduler(workers=workers, dt=0.01, t_max=0.1)

        # Run simulation
        diagnostics = scheduler.run()

        assert len(diagnostics) == 10
        assert diagnostics[0]["num_workers"] == 4

        # Check that global aggregation works
        first_step = diagnostics[0]
        assert "biomass_global_mean" in first_step
        assert "biomass_global_min" in first_step
        assert "biomass_global_max" in first_step

        # Mean should be around 25.0 initially (10+20+30+40)/4
        assert 20.0 < first_step["biomass_global_mean"] < 30.0
        assert first_step["biomass_global_min"] < first_step["biomass_global_mean"]
        assert first_step["biomass_global_max"] > first_step["biomass_global_mean"]

    def test_scheduler_get_worker_states(self, ray_context) -> None:
        """Test retrieving worker states."""
        grid = GridInfo(lat_min=0.0, lat_max=10.0, lon_min=0.0, lon_max=20.0, nlat=10, nlon=20)
        kernel = Kernel([compute_recruitment_2d, compute_mortality, compute_growth])
        params = {"R": 5.0, "lambda": 0.1}

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

        initial_state = {"biomass": jnp.ones((10, 20)) * 42.0}
        ray.get(worker.set_initial_state.remote(initial_state))

        scheduler = EventScheduler(workers=[worker], dt=0.1, t_max=1.0)

        # Get states
        states = scheduler.get_worker_states()

        assert len(states) == 1
        assert "biomass" in states[0]
        assert jnp.allclose(states[0]["biomass"], 42.0)


@pytest.mark.integration
class TestEventSchedulerLongRun:
    """Test scheduler for longer simulations."""

    def test_convergence_with_scheduler(self, ray_context) -> None:
        """Test that simulation converges to equilibrium using scheduler."""
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

        initial_state = {"biomass": jnp.zeros((10, 20))}
        ray.get(worker.set_initial_state.remote(initial_state))

        # Run long simulation
        scheduler = EventScheduler(workers=[worker], dt=0.1, t_max=50.0)
        diagnostics = scheduler.run()

        # Should converge to B_eq = R/lambda = 100
        final_biomass = diagnostics[-1]["biomass_global_mean"]
        assert jnp.allclose(final_biomass, 100.0, rtol=0.01)

    def test_scheduler_time_progression(self, ray_context) -> None:
        """Test that time progresses correctly."""
        grid = GridInfo(lat_min=0.0, lat_max=10.0, lon_min=0.0, lon_max=20.0, nlat=10, nlon=20)
        kernel = Kernel([compute_recruitment_2d, compute_mortality, compute_growth])
        params = {"R": 5.0, "lambda": 0.1}

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

        initial_state = {"biomass": jnp.ones((10, 20)) * 10.0}
        ray.get(worker.set_initial_state.remote(initial_state))

        scheduler = EventScheduler(workers=[worker], dt=0.2, t_max=2.0)
        diagnostics = scheduler.run()

        # Check time progression
        assert len(diagnostics) == 10  # 2.0 / 0.2 = 10
        for i, diag in enumerate(diagnostics):
            expected_time = (i + 1) * 0.2
            assert pytest.approx(diag["t"], abs=1e-10) == expected_time

        # Final time
        assert pytest.approx(scheduler.get_current_time(), abs=1e-10) == 2.0
