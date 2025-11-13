"""End-to-end integration tests for complete simulations."""

import jax.numpy as jnp
import pytest
import ray

from seapopym_message import (
    create_distributed_simulation,
    get_global_state,
    initialize_workers,
    setup_and_run,
)
from seapopym_message.core.kernel import Kernel
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
    # ray.shutdown()


@pytest.mark.integration
class TestEndToEndSimulation:
    """Test complete end-to-end simulations."""

    def test_create_distributed_simulation_2x2(self, ray_context) -> None:
        """Test creating a 2x2 distributed simulation."""
        grid = GridInfo(0.0, 10.0, 0.0, 20.0, nlat=20, nlon=40)
        kernel = Kernel([compute_growth])
        params = {"R": 5.0, "lambda": 0.1}

        workers, patches = create_distributed_simulation(
            grid=grid,
            kernel=kernel,
            params=params,
            num_workers_lat=2,
            num_workers_lon=2,
        )

        assert len(workers) == 4
        assert len(patches) == 4
        assert patches[0]["nlat"] == 10
        assert patches[0]["nlon"] == 20

    def test_initialize_and_get_state(self, ray_context) -> None:
        """Test initializing workers and retrieving global state."""
        grid = GridInfo(0.0, 10.0, 0.0, 20.0, nlat=10, nlon=20)
        kernel = Kernel([compute_growth])
        params = {"R": 5.0, "lambda": 0.1}

        workers, patches = create_distributed_simulation(
            grid, kernel, params, num_workers_lat=2, num_workers_lon=2
        )

        # Initialize with uniform values
        def uniform_init(lat_start, lat_end, lon_start, lon_end):
            nlat = lat_end - lat_start
            nlon = lon_end - lon_start
            return {"biomass": jnp.ones((nlat, nlon)) * 42.0}

        initialize_workers(workers, patches, uniform_init)

        # Get global state
        global_state = get_global_state(workers, patches)

        assert global_state["biomass"].shape == (10, 20)
        assert jnp.allclose(global_state["biomass"], 42.0)

    def test_setup_and_run_biology(self, ray_context) -> None:
        """Test complete simulation with biology kernel."""
        grid = GridInfo(0.0, 10.0, 0.0, 20.0, nlat=20, nlon=40)
        kernel = Kernel([compute_recruitment_2d, compute_mortality, compute_growth])
        params = {"R": 10.0, "lambda": 0.1}

        def zero_init(lat_start, lat_end, lon_start, lon_end):
            nlat = lat_end - lat_start
            nlon = lon_end - lon_start
            return {"biomass": jnp.zeros((nlat, nlon))}

        diagnostics, final_state = setup_and_run(
            grid=grid,
            kernel=kernel,
            params=params,
            initial_state_fn=zero_init,
            dt=0.1,
            t_max=1.0,
            num_workers_lat=2,
            num_workers_lon=2,
        )

        # Check results
        assert len(diagnostics) == 10
        assert final_state["biomass"].shape == (20, 40)
        assert jnp.all(final_state["biomass"] > 0)  # Should have grown

    def test_convergence_to_equilibrium(self, ray_context) -> None:
        """Test that simulation converges to expected equilibrium."""
        grid = GridInfo(0.0, 10.0, 0.0, 20.0, nlat=10, nlon=20)
        kernel = Kernel([compute_recruitment_2d, compute_mortality, compute_growth])
        params = {"R": 10.0, "lambda": 0.1}

        def zero_init(lat_start, lat_end, lon_start, lon_end):
            nlat = lat_end - lat_start
            nlon = lon_end - lon_start
            return {"biomass": jnp.zeros((nlat, nlon))}

        diagnostics, final_state = setup_and_run(
            grid=grid,
            kernel=kernel,
            params=params,
            initial_state_fn=zero_init,
            dt=0.1,
            t_max=50.0,
            num_workers_lat=1,
            num_workers_lon=1,
        )

        # Should converge to B_eq = R/lambda = 100
        final_biomass = diagnostics[-1]["biomass_global_mean"]
        assert jnp.allclose(final_biomass, 100.0, rtol=0.01)

    def test_diffusion_mass_conservation(self, ray_context) -> None:
        """Test that diffusion conserves mass."""
        grid = GridInfo(0.0, 10.0, 0.0, 20.0, nlat=20, nlon=40)
        kernel = Kernel([compute_diffusion_2d])
        params = {"D": 1.0, "dx": grid.dx}

        # Initial condition: Gaussian blob
        def gaussian_init(lat_start, lat_end, lon_start, lon_end):
            nlat = lat_end - lat_start
            nlon = lon_end - lon_start
            center_lat = (lat_start + lat_end) // 2
            center_lon = (lon_start + lon_end) // 2
            i_coords = jnp.arange(nlat)
            j_coords = jnp.arange(nlon)
            grid_i, grid_j = jnp.meshgrid(i_coords, j_coords, indexing="ij")
            biomass = 100.0 * jnp.exp(
                -((grid_i - center_lat) ** 2 + (grid_j - center_lon) ** 2) / 20.0
            )
            return {"biomass": biomass}

        diagnostics, final_state = setup_and_run(
            grid=grid,
            kernel=kernel,
            params=params,
            initial_state_fn=gaussian_init,
            dt=0.01,
            t_max=0.1,
            num_workers_lat=2,
            num_workers_lon=2,
        )

        # Check mass conservation
        initial_mean = diagnostics[0]["biomass_global_mean"]
        final_mean = diagnostics[-1]["biomass_global_mean"]
        mass_change = abs(final_mean - initial_mean) / initial_mean

        assert mass_change < 0.01  # Less than 1% change

    def test_periodic_boundaries(self, ray_context) -> None:
        """Test periodic boundary conditions in longitude."""
        grid = GridInfo(0.0, 10.0, 0.0, 20.0, nlat=10, nlon=20)
        kernel = Kernel([compute_diffusion_2d])
        params = {"D": 1.0, "dx": grid.dx}

        def uniform_init(lat_start, lat_end, lon_start, lon_end):
            nlat = lat_end - lat_start
            nlon = lon_end - lon_start
            return {"biomass": jnp.ones((nlat, nlon)) * 50.0}

        # Test with periodic boundaries
        workers_periodic, patches_periodic = create_distributed_simulation(
            grid=grid,
            kernel=kernel,
            params=params,
            num_workers_lat=2,
            num_workers_lon=4,
            periodic_lon=True,
        )

        # Check that leftmost and rightmost workers are connected
        # Worker 0 (top-left) should have worker 3 (top-right) as west neighbor
        assert patches_periodic[0]["neighbors"]["west"] == 3
        # Worker 3 (top-right) should have worker 0 (top-left) as east neighbor
        assert patches_periodic[3]["neighbors"]["east"] == 0


@pytest.mark.integration
class TestMultiWorkerScaling:
    """Test simulation with different numbers of workers."""

    def test_consistent_results_across_decompositions(self, ray_context) -> None:
        """Test that different domain decompositions give same results."""
        grid = GridInfo(0.0, 10.0, 0.0, 20.0, nlat=20, nlon=40)
        kernel = Kernel([compute_recruitment_2d, compute_mortality, compute_growth])
        params = {"R": 10.0, "lambda": 0.1}

        def init_fn(lat_start, lat_end, lon_start, lon_end):
            return {"biomass": jnp.zeros((lat_end - lat_start, lon_end - lon_start))}

        # Run with different decompositions
        _, state_1x1 = setup_and_run(
            grid, kernel, params, init_fn, dt=0.1, t_max=5.0, num_workers_lat=1, num_workers_lon=1
        )

        _, state_2x2 = setup_and_run(
            grid, kernel, params, init_fn, dt=0.1, t_max=5.0, num_workers_lat=2, num_workers_lon=2
        )

        # Results should be very close (within numerical precision)
        assert jnp.allclose(state_1x1["biomass"], state_2x2["biomass"], rtol=1e-5)
