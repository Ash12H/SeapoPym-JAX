"""Integration tests for EventScheduler with TransportWorker.

Tests verify:
- Biology-only mode (transport disabled)
- Transport-only mode (no biology growth)
- Coupled biology + transport
- Mass conservation during collect/redistribute
- Multi-worker coordination
"""

import jax.numpy as jnp
import ray

from seapopym_message.core.kernel import Kernel
from seapopym_message.core.unit import unit
from seapopym_message.distributed.scheduler import EventScheduler
from seapopym_message.distributed.worker import CellWorker2D
from seapopym_message.transport.worker import TransportWorker
from seapopym_message.utils.grid import GridInfo


# Simple growth compute unit for testing
@unit(name="simple_growth", inputs=["biomass"], outputs=["biomass"], scope="local", compiled=False)
def simple_growth(biomass, dt, params, forcings=None):
    """Simple exponential growth: dB/dt = r*B."""
    r = params.get("growth_rate", 0.0)
    biomass_new = biomass * jnp.exp(r * dt)
    return biomass_new


class TestSchedulerTransportIntegration:
    """Integration tests for scheduler with transport."""

    def setup_method(self):
        """Initialize Ray before each test."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    def test_biology_only_no_transport(self):
        """Test 1: Biology only (transport disabled).

        Verify that simulation runs without transport enabled.
        """
        # Grid configuration (2x2 workers, 20x20 global grid)
        global_nlat, global_nlon = 20, 20
        n_workers_lat, n_workers_lon = 2, 2
        nlat_per_worker = global_nlat // n_workers_lat
        nlon_per_worker = global_nlon // n_workers_lon

        grid_info = GridInfo(
            lat_min=-30.0,
            lat_max=30.0,
            lon_min=0.0,
            lon_max=360.0,
            nlat=global_nlat,
            nlon=global_nlon,
        )

        # Create kernel with simple growth
        kernel = Kernel([simple_growth])

        # Create workers
        workers = []
        for i in range(n_workers_lat):
            for j in range(n_workers_lon):
                lat_start = i * nlat_per_worker
                lat_end = (i + 1) * nlat_per_worker
                lon_start = j * nlon_per_worker
                lon_end = (j + 1) * nlon_per_worker

                worker = CellWorker2D.remote(
                    worker_id=i * n_workers_lon + j,
                    grid_info=grid_info,
                    lat_start=lat_start,
                    lat_end=lat_end,
                    lon_start=lon_start,
                    lon_end=lon_end,
                    kernel=kernel,
                    params={"growth_rate": 0.01},  # 1% growth per timestep
                )

                # Set initial biomass
                initial_biomass = jnp.ones((nlat_per_worker, nlon_per_worker)) * 10.0
                ray.get(worker.set_initial_state.remote({"biomass": initial_biomass}))

                workers.append(worker)

        # Create scheduler WITHOUT transport
        scheduler = EventScheduler(
            workers=workers, dt=3600.0, t_max=7200.0, transport_enabled=False
        )

        # Run simulation (2 timesteps)
        diagnostics = scheduler.run()

        # Verify results
        assert len(diagnostics) == 2
        assert diagnostics[0]["t"] == 3600.0
        assert diagnostics[1]["t"] == 7200.0

        # Verify that biomass increased (growth)
        assert diagnostics[1]["biomass_global_mean"] > diagnostics[0]["biomass_global_mean"]

        # No transport diagnostics
        assert "transport" not in diagnostics[0]

    def test_collect_redistribute_conservation(self):
        """Test collect and redistribute preserve mass.

        Verify that collecting biomass from workers and redistributing
        it back conserves total mass perfectly.
        """
        # Grid configuration
        global_nlat, global_nlon = 20, 20
        n_workers_lat, n_workers_lon = 2, 2
        nlat_per_worker = global_nlat // n_workers_lat
        nlon_per_worker = global_nlon // n_workers_lon

        grid_info = GridInfo(
            lat_min=-30.0,
            lat_max=30.0,
            lon_min=0.0,
            lon_max=360.0,
            nlat=global_nlat,
            nlon=global_nlon,
        )

        # Create simple kernel (no growth for this test)
        kernel = Kernel([simple_growth])

        # Create workers
        workers = []
        for i in range(n_workers_lat):
            for j in range(n_workers_lon):
                lat_start = i * nlat_per_worker
                lat_end = (i + 1) * nlat_per_worker
                lon_start = j * nlon_per_worker
                lon_end = (j + 1) * nlon_per_worker

                worker = CellWorker2D.remote(
                    worker_id=i * n_workers_lon + j,
                    grid_info=grid_info,
                    lat_start=lat_start,
                    lat_end=lat_end,
                    lon_start=lon_start,
                    lon_end=lon_end,
                    kernel=kernel,
                    params={"growth_rate": 0.0},  # No growth
                )

                # Set initial biomass with different values per worker
                initial_biomass = jnp.ones((nlat_per_worker, nlon_per_worker)) * (10.0 + i + j)
                ray.get(worker.set_initial_state.remote({"biomass": initial_biomass}))

                workers.append(worker)

        # Create TransportWorker (plane grid for simplicity)
        transport_worker = TransportWorker.remote(
            grid_type="plane",
            nlat=global_nlat,
            nlon=global_nlon,
            dx=10e3,
            dy=10e3,
            lat_bc="closed",
            lon_bc="closed",
        )

        # Create scheduler WITH transport
        scheduler = EventScheduler(
            workers=workers,
            dt=100.0,
            t_max=100.0,
            transport_worker=transport_worker,
            transport_enabled=True,
            global_nlat=global_nlat,
            global_nlon=global_nlon,
            forcing_params={"horizontal_diffusivity": 100.0},  # Small diffusion
        )

        # Get initial mass from workers
        initial_states = ray.get([w.get_state.remote() for w in workers])
        initial_mass = sum(float(jnp.sum(s["biomass"])) for s in initial_states)

        # Collect global biomass
        biomass_global = scheduler._collect_global_biomass()
        collected_mass = float(jnp.sum(biomass_global))

        # Check collection conserves mass
        assert (
            abs(collected_mass - initial_mass) < 1e-6
        ), f"Collection lost mass: {initial_mass:.6f} -> {collected_mass:.6f}"

        # Redistribute
        scheduler._redistribute_biomass(biomass_global)

        # Get final mass
        final_states = ray.get([w.get_state.remote() for w in workers])
        final_mass = sum(float(jnp.sum(s["biomass"])) for s in final_states)

        # Check redistribution conserves mass
        assert (
            abs(final_mass - initial_mass) < 1e-6
        ), f"Redistribution lost mass: {initial_mass:.6f} -> {final_mass:.6f}"

    def test_coupled_biology_transport(self):
        """Test 3: Coupled biology + transport.

        Verify that biology and transport phases execute correctly together
        and that diagnostics include both components.
        """
        # Grid configuration (small for speed)
        global_nlat, global_nlon = 10, 10
        n_workers_lat, n_workers_lon = 2, 2
        nlat_per_worker = global_nlat // n_workers_lat
        nlon_per_worker = global_nlon // n_workers_lon

        grid_info = GridInfo(
            lat_min=-30.0,
            lat_max=30.0,
            lon_min=0.0,
            lon_max=360.0,
            nlat=global_nlat,
            nlon=global_nlon,
        )

        # Create kernel with growth
        kernel = Kernel([simple_growth])

        # Create workers
        workers = []
        for i in range(n_workers_lat):
            for j in range(n_workers_lon):
                lat_start = i * nlat_per_worker
                lat_end = (i + 1) * nlat_per_worker
                lon_start = j * nlon_per_worker
                lon_end = (j + 1) * nlon_per_worker

                worker = CellWorker2D.remote(
                    worker_id=i * n_workers_lon + j,
                    grid_info=grid_info,
                    lat_start=lat_start,
                    lat_end=lat_end,
                    lon_start=lon_start,
                    lon_end=lon_end,
                    kernel=kernel,
                    params={"growth_rate": 0.01},
                )

                # Initial biomass
                initial_biomass = jnp.ones((nlat_per_worker, nlon_per_worker)) * 5.0
                ray.get(worker.set_initial_state.remote({"biomass": initial_biomass}))

                workers.append(worker)

        # Create TransportWorker
        transport_worker = TransportWorker.remote(
            grid_type="plane",
            nlat=global_nlat,
            nlon=global_nlon,
            dx=10e3,
            dy=10e3,
            lat_bc="closed",
            lon_bc="closed",
        )

        # Create scheduler WITH transport
        scheduler = EventScheduler(
            workers=workers,
            dt=100.0,
            t_max=300.0,
            transport_worker=transport_worker,
            transport_enabled=True,
            global_nlat=global_nlat,
            global_nlon=global_nlon,
            forcing_params={"horizontal_diffusivity": 100.0},
        )

        # Run simulation (3 timesteps)
        diagnostics = scheduler.run()

        # Verify results
        assert len(diagnostics) == 3

        # Check that transport diagnostics are present
        for diag in diagnostics:
            assert "transport" in diag, "Transport diagnostics missing"
            assert "conservation_fraction" in diag["transport"]
            assert "mode" in diag["transport"]
            assert diag["transport"]["mode"] == "physics"

            # Check conservation
            conservation = diag["transport"]["conservation_fraction"]
            assert conservation > 0.99, f"Poor conservation: {conservation:.4f}"

        # Check that biomass evolved (either by growth or transport or both)
        assert diagnostics[-1]["biomass_global_mean"] != diagnostics[0]["biomass_global_mean"]
