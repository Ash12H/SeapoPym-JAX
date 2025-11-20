"""Unit tests for EventScheduler generic field collection/redistribution."""

import jax.numpy as jnp
import pytest
import ray

from seapopym_message.core.kernel import Kernel
from seapopym_message.distributed.scheduler import EventScheduler
from seapopym_message.distributed.transport_config import FieldConfig, TransportConfig
from seapopym_message.distributed.worker import CellWorker2D
from seapopym_message.transport.worker import TransportWorker
from seapopym_message.utils.grid import SphericalGridInfo


@pytest.mark.unit
class TestSchedulerFieldCollection:
    """Test EventScheduler generic field collection/redistribution methods."""

    def test_collect_redistribute_field_conservation(self):
        """Collect then redistribute should conserve total field mass."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Setup: 2x2 workers on 20x20 grid
        nlat_global, nlon_global = 20, 20
        n_ages = 11

        # Create workers
        grid = SphericalGridInfo(
            lat_min=-60, lat_max=60, lon_min=0, lon_max=360, nlat=nlat_global, nlon=nlon_global
        )
        kernel = Kernel([])  # Empty kernel for this test
        params = {"age": n_ages}

        workers = []
        for i in range(2):
            for j in range(2):
                lat_start = i * 10
                lat_end = (i + 1) * 10
                lon_start = j * 10
                lon_end = (j + 1) * 10

                worker = CellWorker2D.remote(
                    worker_id=i * 2 + j,
                    grid_info=grid,
                    lat_start=lat_start,
                    lat_end=lat_end,
                    lon_start=lon_start,
                    lon_end=lon_end,
                    kernel=kernel,
                    params=params,
                )
                workers.append(worker)

        # Create transport worker (plane grid for simplicity)
        transport_worker = TransportWorker.remote(
            grid_type="plane",
            nlat=nlat_global,
            nlon=nlon_global,
            dx=10e3,
            dy=10e3,
            lat_bc="closed",
            lon_bc="closed",
        )

        # Create transport config for production field
        transport_config = TransportConfig(
            fields=[FieldConfig(name="production", dims=["age", "Y", "X"])]
        )

        # Create scheduler
        scheduler = EventScheduler(
            workers=workers,
            dt=3600.0,
            t_max=3600.0,
            forcing_params={"age": n_ages},
            transport_worker=transport_worker,
            transport_config=transport_config,
            global_nlat=nlat_global,
            global_nlon=nlon_global,
        )

        # Set initial production in each worker (different values for each patch)
        initial_mass_total = 0.0
        for idx, worker in enumerate(workers):
            production_patch = jnp.zeros((n_ages, 10, 10))
            # Age class 5 has biomass proportional to worker index
            production_patch = production_patch.at[5].set(jnp.ones((10, 10)) * (idx + 1) * 2.0)
            ray.get(worker.set_field.remote("production", production_patch))

            # Calculate initial mass for this patch
            initial_mass_total += float(jnp.sum(production_patch))

        # Collect global production using generic method
        production_global = scheduler._collect_global_field(
            "production", (n_ages, nlat_global, nlon_global)
        )

        # Check global production shape
        assert production_global.shape == (n_ages, nlat_global, nlon_global)

        # Check total mass conservation after collection
        collected_mass = float(jnp.sum(production_global))
        assert jnp.isclose(collected_mass, initial_mass_total, rtol=1e-5)

        # Redistribute production back to workers using generic method
        scheduler._redistribute_field("production", production_global)

        # Collect again and verify conservation
        production_global_2 = scheduler._collect_global_field(
            "production", (n_ages, nlat_global, nlon_global)
        )
        redistributed_mass = float(jnp.sum(production_global_2))

        assert jnp.isclose(redistributed_mass, initial_mass_total, rtol=1e-5)
        assert jnp.allclose(production_global_2, production_global, rtol=1e-5)

    def test_collect_field_error_handling(self):
        """Test that collecting non-existent field raises error."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Setup: 2 workers
        nlat_global, nlon_global = 10, 10

        grid = SphericalGridInfo(lat_min=-60, lat_max=60, lon_min=0, lon_max=360, nlat=10, nlon=10)
        kernel = Kernel([])
        params = {}

        workers = [
            CellWorker2D.remote(
                worker_id=0,
                grid_info=grid,
                lat_start=0,
                lat_end=5,
                lon_start=0,
                lon_end=10,
                kernel=kernel,
                params=params,
            ),
            CellWorker2D.remote(
                worker_id=1,
                grid_info=grid,
                lat_start=5,
                lat_end=10,
                lon_start=0,
                lon_end=10,
                kernel=kernel,
                params=params,
            ),
        ]

        transport_worker = TransportWorker.remote(
            grid_type="plane", nlat=10, nlon=10, dx=10e3, dy=10e3, lat_bc="closed", lon_bc="closed"
        )

        transport_config = TransportConfig(
            fields=[FieldConfig(name="temperature", dims=["Y", "X"])]
        )

        scheduler = EventScheduler(
            workers=workers,
            dt=3600.0,
            t_max=3600.0,
            transport_worker=transport_worker,
            transport_config=transport_config,
            global_nlat=nlat_global,
            global_nlon=nlon_global,
        )

        # Try to collect field that doesn't exist in workers
        # Should raise ValueError because workers return empty arrays
        with pytest.raises(ValueError, match="returned empty patch"):
            scheduler._collect_global_field("temperature", (nlat_global, nlon_global))
