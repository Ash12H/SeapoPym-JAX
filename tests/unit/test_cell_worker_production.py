"""Unit tests for CellWorker2D production field management (Phase 3)."""

import jax.numpy as jnp
import pytest
import ray

from seapopym_message.core.kernel import Kernel
from seapopym_message.distributed.worker import CellWorker2D
from seapopym_message.utils.grid import SphericalGridInfo


@pytest.mark.unit
class TestCellWorkerProduction:
    """Test CellWorker2D production get/set methods."""

    def test_worker_get_field_production_initial(self):
        """get_field('production') should return empty array if not initialized."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Create minimal kernel and worker
        grid = SphericalGridInfo(lat_min=-60, lat_max=60, lon_min=0, lon_max=360, nlat=20, nlon=40)
        kernel = Kernel([])  # Empty kernel
        params = {"age": 11}

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

        # Get production before setting anything
        production = ray.get(worker.get_field.remote("production"))

        # Should return empty array since field doesn't exist yet
        assert production.size == 0

    def test_worker_set_get_field_production(self):
        """set_field then get_field should return the same thing."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Create worker
        grid = SphericalGridInfo(lat_min=-60, lat_max=60, lon_min=0, lon_max=360, nlat=20, nlon=40)
        kernel = Kernel([])
        params = {"age": 11}

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

        # Create production field with non-zero values
        n_ages = 11
        nlat = 10
        nlon = 20
        production_test = jnp.zeros((n_ages, nlat, nlon))
        production_test = production_test.at[3].set(jnp.ones((nlat, nlon)) * 2.5)
        production_test = production_test.at[7].set(jnp.ones((nlat, nlon)) * 1.8)

        # Set production using generic method
        ray.get(worker.set_field.remote("production", production_test))

        # Get production back using generic method
        production_retrieved = ray.get(worker.get_field.remote("production"))

        # Should match
        assert production_retrieved.shape == production_test.shape
        assert jnp.allclose(production_retrieved, production_test)

        # Verify specific age classes
        assert jnp.allclose(production_retrieved[3], 2.5)
        assert jnp.allclose(production_retrieved[7], 1.8)
        assert jnp.allclose(production_retrieved[0], 0.0)

    def test_worker_field_different_shapes(self):
        """Test with different field shapes."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Create worker
        grid = SphericalGridInfo(lat_min=-60, lat_max=60, lon_min=0, lon_max=360, nlat=20, nlon=40)
        kernel = Kernel([])
        params = {}

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

        # Test setting/getting field with n_ages=15
        n_ages = 15
        production_test = jnp.ones((n_ages, 10, 20)) * 3.0
        ray.get(worker.set_field.remote("production", production_test))

        production_retrieved = ray.get(worker.get_field.remote("production"))
        assert production_retrieved.shape == (15, 10, 20)
        assert jnp.allclose(production_retrieved, 3.0)

        # Test with different field (biomass) - 2D
        biomass_test = jnp.ones((10, 20)) * 5.0
        ray.get(worker.set_field.remote("biomass", biomass_test))

        biomass_retrieved = ray.get(worker.get_field.remote("biomass"))
        assert biomass_retrieved.shape == (10, 20)
        assert jnp.allclose(biomass_retrieved, 5.0)
