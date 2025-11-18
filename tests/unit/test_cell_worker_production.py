"""Unit tests for CellWorker2D production field management (Phase 3)."""

import jax.numpy as jnp
import pytest
import ray

from seapopym_message.core.kernel import Kernel
from seapopym_message.distributed.worker import CellWorker2D
from seapopym_message.utils.grid import GridInfo


@pytest.mark.unit
class TestCellWorkerProduction:
    """Test CellWorker2D production get/set methods."""

    def test_worker_get_production_initial(self):
        """get_production should return zeros if not initialized."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Create minimal kernel and worker
        grid = GridInfo(lat_min=-60, lat_max=60, lon_min=0, lon_max=360, nlat=20, nlon=40)
        kernel = Kernel([])  # Empty kernel
        params = {"n_ages": 11}

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
        production = ray.get(worker.get_production.remote())

        # Should return zeros with shape (n_ages, nlat, nlon) = (11, 10, 20)
        assert production.shape == (11, 10, 20)
        assert jnp.allclose(production, 0.0)

    def test_worker_set_get_production(self):
        """set_production then get_production should return the same thing."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Create worker
        grid = GridInfo(lat_min=-60, lat_max=60, lon_min=0, lon_max=360, nlat=20, nlon=40)
        kernel = Kernel([])
        params = {"n_ages": 11}

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

        # Set production
        ray.get(worker.set_production.remote(production_test))

        # Get production back
        production_retrieved = ray.get(worker.get_production.remote())

        # Should match
        assert production_retrieved.shape == production_test.shape
        assert jnp.allclose(production_retrieved, production_test)

        # Verify specific age classes
        assert jnp.allclose(production_retrieved[3], 2.5)
        assert jnp.allclose(production_retrieved[7], 1.8)
        assert jnp.allclose(production_retrieved[0], 0.0)

    def test_worker_get_production_default_n_ages(self):
        """get_production should use n_ages=11 by default if not in params."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Create worker WITHOUT n_ages in params
        grid = GridInfo(lat_min=-60, lat_max=60, lon_min=0, lon_max=360, nlat=20, nlon=40)
        kernel = Kernel([])
        params = {}  # No n_ages specified

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

        production = ray.get(worker.get_production.remote())

        # Should default to n_ages=11
        assert production.shape == (11, 10, 20)

    def test_worker_production_different_n_ages(self):
        """Test with different n_ages parameter."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Create worker with n_ages=15
        grid = GridInfo(lat_min=-60, lat_max=60, lon_min=0, lon_max=360, nlat=20, nlon=40)
        kernel = Kernel([])
        params = {"n_ages": 15}

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

        # Get initial production
        production = ray.get(worker.get_production.remote())

        # Should have 15 age classes
        assert production.shape == (15, 10, 20)

        # Set production with 15 age classes
        production_test = jnp.ones((15, 10, 20)) * 3.0
        ray.get(worker.set_production.remote(production_test))

        production_retrieved = ray.get(worker.get_production.remote())
        assert production_retrieved.shape == (15, 10, 20)
        assert jnp.allclose(production_retrieved, 3.0)
