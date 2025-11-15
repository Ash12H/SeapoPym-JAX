"""Tests for TransportWorker in passthrough mode.

This validates the Ray communication infrastructure before implementing
the actual transport physics.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import ray

from seapopym_message.transport import TransportWorker


@pytest.fixture(scope="module")
def ray_context():
    """Initialize Ray for testing."""
    if not ray.is_initialized():
        ray.init(num_cpus=2, runtime_env=None)
    yield
    ray.shutdown()


class TestTransportWorkerPassthrough:
    """Test TransportWorker in passthrough mode."""

    def test_initialization(self, ray_context):
        """Test that worker initializes correctly."""
        worker = TransportWorker.remote()
        # Simple test: worker exists
        assert worker is not None

    def test_passthrough_conservation(self, ray_context):
        """Test that passthrough mode returns biomass unchanged."""
        # Create test data
        nlat, nlon = 20, 30
        biomass = jnp.ones((nlat, nlon)) * 100.0
        u = jnp.ones((nlat, nlon)) * 1.0
        v = jnp.zeros((nlat, nlon))
        D = 100.0
        dt = 900.0
        dx = dy = 20000.0

        worker = TransportWorker.remote()

        result = ray.get(
            worker.transport_step.remote(
                biomass=biomass,
                u=u,
                v=v,
                D=D,
                dt=dt,
                dx=dx,
                dy=dy,
                mask=None,
            )
        )

        # Biomass should be unchanged
        assert jnp.allclose(result["biomass"], biomass)

        # Diagnostics should show perfect conservation
        assert result["diagnostics"]["mass_error_total"] == 0.0
        assert result["diagnostics"]["mode"] == "passthrough"

    def test_passthrough_with_mask(self, ray_context):
        """Test passthrough with ocean mask."""
        nlat, nlon = 30, 40

        # Create mask with island
        mask = jnp.ones((nlat, nlon), dtype=bool)
        mask = mask.at[10:20, 15:25].set(False)

        # Biomass only in ocean
        biomass = jnp.where(mask, 50.0, 0.0)

        u = jnp.ones((nlat, nlon)) * 0.5
        v = jnp.zeros((nlat, nlon))

        worker = TransportWorker.remote()

        result = ray.get(
            worker.transport_step.remote(
                biomass=biomass,
                u=u,
                v=v,
                D=0.0,
                dt=1000.0,
                dx=10000.0,
                dy=10000.0,
                mask=mask,
            )
        )

        # Biomass should be unchanged
        assert jnp.allclose(result["biomass"], biomass)

    def test_passthrough_multiple_steps(self, ray_context):
        """Test multiple sequential calls (infrastructure stress test)."""
        nlat, nlon = 15, 25
        biomass = jnp.array(np.random.rand(nlat, nlon) * 100.0)
        mass_initial = float(jnp.sum(biomass))

        u = jnp.ones((nlat, nlon)) * 2.0
        v = jnp.ones((nlat, nlon)) * 1.0
        D = 50.0

        worker = TransportWorker.remote()

        # Run 100 steps
        for _ in range(100):
            result = ray.get(
                worker.transport_step.remote(
                    biomass=biomass,
                    u=u,
                    v=v,
                    D=D,
                    dt=600.0,
                    dx=15000.0,
                    dy=15000.0,
                    mask=None,
                )
            )

            biomass = result["biomass"]

            # Each step should preserve mass
            assert result["diagnostics"]["mass_error_total"] == 0.0

        # After 100 steps, biomass should still be unchanged
        mass_final = float(jnp.sum(biomass))
        assert abs(mass_final - mass_initial) < 1e-6

    def test_ray_communication(self, ray_context):
        """Test Ray actor communication with large arrays."""
        # Large grid to test Ray serialization
        nlat, nlon = 100, 200
        biomass = jnp.ones((nlat, nlon)) * 1000.0

        u = jnp.zeros((nlat, nlon))
        v = jnp.zeros((nlat, nlon))

        worker = TransportWorker.remote()

        result = ray.get(
            worker.transport_step.remote(
                biomass=biomass,
                u=u,
                v=v,
                D=0.0,
                dt=1.0,
                dx=1000.0,
                dy=1000.0,
                mask=None,
            )
        )

        # Verify data integrity after Ray serialization
        assert result["biomass"].shape == (nlat, nlon)
        assert jnp.allclose(result["biomass"], biomass)
        assert result["diagnostics"]["mass_before"] == 1000.0 * nlat * nlon
        assert result["diagnostics"]["mass_after"] == 1000.0 * nlat * nlon
