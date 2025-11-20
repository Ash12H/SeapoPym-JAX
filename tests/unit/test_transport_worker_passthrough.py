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
        worker = TransportWorker.remote(
            grid_type="plane",
            nlat=10,
            nlon=10,
            dx=10000.0,
            dy=10000.0,
        )
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

        worker = TransportWorker.remote(
            grid_type="plane",
            nlat=nlat,
            nlon=nlon,
            dx=dx,
            dy=dy,
            lat_bc="closed",
            lon_bc="closed",
        )

        result = ray.get(
            worker.transport_step.remote(
                biomass=biomass,
                u=u,
                v=v,
                D=D,
                dt=dt,
                _dx=dx,
                _dy=dy,
                mask=None,
            )
        )

        # Mass should be conserved (biomass may change due to transport)
        mass_before = result["diagnostics"]["mass_before"]
        mass_error = result["diagnostics"]["mass_error_total"]

        # Conservation check: relative error < 1%
        assert abs(mass_error) < 0.01 * mass_before
        assert result["diagnostics"]["mode"] == "physics"

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

        worker = TransportWorker.remote(
            grid_type="plane",
            nlat=nlat,
            nlon=nlon,
            dx=10000.0,
            dy=10000.0,
            lat_bc="closed",
            lon_bc="closed",
        )

        result = ray.get(
            worker.transport_step.remote(
                biomass=biomass,
                u=u,
                v=v,
                D=0.0,
                dt=1000.0,
                _dx=10000.0,
                _dy=10000.0,
                mask=mask,
            )
        )

        # Mass should be conserved
        mass_before = result["diagnostics"]["mass_before"]
        mass_error = result["diagnostics"]["mass_error_total"]

        # Conservation check: relative error < 1%
        assert abs(mass_error) < 0.01 * mass_before

    def test_passthrough_multiple_steps(self, ray_context):
        """Test multiple sequential calls (infrastructure stress test)."""
        nlat, nlon = 15, 25
        biomass = jnp.array(np.random.rand(nlat, nlon) * 100.0)
        mass_initial = float(jnp.sum(biomass))

        u = jnp.ones((nlat, nlon)) * 2.0
        v = jnp.ones((nlat, nlon)) * 1.0
        D = 50.0

        worker = TransportWorker.remote(
            grid_type="plane",
            nlat=nlat,
            nlon=nlon,
            dx=15000.0,
            dy=15000.0,
            lat_bc="closed",
            lon_bc="closed",
        )

        # Run 100 steps
        for _ in range(100):
            result = ray.get(
                worker.transport_step.remote(
                    biomass=biomass,
                    u=u,
                    v=v,
                    D=D,
                    dt=600.0,
                    _dx=15000.0,
                    _dy=15000.0,
                    mask=None,
                )
            )

            biomass = result["biomass"]

            # Each step should preserve mass (allow 1% error)
            mass_before = result["diagnostics"]["mass_before"]
            mass_error = result["diagnostics"]["mass_error_total"]
            assert abs(mass_error) < 0.01 * mass_before

        # After 100 steps, mass should still be conserved (cumulative error < 5%)
        mass_final = float(jnp.sum(biomass))
        assert abs(mass_final - mass_initial) / mass_initial < 0.05

    def test_ray_communication(self, ray_context):
        """Test Ray actor communication with large arrays."""
        # Large grid to test Ray serialization
        nlat, nlon = 100, 200
        biomass = jnp.ones((nlat, nlon)) * 1000.0

        u = jnp.zeros((nlat, nlon))
        v = jnp.zeros((nlat, nlon))

        worker = TransportWorker.remote(
            grid_type="plane",
            nlat=nlat,
            nlon=nlon,
            dx=1000.0,
            dy=1000.0,
            lat_bc="closed",
            lon_bc="closed",
        )

        result = ray.get(
            worker.transport_step.remote(
                biomass=biomass,
                u=u,
                v=v,
                D=0.0,
                dt=1.0,
                _dx=1000.0,
                _dy=1000.0,
                mask=None,
            )
        )

        # Verify data integrity after Ray serialization
        assert result["biomass"].shape == (nlat, nlon)

        # Mass conservation check
        # Note: We check conservation (mass_error), not absolute mass value
        # because floating-point arithmetic introduces small rounding errors
        # when computing cell areas and summing over 20,000 cells
        mass_before = result["diagnostics"]["mass_before"]
        mass_error = result["diagnostics"]["mass_error_total"]

        # Verify mass is in expected range (allow 0.01% error for float arithmetic)
        expected_mass = 1000.0 * nlat * nlon * 1000.0 * 1000.0
        assert abs(mass_before - expected_mass) / expected_mass < 1e-4

        # Main check: mass conservation (< 1% error)
        assert abs(mass_error) < 0.01 * mass_before
