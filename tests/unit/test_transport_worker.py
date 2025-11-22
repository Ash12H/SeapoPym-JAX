"""Unit tests for TransportWorker with physics implementation.

Tests verify:
- Worker initialization
- Transport step execution
- Mass conservation (short and long term)
- **CRITICAL**: 10-day conservation test (>99% target)
- Land masking
- Different grid types
- Diagnostics
"""

import jax.numpy as jnp
import pytest
import ray

from seapopym_message.transport.worker import TransportWorker


class TestTransportWorkerInitialization:
    """Tests for worker initialization."""

    def test_worker_spherical_grid_init(self):
        """Test initialization with spherical grid."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        worker = TransportWorker.remote(
            grid_type="spherical",
            lat_min=-60.0,
            lat_max=60.0,
            lon_min=0.0,
            lon_max=360.0,
            nlat=120,
            nlon=360,
            lat_bc="closed",
            lon_bc="periodic",
        )

        # Should initialize without errors
        assert worker is not None

    def test_worker_plane_grid_init(self):
        """Test initialization with plane grid."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        worker = TransportWorker.remote(
            grid_type="plane",
            nlat=100,
            nlon=100,
            dx=10e3,
            dy=10e3,
            lat_bc="closed",
            lon_bc="closed",
        )

        assert worker is not None

    def test_worker_invalid_grid_type(self):
        """Test that invalid grid type raises error."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        worker = TransportWorker.remote(grid_type="invalid")

        # Should raise error when trying to use (ValueError, RayTaskError, or ActorDiedError)
        with pytest.raises(
            (ValueError, ray.exceptions.RayTaskError, ray.exceptions.ActorDiedError)
        ):
            ray.get(
                worker.transport_step.remote(
                    biomass=jnp.ones((10, 10)),
                    u=jnp.zeros((10, 10)),
                    v=jnp.zeros((10, 10)),
                    D=1000.0,
                    dt=3600.0,
                )
            )


class TestTransportWorkerBasics:
    """Basic transport tests."""

    def test_worker_transport_step_executes(self):
        """Test that transport step executes and returns results."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        worker = TransportWorker.remote(
            grid_type="plane",
            nlat=10,
            nlon=10,
            dx=10e3,
            dy=10e3,
            lat_bc="closed",
            lon_bc="periodic",
        )

        biomass = jnp.ones((10, 10), dtype=jnp.float32) * 5.0
        u = jnp.ones((10, 10), dtype=jnp.float32) * 0.1
        v = jnp.zeros((10, 10), dtype=jnp.float32)

        result = ray.get(
            worker.transport_step.remote(
                biomass=biomass,
                u=u,
                v=v,
                D=1000.0,
                dt=3600.0,
            )
        )

        # Check result structure
        assert "biomass" in result
        assert "diagnostics" in result
        assert result["biomass"].shape == (10, 10)
        assert result["diagnostics"]["mode"] == "physics"

    def test_worker_zero_velocity_conservation(self):
        """Test conservation with zero velocity."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        worker = TransportWorker.remote(
            grid_type="plane",
            nlat=20,
            nlon=20,
            dx=10e3,
            dy=10e3,
            lat_bc="closed",
            lon_bc="closed",
        )

        biomass = jnp.ones((20, 20), dtype=jnp.float32) * 10.0
        u = jnp.zeros((20, 20), dtype=jnp.float32)
        v = jnp.zeros((20, 20), dtype=jnp.float32)

        result = ray.get(
            worker.transport_step.remote(
                biomass=biomass,
                u=u,
                v=v,
                D=500.0,  # Small diffusion
                dt=100.0,  # Small timestep
            )
        )

        # Should conserve mass nearly perfectly
        diag = result["diagnostics"]
        assert diag["conservation_fraction"] > 0.999


class TestTransportWorkerConservation:
    """Mass conservation tests."""

    def test_worker_short_term_conservation(self):
        """Test conservation over 1 hour (single step)."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        worker = TransportWorker.remote(
            grid_type="spherical",
            lat_min=-30.0,
            lat_max=30.0,
            lon_min=0.0,
            lon_max=360.0,
            nlat=60,
            nlon=120,
            lat_bc="closed",
            lon_bc="periodic",
        )

        # Non-uniform biomass
        biomass = jnp.ones((60, 120), dtype=jnp.float32) * 2.0
        biomass = biomass.at[30, 60].set(50.0)  # Spike

        # Moderate velocity
        u = jnp.ones((60, 120), dtype=jnp.float32) * 0.1
        v = jnp.zeros((60, 120), dtype=jnp.float32)

        result = ray.get(
            worker.transport_step.remote(
                biomass=biomass,
                u=u,
                v=v,
                D=1000.0,
                dt=3600.0,  # 1 hour
            )
        )

        diag = result["diagnostics"]

        # Should conserve mass well in single step
        assert diag["conservation_fraction"] > 0.99
        assert diag["stability_ok"] is True

    def test_worker_conservation_10_days(self):
        """CRITICAL TEST: Conservation over 10 days (240 timesteps).

        This is the primary validation criterion from the implementation plan.
        Target: >99% mass conservation over 240 timesteps at 1h each.
        """
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Realistic SEAPOPYM-like configuration
        worker = TransportWorker.remote(
            grid_type="spherical",
            lat_min=-60.0,
            lat_max=60.0,
            lon_min=0.0,
            lon_max=360.0,
            nlat=60,  # Coarser grid for speed (2° resolution)
            nlon=80,
            lat_bc="closed",
            lon_bc="periodic",
        )

        # Initial biomass field
        nlat, nlon = 60, 80
        biomass = jnp.ones((nlat, nlon), dtype=jnp.float32) * 5.0
        # Add some spatial structure
        biomass = biomass.at[25:35, 30:50].set(20.0)  # High biomass region

        # Realistic velocities (0.1 m/s ~ 10 km/day)
        u = jnp.ones((nlat, nlon), dtype=jnp.float32) * 0.05
        v = jnp.ones((nlat, nlon), dtype=jnp.float32) * 0.02

        # Horizontal diffusion coefficient
        D = 1000.0  # m²/s, typical for mesoscale

        # Time parameters
        dt = 3600.0  # 1 hour timesteps
        n_steps = 240  # 10 days

        # Track mass over time
        masses = []
        current_biomass = biomass

        for step in range(n_steps):
            result = ray.get(
                worker.transport_step.remote(
                    biomass=current_biomass,
                    u=u,
                    v=v,
                    D=D,
                    dt=dt,
                )
            )

            current_biomass = result["biomass"]
            masses.append(result["diagnostics"]["mass_after"])

            # Check stability
            assert result["diagnostics"]["stability_ok"], f"Unstable at step {step}"

        # Final conservation check
        mass_initial = masses[0]
        mass_final = masses[-1]
        conservation = mass_final / mass_initial

        # PRIMARY SUCCESS CRITERION
        assert conservation > 0.99, (
            f"10-day conservation FAILED: {conservation:.4f} < 0.99. "
            f"Initial={mass_initial:.2e}, Final={mass_final:.2e}"
        )

        # Additional diagnostics
        mass_drift = abs(conservation - 1.0)
        assert mass_drift < 0.01, f"Mass drift {mass_drift:.4f} exceeds 1%"

        print(f"10-day conservation: {conservation:.6f} ({conservation * 100:.4f}%)")


class TestTransportWorkerMasking:
    """Tests for land masking."""

    def test_worker_with_land_mask(self):
        """Test that land cells remain zero."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        worker = TransportWorker.remote(
            grid_type="plane",
            nlat=20,
            nlon=20,
            dx=10e3,
            dy=10e3,
            lat_bc="closed",
            lon_bc="closed",
        )

        # Biomass with ocean values
        biomass = jnp.ones((20, 20), dtype=jnp.float32) * 10.0

        # Mask: land in center
        mask = jnp.ones((20, 20), dtype=jnp.float32)
        mask = mask.at[8:12, 8:12].set(0.0)  # Land square

        u = jnp.ones((20, 20), dtype=jnp.float32) * 0.5
        v = jnp.zeros((20, 20), dtype=jnp.float32)

        result = ray.get(
            worker.transport_step.remote(
                biomass=biomass,
                u=u,
                v=v,
                D=1000.0,
                dt=3600.0,
                mask=mask,
            )
        )

        biomass_final = result["biomass"]

        # Land cells should be zero
        assert jnp.all(biomass_final[8:12, 8:12] == 0.0)

        # Ocean cells should have non-zero values
        assert jnp.any(biomass_final[0:5, 0:5] > 0)


class TestTransportWorkerGridTypes:
    """Tests for different grid configurations."""

    def test_worker_spherical_vs_plane_behavior(self):
        """Test that spherical and plane grids handle geometry correctly."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Test with spherical grid at different latitudes
        nlat, nlon = 30, 40

        worker_spherical = TransportWorker.remote(
            grid_type="spherical",
            lat_min=-60.0,
            lat_max=60.0,
            lon_min=0.0,
            lon_max=360.0,
            nlat=nlat,
            nlon=nlon,
            lat_bc="closed",
            lon_bc="periodic",
        )

        # Non-uniform biomass with gradient in latitude
        biomass = jnp.ones((nlat, nlon), dtype=jnp.float32) * 5.0
        # Add latitude-dependent structure
        for i in range(nlat):
            biomass = biomass.at[i, :].set(5.0 + i * 0.5)

        u = jnp.ones((nlat, nlon), dtype=jnp.float32) * 0.1
        v = jnp.zeros((nlat, nlon), dtype=jnp.float32)

        result = ray.get(
            worker_spherical.transport_step.remote(biomass=biomass, u=u, v=v, D=1000.0, dt=3600.0)
        )

        # Should execute successfully and conserve mass
        assert result["diagnostics"]["conservation_fraction"] > 0.95
        assert result["diagnostics"]["grid_type"] == "spherical"


class TestTransportWorkerDiagnostics:
    """Tests for diagnostic outputs."""

    def test_worker_diagnostics_complete(self):
        """Test that all expected diagnostics are present."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        worker = TransportWorker.remote(
            grid_type="plane",
            nlat=10,
            nlon=10,
            dx=10e3,
            dy=10e3,
        )

        biomass = jnp.ones((10, 10), dtype=jnp.float32) * 5.0
        u = jnp.ones((10, 10), dtype=jnp.float32) * 0.1
        v = jnp.zeros((10, 10), dtype=jnp.float32)

        result = ray.get(
            worker.transport_step.remote(biomass=biomass, u=u, v=v, D=1000.0, dt=1000.0)
        )

        diag = result["diagnostics"]

        # Check all expected fields
        expected_fields = [
            "mass_before",
            "mass_after",
            "mass_error_total",
            "conservation_fraction",
            "mass_error_advection",
            "conservation_advection",
            "max_velocity",
            "cfl_advection",
            "cfl_diffusion",
            "stability_ok",
            "dt_max_diffusion",
            "compute_time_s",
            "compute_time_advection_s",
            "compute_time_diffusion_s",
            "mode",
            "grid_type",
        ]

        for field in expected_fields:
            assert field in diag, f"Missing diagnostic field: {field}"

        # Check values are reasonable
        assert diag["mode"] == "physics"
        assert diag["compute_time_s"] > 0
        assert diag["conservation_fraction"] > 0
