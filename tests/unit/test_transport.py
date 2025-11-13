"""Tests for transport Units."""

import jax.numpy as jnp
import pytest

from seapopym_message.core.kernel import Kernel
from seapopym_message.kernels.transport import compute_diffusion_2d, compute_diffusion_simple


@pytest.mark.unit
class TestDiffusionUnit:
    """Test diffusion Unit."""

    def test_diffusion_simple_no_halo(self) -> None:
        """Test simple diffusion without halo (Neumann BC)."""
        # Create a peak in the center
        biomass = jnp.array([[10.0, 10.0, 10.0], [10.0, 100.0, 10.0], [10.0, 10.0, 10.0]])

        params = {"D": 1.0, "dx": 1.0}
        dt = 0.01

        state = {"biomass": biomass}
        result = compute_diffusion_simple.execute(state, dt=dt, params=params)

        # Peak should diffuse outward
        assert "biomass" in result
        # Center should decrease
        assert result["biomass"][1, 1] < biomass[1, 1]
        # Neighbors should increase
        assert result["biomass"][0, 1] > biomass[0, 1]
        assert result["biomass"][2, 1] > biomass[2, 1]
        assert result["biomass"][1, 0] > biomass[1, 0]
        assert result["biomass"][1, 2] > biomass[1, 2]

    def test_diffusion_mass_conservation(self) -> None:
        """Test that diffusion conserves total mass."""
        biomass = jnp.array([[10.0, 50.0, 10.0], [20.0, 30.0, 40.0], [15.0, 25.0, 35.0]])

        params = {"D": 10.0, "dx": 1000.0}
        dt = 10.0

        state = {"biomass": biomass}
        initial_mass = jnp.sum(biomass)

        result = compute_diffusion_simple.execute(state, dt=dt, params=params)

        final_mass = jnp.sum(result["biomass"])

        # Mass should be conserved (within numerical precision)
        assert jnp.allclose(final_mass, initial_mass, rtol=1e-5)

    def test_diffusion_with_halos(self) -> None:
        """Test diffusion with halo exchange."""
        biomass = jnp.array([[10.0, 10.0], [10.0, 10.0]])

        # Neighbors have higher values
        halo_north = {"biomass": jnp.array([50.0, 50.0])}
        halo_south = {"biomass": jnp.array([50.0, 50.0])}
        halo_east = {"biomass": jnp.array([50.0, 50.0])}
        halo_west = {"biomass": jnp.array([50.0, 50.0])}

        params = {"D": 1.0, "dx": 1.0}
        dt = 0.01

        state = {"biomass": biomass}
        result = compute_diffusion_2d.execute(
            state,
            dt=dt,
            params=params,
            halo_north=halo_north,
            halo_south=halo_south,
            halo_east=halo_east,
            halo_west=halo_west,
        )

        # All cells should increase (diffusion from high-value neighbors)
        assert jnp.all(result["biomass"] > biomass)

    def test_diffusion_uniform_field(self) -> None:
        """Test that uniform field remains uniform."""
        biomass = jnp.full((5, 5), 42.0)

        params = {"D": 100.0, "dx": 1000.0}
        dt = 100.0

        state = {"biomass": biomass}
        result = compute_diffusion_simple.execute(state, dt=dt, params=params)

        # Laplacian of uniform field is zero, so no change
        assert jnp.allclose(result["biomass"], biomass, atol=1e-5)

    def test_diffusion_anisotropic(self) -> None:
        """Test diffusion with different dx and dy."""
        biomass = jnp.array([[10.0, 10.0, 10.0], [10.0, 100.0, 10.0], [10.0, 10.0, 10.0]])

        params_isotropic = {"D": 1.0, "dx": 1.0, "dy": 1.0}
        params_anisotropic = {"D": 1.0, "dx": 1.0, "dy": 2.0}  # Larger dy

        dt = 0.01
        state1 = {"biomass": biomass.copy()}
        state2 = {"biomass": biomass.copy()}

        result1 = compute_diffusion_simple.execute(state1, dt=dt, params=params_isotropic)
        result2 = compute_diffusion_simple.execute(state2, dt=dt, params=params_anisotropic)

        # Results should differ
        assert not jnp.allclose(result1["biomass"], result2["biomass"])


@pytest.mark.unit
class TestDiffusionStability:
    """Test numerical stability of diffusion."""

    def test_cfl_condition_warning(self) -> None:
        """Test that violating CFL condition can cause instability.

        CFL condition for diffusion: D * dt / dx² ≤ 0.5 (in 2D)
        """
        biomass = jnp.array([[10.0, 50.0, 10.0], [10.0, 50.0, 10.0], [10.0, 50.0, 10.0]])

        # Stable: D * dt / dx² = 1 * 0.1 / 1² = 0.1 < 0.5
        params_stable = {"D": 1.0, "dx": 1.0}
        dt_stable = 0.1

        state_stable = {"biomass": biomass.copy()}
        result_stable = compute_diffusion_simple.execute(
            state_stable, dt=dt_stable, params=params_stable
        )

        # Should not have negative values or oscillations
        assert jnp.all(result_stable["biomass"] >= 0.0)

    def test_diffusion_positivity(self) -> None:
        """Test that diffusion preserves non-negativity."""
        biomass = jnp.array([[0.0, 10.0, 0.0], [0.0, 10.0, 0.0], [0.0, 10.0, 0.0]])

        params = {"D": 1.0, "dx": 1.0}
        dt = 0.01

        state = {"biomass": biomass}
        result = compute_diffusion_simple.execute(state, dt=dt, params=params)

        # All values should remain non-negative
        assert jnp.all(result["biomass"] >= 0.0)


@pytest.mark.unit
class TestDiffusionIntegration:
    """Integration tests with Kernel."""

    def test_diffusion_in_kernel(self) -> None:
        """Test diffusion as part of a Kernel."""
        kernel = Kernel([compute_diffusion_simple])

        biomass = jnp.array([[10.0, 50.0, 10.0], [10.0, 50.0, 10.0], [10.0, 50.0, 10.0]])

        state = {"biomass": biomass}
        params = {"D": 10.0, "dx": 1000.0}
        dt = 100.0

        # Execute via kernel (local phase since we're using simple version)
        state = kernel.execute_local_phase(state, dt=dt, params=params)

        # Peak should diffuse
        assert state["biomass"][0, 1] < biomass[0, 1]

    def test_diffusion_global_kernel(self) -> None:
        """Test diffusion_2d requires global phase."""
        kernel = Kernel([compute_diffusion_2d])

        # Verify it's recognized as global
        assert len(kernel.global_units) == 1
        assert len(kernel.local_units) == 0
        assert kernel.has_global_units() is True

        biomass = jnp.array([[10.0, 10.0], [10.0, 10.0]])
        state = {"biomass": biomass}
        params = {"D": 1.0, "dx": 1.0}
        dt = 0.01

        # Execute global phase (no halos, so Neumann BC)
        state = kernel.execute_global_phase(state, dt=dt, params=params, neighbor_data=None)

        assert state["biomass"].shape == (2, 2)


@pytest.mark.unit
class TestDiffusionPhysics:
    """Test physical correctness of diffusion."""

    def test_gaussian_spreading(self) -> None:
        """Test that a Gaussian peak spreads over time."""
        # Create a peak
        biomass = jnp.zeros((7, 7))
        biomass = biomass.at[3, 3].set(100.0)

        params = {"D": 1.0, "dx": 1.0}
        dt = 0.01

        state = {"biomass": biomass}

        # Run multiple steps
        for _ in range(10):
            state = compute_diffusion_simple.execute(state, dt=dt, params=params)

        # Peak should decrease
        assert state["biomass"][3, 3] < 100.0
        # Should spread in all directions
        assert state["biomass"][2, 3] > 0.0  # north
        assert state["biomass"][4, 3] > 0.0  # south
        assert state["biomass"][3, 2] > 0.0  # west
        assert state["biomass"][3, 4] > 0.0  # east
        # Total mass approximately conserved
        assert jnp.allclose(jnp.sum(state["biomass"]), 100.0, rtol=0.01)

    def test_diffusion_timescale(self) -> None:
        """Test characteristic diffusion timescale.

        Diffusion timescale: τ = L² / D
        For L=1m, D=1m²/s: τ = 1s
        """
        # Two adjacent cells with different values
        biomass = jnp.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        params = {"D": 1.0, "dx": 1.0}
        dt = 0.01
        t_total = 1.0  # Run for 1 second
        n_steps = int(t_total / dt)

        state = {"biomass": biomass}

        for _ in range(n_steps):
            state = compute_diffusion_simple.execute(state, dt=dt, params=params)

        # After one timescale, concentration difference should be significantly reduced
        # (exact solution involves error functions, here we just check qualitative behavior)
        assert state["biomass"][1, 0] < 50.0  # Source cell decreased
        assert state["biomass"][1, 1] > 10.0  # Neighbor increased significantly
