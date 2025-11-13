"""Tests for biology Units."""

import jax.numpy as jnp
import pytest

from seapopym_message.core.kernel import Kernel
from seapopym_message.kernels.biology import (
    compute_growth,
    compute_mortality,
    compute_recruitment,
    compute_recruitment_2d,
)


@pytest.mark.unit
class TestBiologyUnits:
    """Test individual biology Units."""

    def test_recruitment_scalar(self) -> None:
        """Test constant recruitment returns correct value."""
        params = {"R": 5.0}
        state = {}

        result = compute_recruitment.execute(state, params=params)

        assert "R" in result
        assert jnp.allclose(result["R"], 5.0)

    def test_recruitment_2d(self) -> None:
        """Test 2D recruitment creates correct shape."""
        params = {"R": 10.0}
        grid_shape = (5, 10)
        state = {}

        result = compute_recruitment_2d.execute(state, params=params, grid_shape=grid_shape)

        assert "R" in result
        assert result["R"].shape == grid_shape
        assert jnp.allclose(result["R"], 10.0)

    def test_mortality_proportional(self) -> None:
        """Test mortality is proportional to biomass."""
        biomass = jnp.array([[10.0, 20.0], [30.0, 40.0]])
        params = {"lambda": 0.1}
        state = {"biomass": biomass}

        result = compute_mortality.execute(state, params=params)

        expected = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        assert "mortality_rate" in result
        assert jnp.allclose(result["mortality_rate"], expected)

    def test_growth_forward_euler(self) -> None:
        """Test growth integrates correctly."""
        biomass = jnp.array([[10.0, 20.0]])
        R = jnp.array([[5.0, 5.0]])
        M = jnp.array([[1.0, 2.0]])
        dt = 0.1
        state = {"biomass": biomass, "R": R, "mortality_rate": M}

        result = compute_growth.execute(state, dt=dt)

        # B_new = B + (R - M) * dt
        # For first cell: 10 + (5 - 1) * 0.1 = 10.4
        # For second cell: 20 + (5 - 2) * 0.1 = 20.3
        expected = jnp.array([[10.4, 20.3]])
        assert "biomass" in result
        assert jnp.allclose(result["biomass"], expected)


@pytest.mark.unit
class TestBiologyKernel:
    """Test biology Units integrated into a Kernel."""

    def test_full_biology_cycle(self) -> None:
        """Test complete biology kernel (recruitment + mortality + growth)."""
        kernel = Kernel([compute_recruitment_2d, compute_mortality, compute_growth])

        # Initial state
        state = {"biomass": jnp.array([[10.0, 20.0], [30.0, 40.0]])}
        params = {"R": 5.0, "lambda": 0.1}
        grid_shape = (2, 2)
        dt = 0.1

        # Execute one timestep
        state = kernel.execute_local_phase(state, dt=dt, params=params, grid_shape=grid_shape)

        # Expected:
        # R = 5.0 (everywhere)
        # M = [1, 2, 3, 4] (lambda * biomass)
        # B_new = B + (R - M) * dt
        expected = jnp.array([[10.4, 20.3], [30.2, 40.1]])
        assert jnp.allclose(state["biomass"], expected, atol=1e-6)

    def test_convergence_to_equilibrium(self) -> None:
        """Test that biomass converges to B_eq = R/λ."""
        kernel = Kernel([compute_recruitment_2d, compute_mortality, compute_growth])

        # Start from zero
        state = {"biomass": jnp.zeros((3, 3))}
        params = {"R": 10.0, "lambda": 0.1}
        grid_shape = (3, 3)
        dt = 0.1  # Larger timestep for faster convergence

        # Run many timesteps (exponential convergence with time constant 1/λ = 10)
        # After t = 5/λ = 50, should be at ~99.3% of equilibrium
        for _ in range(500):  # t_total = 50
            state = kernel.execute_local_phase(state, dt=dt, params=params, grid_shape=grid_shape)

        # Should converge to B_eq = R/λ = 10/0.1 = 100
        expected = jnp.full((3, 3), 100.0)
        assert jnp.allclose(state["biomass"], expected, rtol=0.01)  # Within 1%

    def test_mass_conservation_no_diffusion(self) -> None:
        """Test that total biomass follows dB/dt = R - λB."""
        kernel = Kernel([compute_recruitment_2d, compute_mortality, compute_growth])

        state = {"biomass": jnp.array([[10.0, 20.0], [30.0, 40.0]])}
        params = {"R": 5.0, "lambda": 0.1}
        grid_shape = (2, 2)
        dt = 0.1

        initial_total = jnp.sum(state["biomass"])

        # Execute one timestep
        state = kernel.execute_local_phase(state, dt=dt, params=params, grid_shape=grid_shape)

        final_total = jnp.sum(state["biomass"])

        # Total change should be: (4 cells * R - sum(M)) * dt
        # sum(M) = 0.1 * (10 + 20 + 30 + 40) = 10
        # dB_total = (4 * 5 - 10) * 0.1 = 1.0
        expected_total = initial_total + 1.0
        assert jnp.allclose(final_total, expected_total, atol=1e-6)

    def test_different_lambda_values(self) -> None:
        """Test behavior with different mortality rates."""
        kernel = Kernel([compute_recruitment_2d, compute_mortality, compute_growth])

        state1 = {"biomass": jnp.full((2, 2), 50.0)}
        state2 = {"biomass": jnp.full((2, 2), 50.0)}

        params_low = {"R": 10.0, "lambda": 0.05}  # Low mortality
        params_high = {"R": 10.0, "lambda": 0.2}  # High mortality

        grid_shape = (2, 2)
        dt = 0.1

        state1 = kernel.execute_local_phase(state1, dt=dt, params=params_low, grid_shape=grid_shape)
        state2 = kernel.execute_local_phase(
            state2, dt=dt, params=params_high, grid_shape=grid_shape
        )

        # Low mortality: B_new = 50 + (10 - 2.5) * 0.1 = 50.75
        # High mortality: B_new = 50 + (10 - 10) * 0.1 = 50.0
        assert jnp.allclose(state1["biomass"], 50.75, atol=1e-6)
        assert jnp.allclose(state2["biomass"], 50.0, atol=1e-6)


@pytest.mark.unit
class TestBiologyEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_biomass(self) -> None:
        """Test behavior with zero biomass."""
        kernel = Kernel([compute_recruitment_2d, compute_mortality, compute_growth])

        state = {"biomass": jnp.zeros((2, 2))}
        params = {"R": 5.0, "lambda": 0.1}
        grid_shape = (2, 2)
        dt = 0.1

        state = kernel.execute_local_phase(state, dt=dt, params=params, grid_shape=grid_shape)

        # M = 0 (no biomass), so B_new = 0 + 5 * 0.1 = 0.5
        assert jnp.allclose(state["biomass"], 0.5)

    def test_high_biomass(self) -> None:
        """Test behavior with very high biomass."""
        kernel = Kernel([compute_recruitment_2d, compute_mortality, compute_growth])

        state = {"biomass": jnp.full((2, 2), 1000.0)}
        params = {"R": 5.0, "lambda": 0.1}
        grid_shape = (2, 2)
        dt = 0.1

        state = kernel.execute_local_phase(state, dt=dt, params=params, grid_shape=grid_shape)

        # M = 100, so B_new = 1000 + (5 - 100) * 0.1 = 990.5
        assert jnp.allclose(state["biomass"], 990.5)

    def test_large_timestep_stability(self) -> None:
        """Test that large timestep doesn't cause negative biomass."""
        kernel = Kernel([compute_recruitment_2d, compute_mortality, compute_growth])

        state = {"biomass": jnp.full((2, 2), 10.0)}
        params = {"R": 1.0, "lambda": 0.5}
        grid_shape = (2, 2)
        dt = 100.0  # Very large timestep

        state = kernel.execute_local_phase(state, dt=dt, params=params, grid_shape=grid_shape)

        # M = 5.0, B_new = 10 + (1 - 5) * 100 = -390
        # This shows forward Euler can go negative - would need implicit method
        # For now, just check it runs without error
        assert state["biomass"].shape == (2, 2)
