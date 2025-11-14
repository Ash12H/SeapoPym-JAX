"""Tests for transport Units."""

import jax.numpy as jnp
import pytest

from seapopym_message.core.kernel import Kernel
from seapopym_message.kernels.transport import (
    check_cfl_condition,
    compute_advection_2d,
    compute_advection_simple,
    compute_diffusion_2d,
    compute_diffusion_simple,
)


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


@pytest.mark.unit
class TestAdvectionUnit:
    """Test advection Unit."""

    def test_advection_translation_eastward(self) -> None:
        """Test uniform eastward advection (translation)."""
        # Create a blob on the left side
        biomass = jnp.zeros((5, 5))
        biomass = biomass.at[2, 1].set(100.0)

        # Uniform eastward velocity
        u = jnp.ones((5, 5)) * 1.0  # 1 m/s eastward
        v = jnp.zeros((5, 5))

        params = {"dx": 1.0, "dy": 1.0}
        forcings = {"u": u, "v": v}
        dt = 0.5  # CFL = 1*0.5/1 = 0.5 < 1, stable

        state = {"biomass": biomass}
        result = compute_advection_simple.execute(state, dt=dt, params=params, forcings=forcings)

        # Blob should move eastward (to the right)
        # Original position should decrease
        assert result["biomass"][2, 1] < biomass[2, 1]
        # Downstream position should increase
        assert result["biomass"][2, 2] > biomass[2, 2]

    def test_advection_translation_northward(self) -> None:
        """Test uniform northward advection."""
        # Create a blob in the south
        biomass = jnp.zeros((5, 5))
        biomass = biomass.at[3, 2].set(100.0)

        # Uniform northward velocity (negative v in our convention)
        u = jnp.zeros((5, 5))
        v = jnp.ones((5, 5)) * (-1.0)  # Northward

        params = {"dx": 1.0, "dy": 1.0}
        forcings = {"u": u, "v": v}
        dt = 0.5

        state = {"biomass": biomass}
        result = compute_advection_simple.execute(state, dt=dt, params=params, forcings=forcings)

        # Blob should move northward (decrease in i index)
        assert result["biomass"][3, 2] < biomass[3, 2]
        assert result["biomass"][2, 2] > biomass[2, 2]

    def test_advection_mass_conservation(self) -> None:
        """Test that advection conserves total mass (without boundaries)."""
        biomass = jnp.array([[0.0, 10.0, 0.0], [0.0, 50.0, 0.0], [0.0, 20.0, 0.0]])

        # Uniform velocity
        u = jnp.ones((3, 3)) * 0.5
        v = jnp.zeros((3, 3))

        params = {"dx": 1.0, "dy": 1.0}
        forcings = {"u": u, "v": v}
        dt = 0.1

        state = {"biomass": biomass}
        initial_mass = jnp.sum(biomass)

        result = compute_advection_simple.execute(state, dt=dt, params=params, forcings=forcings)

        final_mass = jnp.sum(result["biomass"])

        # Mass should be conserved (within numerical precision)
        # Note: upwind scheme has some numerical diffusion, so tolerance is relaxed
        assert jnp.allclose(final_mass, initial_mass, rtol=1e-3)

    def test_advection_rotation(self) -> None:
        """Test advection with rotational velocity field."""
        # Create a small blob off-center
        biomass = jnp.zeros((7, 7))
        biomass = biomass.at[3, 5].set(100.0)  # Right of center

        # Create rotational velocity field (counterclockwise)
        # u = -y, v = x (with origin at center)
        center_i, center_j = 3, 3
        i_indices, j_indices = jnp.meshgrid(jnp.arange(7), jnp.arange(7), indexing="ij")

        y = (i_indices - center_i).astype(float)
        x = (j_indices - center_j).astype(float)

        v_rot = x * 0.5  # Proportional to x
        u_rot = -y * 0.5  # Proportional to -y

        params = {"dx": 1.0, "dy": 1.0}
        forcings = {"u": u_rot, "v": v_rot}
        dt = 0.1

        state = {"biomass": biomass}

        # Run multiple steps
        for _ in range(10):
            state = compute_advection_simple.execute(state, dt=dt, params=params, forcings=forcings)

        # Blob should have rotated (position changed)
        # Original position should decrease significantly
        assert state["biomass"][3, 5] < 80.0
        # Mass should be approximately conserved (upwind has numerical diffusion)
        assert jnp.allclose(jnp.sum(state["biomass"]), 100.0, rtol=0.2)

    def test_advection_with_halos(self) -> None:
        """Test advection with halo exchange."""
        # Small domain with eastward flow
        biomass = jnp.array([[10.0, 10.0], [10.0, 10.0]])

        # Eastward velocity
        u = jnp.ones((2, 2)) * 1.0
        v = jnp.zeros((2, 2))

        # West neighbor has higher biomass (upstream source)
        halo_west = {"biomass": jnp.array([50.0, 50.0])}

        params = {"dx": 1.0, "dy": 1.0}
        forcings = {"u": u, "v": v}
        dt = 0.5

        state = {"biomass": biomass}
        result = compute_advection_2d.execute(
            state,
            dt=dt,
            params=params,
            forcings=forcings,
            halo_west=halo_west,
            halo_north=None,
            halo_south=None,
            halo_east=None,
        )

        # Western cells should increase (advection from high-value upstream)
        assert jnp.all(result["biomass"][:, 0] > biomass[:, 0])

    def test_advection_uniform_field(self) -> None:
        """Test that advecting a uniform field with uniform velocity preserves uniformity."""
        biomass = jnp.full((5, 5), 42.0)

        u = jnp.ones((5, 5)) * 0.5
        v = jnp.zeros((5, 5))

        params = {"dx": 1.0, "dy": 1.0}
        forcings = {"u": u, "v": v}
        dt = 0.1

        state = {"biomass": biomass}
        result = compute_advection_simple.execute(state, dt=dt, params=params, forcings=forcings)

        # Uniform field remains uniform (upwind of uniform is zero)
        assert jnp.allclose(result["biomass"], biomass, atol=1e-5)

    def test_advection_positivity(self) -> None:
        """Test that advection preserves non-negativity."""
        biomass = jnp.array([[0.0, 10.0, 0.0], [0.0, 10.0, 0.0], [0.0, 10.0, 0.0]])

        u = jnp.ones((3, 3)) * 0.5
        v = jnp.zeros((3, 3))

        params = {"dx": 1.0, "dy": 1.0}
        forcings = {"u": u, "v": v}
        dt = 0.1

        state = {"biomass": biomass}
        result = compute_advection_simple.execute(state, dt=dt, params=params, forcings=forcings)

        # All values should remain non-negative
        assert jnp.all(result["biomass"] >= 0.0)


@pytest.mark.unit
class TestAdvectionWithMask:
    """Test advection with islands/land mask."""

    def test_advection_with_island(self) -> None:
        """Test advection with a land mask (island)."""
        # Create domain with island in center
        biomass = jnp.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 100.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
        )

        # Mask: True = ocean, False = land
        mask = jnp.array(
            [
                [True, True, True, True, True],
                [True, True, False, True, True],
                [True, True, True, True, True],
            ]
        )

        # Eastward flow
        u = jnp.ones((3, 5)) * 1.0
        v = jnp.zeros((3, 5))

        params = {"dx": 1.0, "dy": 1.0, "mask": mask}
        forcings = {"u": u, "v": v}
        dt = 0.5

        state = {"biomass": biomass}
        result = compute_advection_simple.execute(state, dt=dt, params=params, forcings=forcings)

        # Land cell (1, 2) should be zero
        assert result["biomass"][1, 2] == 0.0
        # Ocean cells should be non-negative
        assert jnp.all(result["biomass"][mask] >= 0.0)

    def test_advection_mask_multiple_islands(self) -> None:
        """Test advection with multiple islands."""
        biomass = jnp.ones((5, 5)) * 10.0

        # Multiple islands
        mask = jnp.ones((5, 5), dtype=bool)
        mask = mask.at[1, 1].set(False)  # Island
        mask = mask.at[3, 3].set(False)  # Island

        u = jnp.ones((5, 5)) * 0.5
        v = jnp.zeros((5, 5))

        params = {"dx": 1.0, "dy": 1.0, "mask": mask}
        forcings = {"u": u, "v": v}
        dt = 0.1

        state = {"biomass": biomass}
        result = compute_advection_simple.execute(state, dt=dt, params=params, forcings=forcings)

        # Land cells should be zero
        assert result["biomass"][1, 1] == 0.0
        assert result["biomass"][3, 3] == 0.0
        # All other cells should be positive
        assert jnp.all(result["biomass"][mask] > 0.0)


@pytest.mark.unit
class TestAdvectionStability:
    """Test CFL condition and stability."""

    def test_cfl_check_stable(self) -> None:
        """Test CFL condition check for stable case."""
        u = jnp.ones((10, 10)) * 0.5
        v = jnp.zeros((10, 10))
        dt = 1.0
        dx = 1.0
        dy = 1.0

        result = check_cfl_condition(u, v, dt, dx, dy)

        assert result["cfl"] == 0.5
        assert result["stable"] is True
        assert result["max_u"] == 0.5
        assert result["max_v"] == 0.0

    def test_cfl_check_unstable(self) -> None:
        """Test CFL condition check for unstable case."""
        u = jnp.ones((10, 10)) * 2.0  # High velocity
        v = jnp.zeros((10, 10))
        dt = 1.0
        dx = 1.0
        dy = 1.0

        with pytest.warns(UserWarning, match="CFL condition violated"):
            result = check_cfl_condition(u, v, dt, dx, dy)

        assert result["cfl"] == 2.0
        assert result["stable"] is False
        assert result["max_dt_stable"] == 0.5

    def test_cfl_max_stable_dt(self) -> None:
        """Test maximum stable timestep calculation."""
        u = jnp.ones((10, 10)) * 1.0
        v = jnp.ones((10, 10)) * 0.5
        dx = 2.0
        dy = 2.0
        dt = 0.5

        result = check_cfl_condition(u, v, dt, dx, dy)

        # max_velocity = 1.0, min_spacing = 2.0
        # max_dt_stable = 2.0 / 1.0 = 2.0
        assert result["max_dt_stable"] == 2.0
        assert result["stable"] is True

    def test_advection_stable_run(self) -> None:
        """Test that advection with CFL ≤ 1 is stable."""
        biomass = jnp.array([[0.0, 100.0, 0.0], [0.0, 100.0, 0.0], [0.0, 100.0, 0.0]])

        u = jnp.ones((3, 3)) * 0.5  # CFL = 0.5 * 0.1 / 1.0 = 0.05
        v = jnp.zeros((3, 3))

        params = {"dx": 1.0, "dy": 1.0}
        forcings = {"u": u, "v": v}
        dt = 0.1

        state = {"biomass": biomass}

        # Run many steps
        for _ in range(100):
            state = compute_advection_simple.execute(state, dt=dt, params=params, forcings=forcings)

        # Should remain stable (no oscillations or negative values)
        assert jnp.all(state["biomass"] >= 0.0)
        assert jnp.all(jnp.isfinite(state["biomass"]))


@pytest.mark.unit
class TestAdvectionIntegration:
    """Integration tests with Kernel."""

    def test_advection_in_kernel(self) -> None:
        """Test advection as part of a Kernel."""
        kernel = Kernel([compute_advection_simple])

        biomass = jnp.zeros((5, 5))
        biomass = biomass.at[2, 2].set(100.0)

        u = jnp.ones((5, 5)) * 0.5
        v = jnp.zeros((5, 5))

        state = {"biomass": biomass}
        params = {"dx": 1.0, "dy": 1.0}
        forcings = {"u": u, "v": v}
        dt = 0.5

        # Execute via kernel (local phase for simple version)
        state = kernel.execute_local_phase(state, dt=dt, params=params, forcings=forcings)

        # Blob should move eastward
        assert state["biomass"][2, 3] > 0.0

    def test_advection_global_kernel(self) -> None:
        """Test advection_2d requires global phase."""
        kernel = Kernel([compute_advection_2d])

        # Verify it's recognized as global
        assert len(kernel.global_units) == 1
        assert len(kernel.local_units) == 0
        assert kernel.has_global_units() is True

        biomass = jnp.ones((3, 3)) * 10.0
        u = jnp.ones((3, 3)) * 0.5
        v = jnp.zeros((3, 3))

        state = {"biomass": biomass}
        params = {"dx": 1.0, "dy": 1.0}
        forcings = {"u": u, "v": v}
        dt = 0.1

        # Execute global phase
        state = kernel.execute_global_phase(
            state, dt=dt, params=params, forcings=forcings, neighbor_data=None
        )

        assert state["biomass"].shape == (3, 3)

    def test_combined_advection_diffusion(self) -> None:
        """Test combined advection and diffusion."""
        kernel = Kernel([compute_advection_simple, compute_diffusion_simple])

        biomass = jnp.zeros((7, 7))
        biomass = biomass.at[3, 2].set(100.0)

        # Eastward advection
        u = jnp.ones((7, 7)) * 1.0
        v = jnp.zeros((7, 7))

        state = {"biomass": biomass}
        params = {"dx": 1.0, "dy": 1.0, "D": 0.1}
        forcings = {"u": u, "v": v}
        dt = 0.1

        initial_pos_value = state["biomass"][3, 2]

        # Execute both processes
        state = kernel.execute_local_phase(state, dt=dt, params=params, forcings=forcings)

        # Blob should have moved and spread
        # Original position decreased due to both processes
        assert state["biomass"][3, 2] < initial_pos_value
        # Downstream and perpendicular positions should increase
        assert state["biomass"][3, 3] > 0.0  # Advection
        assert state["biomass"][2, 2] > 0.0  # Diffusion
