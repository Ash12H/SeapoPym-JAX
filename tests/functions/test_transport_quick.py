"""Tests for the QUICK advection scheme (transport_tendency_quick)."""

import jax
import jax.numpy as jnp
import pytest

from seapopym.functions.transport import (
    BoundaryType,
    transport_tendency,
    transport_tendency_quick,
)


@pytest.fixture
def uniform_grid():
    """Create a simple uniform 8x8 grid with 1 km spacing."""
    ny, nx = 8, 8
    dx = jnp.full((ny, nx), 1000.0)
    dy = jnp.full((ny, nx), 1000.0)
    return {
        "ny": ny,
        "nx": nx,
        "dx": dx,
        "dy": dy,
        "face_height": dy,
        "face_width": dx,
        "cell_area": dx * dy,
        "mask": jnp.ones((ny, nx)),
    }


class TestQuickConstantField:
    """A spatially constant field has zero advection and diffusion tendency."""

    def test_constant_field_zero_tendency(self, uniform_grid):
        g = uniform_grid
        state = jnp.full((g["ny"], g["nx"]), 3.14)
        u = jnp.full((g["ny"], g["nx"]), 0.5)
        v = jnp.full((g["ny"], g["nx"]), -0.3)

        adv, diff = transport_tendency_quick(
            state,
            u,
            v,
            100.0,
            g["dx"],
            g["dy"],
            g["face_height"],
            g["face_width"],
            g["cell_area"],
            g["mask"],
            bc_north=BoundaryType.PERIODIC,
            bc_south=BoundaryType.PERIODIC,
            bc_east=BoundaryType.PERIODIC,
            bc_west=BoundaryType.PERIODIC,
        )
        assert jnp.allclose(adv, 0.0, atol=1e-10)
        assert jnp.allclose(diff, 0.0, atol=1e-10)


class TestQuickMassConservation:
    """Total mass is conserved under periodic BC with no diffusion."""

    def test_periodic_mass_conservation(self, uniform_grid):
        g = uniform_grid
        key = jax.random.key(42)
        state = jax.random.uniform(key, (g["ny"], g["nx"]))
        u = jnp.full((g["ny"], g["nx"]), 0.1)
        v = jnp.full((g["ny"], g["nx"]), 0.05)

        adv, _ = transport_tendency_quick(
            state,
            u,
            v,
            0.0,  # no diffusion
            g["dx"],
            g["dy"],
            g["face_height"],
            g["face_width"],
            g["cell_area"],
            g["mask"],
            bc_north=BoundaryType.PERIODIC,
            bc_south=BoundaryType.PERIODIC,
            bc_east=BoundaryType.PERIODIC,
            bc_west=BoundaryType.PERIODIC,
        )
        # Sum of tendencies * area = 0 (no net flux in/out). Use a relative
        # tolerance because float32 round-off on sums of ~1e3 terms is ~1e-4.
        total_mass_rate = float(jnp.sum(adv * g["cell_area"]))
        flux_scale = float(jnp.sum(jnp.abs(adv * g["cell_area"])))
        assert abs(total_mass_rate) / flux_scale < 1e-5


class TestQuickAccuracy:
    """QUICK should be strictly more accurate than upwind on a smooth field."""

    @staticmethod
    def _advect_gaussian(scheme_fn, n, t_final, sigma=0.1, cfl=0.3):
        """Advect a Gaussian with uniform flow using Heun RK2.

        RK2 is required because explicit Euler + QUICK is unconditionally
        unstable. Upwind is also integrated with RK2 here so the comparison
        reflects only the spatial discretization error.
        """
        dx = dy = 1.0 / n
        x = (jnp.arange(n) + 0.5) / n
        y = (jnp.arange(n) + 0.5) / n
        xx, yy = jnp.meshgrid(x, y)

        state0 = jnp.exp(-((xx - 0.3) ** 2 + (yy - 0.5) ** 2) / (2 * sigma**2))
        u = jnp.full((n, n), 1.0)
        v = jnp.zeros((n, n))

        dx_arr = jnp.full((n, n), dx)
        dy_arr = jnp.full((n, n), dy)
        mask = jnp.ones((n, n))
        area = dx_arr * dy_arr

        dt_target = cfl * dx / 1.0
        n_steps = max(1, int(t_final / dt_target))
        dt = t_final / n_steps

        def tendency(s):
            adv, _ = scheme_fn(
                s,
                u,
                v,
                0.0,
                dx_arr,
                dy_arr,
                dy_arr,
                dx_arr,
                area,
                mask,
                bc_north=BoundaryType.PERIODIC,
                bc_south=BoundaryType.PERIODIC,
                bc_east=BoundaryType.PERIODIC,
                bc_west=BoundaryType.PERIODIC,
            )
            return adv

        @jax.jit
        def step(state):
            k1 = tendency(state)
            k2 = tendency(state + dt * k1)
            return state + 0.5 * dt * (k1 + k2)

        state = state0
        for _ in range(n_steps):
            state = step(state)

        total_time = dt * n_steps
        xs = (xx - 0.3 - total_time) % 1.0
        xs = jnp.where(xs > 0.5, xs - 1.0, xs)
        state_true = jnp.exp(-(xs**2 + (yy - 0.5) ** 2) / (2 * sigma**2))
        return float(jnp.sqrt(jnp.mean((state - state_true) ** 2)))

    def test_quick_more_accurate_than_upwind_smooth(self):
        """On a well-resolved smooth Gaussian, QUICK RMSE should be substantially
        better than upwind. Overall order is limited by explicit Euler (1st order
        in time) so the theoretical 3rd-order spatial gain shows as a ~1.5-2x
        error reduction, not a pure order-of-magnitude win.
        """
        err_up = self._advect_gaussian(transport_tendency, 64, t_final=0.3)
        err_qk = self._advect_gaussian(transport_tendency_quick, 64, t_final=0.3)
        assert err_qk < 0.75 * err_up, f"QUICK error ({err_qk:.4f}) not better enough than upwind ({err_up:.4f})"


class TestQuickDifferentiability:
    """QUICK must be differentiable under jax.grad."""

    def test_grad_finite(self, uniform_grid):
        g = uniform_grid
        key = jax.random.key(0)
        state = jax.random.uniform(key, (g["ny"], g["nx"]))
        u = jnp.full((g["ny"], g["nx"]), 0.2)
        v = jnp.full((g["ny"], g["nx"]), 0.1)

        def loss(s):
            adv, diff = transport_tendency_quick(
                s,
                u,
                v,
                10.0,
                g["dx"],
                g["dy"],
                g["face_height"],
                g["face_width"],
                g["cell_area"],
                g["mask"],
                bc_north=BoundaryType.PERIODIC,
                bc_south=BoundaryType.PERIODIC,
                bc_east=BoundaryType.PERIODIC,
                bc_west=BoundaryType.PERIODIC,
            )
            return jnp.sum((adv + diff) ** 2)

        grad = jax.grad(loss)(state)
        assert grad.shape == state.shape
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert bool(jnp.any(grad != 0))


class TestQuickClosedBoundary:
    """QUICK with CLOSED BCs should not emit mass across the boundary."""

    def test_closed_boundary_mass_bounded(self, uniform_grid):
        g = uniform_grid
        key = jax.random.key(1)
        state = jax.random.uniform(key, (g["ny"], g["nx"]))
        u = jnp.full((g["ny"], g["nx"]), 0.5)
        v = jnp.full((g["ny"], g["nx"]), 0.3)

        adv, _ = transport_tendency_quick(
            state,
            u,
            v,
            0.0,
            g["dx"],
            g["dy"],
            g["face_height"],
            g["face_width"],
            g["cell_area"],
            g["mask"],
            bc_north=BoundaryType.CLOSED,
            bc_south=BoundaryType.CLOSED,
            bc_east=BoundaryType.CLOSED,
            bc_west=BoundaryType.CLOSED,
        )
        # Net mass rate should be 0 for CLOSED boundaries (no flux in/out).
        # Relative tolerance for float32 round-off.
        total_mass_rate = float(jnp.sum(adv * g["cell_area"]))
        flux_scale = float(jnp.sum(jnp.abs(adv * g["cell_area"])))
        assert abs(total_mass_rate) / flux_scale < 1e-5
