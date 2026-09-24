"""Tests for the MUSCL advection scheme with the van Leer limiter (transport_tendency_muscl)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from seapopym.functions.transport import (
    BoundaryType,
    transport_tendency,
    transport_tendency_muscl,
)

CLOSED = {"bc_north": 0, "bc_south": 0, "bc_east": 0, "bc_west": 0}
PERIODIC = {"bc_north": 2, "bc_south": 2, "bc_east": 2, "bc_west": 2}


def lat_lon_grid(ny: int, nx: int) -> dict[str, jnp.ndarray]:
    """Return the metrics of a 1-degree latitude-longitude grid between 30N and 30N + ny."""
    latitude = 30.5 + jnp.arange(ny)
    dx = jnp.broadcast_to((111e3 * jnp.cos(jnp.deg2rad(latitude)))[:, None], (ny, nx))
    dy = jnp.full((ny, nx), 111e3)
    return {"dx": dx, "dy": dy, "face_height": dy, "face_width": dx, "cell_area": dx * dy}


def irregular_mask(ny: int, nx: int) -> jnp.ndarray:
    """Return an ocean mask with an island and a peninsula."""
    mask = np.ones((ny, nx))
    mask[3:5, 3:6] = 0.0
    mask[0:4, 9] = 0.0
    mask[7, 0:2] = 0.0
    return jnp.asarray(mask)


def muscl(state, u, v, grid, mask, D=0.0, **bc):
    return transport_tendency_muscl(
        state,
        u,
        v,
        D,
        grid["dx"],
        grid["dy"],
        grid["face_height"],
        grid["face_width"],
        grid["cell_area"],
        mask,
        **bc,
    )


def random_fields(ny: int, nx: int, seed: int = 0) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return a positive field and two velocity components of either sign."""
    k1, k2, k3 = jax.random.split(jax.random.key(seed), 3)
    state = jax.random.uniform(k1, (ny, nx), minval=0.1, maxval=2.0)
    u = jax.random.uniform(k2, (ny, nx), minval=-0.5, maxval=0.5)
    v = jax.random.uniform(k3, (ny, nx), minval=-0.5, maxval=0.5)
    return state, u, v


class TestMusclExactZeros:
    """Cases where the tendency must vanish exactly, not approximately."""

    @pytest.mark.parametrize("bc", [CLOSED, PERIODIC, {"bc_north": 1, "bc_south": 1, "bc_east": 1, "bc_west": 1}])
    def test_zero_velocity_gives_exactly_zero(self, bc):
        grid = lat_lon_grid(10, 12)
        state, _, _ = random_fields(10, 12)
        zero = jnp.zeros((10, 12))
        adv, diff = muscl(state, zero, zero, grid, irregular_mask(10, 12), **bc)
        assert np.array_equal(np.asarray(adv), np.zeros((10, 12)))
        assert np.array_equal(np.asarray(diff), np.zeros((10, 12)))

    def test_constant_field_in_uniform_periodic_flow(self):
        grid = {k: jnp.full((8, 8), 1000.0) for k in ("dx", "dy", "face_height", "face_width")}
        grid["cell_area"] = grid["dx"] * grid["dy"]
        state = jnp.full((8, 8), 3.14)
        adv, diff = muscl(
            state, jnp.full((8, 8), 0.5), jnp.full((8, 8), -0.3), grid, jnp.ones((8, 8)), 100.0, **PERIODIC
        )
        assert np.array_equal(np.asarray(adv), np.zeros((8, 8)))
        assert np.array_equal(np.asarray(diff), np.zeros((8, 8)))

    def test_open_edges_let_a_uniform_field_through_unchanged(self):
        # Inflow on the west edge carries the edge value, outflow on the east edge the
        # same: a uniform field in a uniform eastward flow does not change.
        grid = lat_lon_grid(6, 9)
        state = jnp.full((6, 9), 2.0)
        adv, _ = muscl(
            state,
            jnp.full((6, 9), 0.3),
            jnp.zeros((6, 9)),
            grid,
            jnp.ones((6, 9)),
            bc_north=0,
            bc_south=0,
            bc_east=1,
            bc_west=1,
        )
        assert np.allclose(np.asarray(adv), 0.0, atol=1e-12)


class TestMusclConservation:
    """Mass changes only through open edges."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_closed_basin_on_a_lat_lon_grid(self, seed):
        # Face lengths vary from row to row here; every face flux is computed once, so
        # what leaves a cell enters its neighbour and the sum telescopes.
        grid = lat_lon_grid(10, 12)
        state, u, v = random_fields(10, 12, seed)
        adv, diff = muscl(state, u, v, grid, irregular_mask(10, 12), D=500.0, **CLOSED)
        for rate in (adv, diff):
            mass_rate = rate * grid["cell_area"]
            assert abs(float(jnp.sum(mass_rate))) / float(jnp.sum(jnp.abs(mass_rate))) < 1e-6

    def test_no_flux_across_a_masked_column(self):
        # A land column splits the domain in two basins: each keeps its own mass.
        grid = lat_lon_grid(8, 11)
        state, u, v = random_fields(8, 11, seed=3)
        mask = jnp.ones((8, 11)).at[:, 5].set(0.0)
        adv, _ = muscl(state, u, v, grid, mask, **CLOSED)
        mass_rate = adv * grid["cell_area"]
        scale = float(jnp.sum(jnp.abs(mass_rate)))
        assert abs(float(jnp.sum(mass_rate[:, :5]))) / scale < 1e-6
        assert abs(float(jnp.sum(mass_rate[:, 6:]))) / scale < 1e-6
        assert np.array_equal(np.asarray(adv[:, 5]), np.zeros(8))

    def test_masked_cells_never_change(self):
        grid = lat_lon_grid(10, 12)
        state, u, v = random_fields(10, 12, seed=4)
        mask = irregular_mask(10, 12)
        adv, diff = muscl(state, u, v, grid, mask, D=500.0, **CLOSED)
        land = np.asarray(mask) == 0
        assert (np.asarray(adv)[land] == 0.0).all()
        assert (np.asarray(diff)[land] == 0.0).all()


def advect_with_heun(scheme, state, u, v, dx, n_steps, dt, bc):
    """Integrate advection only with Heun's RK2 on a uniform grid of spacing dx."""
    metric = jnp.full(state.shape, dx)
    ones = jnp.ones(state.shape)

    def tendency(s):
        adv, _ = scheme(s, u, v, 0.0, metric, metric, metric, metric, metric * metric, ones, **bc)
        return adv

    @jax.jit
    def step(s):
        k1 = tendency(s)
        k2 = tendency(s + dt * k1)
        return s + 0.5 * dt * (k1 + k2)

    for _ in range(n_steps):
        state = step(state)
    return state


class TestMusclShape:
    """The limiter keeps the field within its bounds and sharper than upwind."""

    def test_a_top_hat_stays_within_its_bounds(self):
        # Uniform diagonal flow at the stated Courant limit, two full crossings.
        n = 32
        dx = 1.0 / n
        x = (jnp.arange(n) + 0.5) / n
        xx, yy = jnp.meshgrid(x, x)
        top_hat = ((jnp.abs(xx - 0.3) < 0.12) & (jnp.abs(yy - 0.3) < 0.12)).astype(jnp.float32)
        speed = 1.0
        dt = 0.4 * dx / (2 * speed)  # (|u| / dx + |v| / dy) * dt = 0.4
        n_steps = int(round(2.0 / dt))
        end = advect_with_heun(
            transport_tendency_muscl,
            top_hat,
            jnp.full((n, n), speed),
            jnp.full((n, n), speed),
            dx,
            n_steps,
            dt,
            PERIODIC,
        )
        assert float(end.min()) >= -1e-6
        assert float(end.max()) <= 1.0 + 1e-6
        assert abs(float(end.sum()) - float(top_hat.sum())) / float(top_hat.sum()) < 1e-5

    def test_more_accurate_than_upwind_on_a_gaussian(self):
        n = 40
        dx = 1.0 / n
        x = (jnp.arange(n) + 0.5) / n
        xx, yy = jnp.meshgrid(x, x)
        gaussian = jnp.exp(-((xx - 0.3) ** 2 + (yy - 0.5) ** 2) / (2 * 0.08**2))
        u, v = jnp.ones((n, n)), jnp.zeros((n, n))
        dt = 0.3 * dx
        n_steps = int(round(1.0 / dt))
        dt = 1.0 / n_steps  # exactly one crossing, back to the start
        errors = {}
        for name, scheme in (("muscl", transport_tendency_muscl), ("upwind", transport_tendency)):
            end = advect_with_heun(scheme, gaussian, u, v, dx, n_steps, dt, PERIODIC)
            errors[name] = float(jnp.abs(end - gaussian).sum())
        assert errors["muscl"] < 0.3 * errors["upwind"]


class TestMusclGradient:
    def test_gradient_is_finite_on_a_flat_field(self):
        # A flat field is where the ratio form of a limiter divides zero by zero.
        grid = lat_lon_grid(6, 6)
        u = jnp.full((6, 6), 0.2)
        v = jnp.full((6, 6), -0.1)

        def total(s):
            adv, _ = muscl(s, u, v, grid, jnp.ones((6, 6)), **CLOSED)
            return jnp.sum(adv**2)

        gradient = jax.grad(total)(jnp.full((6, 6), 1.5))
        assert bool(jnp.isfinite(gradient).all())


def test_the_boundary_type_values_match_the_other_schemes():
    assert (BoundaryType.CLOSED, BoundaryType.OPEN, BoundaryType.PERIODIC) == (0, 1, 2)
