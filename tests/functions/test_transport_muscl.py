"""Tests for the MUSCL advection scheme with the van Leer limiter (transport_tendency_muscl)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from seapopym.functions.transport import (
    BoundaryType,
    _get_neighbor,
    transport_tendency,
    transport_tendency_muscl,
    transport_tendency_muscl_faces,
    transport_tendency_muscl_faces_exterior,
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


def muscl_faces(state, u_face, v_face, grid, mask, D=0.0, **bc):
    return transport_tendency_muscl_faces(
        state,
        u_face,
        v_face,
        D,
        grid["dx"],
        grid["dy"],
        grid["face_height"],
        grid["face_width"],
        grid["cell_area"],
        mask,
        **bc,
    )


class TestMusclFaces:
    """The same scheme, with the face velocities given rather than averaged."""

    @pytest.mark.parametrize("bc", [CLOSED, PERIODIC])
    def test_the_mean_of_the_centres_gives_the_centre_scheme_to_the_bit(self, bc):
        grid = lat_lon_grid(10, 12)
        state, u, v = random_fields(10, 12, seed=11)
        mask = irregular_mask(10, 12)
        u_face = 0.5 * (u + _get_neighbor(u, "east", bc["bc_east"]))
        v_face = 0.5 * (v + _get_neighbor(v, "north", bc["bc_north"]))
        centre = muscl(state, u, v, grid, mask, D=300.0, **bc)
        faces = muscl_faces(state, u_face, v_face, grid, mask, D=300.0, **bc)
        for a, b in zip(centre, faces, strict=True):
            assert np.array_equal(np.asarray(a), np.asarray(b))

    def test_a_uniform_field_stays_uniform_in_a_flow_without_divergence(self):
        # Face velocities from a streamfunction on the corners: what enters a cell
        # through two faces leaves it through the other two, exactly. The
        # streamfunction is zero on the boundary, so no flux crosses the closed edges.
        ny, nx, spacing = 10, 12, 1000.0
        rng = np.random.default_rng(2)
        psi = np.zeros((ny + 1, nx + 1))
        psi[1:-1, 1:-1] = rng.normal(0.0, 100.0, (ny - 1, nx - 1))
        # Corner (j, i) is the south-west corner of cell (j, i).
        u_face = -(psi[1:, 1:] - psi[:-1, 1:]) / spacing
        v_face = (psi[1:, 1:] - psi[1:, :-1]) / spacing
        metric = jnp.full((ny, nx), spacing)
        grid = {"dx": metric, "dy": metric, "face_height": metric, "face_width": metric, "cell_area": metric * metric}
        state = jnp.full((ny, nx), 2.5)
        adv, _ = muscl_faces(state, jnp.asarray(u_face), jnp.asarray(v_face), grid, jnp.ones((ny, nx)), **CLOSED)
        # Against the tendency one face alone would give, rounding in float32.
        one_face = 2.5 * float(np.abs(u_face).max()) / spacing
        assert float(jnp.abs(adv).max()) < 1e-5 * one_face

    def test_a_closed_basin_keeps_its_mass(self):
        grid = lat_lon_grid(10, 12)
        state, u, v = random_fields(10, 12, seed=13)
        adv, diff = muscl_faces(state, u, v, grid, irregular_mask(10, 12), D=500.0, **CLOSED)
        for rate in (adv, diff):
            mass_rate = rate * grid["cell_area"]
            assert abs(float(jnp.sum(mass_rate))) / float(jnp.sum(jnp.abs(mass_rate))) < 1e-6


OPEN = {"bc_north": 1, "bc_south": 1, "bc_east": 1, "bc_west": 1}


def muscl_exterior(state, exterior, u_face, v_face, grid, mask, D=0.0, **bc):
    return transport_tendency_muscl_faces_exterior(
        state,
        exterior,
        u_face,
        v_face,
        D,
        grid["dx"],
        grid["dy"],
        grid["face_height"],
        grid["face_width"],
        grid["cell_area"],
        mask,
        **bc,
    )


def edges(ny: int, nx: int) -> np.ndarray:
    """Return the cells on the edge of the grid."""
    edge = np.zeros((ny, nx), dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    return edge


class TestMusclFacesExterior:
    """What flows in through an open edge carries the exterior value; nothing else changes."""

    @pytest.mark.parametrize("bc", [CLOSED, PERIODIC])
    def test_without_open_edges_the_exterior_is_never_read(self, bc):
        grid = lat_lon_grid(10, 12)
        state, u, v = random_fields(10, 12, seed=21)
        exterior, _, _ = random_fields(10, 12, seed=22)
        mask = irregular_mask(10, 12)
        faces = muscl_faces(state, u, v, grid, mask, D=300.0, **bc)
        outside = muscl_exterior(state, 100.0 * exterior, u, v, grid, mask, D=300.0, **bc)
        for a, b in zip(faces, outside, strict=True):
            assert np.array_equal(np.asarray(a), np.asarray(b))

    def test_open_edges_change_only_the_edge_cells(self):
        grid = lat_lon_grid(10, 12)
        state, u, v = random_fields(10, 12, seed=23)
        exterior, _, _ = random_fields(10, 12, seed=24)
        mask = irregular_mask(10, 12)
        adv_faces, diff_faces = muscl_faces(state, u, v, grid, mask, D=300.0, **OPEN)
        adv, diff = muscl_exterior(state, exterior, u, v, grid, mask, D=300.0, **OPEN)
        inner = ~edges(10, 12)
        assert np.array_equal(np.asarray(adv)[inner], np.asarray(adv_faces)[inner])
        assert np.array_equal(np.asarray(diff), np.asarray(diff_faces))
        assert not np.array_equal(np.asarray(adv), np.asarray(adv_faces))

    def test_a_field_equal_to_its_exterior_goes_through_unchanged(self):
        grid = lat_lon_grid(6, 9)
        state = jnp.full((6, 9), 2.0)
        adv, _ = muscl_exterior(
            state,
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

    @pytest.mark.parametrize("sign", [1.0, -1.0])
    def test_the_mass_changes_by_what_flows_in_minus_what_flows_out(self, sign):
        # A uniform flow across the grid, west to east or back: it enters through one
        # open edge with the exterior value and leaves through the other with the edge
        # value, which the flat edge columns hold at first order.
        ny, nx = 5, 8
        grid = lat_lon_grid(ny, nx)
        state = jnp.broadcast_to(jnp.linspace(1.0, 2.0, nx)[None, :], (ny, nx))
        exterior = jnp.full((ny, nx), 5.0)
        speed = 0.3 * sign
        adv, _ = muscl_exterior(
            state,
            exterior,
            jnp.full((ny, nx), speed),
            jnp.zeros((ny, nx)),
            grid,
            jnp.ones((ny, nx)),
            bc_north=0,
            bc_south=0,
            bc_east=1,
            bc_west=1,
        )
        height = grid["face_height"][:, 0]
        leaving = -1 if sign > 0 else 0
        flowing_in = abs(speed) * float(jnp.sum(height * 5.0))
        flowing_out = abs(speed) * float(jnp.sum(height * state[:, leaving]))
        mass_rate = float(jnp.sum(adv * grid["cell_area"]))
        assert mass_rate == pytest.approx(flowing_in - flowing_out, rel=1e-5)

    def test_an_edge_cell_walled_on_two_sides_no_longer_feeds_itself(self):
        # An edge cell with land to the north and west: the flow enters through its open
        # east face and leaves slowly through the south. With the edge value outside, the
        # more it holds the more flows in; with an exterior, what flows in is fixed.
        ny, nx = 3, 4
        grid = lat_lon_grid(ny, nx)
        mask = jnp.ones((ny, nx)).at[2, 3].set(0.0).at[1, 2].set(0.0)
        u_face = jnp.zeros((ny, nx)).at[1, 3].set(-0.3)
        v_face = jnp.zeros((ny, nx)).at[0, 3].set(-0.1)
        bc = {"bc_north": 0, "bc_south": 0, "bc_east": 1, "bc_west": 0}
        exterior = jnp.full((ny, nx), 0.5)

        def rate(value, with_exterior):
            state = jnp.ones((ny, nx)).at[1, 3].set(value)
            if with_exterior:
                adv, _ = muscl_exterior(state, exterior, u_face, v_face, grid, mask, **bc)
            else:
                adv, _ = muscl_faces(state, u_face, v_face, grid, mask, **bc)
            return float(adv[1, 3])

        assert rate(20.0, False) > rate(10.0, False) > 0.0
        assert rate(20.0, True) < rate(10.0, True) < 0.0
