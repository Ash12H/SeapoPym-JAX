"""Tests for Numba transport kernels."""

import numpy as np
import pytest

from seapopym.transport.numba_kernels import advection_flux_numba, diffusion_flux_numba


@pytest.fixture
def grid_setup():
    ny, nx = 5, 5
    state = np.zeros((ny, nx), dtype=np.float32)
    state[2, 2] = 10.0  # Source in center

    ew_area = np.ones((ny, nx), dtype=np.float32) * 10.0
    ns_area = np.ones((ny, nx), dtype=np.float32) * 10.0
    mask = np.ones((ny, nx), dtype=np.float32)
    bc = np.array([0, 0, 0, 0], dtype=np.int32)  # Closed boundaries: N, S, E, W

    flux_e = np.zeros((ny, nx), dtype=np.float32)
    flux_w = np.zeros((ny, nx), dtype=np.float32)
    flux_n = np.zeros((ny, nx), dtype=np.float32)
    flux_s = np.zeros((ny, nx), dtype=np.float32)

    return state, ew_area, ns_area, mask, bc, flux_e, flux_w, flux_n, flux_s


def test_advection_flux_numba_zonal(grid_setup):
    state, ew_area, ns_area, mask, bc, fe, fw, fn, fs = grid_setup

    # Uniform velocity East
    u = np.ones_like(state) * 1.0
    v = np.zeros_like(state)

    # Numba guvectorize requires calling with matching dimensions and types
    # or let numpy broadcasting handle it.

    advection_flux_numba(state, u, v, ew_area, ns_area, mask, bc, fe, fw, fn, fs)

    # Check flux out of center (2,2) to East (2,3)
    # u > 0, so flux E at (2,2) uses state at (2,2) = 10.0
    # flux_e[j, i] corresponds to flux across east face of cell i
    # flux = u * c_up * area
    # flux_e[2, 2] = 1.0 * 10.0 * 10.0 = 100.0
    assert fe[2, 2] == 100.0

    # Flux West into center (2,2) from (2,1)
    # flux_w[2, 2] corresponds to flux across west face of cell i
    # flux = u * c_up * area
    # u > 0, so upwind is west neighbor (2,1) which is 0
    assert fw[2, 2] == 0.0

    # Flux East out of (2,3) should be zero since (2,3) is zero
    assert fe[2, 3] == 0.0


def test_diffusion_flux_numba(grid_setup):
    state, ew_area, ns_area, mask, bc, fe, fw, fn, fs = grid_setup

    D = np.ones_like(state) * 10.0
    dx = np.ones_like(state) * 100.0
    dy = np.ones_like(state) * 100.0

    diffusion_flux_numba(state, D, dx, dy, ew_area, ns_area, mask, bc, fe, fw, fn, fs)

    # Flux is -D * grad * Area
    # At (2,2), c=10.
    # East neighbor (2,3), c=0.
    # grad_east = (0 - 10) / dx_face = -10 / 100 = -0.1
    # flux_e[2, 2] = -D * grad * Area = -10 * (-0.1) * 10 = 10.0
    assert fe[2, 2] == pytest.approx(10.0)

    # West neighbor (2,1), c=0.
    # grad_west = (10 - 0) / dx_face = 10 / 100 = 0.1
    # flux_w[2, 2] = -D * grad * Area = -10 * (0.1) * 10 = -10.0
    # Wait, check definition in kernel:
    # flux_w[j, i] = -d_face * grad * area
    # grad = (c_center - c_west) / dx
    # = (10 - 0) / 100 = 0.1
    # flux_w = -10 * 0.1 * 10 = -10.0
    # Correct.
    assert fw[2, 2] == pytest.approx(-10.0)
