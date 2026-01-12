"""Tests for transport stability module."""

import numpy as np
import pytest
import xarray as xr

from seapopym.transport.stability import check_diffusion_stability, compute_advection_cfl


def test_check_diffusion_stability():
    # Case 1: Stable
    D = 100.0
    dx = 1000.0
    dy = 1000.0
    dt = 100.0
    # Stability: dt <= min(dx^2, dy^2) / (4*D)
    # 100 <= 1000^2 / 400 = 1000000 / 400 = 2500
    res = check_diffusion_stability(D, dx, dy, dt)
    assert res["is_stable"]
    assert res["cfl_diffusion"] <= 0.25
    assert not np.isinf(res["margin"])

    # Case 2: Unstable
    dt_unstable = 3000.0
    res = check_diffusion_stability(D, dx, dy, dt_unstable)
    assert not res["is_stable"]
    assert res["dt_max"] == 2500.0

    # Case 3: Zero diffusion
    res = check_diffusion_stability(0.0, dx, dy, dt)
    assert res["is_stable"]
    assert res["dt_max"] == float("inf")

    # Case 4: DataArray inputs
    D_da = xr.DataArray([100.0, 200.0], dims="x")
    dx_da = xr.DataArray([1000.0, 1000.0], dims="x")
    dt_da = xr.DataArray(100.0)
    res = check_diffusion_stability(D_da, dx_da, dy, dt_da)
    assert res["is_stable"]
    assert res["D_max"] == 200.0


def test_compute_advection_cfl():
    # CFL = |u|*dt/dx + |v|*dt/dy

    # Case 1: Stable
    u = 1.0
    v = 1.0
    dx = 100.0
    dy = 100.0
    dt = 10.0
    # CFL = 1*10/100 + 1*10/100 = 0.1 + 0.1 = 0.2 <= 1.0
    res = compute_advection_cfl(u, v, dx, dy, dt)
    assert res["is_stable"]
    assert res["cfl_max"] == pytest.approx(0.2)

    # Case 2: Unstable
    dt_unstable = 60.0
    # CFL = 0.6 + 0.6 = 1.2 > 1.0
    res = compute_advection_cfl(u, v, dx, dy, dt_unstable)
    assert not res["is_stable"]
    assert res["cfl_max"] == pytest.approx(1.2)

    # Case 3: DataArray
    u_da = xr.DataArray([1.0, 2.0], dims="x")
    v_da = xr.DataArray([0.0, 0.0], dims="x")  # Pure zonal
    dx_da = xr.DataArray([100.0, 100.0], dims="x")
    dt = 10.0
    # CFL at x=0: 1*10/100 = 0.1
    # CFL at x=1: 2*10/100 = 0.2
    # Max is 0.2
    res = compute_advection_cfl(u_da, v_da, dx_da, dy, dt)
    assert res["cfl_max"] == pytest.approx(0.2)
    assert res["u_max"] == 2.0
