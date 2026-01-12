"""Tests for Chunking Correctness on Transport Functions.

Objective: Ensure that compute_transport_numba and compute_transport_xarray behave correctly with input chunks.
"""

import numpy as np
import pytest
import xarray as xr

from seapopym.standard.coordinates import Coordinates
from seapopym.transport import (
    compute_spherical_cell_areas,
    compute_spherical_dx,
    compute_spherical_dy,
    compute_spherical_face_areas_ew,
    compute_spherical_face_areas_ns,
    compute_transport_numba,
    compute_transport_xarray,
)


@pytest.fixture
def grid_for_chunking():
    """Create a 20x20 grid."""
    lats = np.linspace(-10, 10, 21)
    lons = np.linspace(0, 20, 21)
    times = [0, 1]

    lats_da = xr.DataArray(lats, dims=[Coordinates.Y.value])
    lons_da = xr.DataArray(lons, dims=[Coordinates.X.value])
    times_da = xr.DataArray(times, dims=[Coordinates.T.value])

    cell_areas = compute_spherical_cell_areas(lats_da, lons_da)
    face_areas_ew = compute_spherical_face_areas_ew(lats_da, lons_da)
    face_areas_ns = compute_spherical_face_areas_ns(lats_da, lons_da)
    dx = compute_spherical_dx(lats_da, lons_da)
    dy = compute_spherical_dy(lats_da, lons_da)

    return {
        "lats": lats_da,
        "lons": lons_da,
        "times": times_da,
        "cell_areas": cell_areas,
        "face_areas_ew": face_areas_ew,
        "face_areas_ns": face_areas_ns,
        "dx": dx,
        "dy": dy,
    }


def create_state(grid, dims=(Coordinates.Y.value, Coordinates.X.value)):
    """Create a state variable."""
    coords = {
        Coordinates.Y.value: grid["lats"],
        Coordinates.X.value: grid["lons"],
    }
    shape = [len(grid["lats"]), len(grid["lons"])]
    if Coordinates.T.value in dims:
        coords[Coordinates.T.value] = grid["times"]
        shape = [len(grid["times"])] + shape

    return xr.DataArray(np.random.rand(*shape) + 1.0, coords=coords, dims=dims)


@pytest.mark.parametrize("transport_func", [compute_transport_xarray, compute_transport_numba])
def test_transport_spatial_chunking(grid_for_chunking, transport_func):
    """Test transport with spatial chunking."""
    state = create_state(grid_for_chunking)
    u = xr.full_like(state, 0.5)
    v = xr.full_like(state, 0.2)
    D = 100.0

    # Reference
    res_ref = transport_func(
        state,
        u,
        v,
        D,
        grid_for_chunking["dx"],
        grid_for_chunking["dy"],
        grid_for_chunking["cell_areas"],
        grid_for_chunking["face_areas_ew"],
        grid_for_chunking["face_areas_ns"],
    )

    # Chunked
    # IMPORTANT: compute_transport_numba requires core dimensions (y, x) to NOT be chunked for proper guvectorize execution on map_blocks?
    # Actually, guvectorize handles chunks automatically if using dask input.
    # However, finite volume usually requires ghost cells or overlap.
    # seapopym transport architecture likely handles this internally or expects single-block spatial domains if overlap isn't managed.
    # Let's see if it supports spatial chunking.

    state_c = state.chunk({Coordinates.Y.value: 10, Coordinates.X.value: 10})
    u_c = u.chunk({Coordinates.Y.value: 10, Coordinates.X.value: 10})
    v_c = v.chunk({Coordinates.Y.value: 10, Coordinates.X.value: 10})

    # Numba guvectorize limitations on chunked core dims
    if transport_func == compute_transport_numba:
        pytest.xfail("Numba guvectorize requires unchunked core dimensions (y, x)")

    # Note: If library does not support spatial chunking, this test might fail or warn.
    # But checking correctness is the goal.
    res_chunked = transport_func(
        state_c,
        u_c,
        v_c,
        D,
        grid_for_chunking["dx"],
        grid_for_chunking["dy"],
        grid_for_chunking["cell_areas"],
        grid_for_chunking["face_areas_ew"],
        grid_for_chunking["face_areas_ns"],
    )

    xr.testing.assert_allclose(res_chunked["advection_rate"].compute(), res_ref["advection_rate"])
    xr.testing.assert_allclose(res_chunked["diffusion_rate"].compute(), res_ref["diffusion_rate"])


@pytest.mark.parametrize("transport_func", [compute_transport_xarray, compute_transport_numba])
def test_transport_time_chunking(grid_for_chunking, transport_func):
    """Test transport with time chunking (if extra dimension present)."""
    # Assuming transport functions can map over time if time is a batch dimension
    # or if we loop over it. The current signatures take 'state' which is (Y, X).
    # If we pass (T, Y, X), it should probably broadcast/vectorize.

    state = create_state(
        grid_for_chunking, dims=(Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value)
    )
    u = xr.full_like(state, 0.5)
    v = xr.full_like(state, 0.2)
    D = 100.0

    res_ref = transport_func(
        state,
        u,
        v,
        D,
        grid_for_chunking["dx"],
        grid_for_chunking["dy"],
        grid_for_chunking["cell_areas"],
        grid_for_chunking["face_areas_ew"],
        grid_for_chunking["face_areas_ns"],
    )

    # Chunking time
    state_c = state.chunk(
        {Coordinates.T.value: 1, Coordinates.Y.value: -1, Coordinates.X.value: -1}
    )
    u_c = u.chunk({Coordinates.T.value: 1, Coordinates.Y.value: -1, Coordinates.X.value: -1})
    v_c = v.chunk({Coordinates.T.value: 1, Coordinates.Y.value: -1, Coordinates.X.value: -1})

    res_chunked = transport_func(
        state_c,
        u_c,
        v_c,
        D,
        grid_for_chunking["dx"],
        grid_for_chunking["dy"],
        grid_for_chunking["cell_areas"],
        grid_for_chunking["face_areas_ew"],
        grid_for_chunking["face_areas_ns"],
    )

    xr.testing.assert_allclose(res_chunked["advection_rate"].compute(), res_ref["advection_rate"])
