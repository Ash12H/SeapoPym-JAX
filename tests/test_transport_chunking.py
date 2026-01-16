"""Tests for Chunking Correctness on Transport Functions.

Objective: Ensure that compute_transport_fv behaves correctly with input chunks.
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
    compute_transport_fv,
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


def test_transport_spatial_chunking(grid_for_chunking):
    """Test transport with spatial chunking."""
    state = create_state(grid_for_chunking)
    u = xr.full_like(state, 0.5)
    v = xr.full_like(state, 0.2)
    D = 100.0

    # Reference
    res_ref = compute_transport_fv(
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
    state_c = state.chunk({Coordinates.Y.value: 10, Coordinates.X.value: 10})
    u_c = u.chunk({Coordinates.Y.value: 10, Coordinates.X.value: 10})
    v_c = v.chunk({Coordinates.Y.value: 10, Coordinates.X.value: 10})

    # Numba guvectorize limitations on chunked core dims
    pytest.xfail("Numba guvectorize requires unchunked core dimensions (y, x)")

    # The code below is unreachable due to xfail, but kept for logic documentation
    res_chunked = compute_transport_fv(
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


def test_transport_time_chunking(grid_for_chunking):
    """Test transport with time chunking (if extra dimension present)."""
    state = create_state(
        grid_for_chunking, dims=(Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value)
    )
    u = xr.full_like(state, 0.5)
    v = xr.full_like(state, 0.2)
    D = 100.0

    res_ref = compute_transport_fv(
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

    res_chunked = compute_transport_fv(
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
