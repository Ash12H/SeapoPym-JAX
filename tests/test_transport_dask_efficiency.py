"""Tests for Transport task graph efficiency.

Objective: Ensure no task explosions or implicit rechunking in transport.
"""

import dask.array as da
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
def grid_efficiency():
    """Create a 100x100 grid for efficiency testing."""
    lats = np.linspace(-40, 40, 100)
    lons = np.linspace(140, 220, 100)
    times = np.arange(5)

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


def count_tasks(dask_obj):
    """Count the number of tasks in a dask graph."""
    if isinstance(dask_obj, xr.DataArray | xr.Dataset):
        return len(dask_obj.__dask_graph__())
    elif hasattr(dask_obj, "dask"):
        return len(dask_obj.dask)
    return 0


@pytest.mark.parametrize("transport_f", [compute_transport_xarray, compute_transport_numba])
def test_transport_task_scaling(grid_efficiency, transport_f):
    """Verify task scaling with time chunks."""
    # Scale with number of chunks
    n_times = 5
    dims = (Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value)
    shape = (n_times, 100, 100)

    # Grid data (static)
    dx = grid_efficiency["dx"]
    dy = grid_efficiency["dy"]
    areas = grid_efficiency["cell_areas"]
    Few = grid_efficiency["face_areas_ew"]
    Fns = grid_efficiency["face_areas_ns"]

    def run_transport(n_chunks):
        chunk_size = 5 // n_chunks

        state = xr.DataArray(da.ones(shape, chunks=(chunk_size, 100, 100)), dims=dims)
        u = xr.DataArray(da.ones(shape, chunks=(chunk_size, 100, 100)), dims=dims)
        v = xr.DataArray(da.ones(shape, chunks=(chunk_size, 100, 100)), dims=dims)
        D = 100.0

        # We need to broadcast static grid fields if using dask inputs with time?
        # The function handles static grid internally via broadcasting.

        res = transport_f(state, u, v, D, dx, dy, areas, Few, Fns)
        return res["advection_rate"]

    # 1 Chunk
    res_1 = run_transport(1)
    tasks_1 = count_tasks(res_1)

    # 5 Chunks
    res_5 = run_transport(5)
    tasks_5 = count_tasks(res_5)

    ratio = tasks_5 / tasks_1

    print(f"Tasks-1: {tasks_1}, Tasks-5: {tasks_5}, Ratio: {ratio}")

    # Should be close to 5
    # Be lenient: > 3 and < 8
    assert ratio > 3.0
    assert ratio < 8.0


@pytest.mark.parametrize("transport_f", [compute_transport_xarray, compute_transport_numba])
def test_transport_no_rechunking(grid_efficiency, transport_f):
    """Ensure no implicit rechunking."""
    dims = (Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value)
    shape = (5, 100, 100)
    chunk_size = 1

    state = xr.DataArray(da.ones(shape, chunks=(chunk_size, 100, 100)), dims=dims)
    u = xr.DataArray(da.ones(shape, chunks=(chunk_size, 100, 100)), dims=dims)
    v = xr.DataArray(da.ones(shape, chunks=(chunk_size, 100, 100)), dims=dims)
    D = 100.0

    res = transport_f(
        state,
        u,
        v,
        D,
        grid_efficiency["dx"],
        grid_efficiency["dy"],
        grid_efficiency["cell_areas"],
        grid_efficiency["face_areas_ew"],
        grid_efficiency["face_areas_ns"],
    )["advection_rate"]

    graph = res.__dask_graph__()

    if hasattr(graph, "layers"):
        layer_names = list(graph.layers.keys())
        rechunk_ops = [k for k in layer_names if "rechunk" in k]
    else:
        rechunk_ops = [k for k in list(graph.keys()) if isinstance(k, str) and "rechunk" in k]

    if rechunk_ops:
        pytest.fail(f"Rechunking detected: {rechunk_ops}")
