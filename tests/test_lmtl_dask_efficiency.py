"""Tests for Dask task graph efficiency in LMTL functions.

Objective: Detect task explosions (e.g. O(N^2) scaling) and unexpected rechunking operations.
"""

import dask
import numpy as np
import xarray as xr

from seapopym.lmtl.core import compute_production_dynamics
from seapopym.standard.coordinates import Coordinates


def count_tasks(dask_obj):
    """Count the number of tasks in a dask graph."""
    if isinstance(dask_obj, xr.DataArray | xr.Dataset):
        return len(dask_obj.__dask_graph__())
    elif hasattr(dask_obj, "dask"):
        return len(dask_obj.dask)
    return 0


def create_production_array(n_cohorts=100, shape=(100, 100), n_times=5):
    """Helper to create a dummy production array."""
    times = np.arange(n_times)
    lats = np.linspace(-40, 40, shape[0])
    lons = np.linspace(140, 220, shape[1])
    cohorts = np.arange(n_cohorts, dtype=float)

    return xr.DataArray(
        dask.array.ones((n_times, shape[0], shape[1], n_cohorts), chunks=(1, 50, 50, 10)),
        coords={
            Coordinates.T: times,
            Coordinates.Y: lats,
            Coordinates.X: lons,
            "cohort": cohorts,
        },
        dims=[Coordinates.T, Coordinates.Y, Coordinates.X, "cohort"],
        name="production",
    )


def test_production_task_count_linear():
    """Verify that task count scales linearly with the number of chunks."""
    # We vary the number of chunks in the 'cohort' dimension
    # Total cohorts = 100

    # Large chunks -> fewer chunks -> fewer tasks
    chunks_large = {"cohort": 50}  # 2 chunks
    prod_large = create_production_array(n_cohorts=100).chunk(chunks_large)

    # Small chunks -> more chunks -> more tasks
    chunks_small = {"cohort": 10}  # 10 chunks
    prod_small = create_production_array(n_cohorts=100).chunk(chunks_small)

    # Inputs
    recruitment_age = xr.DataArray(
        np.ones((5, 100, 100)) * 5, dims=[Coordinates.T, Coordinates.Y, Coordinates.X]
    ).chunk({Coordinates.T: 1, Coordinates.Y: 50, Coordinates.X: 50})
    cohorts = prod_large.coords["cohort"]
    dt = 1.0

    # Compute graph
    res_large = compute_production_dynamics(prod_large, recruitment_age, cohorts, dt)[
        "production_tendency"
    ]
    res_small = compute_production_dynamics(prod_small, recruitment_age, cohorts, dt)[
        "production_tendency"
    ]

    tasks_large = count_tasks(res_large)
    tasks_small = count_tasks(res_small)

    ratio_chunks = (100 / 10) / (100 / 50)  # 10 / 2 = 5
    ratio_tasks = tasks_small / tasks_large

    # We expect roughly linear scaling.
    # Allow some overhead (offset), so check if ratio is roughly consistent
    # It won't be exactly 5.0 because of common tasks (like coords), but should be close.
    # If it were O(N^2), ratio would be ~25.

    print(f"Tasks (2 chunks): {tasks_large}")
    print(f"Tasks (10 chunks): {tasks_small}")
    print(f"Ratio Tasks: {ratio_tasks:.2f} (Expected ~{ratio_chunks})")

    assert (
        ratio_tasks < ratio_chunks * 1.5
    ), "Task count scaling looks super-linear (potential explosion)"
    assert ratio_tasks > ratio_chunks * 0.5, "Task count scaling looks too flat (unexpected)"


def test_minimal_rechunking():
    """Ensure that compute_production_dynamics rechunking is minimal (linear O(N)).

    Standard .shift() implies communication across chunk boundaries, which Dask
    may label as 'rechunk-merge'. We expect O(N_chunks) such operations, not O(N^2).
    """
    chunks = {"cohort": 5}
    prod = create_production_array(n_cohorts=20).chunk(chunks)  # 4 chunks
    recruitment_age = xr.DataArray(
        np.ones((5, 100, 100)) * 5, dims=[Coordinates.T, Coordinates.Y, Coordinates.X]
    ).chunk({Coordinates.T: 1, Coordinates.Y: 50, Coordinates.X: 50})
    cohorts = prod.coords["cohort"]
    dt = 1.0

    res = compute_production_dynamics(prod, recruitment_age, cohorts, dt)["production_tendency"]

    graph = res.__dask_graph__()
    # In recent Dask versions, rechunk operations might be named differently or hidden in layers.
    # We check keys.
    keys = list(graph.keys())
    # Note: HighLevelGraph layers might be named "rechunk-merge-..."
    if hasattr(graph, "layers"):
        layer_names = list(graph.layers.keys())
        # Filter for rechunk ops
        rechunk_ops = [k for k in layer_names if "rechunk" in k]
    else:
        rechunk_ops = [k for k in keys if isinstance(k, str) and "rechunk" in k]

    num_rechunks = len(rechunk_ops)
    num_chunks = 20 / 5  # 4

    # We expect roughly 1 or 2 rechunk layers per shift operation.
    # There are 2 shifts (influx, prev_recruitment).
    # Plus potentially broadcast alignment.
    # If it were O(all-to-all), it would be huge or complex.
    # Count of 5 for 4 chunks is totally acceptable (linear).

    print(f"Rechunk ops count: {num_rechunks}")

    # Assert reasonable limit
    assert (
        num_rechunks <= num_chunks * 3
    ), f"Too many rechunk ops ({num_rechunks}) for {num_chunks} chunks"
