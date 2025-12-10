"""Performance benchmark suite for Seapopym transport and biology computations."""

import os
import sys
import time
import tracemalloc

# Add current directory to path
sys.path.append(os.getcwd())

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from seapopym.lmtl.core import compute_production_dynamics  # noqa: E402
from seapopym.standard.coordinates import Coordinates  # noqa: E402
from seapopym.transport.boundary import BoundaryConditions, BoundaryType  # noqa: E402
from seapopym.transport.core import (  # noqa: E402
    compute_advection_tendency,
    compute_diffusion_tendency,
)
from seapopym.transport.grid import (  # noqa: E402
    compute_spherical_cell_areas,
    compute_spherical_dx,
    compute_spherical_dy,
    compute_spherical_face_areas_ew,
    compute_spherical_face_areas_ns,
)


def benchmark() -> None:
    """Run performance benchmarks for transport and biology computations."""
    print("--- Seapopym Transport Benchmark ---")

    # Setup Grid
    # 1 degree resolution: 180x360
    # 1/4 degree resolution: 720x1440 (more realistic for global high res)
    # Let's start with 1 degree for quick test, then maybe scale up
    nlat, nlon = 180, 360
    lats = np.linspace(-89.5, 89.5, nlat)
    lons = np.linspace(0.5, 359.5, nlon)

    lats_da = xr.DataArray(lats, dims=Coordinates.Y.value, coords={Coordinates.Y.value: lats})
    lons_da = xr.DataArray(lons, dims=Coordinates.X.value, coords={Coordinates.X.value: lons})

    print(f"Grid Size: {nlat}x{nlon}")
    print("Computing grid metrics...")

    cell_areas = compute_spherical_cell_areas(lats_da, lons_da)
    face_areas_ew = compute_spherical_face_areas_ew(lats_da, lons_da)
    face_areas_ns = compute_spherical_face_areas_ns(lats_da, lons_da)

    # Setup State and Velocity
    # Typical dims: Time, Functional Group, Cohort, Depth, Lat, Lon
    # But transport is usually per-layer or 3D.
    # Let's assume (Time, Depth, Lat, Lon) or even (Time, Cohort, Lat, Lon)
    # Transport function handles extra dims automatically.

    ntime = 5
    ndepth = 3
    ncohort = 4

    dims = ("time", "depth", "cohort", Coordinates.Y.value, Coordinates.X.value)
    shape = (ntime, ndepth, ncohort, nlat, nlon)

    coord_dict = {
        "time": np.arange(ntime),
        "depth": np.arange(ndepth),
        "cohort": np.arange(ncohort),
        Coordinates.Y.value: lats,
        Coordinates.X.value: lons,
    }

    print(f"Creating Data with shape {shape} (Total elements: {np.prod(shape):,})...")
    state = xr.DataArray(np.random.rand(*shape).astype(np.float32), dims=dims, coords=coord_dict)

    # Velocity usually (Time, Depth, Lat, Lon)
    u_dims = ("time", "depth", Coordinates.Y.value, Coordinates.X.value)
    u_shape = (ntime, ndepth, nlat, nlon)
    u_coords = {k: v for k, v in coord_dict.items() if k in u_dims}

    u = xr.DataArray(np.random.rand(*u_shape).astype(np.float32), dims=u_dims, coords=u_coords)
    v = xr.DataArray(np.random.rand(*u_shape).astype(np.float32), dims=u_dims, coords=u_coords)

    bc = BoundaryConditions(
        north=BoundaryType.CLOSED,
        south=BoundaryType.CLOSED,
        east=BoundaryType.PERIODIC,
        west=BoundaryType.PERIODIC,
    )

    print("\n--- Benchmarking Advection ---")
    tracemalloc.start()
    start_time = time.time()

    res = compute_advection_tendency(
        state=state,
        u=u,
        v=v,
        cell_areas=cell_areas,
        face_areas_ew=face_areas_ew,
        face_areas_ns=face_areas_ns,
        boundary_conditions=bc,
    )

    # Force computation
    # The output should have same dims as state
    _ = res["advection_rate"].values

    end_time = time.time()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    duration = end_time - start_time
    print(f"Advection Execution Time: {duration:.4f} s")
    print(f"Memory Peak: {peak / 1024 / 1024:.2f} MB")
    print(f"Throughput: {np.prod(shape) / duration / 1e6:.2f} M_elements/s")

    print("\n--- Benchmarking Diffusion ---")
    # Diffusion needs dx, dy, D
    dx = compute_spherical_dx(lats_da, lons_da)
    dy = compute_spherical_dy(lats_da, lons_da)
    D = 1000.0

    tracemalloc.start()
    start_time = time.time()

    res_diff = compute_diffusion_tendency(state=state, D=D, dx=dx, dy=dy, boundary_conditions=bc)
    _ = res_diff["diffusion_rate"].values

    end_time = time.time()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    duration = end_time - start_time
    print(f"Diffusion Execution Time: {duration:.4f} s")
    print(f"Memory Peak: {peak / 1024 / 1024:.2f} MB")

    print("\n--- Benchmarking Biology (Production Dynamics) ---")

    # Needs: production, recruitment_age, cohort_ages, dt
    # Production is state (same shape)
    # Recruitment age (Time, Lat, Lon)? Or just scalar?
    # Dim handling in compute_production_dynamics relies on xarray broadcasting

    recruitment_age = xr.DataArray(
        np.ones((ntime, nlat, nlon)) * 10.0, dims=("time", Coordinates.Y.value, Coordinates.X.value)
    )
    # Cohort ages: (cohort)
    cohort_ages = xr.DataArray(
        np.linspace(0, 100, ncohort), dims="cohort", coords={"cohort": coord_dict["cohort"]}
    )
    dt = 86400.0

    tracemalloc.start()
    start_time = time.time()

    res_bio = compute_production_dynamics(
        production=state, recruitment_age=recruitment_age, cohort_ages=cohort_ages, dt=dt
    )
    # Output is dict with multiple arrays
    for v in res_bio.values():
        _ = v.values

    end_time = time.time()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    duration = end_time - start_time
    print(f"Biology Execution Time: {duration:.4f} s")
    print(f"Memory Peak: {peak / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    benchmark()
