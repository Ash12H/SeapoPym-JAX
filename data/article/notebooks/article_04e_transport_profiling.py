# %% [markdown]
"""
# Transport Function Profiling

Microbenchmark to identify bottlenecks in `compute_transport_fv`.

Measures time spent in each phase:
1. Data preparation (fillna, ensure_dataarray, face areas)
2. Advection kernel (Numba)
3. Diffusion kernel (Numba)
4. Divergence computation
5. Transpose
"""

# %%
import time
import numpy as np
import xarray as xr
from functools import wraps

from seapopym.standard.coordinates import Coordinates, GridPosition
from seapopym.transport import (
    compute_spherical_cell_areas,
    compute_spherical_dx,
    compute_spherical_dy,
    compute_spherical_face_areas_ew,
    compute_spherical_face_areas_ns,
)
from seapopym.transport.numba_kernels import advection_flux_numba, diffusion_flux_numba

print("✅ Imports loaded")

# %% [markdown]
"""
## Setup Test Data
"""

# %%
# Grid size (similar to Pacific)
NY, NX = 121, 191  # Pacific grid
N_COHORTS = 12
N_ITERATIONS = 100  # For timing stability

# Coordinates
lat = xr.DataArray(np.linspace(-60, 60, NY), dims=["y"])
lon = xr.DataArray(np.linspace(110, 290, NX), dims=["x"])

# Grid metrics
cell_areas = compute_spherical_cell_areas(lat, lon).values
face_areas_ew = compute_spherical_face_areas_ew(lat, lon)
face_areas_ns = compute_spherical_face_areas_ns(lat, lon)
dx = compute_spherical_dx(lat, lon).values
dy = compute_spherical_dy(lat, lon).values

# State (cohort, Y, X) - core dims at end
state = np.random.rand(N_COHORTS, NY, NX).astype(np.float64)
state_xa = xr.DataArray(state, dims=["cohort", "y", "x"])

# Velocities (Y, X)
u = np.random.uniform(-0.5, 0.5, (NY, NX)).astype(np.float64)
v = np.random.uniform(-0.5, 0.5, (NY, NX)).astype(np.float64)

# Mask
mask = np.ones((NY, NX), dtype=np.float64)
mask[40:50, 80:100] = 0  # Some land

# Parameters
D = 500.0  # Diffusion coefficient
bc = np.array([0, 0, 0, 0], dtype=np.int32)  # Closed boundaries

# Extract face areas for Numba (aligned with cell centers)
x_face_dim = GridPosition.get_face_dim(Coordinates.X, GridPosition.LEFT)
y_face_dim = GridPosition.get_face_dim(Coordinates.Y, GridPosition.LEFT)

ew_area = face_areas_ew.isel({x_face_dim: slice(1, None)}).values
ns_area = face_areas_ns.isel({y_face_dim: slice(1, None)}).values

# Prepare D as 2D array
D_2d = np.full((NY, NX), D, dtype=np.float64)

print(f"✅ Test data prepared")
print(f"   State shape: {state.shape}")
print(f"   Grid: {NY} x {NX}")
print(f"   Cohorts: {N_COHORTS}")

# %% [markdown]
"""
## 1. Profile fillna

Measure cost of replacing NaN with 0.
"""

# %%
# Test with NaN
state_with_nan = state.copy()
state_with_nan[0, 10, 20] = np.nan
state_with_nan[5, 50, 80] = np.nan

# Using xarray fillna
state_xa_nan = xr.DataArray(state_with_nan, dims=["cohort", "y", "x"])


def profile_fillna(n_iter):
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        _ = state_xa_nan.fillna(0.0)
        times.append(time.perf_counter() - t0)
    return np.mean(times) * 1000, np.std(times) * 1000


# Warmup
_ = state_xa_nan.fillna(0.0)

mean_ms, std_ms = profile_fillna(N_ITERATIONS)
print(f"📊 fillna: {mean_ms:.3f} ± {std_ms:.3f} ms")


# Compare with np.nan_to_num
def profile_nan_to_num(n_iter):
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        _ = np.nan_to_num(state_with_nan, nan=0.0)
        times.append(time.perf_counter() - t0)
    return np.mean(times) * 1000, np.std(times) * 1000


mean_ms2, std_ms2 = profile_nan_to_num(N_ITERATIONS)
print(f"📊 np.nan_to_num: {mean_ms2:.3f} ± {std_ms2:.3f} ms")

# %% [markdown]
"""
## 2. Profile Numba Kernels (Advection + Diffusion)
"""

# %%
# Output arrays
flux_e = np.zeros((NY, NX), dtype=np.float64)
flux_w = np.zeros((NY, NX), dtype=np.float64)
flux_n = np.zeros((NY, NX), dtype=np.float64)
flux_s = np.zeros((NY, NX), dtype=np.float64)

# Warmup (JIT compilation)
advection_flux_numba(state[0], u, v, ew_area, ns_area, mask, bc, flux_e, flux_w, flux_n, flux_s)
diffusion_flux_numba(
    state[0], D_2d, dx, dy, ew_area, ns_area, mask, bc, flux_e, flux_w, flux_n, flux_s
)


def profile_advection_single(n_iter):
    """Profile advection for a SINGLE 2D slice."""
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        advection_flux_numba(
            state[0], u, v, ew_area, ns_area, mask, bc, flux_e, flux_w, flux_n, flux_s
        )
        times.append(time.perf_counter() - t0)
    return np.mean(times) * 1000, np.std(times) * 1000


def profile_diffusion_single(n_iter):
    """Profile diffusion for a SINGLE 2D slice."""
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        diffusion_flux_numba(
            state[0], D_2d, dx, dy, ew_area, ns_area, mask, bc, flux_e, flux_w, flux_n, flux_s
        )
        times.append(time.perf_counter() - t0)
    return np.mean(times) * 1000, np.std(times) * 1000


mean_adv, std_adv = profile_advection_single(N_ITERATIONS)
print(f"📊 advection_flux_numba (1 slice): {mean_adv:.3f} ± {std_adv:.3f} ms")

mean_diff, std_diff = profile_diffusion_single(N_ITERATIONS)
print(f"📊 diffusion_flux_numba (1 slice): {mean_diff:.3f} ± {std_diff:.3f} ms")


# For all cohorts
def profile_kernels_all_cohorts(n_iter):
    """Profile both kernels for ALL cohorts."""
    times_adv = []
    times_diff = []
    for _ in range(n_iter):
        # Advection
        t0 = time.perf_counter()
        for c in range(N_COHORTS):
            advection_flux_numba(
                state[c], u, v, ew_area, ns_area, mask, bc, flux_e, flux_w, flux_n, flux_s
            )
        times_adv.append(time.perf_counter() - t0)

        # Diffusion
        t0 = time.perf_counter()
        for c in range(N_COHORTS):
            diffusion_flux_numba(
                state[c], D_2d, dx, dy, ew_area, ns_area, mask, bc, flux_e, flux_w, flux_n, flux_s
            )
        times_diff.append(time.perf_counter() - t0)

    return (
        np.mean(times_adv) * 1000,
        np.std(times_adv) * 1000,
        np.mean(times_diff) * 1000,
        np.std(times_diff) * 1000,
    )


mean_adv_all, std_adv_all, mean_diff_all, std_diff_all = profile_kernels_all_cohorts(
    N_ITERATIONS // 10
)
print(f"📊 advection_flux_numba ({N_COHORTS} cohorts): {mean_adv_all:.3f} ± {std_adv_all:.3f} ms")
print(f"📊 diffusion_flux_numba ({N_COHORTS} cohorts): {mean_diff_all:.3f} ± {std_diff_all:.3f} ms")

# %% [markdown]
"""
## 3. Profile Divergence Computation
"""

# %%
# Simulate flux arrays (matching what Numba would produce)
flux_adv_e = np.random.rand(N_COHORTS, NY, NX)
flux_adv_w = np.random.rand(N_COHORTS, NY, NX)
flux_adv_n = np.random.rand(N_COHORTS, NY, NX)
flux_adv_s = np.random.rand(N_COHORTS, NY, NX)


def profile_divergence_numpy(n_iter):
    """Profile divergence using pure NumPy."""
    times = []
    cell_areas_bc = cell_areas[np.newaxis, :, :]  # Broadcast to (1, NY, NX)
    for _ in range(n_iter):
        t0 = time.perf_counter()
        div = (flux_adv_e - flux_adv_w + flux_adv_n - flux_adv_s) / cell_areas_bc
        times.append(time.perf_counter() - t0)
    return np.mean(times) * 1000, np.std(times) * 1000


mean_div, std_div = profile_divergence_numpy(N_ITERATIONS)
print(f"📊 divergence (NumPy, {N_COHORTS} cohorts): {mean_div:.3f} ± {std_div:.3f} ms")

# %% [markdown]
"""
## 4. Profile Transpose
"""

# %%
result = np.random.rand(NY, NX, N_COHORTS)  # Wrong order
result_xa = xr.DataArray(result, dims=["y", "x", "cohort"])


def profile_transpose(n_iter):
    """Profile xarray transpose."""
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        _ = result_xa.transpose("cohort", "y", "x")
        times.append(time.perf_counter() - t0)
    return np.mean(times) * 1000, np.std(times) * 1000


mean_tr, std_tr = profile_transpose(N_ITERATIONS)
print(f"📊 transpose (xarray): {mean_tr:.3f} ± {std_tr:.3f} ms")


# NumPy transpose for comparison
def profile_transpose_numpy(n_iter):
    """Profile NumPy transpose."""
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        _ = result.transpose(2, 0, 1)
        times.append(time.perf_counter() - t0)
    return np.mean(times) * 1000, np.std(times) * 1000


mean_tr_np, std_tr_np = profile_transpose_numpy(N_ITERATIONS)
print(f"📊 transpose (NumPy): {mean_tr_np:.3f} ± {std_tr_np:.3f} ms")

# %% [markdown]
"""
## 5. Profile Full compute_transport_fv
"""

# %%
from seapopym.transport import compute_transport_fv, BoundaryType

# Prepare xarray inputs
state_xa = xr.DataArray(state, dims=["cohort", "y", "x"])
u_xa = xr.DataArray(u, dims=["y", "x"])
v_xa = xr.DataArray(v, dims=["y", "x"])
dx_xa = xr.DataArray(dx, dims=["y", "x"])
dy_xa = xr.DataArray(dy, dims=["y", "x"])
cell_areas_xa = xr.DataArray(cell_areas, dims=["y", "x"])
mask_xa = xr.DataArray(mask, dims=["y", "x"])

# Warmup
_ = compute_transport_fv(
    state_xa,
    u_xa,
    v_xa,
    D,
    dx_xa,
    dy_xa,
    cell_areas_xa,
    face_areas_ew,
    face_areas_ns,
    mask_xa,
    BoundaryType.CLOSED,
    BoundaryType.CLOSED,
    BoundaryType.CLOSED,
    BoundaryType.CLOSED,
)


def profile_full_transport(n_iter):
    """Profile full compute_transport_fv."""
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        _ = compute_transport_fv(
            state_xa,
            u_xa,
            v_xa,
            D,
            dx_xa,
            dy_xa,
            cell_areas_xa,
            face_areas_ew,
            face_areas_ns,
            mask_xa,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
        )
        times.append(time.perf_counter() - t0)
    return np.mean(times) * 1000, np.std(times) * 1000


mean_full, std_full = profile_full_transport(N_ITERATIONS // 10)
print(f"📊 compute_transport_fv (full): {mean_full:.3f} ± {std_full:.3f} ms")

# %% [markdown]
"""
## Summary
"""

# %%
print("\n" + "=" * 60)
print("📊 COMPONENT PROFILING SUMMARY")
print("=" * 60)
print(f"Grid: {NY} x {NX}, Cohorts: {N_COHORTS}")
print()
print("Individual components:")
print(f"  fillna:           {mean_ms:.3f} ms")
print(f"  advection (all):  {mean_adv_all:.3f} ms")
print(f"  diffusion (all):  {mean_diff_all:.3f} ms")
print(f"  divergence:       {mean_div:.3f} ms")
print(f"  transpose:        {mean_tr:.3f} ms")
print()
print(f"Full transport:     {mean_full:.3f} ms")
print()

# Estimate overhead
numba_time = mean_adv_all + mean_diff_all
overhead = mean_full - numba_time - mean_div
print(f"Numba kernels:      {numba_time:.3f} ms ({100 * numba_time / mean_full:.1f}%)")
print(f"Divergence:         {mean_div:.3f} ms ({100 * mean_div / mean_full:.1f}%)")
print(f"Overhead (est):     {overhead:.3f} ms ({100 * overhead / mean_full:.1f}%)")

# %% [markdown]
"""
## 6. Compare Original vs Optimized
"""

# %%
from seapopym.transport import compute_transport_fv_optimized, _prepare_face_areas

# Pre-compute face areas (for repeated calls optimization)
precomputed_faces = _prepare_face_areas(
    face_areas_ew, face_areas_ns, Coordinates.X.value, Coordinates.Y.value
)

# Warmup optimized version
_ = compute_transport_fv_optimized(
    state_xa,
    u_xa,
    v_xa,
    D,
    dx_xa,
    dy_xa,
    cell_areas_xa,
    face_areas_ew,
    face_areas_ns,
    mask_xa,
    BoundaryType.CLOSED,
    BoundaryType.CLOSED,
    BoundaryType.CLOSED,
    BoundaryType.CLOSED,
    skip_nan_check=True,
    precomputed_face_areas=precomputed_faces,
)


def profile_optimized_transport(n_iter):
    """Profile optimized compute_transport_fv."""
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        _ = compute_transport_fv_optimized(
            state_xa,
            u_xa,
            v_xa,
            D,
            dx_xa,
            dy_xa,
            cell_areas_xa,
            face_areas_ew,
            face_areas_ns,
            mask_xa,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            BoundaryType.CLOSED,
            skip_nan_check=True,
            precomputed_face_areas=precomputed_faces,
        )
        times.append(time.perf_counter() - t0)
    return np.mean(times) * 1000, np.std(times) * 1000


mean_opt, std_opt = profile_optimized_transport(N_ITERATIONS // 10)

# %%
# Verify results are identical
result_original = compute_transport_fv(
    state_xa,
    u_xa,
    v_xa,
    D,
    dx_xa,
    dy_xa,
    cell_areas_xa,
    face_areas_ew,
    face_areas_ns,
    mask_xa,
    BoundaryType.CLOSED,
    BoundaryType.CLOSED,
    BoundaryType.CLOSED,
    BoundaryType.CLOSED,
)

result_optimized = compute_transport_fv_optimized(
    state_xa,
    u_xa,
    v_xa,
    D,
    dx_xa,
    dy_xa,
    cell_areas_xa,
    face_areas_ew,
    face_areas_ns,
    mask_xa,
    BoundaryType.CLOSED,
    BoundaryType.CLOSED,
    BoundaryType.CLOSED,
    BoundaryType.CLOSED,
    skip_nan_check=True,
    precomputed_face_areas=precomputed_faces,
)

# Check numerical equivalence
adv_diff = np.abs(
    result_original["advection_rate"].values - result_optimized["advection_rate"].values
).max()
diff_diff = np.abs(
    result_original["diffusion_rate"].values - result_optimized["diffusion_rate"].values
).max()

print("\n" + "=" * 60)
print("📊 COMPARISON: ORIGINAL vs OPTIMIZED")
print("=" * 60)
print()
print(f"Original:   {mean_full:.3f} ± {std_full:.3f} ms")
print(f"Optimized:  {mean_opt:.3f} ± {std_opt:.3f} ms")
print()
speedup = mean_full / mean_opt
print(f"Speedup:    {speedup:.2f}x ({(1 - mean_opt / mean_full) * 100:.1f}% faster)")
print()
print("Numerical validation:")
print(f"  Max advection diff:  {adv_diff:.2e}")
print(f"  Max diffusion diff:  {diff_diff:.2e}")
print(f"  Results identical:   {'✅ YES' if adv_diff < 1e-10 and diff_diff < 1e-10 else '❌ NO'}")
