# %% [markdown]
"""
# Test: guvectorize with target="parallel"

This should give us the best of both worlds:
- Automatic broadcasting over extra dimensions (cohorts, etc.)
- Parallel execution of each 2D slice
"""

# %%
import time
import numpy as np
import xarray as xr
from numba import guvectorize, float64, int32

from seapopym.standard.coordinates import Coordinates, GridPosition
from seapopym.transport import (
    compute_spherical_cell_areas,
    compute_spherical_dx,
    compute_spherical_dy,
    compute_spherical_face_areas_ew,
    compute_spherical_face_areas_ns,
)
from seapopym.transport.numba_kernels import transport_tendency_numba

print("✅ Imports loaded")

# %%
NY, NX = 121, 191
N_COHORTS = 12
N_ITERATIONS = 20

lat = xr.DataArray(np.linspace(-60, 60, NY), dims=["y"])
lon = xr.DataArray(np.linspace(110, 290, NX), dims=["x"])

cell_areas = compute_spherical_cell_areas(lat, lon).values
face_areas_ew = compute_spherical_face_areas_ew(lat, lon)
face_areas_ns = compute_spherical_face_areas_ns(lat, lon)
dx = compute_spherical_dx(lat, lon).values
dy = compute_spherical_dy(lat, lon).values

x_face_dim = GridPosition.get_face_dim(Coordinates.X, GridPosition.LEFT)
y_face_dim = GridPosition.get_face_dim(Coordinates.Y, GridPosition.LEFT)
ew_area = face_areas_ew.isel({x_face_dim: slice(1, None)}).values
ns_area = face_areas_ns.isel({y_face_dim: slice(1, None)}).values

state = np.random.rand(N_COHORTS, NY, NX).astype(np.float64)
u = np.random.uniform(-0.5, 0.5, (NY, NX)).astype(np.float64)
v = np.random.uniform(-0.5, 0.5, (NY, NX)).astype(np.float64)
D_2d = np.full((NY, NX), 500.0, dtype=np.float64)
mask = np.ones((NY, NX), dtype=np.float64)
mask[40:50, 80:100] = 0
bc = np.array([0, 0, 0, 0], dtype=np.int32)

print(f"✅ Test data: {NY}x{NX}, {N_COHORTS} cohorts")

# %% [markdown]
"""
## Reference: Current guvectorize (nopython)
"""

# %%
adv_ref = np.zeros((NY, NX), dtype=np.float64)
diff_ref = np.zeros((NY, NX), dtype=np.float64)
transport_tendency_numba(
    state[0], u, v, D_2d, dx, dy, cell_areas, ew_area, ns_area, mask, bc, adv_ref, diff_ref
)


def benchmark_current(n_iter):
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        for c in range(N_COHORTS):
            transport_tendency_numba(
                state[c],
                u,
                v,
                D_2d,
                dx,
                dy,
                cell_areas,
                ew_area,
                ns_area,
                mask,
                bc,
                adv_ref,
                diff_ref,
            )
        times.append(time.perf_counter() - t0)
    return np.mean(times) * 1000, np.std(times) * 1000


mean_ref, std_ref = benchmark_current(N_ITERATIONS)
print(f"📊 Reference (guvectorize nopython): {mean_ref:.3f} ± {std_ref:.3f} ms")

# Store reference
adv_reference = np.zeros((N_COHORTS, NY, NX), dtype=np.float64)
diff_reference = np.zeros((N_COHORTS, NY, NX), dtype=np.float64)
for c in range(N_COHORTS):
    transport_tendency_numba(
        state[c],
        u,
        v,
        D_2d,
        dx,
        dy,
        cell_areas,
        ew_area,
        ns_area,
        mask,
        bc,
        adv_reference[c],
        diff_reference[c],
    )

# %% [markdown]
"""
## guvectorize with target="parallel"
"""


# %%
@guvectorize(
    [
        (
            float64[:, :],
            float64[:, :],
            float64[:, :],
            float64[:, :],
            float64[:, :],
            float64[:, :],
            float64[:, :],
            float64[:, :],
            float64[:, :],
            float64[:, :],
            int32[:],
            float64[:, :],
            float64[:, :],
        )
    ],
    "(y,x),(y,x),(y,x),(y,x),(y,x),(y,x),(y,x),(y,x),(y,x),(y,x),(b)->(y,x),(y,x)",
    nopython=True,
    cache=True,
    fastmath=True,
    target="parallel",  # <-- This is the key!
)
def transport_tendency_parallel(
    state,
    u,
    v,
    D,
    dx,
    dy,
    cell_areas,
    ew_area,
    ns_area,
    mask,
    bc,
    advection_tendency,
    diffusion_tendency,
):
    """Transport with target='parallel' - parallel execution of broadcast slices."""
    ny, nx = state.shape
    bc_north = bc[0]
    bc_south = bc[1]
    bc_east = bc[2]
    bc_west = bc[3]

    for j in range(ny):
        for i in range(nx):
            advection_tendency[j, i] = 0.0
            diffusion_tendency[j, i] = 0.0

            if mask[j, i] == 0:
                continue

            c_center = state[j, i]
            u_center = u[j, i]
            v_center = v[j, i]
            d_center = D[j, i]
            dx_center = dx[j, i]
            dy_center = dy[j, i]
            area = cell_areas[j, i]

            adv_div = 0.0
            diff_div = 0.0

            # East
            if i + 1 < nx:
                ip1 = i + 1
            elif bc_east == 2:
                ip1 = 0
            elif bc_east == 1:
                ip1 = i
            else:
                ip1 = -1

            if i - 1 >= 0:
                im1 = i - 1
            elif bc_west == 2:
                im1 = nx - 1
            elif bc_west == 1:
                im1 = i
            else:
                im1 = -1

            if j + 1 < ny:
                jp1 = j + 1
            elif bc_north == 2:
                jp1 = 0
            elif bc_north == 1:
                jp1 = j
            else:
                jp1 = -1

            if j - 1 >= 0:
                jm1 = j - 1
            elif bc_south == 2:
                jm1 = ny - 1
            elif bc_south == 1:
                jm1 = j
            else:
                jm1 = -1

            # East face
            if ip1 != -1 and mask[j, ip1] != 0:
                c_east = state[j, ip1]
                u_east = u[j, ip1]
                d_east = D[j, ip1]
                dx_east = dx[j, ip1]
                face_area = ew_area[j, i]
                u_face = 0.5 * (u_center + u_east)
                if u_face > 0:
                    c_up = c_center
                else:
                    c_up = c_east
                adv_div = adv_div + u_face * c_up * face_area
                d_face = 0.5 * (d_center + d_east)
                dx_face = 0.5 * (dx_center + dx_east)
                diff_div = diff_div - d_face * (c_east - c_center) / dx_face * face_area

            # West face
            if im1 != -1 and mask[j, im1] != 0:
                c_west = state[j, im1]
                u_west = u[j, im1]
                d_west = D[j, im1]
                dx_west = dx[j, im1]
                face_area = ew_area[j, im1]
                u_face = 0.5 * (u_west + u_center)
                if u_face > 0:
                    c_up = c_west
                else:
                    c_up = c_center
                adv_div = adv_div - u_face * c_up * face_area
                d_face = 0.5 * (d_west + d_center)
                dx_face = 0.5 * (dx_west + dx_center)
                diff_div = diff_div + d_face * (c_center - c_west) / dx_face * face_area

            # North face
            if jp1 != -1 and mask[jp1, i] != 0:
                c_north = state[jp1, i]
                v_north = v[jp1, i]
                d_north = D[jp1, i]
                dy_north = dy[jp1, i]
                face_area = ns_area[j, i]
                v_face = 0.5 * (v_center + v_north)
                if v_face > 0:
                    c_up = c_center
                else:
                    c_up = c_north
                adv_div = adv_div + v_face * c_up * face_area
                d_face = 0.5 * (d_center + d_north)
                dy_face = 0.5 * (dy_center + dy_north)
                diff_div = diff_div - d_face * (c_north - c_center) / dy_face * face_area

            # South face
            if jm1 != -1 and mask[jm1, i] != 0:
                c_south = state[jm1, i]
                v_south = v[jm1, i]
                d_south = D[jm1, i]
                dy_south = dy[jm1, i]
                face_area = ns_area[jm1, i]
                v_face = 0.5 * (v_south + v_center)
                if v_face > 0:
                    c_up = c_south
                else:
                    c_up = c_center
                adv_div = adv_div - v_face * c_up * face_area
                d_face = 0.5 * (d_south + d_center)
                dy_face = 0.5 * (dy_south + dy_center)
                diff_div = diff_div + d_face * (c_center - c_south) / dy_face * face_area

            advection_tendency[j, i] = -adv_div / area * mask[j, i]
            diffusion_tendency[j, i] = -diff_div / area * mask[j, i]


# %%
# Warmup
adv_par = np.zeros((NY, NX), dtype=np.float64)
diff_par = np.zeros((NY, NX), dtype=np.float64)
transport_tendency_parallel(
    state[0], u, v, D_2d, dx, dy, cell_areas, ew_area, ns_area, mask, bc, adv_par, diff_par
)


def benchmark_parallel(n_iter):
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        for c in range(N_COHORTS):
            transport_tendency_parallel(
                state[c],
                u,
                v,
                D_2d,
                dx,
                dy,
                cell_areas,
                ew_area,
                ns_area,
                mask,
                bc,
                adv_par,
                diff_par,
            )
        times.append(time.perf_counter() - t0)
    return np.mean(times) * 1000, np.std(times) * 1000


mean_par, std_par = benchmark_parallel(N_ITERATIONS)

# Validate
adv_par_results = np.zeros((N_COHORTS, NY, NX), dtype=np.float64)
diff_par_results = np.zeros((N_COHORTS, NY, NX), dtype=np.float64)
for c in range(N_COHORTS):
    transport_tendency_parallel(
        state[c],
        u,
        v,
        D_2d,
        dx,
        dy,
        cell_areas,
        ew_area,
        ns_area,
        mask,
        bc,
        adv_par_results[c],
        diff_par_results[c],
    )

adv_diff = np.abs(adv_reference - adv_par_results).max()
diff_diff = np.abs(diff_reference - diff_par_results).max()

print(f"📊 guvectorize target='parallel': {mean_par:.3f} ± {std_par:.3f} ms")
print(f"   Speedup vs reference: {mean_ref / mean_par:.2f}x")
print(f"   Max diff: adv={adv_diff:.2e}, diff={diff_diff:.2e}")
print(f"   Valid: {'✅' if adv_diff < 1e-10 and diff_diff < 1e-10 else '❌'}")

# %% [markdown]
"""
## Test with broadcasting on 3D array directly
"""

# %%
# Now test with the full 3D array - broadcasting should work automatically
print("\n📊 Testing 3D broadcasting...")

adv_3d = np.zeros((N_COHORTS, NY, NX), dtype=np.float64)
diff_3d = np.zeros((N_COHORTS, NY, NX), dtype=np.float64)

# Single call with 3D state - should broadcast automatically
t0 = time.perf_counter()
transport_tendency_parallel(
    state, u, v, D_2d, dx, dy, cell_areas, ew_area, ns_area, mask, bc, adv_3d, diff_3d
)
t_3d = (time.perf_counter() - t0) * 1000

# Validate
adv_diff_3d = np.abs(adv_reference - adv_3d).max()
diff_diff_3d = np.abs(diff_reference - diff_3d).max()

print(f"   Single 3D call: {t_3d:.3f} ms")
print(f"   Max diff: adv={adv_diff_3d:.2e}, diff={diff_diff_3d:.2e}")
print(f"   Valid: {'✅' if adv_diff_3d < 1e-10 and diff_diff_3d < 1e-10 else '❌'}")


# Benchmark 3D call
def benchmark_3d(n_iter):
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        transport_tendency_parallel(
            state, u, v, D_2d, dx, dy, cell_areas, ew_area, ns_area, mask, bc, adv_3d, diff_3d
        )
        times.append(time.perf_counter() - t0)
    return np.mean(times) * 1000, np.std(times) * 1000


mean_3d, std_3d = benchmark_3d(N_ITERATIONS)
print(f"📊 3D broadcasting call: {mean_3d:.3f} ± {std_3d:.3f} ms")
print(f"   Speedup vs loop: {mean_par / mean_3d:.2f}x")

# %%
print("\n" + "=" * 60)
print("📊 SUMMARY")
print("=" * 60)
print(f"Reference (guvectorize nopython): {mean_ref:.3f} ms")
print(f"target='parallel' (loop):         {mean_par:.3f} ms (speedup: {mean_ref / mean_par:.2f}x)")
print(f"target='parallel' (3D broadcast): {mean_3d:.3f} ms (speedup: {mean_ref / mean_3d:.2f}x)")
print()
if mean_3d < mean_ref:
    print(f"✅ target='parallel' with 3D broadcast is the best option!")
    print(f"   Total speedup: {mean_ref / mean_3d:.1f}x faster than reference")
