"""Numba-accelerated transport kernels."""

from numba import float32, float64, guvectorize, int32  # type: ignore[import-not-found]

# Signature: (lat, lon), (lat, lon), ... -> (lat, lon)
# We handle multiple inputs: state, u, v, face_u, face_v, areas, etc.
# Actually, to minimize memory, we should recompute local geometry or pass it?
# Passing precomputed geometry is safer for now.

# We need to handle boundary conditions.
# Passing boundary type as integer?
# 0: Closed, 1: Periodic, 2: Open


@guvectorize(
    [
        (
            float32[:, :],
            float32[:, :],
            float32[:, :],
            float32[:, :],
            float32[:, :],
            float32[:, :],
            float32[:, :],
            int32[:],
            float32[:, :],
        ),
        (
            float64[:, :],
            float64[:, :],
            float64[:, :],
            float64[:, :],
            float64[:, :],
            float64[:, :],
            float64[:, :],
            int32[:],
            float64[:, :],
        ),
    ],
    "(y, x), (y, x), (y, x), (y, x), (y, x), (y, x), (y, x), (b) -> (y, x)",
    nopython=True,
    cache=True,
)
def advection_upwind_numba(state, u, v, ew_area, ns_area, cell_area, mask, bc, out):  # type: ignore[no-untyped-def]
    """Compute advection using upwind scheme with Numba.

    Dimensions:
    - y: Latitude (0 to ny-1)
    - x: Longitude (0 to nx-1)

    Args:
        state: Concentration (ny, nx)
        u: Zonal velocity (ny, nx)
        v: Meridional velocity (ny, nx)
        ew_area: East face area of each cell (ny, nx)
        ns_area: North face area of each cell (ny, nx)
        cell_area: Area of each cell (ny, nx)
        mask: Binary mask for valid cells (ny, nx)
        bc: Boundary conditions array [north, south, east, west] (int encodings)
        out: Output array for advection rate (ny, nx)
    """
    ny, nx = state.shape

    # Boundary Codes
    # 0: Closed, 1: Periodic
    bc_north = bc[0]
    bc_south = bc[1]
    bc_east = bc[2]
    bc_west = bc[3]

    for j in range(ny):
        for i in range(nx):
            # --- INDICES ---
            # East Neighbor (i+1)
            ip1 = i + 1
            if ip1 >= nx:
                if bc_east == 1:  # Periodic  # noqa: SIM108
                    ip1 = 0
                else:
                    ip1 = -1  # Closed/Ghost

            # West Neighbor (i-1)
            im1 = i - 1
            if im1 < 0:
                if bc_west == 1:  # Periodic  # noqa: SIM108
                    im1 = nx - 1
                else:
                    im1 = -1

            # North Neighbor (j+1) (Assuming j increases northward? Check coords)
            # Usually index 0 is South (-90) and index N is North (+90)
            # So North is j+1.
            jp1 = j + 1
            if jp1 >= ny:
                if bc_north == 1:  # noqa: SIM108
                    jp1 = 0
                else:
                    jp1 = -1

            # South Neighbor (j-1)
            jm1 = j - 1
            if jm1 < 0:
                if bc_south == 1:  # noqa: SIM108
                    jm1 = ny - 1
                else:
                    jm1 = -1

            # --- PREPARE LOCAL VALUES ---
            c_center = state[j, i]
            u_center = u[j, i]
            v_center = v[j, i]

            # --- EAST FLUX (at i+1/2) ---
            # Velocity at face: Avg(u[i], u[i+1])
            # If Boundary is closed (ip1=-1), u at face is 0.
            if ip1 == -1:
                flux_east = 0.0
            else:
                u_east = u[j, ip1]
                u_face_east = 0.5 * (u_center + u_east)

                # Upwind state
                if u_face_east > 0:  # noqa: SIM108
                    c_up = c_center
                else:
                    c_up = state[j, ip1]

                # Area East (provided as input for cell i)
                area_e = ew_area[j, i]
                flux_east = u_face_east * c_up * area_e

            # --- WEST FLUX (at i-1/2) ---
            # Velocity at face: Avg(u[i-1], u[i])
            if im1 == -1:
                flux_west = 0.0
            else:
                u_west = u[j, im1]
                u_face_west = 0.5 * (u_west + u_center)

                # Upwind state
                if u_face_west > 0:  # noqa: SIM108
                    c_up = state[j, im1]
                else:
                    c_up = c_center

                # Area West is East area of neighbor (i-1)
                # Need to access ew_area[j, im1]
                area_w = ew_area[j, im1]
                flux_west = u_face_west * c_up * area_w

            # --- NORTH FLUX (at j+1/2) ---
            if jp1 == -1:
                flux_north = 0.0
            else:
                v_north = v[jp1, i]
                v_face_north = 0.5 * (v_center + v_north)

                if v_face_north > 0:  # noqa: SIM108
                    c_up = c_center
                else:
                    c_up = state[jp1, i]

                area_n = ns_area[j, i]
                flux_north = v_face_north * c_up * area_n

            # --- SOUTH FLUX (at j-1/2) ---
            if jm1 == -1:
                flux_south = 0.0
            else:
                v_south = v[jm1, i]
                v_face_south = 0.5 * (v_south + v_center)

                if v_face_south > 0:  # noqa: SIM108
                    c_up = state[jm1, i]
                else:
                    c_up = c_center

                area_s = ns_area[jm1, i]
                flux_south = v_face_south * c_up * area_s

            # --- DIVERGENCE ---
            # Div = (East - West + North - South) / Volume
            # Volume = Cell Area
            div = (flux_east - flux_west + flux_north - flux_south) / cell_area[j, i]

            # Tendency = -Divergence
            val = -div

            # Apply Mask
            if mask[j, i] == 0:
                val = 0.0

            out[j, i] = val
