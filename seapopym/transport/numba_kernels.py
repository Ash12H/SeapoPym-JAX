"""New Numba-accelerated transport kernels with separate flux computation."""

from numba import float32, float64, guvectorize, int32  # type: ignore[import-not-found]


@guvectorize(
    [
        (
            float32[:, :],  # state
            float32[:, :],  # u
            float32[:, :],  # v
            float32[:, :],  # ew_area
            float32[:, :],  # ns_area
            float32[:, :],  # mask
            int32[:],  # bc
            float32[:, :],  # flux_east (output)
            float32[:, :],  # flux_west (output)
            float32[:, :],  # flux_north (output)
            float32[:, :],  # flux_south (output)
        ),
        (
            float64[:, :],
            float64[:, :],
            float64[:, :],
            float64[:, :],
            float64[:, :],
            float64[:, :],
            int32[:],
            float64[:, :],
            float64[:, :],
            float64[:, :],
            float64[:, :],
        ),
    ],
    "(y,x),(y,x),(y,x),(y,x),(y,x),(y,x),(b)->(y,x),(y,x),(y,x),(y,x)",
    nopython=True,
    cache=True,
)
def advection_flux_numba(state, u, v, ew_area, ns_area, mask, bc, flux_e, flux_w, flux_n, flux_s):  # type: ignore[no-untyped-def]
    """Compute advection fluxes at all cell faces using upwind scheme.

    Args:
        state: Concentration (ny, nx)
        u: Zonal velocity (ny, nx)
        v: Meridional velocity (ny, nx)
        ew_area: East face area of each cell (ny, nx)
        ns_area: North face area of each cell (ny, nx)
        mask: Binary mask for valid cells (ny, nx)
        bc: Boundary conditions [north, south, east, west]
        flux_e: Output - East flux (ny, nx)
        flux_w: Output - West flux (ny, nx)
        flux_n: Output - North flux (ny, nx)
        flux_s: Output - South flux (ny, nx)
    """
    ny, nx = state.shape

    # Boundary Codes: 0=Closed, 1=Periodic
    bc_north = bc[0]
    bc_south = bc[1]
    bc_east = bc[2]
    bc_west = bc[3]

    for j in range(ny):
        for i in range(nx):
            # Initialize all fluxes to zero
            flux_e[j, i] = 0.0
            flux_w[j, i] = 0.0
            flux_n[j, i] = 0.0
            flux_s[j, i] = 0.0

            # Skip if land cell
            if mask[j, i] == 0:
                continue

            c_center = state[j, i]
            u_center = u[j, i]
            v_center = v[j, i]

            # --- INDICES ---
            # East (i+1)
            ip1 = i + 1
            if ip1 >= nx:
                if bc_east == 1:  # noqa: SIM108
                    ip1 = 0
                else:
                    ip1 = -1  # Closed

            # West (i-1)
            im1 = i - 1
            if im1 < 0:
                if bc_west == 1:  # noqa: SIM108
                    im1 = nx - 1
                else:
                    im1 = -1

            # North (j+1)
            jp1 = j + 1
            if jp1 >= ny:
                if bc_north == 1:  # noqa: SIM108
                    jp1 = 0
                else:
                    jp1 = -1

            # South (j-1)
            jm1 = j - 1
            if jm1 < 0:
                if bc_south == 1:  # noqa: SIM108
                    jm1 = ny - 1
                else:
                    jm1 = -1

            # --- EAST FLUX ---
            if ip1 != -1 and mask[j, ip1] != 0:
                u_east = u[j, ip1]
                u_face_east = 0.5 * (u_center + u_east)

                # Upwind
                if u_face_east > 0:  # noqa: SIM108
                    c_up = c_center
                else:
                    c_up = state[j, ip1]

                flux_e[j, i] = u_face_east * c_up * ew_area[j, i]

            # --- WEST FLUX ---
            if im1 != -1 and mask[j, im1] != 0:
                u_west = u[j, im1]
                u_face_west = 0.5 * (u_west + u_center)

                # Upwind
                if u_face_west > 0:  # noqa: SIM108
                    c_up = state[j, im1]
                else:
                    c_up = c_center

                flux_w[j, i] = u_face_west * c_up * ew_area[j, im1]

            # --- NORTH FLUX ---
            if jp1 != -1 and mask[jp1, i] != 0:
                v_north = v[jp1, i]
                v_face_north = 0.5 * (v_center + v_north)

                if v_face_north > 0:  # noqa: SIM108
                    c_up = c_center
                else:
                    c_up = state[jp1, i]

                flux_n[j, i] = v_face_north * c_up * ns_area[j, i]

            # --- SOUTH FLUX ---
            if jm1 != -1 and mask[jm1, i] != 0:
                v_south = v[jm1, i]
                v_face_south = 0.5 * (v_south + v_center)

                if v_face_south > 0:  # noqa: SIM108
                    c_up = state[jm1, i]
                else:
                    c_up = c_center

                flux_s[j, i] = v_face_south * c_up * ns_area[jm1, i]


@guvectorize(
    [
        (
            float32[:, :],  # state
            float32[:, :],  # D
            float32[:, :],  # dx
            float32[:, :],  # dy
            float32[:, :],  # ew_area
            float32[:, :],  # ns_area
            float32[:, :],  # mask
            int32[:],  # bc
            float32[:, :],  # flux_east (output)
            float32[:, :],  # flux_west (output)
            float32[:, :],  # flux_north (output)
            float32[:, :],  # flux_south (output)
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
            float64[:, :],
            float64[:, :],
            float64[:, :],
        ),
    ],
    "(y,x),(y,x),(y,x),(y,x),(y,x),(y,x),(y,x),(b)->(y,x),(y,x),(y,x),(y,x)",
    nopython=True,
    cache=True,
)
def diffusion_flux_numba(state, D, dx, dy, ew_area, ns_area, mask, bc, flux_e, flux_w, flux_n, flux_s):  # type: ignore[no-untyped-def]
    """Compute diffusion fluxes at all cell faces.

    Args:
        state: Concentration (ny, nx)
        D: Diffusion coefficient (ny, nx)
        dx: Grid spacing x (ny, nx)
        dy: Grid spacing y (ny, nx)
        ew_area: East face area of each cell (ny, nx)
        ns_area: North face area of each cell (ny, nx)
        mask: Binary mask for valid cells (ny, nx)
        bc: Boundary conditions [north, south, east, west]
        flux_e: Output - East flux (ny, nx)
        flux_w: Output - West flux (ny, nx)
        flux_n: Output - North flux (ny, nx)
        flux_s: Output - South flux (ny, nx)
    """
    ny, nx = state.shape

    bc_north = bc[0]
    bc_south = bc[1]
    bc_east = bc[2]
    bc_west = bc[3]

    for j in range(ny):
        for i in range(nx):
            # Initialize
            flux_e[j, i] = 0.0
            flux_w[j, i] = 0.0
            flux_n[j, i] = 0.0
            flux_s[j, i] = 0.0

            if mask[j, i] == 0:
                continue

            c_center = state[j, i]
            d_center = D[j, i]
            dx_center = dx[j, i]
            dy_center = dy[j, i]

            # --- INDICES ---
            ip1 = i + 1
            if ip1 >= nx:
                if bc_east == 1:  # noqa: SIM108
                    ip1 = 0
                else:
                    ip1 = -1

            im1 = i - 1
            if im1 < 0:
                if bc_west == 1:  # noqa: SIM108
                    im1 = nx - 1
                else:
                    im1 = -1

            jp1 = j + 1
            if jp1 >= ny:
                if bc_north == 1:  # noqa: SIM108
                    jp1 = 0
                else:
                    jp1 = -1

            jm1 = j - 1
            if jm1 < 0:
                if bc_south == 1:  # noqa: SIM108
                    jm1 = ny - 1
                else:
                    jm1 = -1

            # --- EAST FLUX ---
            if ip1 != -1 and mask[j, ip1] != 0:
                d_east = D[j, ip1]
                d_face_east = 0.5 * (d_center + d_east)

                dx_east = dx[j, ip1]
                dx_face = 0.5 * (dx_center + dx_east)

                grad_x = (state[j, ip1] - c_center) / dx_face
                flux_e[j, i] = -d_face_east * grad_x * ew_area[j, i]

            # --- WEST FLUX ---
            if im1 != -1 and mask[j, im1] != 0:
                d_west = D[j, im1]
                d_face_west = 0.5 * (d_west + d_center)

                dx_west = dx[j, im1]
                dx_face = 0.5 * (dx_west + dx_center)

                grad_x = (c_center - state[j, im1]) / dx_face
                flux_w[j, i] = -d_face_west * grad_x * ew_area[j, im1]

            # --- NORTH FLUX ---
            if jp1 != -1 and mask[jp1, i] != 0:
                d_north = D[jp1, i]
                d_face_north = 0.5 * (d_center + d_north)

                dy_north = dy[jp1, i]
                dy_face = 0.5 * (dy_center + dy_north)

                grad_y = (state[jp1, i] - c_center) / dy_face
                flux_n[j, i] = -d_face_north * grad_y * ns_area[j, i]

            # --- SOUTH FLUX ---
            if jm1 != -1 and mask[jm1, i] != 0:
                d_south = D[jm1, i]
                d_face_south = 0.5 * (d_south + d_center)

                dy_south = dy[jm1, i]
                dy_face = 0.5 * (dy_south + dy_center)

                grad_y = (c_center - state[jm1, i]) / dy_face
                flux_s[j, i] = -d_face_south * grad_y * ns_area[jm1, i]
