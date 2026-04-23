"""Transport functions for advection and diffusion using finite volume method.

This module provides JAX-compatible transport functions that are:
- Differentiable (compatible with jax.grad)
- Grid-agnostic (works with lat/lon, ORCA, or any structured grid)
- Broadcastable (vmap over non-spatial dimensions like cohorts)

The finite volume method ensures mass conservation. Advection uses an upwind
scheme, diffusion uses centered differences.

Geometry parameters:
- dx, dy: distances between cell centers (for gradient calculation)
- face_height: vertical length of E/W faces (for zonal flux)
- face_width: horizontal length of N/S faces (for meridional flux)
- cell_area: area of each cell (for flux divergence)

For a simple lat/lon grid: face_height = dy, face_width = dx
For curvilinear grids (ORCA): provide e2u, e1v from grid metrics.
"""

from __future__ import annotations

from enum import IntEnum

import jax
import jax.numpy as jnp

from seapopym.blueprint import functional


class BoundaryType(IntEnum):
    """Boundary condition types for transport.

    CLOSED (0): No-flux boundary (Neumann BC)
        - Flux is zero at the boundary
        - Used for solid boundaries (coastlines, domain edges)

    OPEN (1): Zero-gradient boundary
        - Neighbor value equals current cell value
        - Allows advective flux out, zero diffusive flux
        - Used for open ocean boundaries

    PERIODIC (2): Wrap-around boundary
        - West neighbor of first cell = last cell (and vice versa)
        - Used for global models (longitude wrap-around)
    """

    CLOSED = 0
    OPEN = 1
    PERIODIC = 2


# Direction config: (shift, axis, boundary_slice_fn)
# - shift: roll direction (-1 = toward higher index, +1 = toward lower index)
# - axis: -1 for X (east/west), -2 for Y (north/south)
# - boundary_slice: indexer for the boundary edge to fix in non-periodic mode
_DIRECTION_CONFIG = {
    "east": (-1, -1, lambda _: (slice(None), -1)),
    "west": (1, -1, lambda _: (slice(None), 0)),
    "north": (-1, -2, lambda _: (-1, slice(None))),
    "south": (1, -2, lambda _: (0, slice(None))),
}


# =============================================================================
# NEIGHBOR ACCESS FUNCTIONS
# =============================================================================


def _get_neighbor(
    state: jnp.ndarray,
    direction: str,
    bc: int,
) -> jnp.ndarray:
    """Get neighbor values in a given direction with boundary condition handling.

    Args:
        state: Field values (Y, X)
        direction: One of "east", "west", "north", "south"
        bc: Boundary condition (0=CLOSED, 1=OPEN, 2=PERIODIC)

    Returns:
        Neighbor values (Y, X). At the boundary:
        - CLOSED/OPEN: returns current cell value (zero gradient)
        - PERIODIC: wraps around
    """
    shift, axis, bnd_slice_fn = _DIRECTION_CONFIG[direction]
    shifted = jnp.roll(state, shift=shift, axis=axis)
    bnd = bnd_slice_fn(state)
    non_periodic = shifted.at[bnd].set(state[bnd])
    return jax.lax.cond(
        bc == BoundaryType.PERIODIC,
        lambda: shifted,
        lambda: non_periodic,
    )


def _get_boundary_mask(
    ny: int,
    nx: int,
    direction: str,
    bc: int,
) -> jnp.ndarray:
    """Get mask for a boundary (1=interior/periodic, 0=closed boundary).

    For CLOSED boundaries, the boundary edge has no flux.
    For OPEN/PERIODIC, flux is allowed everywhere.

    Args:
        ny: Number of rows.
        nx: Number of columns.
        direction: One of "east", "west", "north", "south"
        bc: Boundary condition (0=CLOSED, 1=OPEN, 2=PERIODIC)
    """
    _shift, _axis, bnd_slice_fn = _DIRECTION_CONFIG[direction]
    mask = jnp.ones((ny, nx))
    bnd = bnd_slice_fn(mask)
    mask_closed = mask.at[bnd].set(0.0)
    return jax.lax.cond(
        bc == BoundaryType.CLOSED,
        lambda: mask_closed,
        lambda: mask,
    )


# =============================================================================
# ADVECTION FLUXES (Upwind scheme)
# =============================================================================


def _compute_advection_fluxes(
    state: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    face_height: jnp.ndarray,
    face_width: jnp.ndarray,
    mask: jnp.ndarray,
    bc_north: int,
    bc_south: int,
    bc_east: int,
    bc_west: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute advection fluxes at all faces using upwind scheme.

    The upwind scheme selects the upstream concentration for stability:
    - If velocity > 0: use concentration from upstream (current) cell
    - If velocity < 0: use concentration from downstream (neighbor) cell

    Args:
        state: Concentration field (Y, X)
        u: Zonal velocity [m/s] (Y, X)
        v: Meridional velocity [m/s] (Y, X)
        face_height: Height of E/W faces [m] (Y, X)
        face_width: Width of N/S faces [m] (Y, X)
        mask: Ocean mask (1=ocean, 0=land) (Y, X)
        bc_*: Boundary conditions (0=CLOSED, 1=OPEN, 2=PERIODIC)

    Returns:
        Tuple of (flux_east, flux_west, flux_north, flux_south)
        All fluxes are in units of [concentration * m³/s]
    """
    ny, nx = state.shape

    # Get neighbor values
    state_east = _get_neighbor(state, "east", bc_east)
    state_west = _get_neighbor(state, "west", bc_west)
    state_north = _get_neighbor(state, "north", bc_north)
    state_south = _get_neighbor(state, "south", bc_south)

    u_east = _get_neighbor(u, "east", bc_east)
    u_west = _get_neighbor(u, "west", bc_west)
    v_north = _get_neighbor(v, "north", bc_north)
    v_south = _get_neighbor(v, "south", bc_south)

    mask_east = _get_neighbor(mask, "east", bc_east)
    mask_west = _get_neighbor(mask, "west", bc_west)
    mask_north = _get_neighbor(mask, "north", bc_north)
    mask_south = _get_neighbor(mask, "south", bc_south)

    # Boundary masks (zero flux at closed boundaries)
    bc_mask_e = _get_boundary_mask(ny, nx, "east", bc_east)
    bc_mask_w = _get_boundary_mask(ny, nx, "west", bc_west)
    bc_mask_n = _get_boundary_mask(ny, nx, "north", bc_north)
    bc_mask_s = _get_boundary_mask(ny, nx, "south", bc_south)

    # --- EAST FACE ---
    # Velocity at face (average of adjacent cells)
    u_face_e = 0.5 * (u + u_east)
    # Upwind concentration (differentiable via jnp.where)
    c_upwind_e = jnp.where(u_face_e > 0, state, state_east)
    # Flux = velocity * concentration * face_area * masks
    flux_east = u_face_e * c_upwind_e * face_height * mask * mask_east * bc_mask_e

    # --- WEST FACE ---
    u_face_w = 0.5 * (u_west + u)
    c_upwind_w = jnp.where(u_face_w > 0, state_west, state)
    flux_west = u_face_w * c_upwind_w * face_height * mask * mask_west * bc_mask_w

    # --- NORTH FACE ---
    v_face_n = 0.5 * (v + v_north)
    c_upwind_n = jnp.where(v_face_n > 0, state, state_north)
    flux_north = v_face_n * c_upwind_n * face_width * mask * mask_north * bc_mask_n

    # --- SOUTH FACE ---
    v_face_s = 0.5 * (v_south + v)
    c_upwind_s = jnp.where(v_face_s > 0, state_south, state)
    flux_south = v_face_s * c_upwind_s * face_width * mask * mask_south * bc_mask_s

    return flux_east, flux_west, flux_north, flux_south


# =============================================================================
# DIFFUSION FLUXES (Centered differences)
# =============================================================================


def _compute_diffusion_fluxes(
    state: jnp.ndarray,
    D: float,
    dx: jnp.ndarray,
    dy: jnp.ndarray,
    face_height: jnp.ndarray,
    face_width: jnp.ndarray,
    mask: jnp.ndarray,
    bc_north: int,
    bc_south: int,
    bc_east: int,
    bc_west: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute diffusion fluxes at all faces using centered differences.

    Diffusion flux follows Fick's law: F = -D * grad(C) * face_area
    The gradient is computed using centered differences between adjacent cells.

    Args:
        state: Concentration field (Y, X)
        D: Diffusion coefficient [m²/s] (scalar, uniform)
        dx: Distance between cell centers in X [m] (Y, X)
        dy: Distance between cell centers in Y [m] (Y, X)
        face_height: Height of E/W faces [m] (Y, X)
        face_width: Width of N/S faces [m] (Y, X)
        mask: Ocean mask (1=ocean, 0=land) (Y, X)
        bc_*: Boundary conditions (0=CLOSED, 1=OPEN, 2=PERIODIC)

    Returns:
        Tuple of (flux_east, flux_west, flux_north, flux_south)
        All fluxes are in units of [concentration * m³/s]
    """
    ny, nx = state.shape

    # Get neighbor values
    state_east = _get_neighbor(state, "east", bc_east)
    state_west = _get_neighbor(state, "west", bc_west)
    state_north = _get_neighbor(state, "north", bc_north)
    state_south = _get_neighbor(state, "south", bc_south)

    dx_east = _get_neighbor(dx, "east", bc_east)
    dx_west = _get_neighbor(dx, "west", bc_west)
    dy_north = _get_neighbor(dy, "north", bc_north)
    dy_south = _get_neighbor(dy, "south", bc_south)

    mask_east = _get_neighbor(mask, "east", bc_east)
    mask_west = _get_neighbor(mask, "west", bc_west)
    mask_north = _get_neighbor(mask, "north", bc_north)
    mask_south = _get_neighbor(mask, "south", bc_south)

    # Boundary masks
    bc_mask_e = _get_boundary_mask(ny, nx, "east", bc_east)
    bc_mask_w = _get_boundary_mask(ny, nx, "west", bc_west)
    bc_mask_n = _get_boundary_mask(ny, nx, "north", bc_north)
    bc_mask_s = _get_boundary_mask(ny, nx, "south", bc_south)

    # --- EAST FACE ---
    # Distance between cell centers
    dx_face_e = 0.5 * (dx + dx_east)
    # Gradient (positive = increasing eastward)
    grad_e = (state_east - state) / dx_face_e
    # Flux = -D * gradient * face_area * masks
    flux_east = -D * grad_e * face_height * mask * mask_east * bc_mask_e

    # --- WEST FACE ---
    dx_face_w = 0.5 * (dx_west + dx)
    grad_w = (state - state_west) / dx_face_w
    flux_west = -D * grad_w * face_height * mask * mask_west * bc_mask_w

    # --- NORTH FACE ---
    dy_face_n = 0.5 * (dy + dy_north)
    grad_n = (state_north - state) / dy_face_n
    flux_north = -D * grad_n * face_width * mask * mask_north * bc_mask_n

    # --- SOUTH FACE ---
    dy_face_s = 0.5 * (dy_south + dy)
    grad_s = (state - state_south) / dy_face_s
    flux_south = -D * grad_s * face_width * mask * mask_south * bc_mask_s

    return flux_east, flux_west, flux_north, flux_south


# =============================================================================
# ADVECTION FLUXES (QUICK scheme — Leonard 1979)
# =============================================================================


def _compute_advection_fluxes_quick(
    state: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    face_height: jnp.ndarray,
    face_width: jnp.ndarray,
    mask: jnp.ndarray,
    bc_north: int,
    bc_south: int,
    bc_east: int,
    bc_west: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute advection fluxes at all faces using the QUICK scheme.

    QUICK (Quadratic Upstream Interpolation for Convective Kinematics,
    Leonard 1979) interpolates the face value using a quadratic fit through
    two upstream cells and one downstream cell. For a face between cells i
    and i+1 in the x-direction:

        u_face > 0:  phi_face = 6/8 * phi_i   + 3/8 * phi_{i+1} - 1/8 * phi_{i-1}
        u_face < 0:  phi_face = 6/8 * phi_{i+1} + 3/8 * phi_i   - 1/8 * phi_{i+2}

    The scheme is 3rd-order accurate on uniform grids (2nd-order on smoothly
    varying grids). It is a linear scheme in the state, so gradients through
    ``state`` are piecewise-linear and fully differentiable in JAX — the only
    source of non-smoothness is the ``jnp.where(u>0, ...)`` branch on velocity,
    matching what the first-order upwind scheme does.

    Boundaries: near non-periodic boundaries, the second-upstream neighbor
    (``phi_{i+2}`` / ``phi_{i-2}``) is obtained by composing ``_get_neighbor``
    twice, which correctly performs zero-gradient extrapolation (the boundary
    cell copies onto itself on each pass). Near a closed/open boundary, QUICK
    thus degenerates gracefully toward a centered 2nd-order stencil without
    introducing a ``where`` branch.

    Caveat: QUICK is not monotonic — it can produce small overshoots and
    undershoots (~5-10%) near sharp gradients. For smooth fields this is
    negligible; for binary/discontinuous fields (e.g. Zalesak), undershoots
    may drive the state slightly negative. Consumers that require strict
    non-negativity should clip downstream.

    Args:
        state: Concentration field (Y, X)
        u: Zonal velocity [m/s] (Y, X)
        v: Meridional velocity [m/s] (Y, X)
        face_height: Height of E/W faces [m] (Y, X)
        face_width: Width of N/S faces [m] (Y, X)
        mask: Ocean mask (1=ocean, 0=land) (Y, X)
        bc_*: Boundary conditions (0=CLOSED, 1=OPEN, 2=PERIODIC)

    Returns:
        Tuple of (flux_east, flux_west, flux_north, flux_south), all in
        [concentration * m^3/s].
    """
    ny, nx = state.shape

    # First-order neighbors
    state_e = _get_neighbor(state, "east", bc_east)
    state_w = _get_neighbor(state, "west", bc_west)
    state_n = _get_neighbor(state, "north", bc_north)
    state_s = _get_neighbor(state, "south", bc_south)

    # Second-order neighbors via composition — yields correct zero-gradient
    # extrapolation near CLOSED/OPEN boundaries, and double roll for PERIODIC.
    state_ee = _get_neighbor(state_e, "east", bc_east)
    state_ww = _get_neighbor(state_w, "west", bc_west)
    state_nn = _get_neighbor(state_n, "north", bc_north)
    state_ss = _get_neighbor(state_s, "south", bc_south)

    u_east = _get_neighbor(u, "east", bc_east)
    u_west = _get_neighbor(u, "west", bc_west)
    v_north = _get_neighbor(v, "north", bc_north)
    v_south = _get_neighbor(v, "south", bc_south)

    mask_east = _get_neighbor(mask, "east", bc_east)
    mask_west = _get_neighbor(mask, "west", bc_west)
    mask_north = _get_neighbor(mask, "north", bc_north)
    mask_south = _get_neighbor(mask, "south", bc_south)

    bc_mask_e = _get_boundary_mask(ny, nx, "east", bc_east)
    bc_mask_w = _get_boundary_mask(ny, nx, "west", bc_west)
    bc_mask_n = _get_boundary_mask(ny, nx, "north", bc_north)
    bc_mask_s = _get_boundary_mask(ny, nx, "south", bc_south)

    # --- EAST FACE (between cell i and i+1 in x) ---
    u_face_e = 0.5 * (u + u_east)
    # u > 0: U = state,   D = state_e, UU = state_w
    # u < 0: U = state_e, D = state,   UU = state_ee
    c_pos_e = (6.0 / 8.0) * state + (3.0 / 8.0) * state_e - (1.0 / 8.0) * state_w
    c_neg_e = (6.0 / 8.0) * state_e + (3.0 / 8.0) * state - (1.0 / 8.0) * state_ee
    c_face_e = jnp.where(u_face_e > 0, c_pos_e, c_neg_e)
    flux_east = u_face_e * c_face_e * face_height * mask * mask_east * bc_mask_e

    # --- WEST FACE (between cell i-1 and i in x) ---
    u_face_w = 0.5 * (u_west + u)
    # u > 0: U = state_w, D = state,   UU = state_ww
    # u < 0: U = state,   D = state_w, UU = state_e
    c_pos_w = (6.0 / 8.0) * state_w + (3.0 / 8.0) * state - (1.0 / 8.0) * state_ww
    c_neg_w = (6.0 / 8.0) * state + (3.0 / 8.0) * state_w - (1.0 / 8.0) * state_e
    c_face_w = jnp.where(u_face_w > 0, c_pos_w, c_neg_w)
    flux_west = u_face_w * c_face_w * face_height * mask * mask_west * bc_mask_w

    # --- NORTH FACE (between cell j and j+1 in y) ---
    v_face_n = 0.5 * (v + v_north)
    # v > 0: U = state,   D = state_n, UU = state_s
    # v < 0: U = state_n, D = state,   UU = state_nn
    c_pos_n = (6.0 / 8.0) * state + (3.0 / 8.0) * state_n - (1.0 / 8.0) * state_s
    c_neg_n = (6.0 / 8.0) * state_n + (3.0 / 8.0) * state - (1.0 / 8.0) * state_nn
    c_face_n = jnp.where(v_face_n > 0, c_pos_n, c_neg_n)
    flux_north = v_face_n * c_face_n * face_width * mask * mask_north * bc_mask_n

    # --- SOUTH FACE (between cell j-1 and j in y) ---
    v_face_s = 0.5 * (v_south + v)
    # v > 0: U = state_s, D = state,   UU = state_ss
    # v < 0: U = state,   D = state_s, UU = state_n
    c_pos_s = (6.0 / 8.0) * state_s + (3.0 / 8.0) * state - (1.0 / 8.0) * state_ss
    c_neg_s = (6.0 / 8.0) * state + (3.0 / 8.0) * state_s - (1.0 / 8.0) * state_n
    c_face_s = jnp.where(v_face_s > 0, c_pos_s, c_neg_s)
    flux_south = v_face_s * c_face_s * face_width * mask * mask_south * bc_mask_s

    return flux_east, flux_west, flux_north, flux_south


# =============================================================================
# MAIN TRANSPORT FUNCTION
# =============================================================================


@functional(
    name="phys:transport_tendency",
    core_dims={
        "state": ["Y", "X"],
        "u": ["Y", "X"],
        "v": ["Y", "X"],
        "dx": ["Y", "X"],
        "dy": ["Y", "X"],
        "face_height": ["Y", "X"],
        "face_width": ["Y", "X"],
        "cell_area": ["Y", "X"],
        "mask": ["Y", "X"],
    },
    out_dims=["Y", "X"],
    outputs=["advection_rate", "diffusion_rate"],
    units={
        "state": "g/m^2",
        "u": "m/s",
        "v": "m/s",
        "D": "m^2/s",
        "dx": "m",
        "dy": "m",
        "face_height": "m",
        "face_width": "m",
        "cell_area": "m^2",
        "mask": "dimensionless",
        "advection_rate": "g/m^2/s",
        "diffusion_rate": "g/m^2/s",
    },
)
def transport_tendency(
    state: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    D: float,
    dx: jnp.ndarray,
    dy: jnp.ndarray,
    face_height: jnp.ndarray,
    face_width: jnp.ndarray,
    cell_area: jnp.ndarray,
    mask: jnp.ndarray,
    bc_north: int = 0,
    bc_south: int = 0,
    bc_east: int = 0,
    bc_west: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute transport tendencies using finite volume method.

    This function computes advection and diffusion tendencies for a scalar
    field using the finite volume method. It is fully differentiable and
    compatible with JAX transformations (jit, grad, vmap).

    The advection uses an upwind scheme for stability. The diffusion uses
    centered differences. Both ensure mass conservation.

    Args:
        state: Concentration field [units] (Y, X)
        u: Zonal (eastward) velocity [m/s] (Y, X)
        v: Meridional (northward) velocity [m/s] (Y, X)
        D: Diffusion coefficient [m²/s] (scalar, uniform)
        dx: Distance between cell centers in X direction [m] (Y, X)
        dy: Distance between cell centers in Y direction [m] (Y, X)
        face_height: Height of E/W faces [m] (Y, X)
            For simple lat/lon grid: face_height = dy
            For ORCA grid: face_height = e2u
        face_width: Width of N/S faces [m] (Y, X)
            For simple lat/lon grid: face_width = dx
            For ORCA grid: face_width = e1v
        cell_area: Area of each cell [m²] (Y, X)
        mask: Ocean/land mask (1=ocean, 0=land) (Y, X)
        bc_north: Northern boundary condition (0=CLOSED, 1=OPEN, 2=PERIODIC)
        bc_south: Southern boundary condition (0=CLOSED, 1=OPEN, 2=PERIODIC)
        bc_east: Eastern boundary condition (0=CLOSED, 1=OPEN, 2=PERIODIC)
        bc_west: Western boundary condition (0=CLOSED, 1=OPEN, 2=PERIODIC)

    Returns:
        Tuple of (advection_rate, diffusion_rate):
        - advection_rate: Tendency from advection [units/s] (Y, X)
        - diffusion_rate: Tendency from diffusion [units/s] (Y, X)

    Example:
        >>> import jax.numpy as jnp
        >>> # Simple 10x10 grid
        >>> ny, nx = 10, 10
        >>> state = jnp.ones((ny, nx))
        >>> u = jnp.full((ny, nx), 0.1)  # 0.1 m/s eastward
        >>> v = jnp.zeros((ny, nx))
        >>> D = 100.0  # 100 m²/s
        >>> dx = jnp.full((ny, nx), 1000.0)  # 1 km
        >>> dy = jnp.full((ny, nx), 1000.0)
        >>> cell_area = dx * dy
        >>> mask = jnp.ones((ny, nx))
        >>> adv, diff = transport_tendency(
        ...     state, u, v, D, dx, dy, dy, dx, cell_area, mask,
        ...     bc_north=0, bc_south=0, bc_east=1, bc_west=1,
        ... )
    """
    # Compute advection fluxes (upwind)
    flux_adv_e, flux_adv_w, flux_adv_n, flux_adv_s = _compute_advection_fluxes(
        state,
        u,
        v,
        face_height,
        face_width,
        mask,
        bc_north,
        bc_south,
        bc_east,
        bc_west,
    )

    # Compute diffusion fluxes (centered)
    flux_diff_e, flux_diff_w, flux_diff_n, flux_diff_s = _compute_diffusion_fluxes(
        state,
        D,
        dx,
        dy,
        face_height,
        face_width,
        mask,
        bc_north,
        bc_south,
        bc_east,
        bc_west,
    )

    # Compute flux divergence
    # Positive flux = leaving the cell, so tendency = -divergence
    adv_divergence = flux_adv_e - flux_adv_w + flux_adv_n - flux_adv_s
    diff_divergence = flux_diff_e - flux_diff_w + flux_diff_n - flux_diff_s

    # Tendency = -divergence / cell_area (with mask)
    # Use jnp.where to avoid division by zero for land cells
    safe_area = jnp.where(cell_area > 0, cell_area, 1.0)

    advection_rate = -adv_divergence / safe_area * mask
    diffusion_rate = -diff_divergence / safe_area * mask

    return advection_rate, diffusion_rate


@functional(
    name="phys:transport_tendency_quick",
    core_dims={
        "state": ["Y", "X"],
        "u": ["Y", "X"],
        "v": ["Y", "X"],
        "dx": ["Y", "X"],
        "dy": ["Y", "X"],
        "face_height": ["Y", "X"],
        "face_width": ["Y", "X"],
        "cell_area": ["Y", "X"],
        "mask": ["Y", "X"],
    },
    out_dims=["Y", "X"],
    outputs=["advection_rate", "diffusion_rate"],
    units={
        "state": "g/m^2",
        "u": "m/s",
        "v": "m/s",
        "D": "m^2/s",
        "dx": "m",
        "dy": "m",
        "face_height": "m",
        "face_width": "m",
        "cell_area": "m^2",
        "mask": "dimensionless",
        "advection_rate": "g/m^2/s",
        "diffusion_rate": "g/m^2/s",
    },
)
def transport_tendency_quick(
    state: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    D: float,
    dx: jnp.ndarray,
    dy: jnp.ndarray,
    face_height: jnp.ndarray,
    face_width: jnp.ndarray,
    cell_area: jnp.ndarray,
    mask: jnp.ndarray,
    bc_north: int = 0,
    bc_south: int = 0,
    bc_east: int = 0,
    bc_west: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute transport tendencies using the QUICK advection scheme.

    Same signature as :func:`transport_tendency` but uses the 3rd-order
    QUICK scheme (Leonard 1979) for advection instead of 1st-order upwind.
    Diffusion still uses centered 2nd-order differences. The scheme is
    fully differentiable in JAX and mass-conservative.

    QUICK is **not** monotonic — expect small undershoots/overshoots near
    sharp gradients. For smooth fields the gain in accuracy over upwind is
    large (~order 3 convergence on smooth solutions, vs. ~order 1 for
    upwind). For binary/discontinuous fields, the effective order is
    limited by the discontinuity itself but numerical diffusion is still
    drastically reduced.

    .. warning::

        **Explicit Euler time integration is unconditionally unstable with
        QUICK.** Von Neumann analysis shows ``max|g|(sigma) > 1`` for any
        ``sigma > 0``. In practice this means a forward-Euler loop blows up
        over long integrations (the growth rate is ~1% per step at CFL 0.3,
        ~10% per step at CFL 1.0).

        You MUST integrate in time with a multi-stage scheme. RK2 (Heun /
        midpoint) is sufficient and stable up to CFL ~0.87 in 1D (CFL ~0.5
        in 2D to be safe). RK3 extends the stability range further.

        Upwind remains stable with explicit Euler at CFL <= 1, so consumers
        switching from ``transport_tendency`` to ``transport_tendency_quick``
        must also switch their time integrator.

    Args: same as :func:`transport_tendency`.

    Returns:
        Tuple of (advection_rate, diffusion_rate) in [units/s].
    """
    flux_adv_e, flux_adv_w, flux_adv_n, flux_adv_s = _compute_advection_fluxes_quick(
        state,
        u,
        v,
        face_height,
        face_width,
        mask,
        bc_north,
        bc_south,
        bc_east,
        bc_west,
    )

    flux_diff_e, flux_diff_w, flux_diff_n, flux_diff_s = _compute_diffusion_fluxes(
        state,
        D,
        dx,
        dy,
        face_height,
        face_width,
        mask,
        bc_north,
        bc_south,
        bc_east,
        bc_west,
    )

    adv_divergence = flux_adv_e - flux_adv_w + flux_adv_n - flux_adv_s
    diff_divergence = flux_diff_e - flux_diff_w + flux_diff_n - flux_diff_s

    safe_area = jnp.where(cell_area > 0, cell_area, 1.0)

    advection_rate = -adv_divergence / safe_area * mask
    diffusion_rate = -diff_divergence / safe_area * mask

    return advection_rate, diffusion_rate
