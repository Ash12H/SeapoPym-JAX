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


# =============================================================================
# NEIGHBOR ACCESS FUNCTIONS
# =============================================================================


def _get_neighbor_east(
    state: jnp.ndarray,
    bc_east: int,
) -> jnp.ndarray:
    """Get eastern neighbor values with boundary condition handling.

    Args:
        state: Field values (Y, X)
        bc_east: Eastern boundary condition (0=CLOSED, 1=OPEN, 2=PERIODIC)

    Returns:
        Eastern neighbor values (Y, X). At eastern boundary:
        - CLOSED/OPEN: returns current cell value (zero gradient)
        - PERIODIC: returns western-most column
    """
    shifted = jnp.roll(state, shift=-1, axis=-1)
    # For non-periodic: replace last column with original (zero gradient)
    non_periodic = shifted.at[:, -1].set(state[:, -1])
    # Select based on boundary condition
    return jax.lax.cond(
        bc_east == BoundaryType.PERIODIC,
        lambda: shifted,
        lambda: non_periodic,
    )


def _get_neighbor_west(
    state: jnp.ndarray,
    bc_west: int,
) -> jnp.ndarray:
    """Get western neighbor values with boundary condition handling.

    Args:
        state: Field values (Y, X)
        bc_west: Western boundary condition (0=CLOSED, 1=OPEN, 2=PERIODIC)

    Returns:
        Western neighbor values (Y, X). At western boundary:
        - CLOSED/OPEN: returns current cell value (zero gradient)
        - PERIODIC: returns eastern-most column
    """
    shifted = jnp.roll(state, shift=1, axis=-1)
    non_periodic = shifted.at[:, 0].set(state[:, 0])
    return jax.lax.cond(
        bc_west == BoundaryType.PERIODIC,
        lambda: shifted,
        lambda: non_periodic,
    )


def _get_neighbor_north(
    state: jnp.ndarray,
    bc_north: int,
) -> jnp.ndarray:
    """Get northern neighbor values with boundary condition handling.

    Args:
        state: Field values (Y, X)
        bc_north: Northern boundary condition (0=CLOSED, 1=OPEN, 2=PERIODIC)

    Returns:
        Northern neighbor values (Y, X). At northern boundary:
        - CLOSED/OPEN: returns current cell value (zero gradient)
        - PERIODIC: returns southern-most row
    """
    shifted = jnp.roll(state, shift=-1, axis=-2)
    non_periodic = shifted.at[-1, :].set(state[-1, :])
    return jax.lax.cond(
        bc_north == BoundaryType.PERIODIC,
        lambda: shifted,
        lambda: non_periodic,
    )


def _get_neighbor_south(
    state: jnp.ndarray,
    bc_south: int,
) -> jnp.ndarray:
    """Get southern neighbor values with boundary condition handling.

    Args:
        state: Field values (Y, X)
        bc_south: Southern boundary condition (0=CLOSED, 1=OPEN, 2=PERIODIC)

    Returns:
        Southern neighbor values (Y, X). At southern boundary:
        - CLOSED/OPEN: returns current cell value (zero gradient)
        - PERIODIC: returns northern-most row
    """
    shifted = jnp.roll(state, shift=1, axis=-2)
    non_periodic = shifted.at[0, :].set(state[0, :])
    return jax.lax.cond(
        bc_south == BoundaryType.PERIODIC,
        lambda: shifted,
        lambda: non_periodic,
    )


def _get_boundary_mask_east(
    ny: int,
    nx: int,
    bc_east: int,
) -> jnp.ndarray:
    """Get mask for eastern boundary (1=interior/periodic, 0=closed boundary).

    For CLOSED boundaries, the eastern-most column has no flux.
    For OPEN/PERIODIC, flux is allowed everywhere.
    """
    mask = jnp.ones((ny, nx))
    mask_closed = mask.at[:, -1].set(0.0)
    return jax.lax.cond(
        bc_east == BoundaryType.CLOSED,
        lambda: mask_closed,
        lambda: mask,
    )


def _get_boundary_mask_west(
    ny: int,
    nx: int,
    bc_west: int,
) -> jnp.ndarray:
    """Get mask for western boundary (1=interior/periodic, 0=closed boundary)."""
    mask = jnp.ones((ny, nx))
    mask_closed = mask.at[:, 0].set(0.0)
    return jax.lax.cond(
        bc_west == BoundaryType.CLOSED,
        lambda: mask_closed,
        lambda: mask,
    )


def _get_boundary_mask_north(
    ny: int,
    nx: int,
    bc_north: int,
) -> jnp.ndarray:
    """Get mask for northern boundary (1=interior/periodic, 0=closed boundary)."""
    mask = jnp.ones((ny, nx))
    mask_closed = mask.at[-1, :].set(0.0)
    return jax.lax.cond(
        bc_north == BoundaryType.CLOSED,
        lambda: mask_closed,
        lambda: mask,
    )


def _get_boundary_mask_south(
    ny: int,
    nx: int,
    bc_south: int,
) -> jnp.ndarray:
    """Get mask for southern boundary (1=interior/periodic, 0=closed boundary)."""
    mask = jnp.ones((ny, nx))
    mask_closed = mask.at[0, :].set(0.0)
    return jax.lax.cond(
        bc_south == BoundaryType.CLOSED,
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
    state_east = _get_neighbor_east(state, bc_east)
    state_west = _get_neighbor_west(state, bc_west)
    state_north = _get_neighbor_north(state, bc_north)
    state_south = _get_neighbor_south(state, bc_south)

    u_east = _get_neighbor_east(u, bc_east)
    u_west = _get_neighbor_west(u, bc_west)
    v_north = _get_neighbor_north(v, bc_north)
    v_south = _get_neighbor_south(v, bc_south)

    mask_east = _get_neighbor_east(mask, bc_east)
    mask_west = _get_neighbor_west(mask, bc_west)
    mask_north = _get_neighbor_north(mask, bc_north)
    mask_south = _get_neighbor_south(mask, bc_south)

    # Boundary masks (zero flux at closed boundaries)
    bc_mask_e = _get_boundary_mask_east(ny, nx, bc_east)
    bc_mask_w = _get_boundary_mask_west(ny, nx, bc_west)
    bc_mask_n = _get_boundary_mask_north(ny, nx, bc_north)
    bc_mask_s = _get_boundary_mask_south(ny, nx, bc_south)

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
    D: jnp.ndarray,
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
        D: Diffusion coefficient [m²/s] (Y, X)
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
    state_east = _get_neighbor_east(state, bc_east)
    state_west = _get_neighbor_west(state, bc_west)
    state_north = _get_neighbor_north(state, bc_north)
    state_south = _get_neighbor_south(state, bc_south)

    D_east = _get_neighbor_east(D, bc_east)
    D_west = _get_neighbor_west(D, bc_west)
    D_north = _get_neighbor_north(D, bc_north)
    D_south = _get_neighbor_south(D, bc_south)

    dx_east = _get_neighbor_east(dx, bc_east)
    dx_west = _get_neighbor_west(dx, bc_west)
    dy_north = _get_neighbor_north(dy, bc_north)
    dy_south = _get_neighbor_south(dy, bc_south)

    mask_east = _get_neighbor_east(mask, bc_east)
    mask_west = _get_neighbor_west(mask, bc_west)
    mask_north = _get_neighbor_north(mask, bc_north)
    mask_south = _get_neighbor_south(mask, bc_south)

    # Boundary masks
    bc_mask_e = _get_boundary_mask_east(ny, nx, bc_east)
    bc_mask_w = _get_boundary_mask_west(ny, nx, bc_west)
    bc_mask_n = _get_boundary_mask_north(ny, nx, bc_north)
    bc_mask_s = _get_boundary_mask_south(ny, nx, bc_south)

    # --- EAST FACE ---
    # Diffusion coefficient at face (harmonic mean for heterogeneous D)
    D_face_e = 0.5 * (D + D_east)
    # Distance between cell centers
    dx_face_e = 0.5 * (dx + dx_east)
    # Gradient (positive = increasing eastward)
    grad_e = (state_east - state) / dx_face_e
    # Flux = -D * gradient * face_area * masks
    flux_east = -D_face_e * grad_e * face_height * mask * mask_east * bc_mask_e

    # --- WEST FACE ---
    D_face_w = 0.5 * (D_west + D)
    dx_face_w = 0.5 * (dx_west + dx)
    grad_w = (state - state_west) / dx_face_w
    flux_west = -D_face_w * grad_w * face_height * mask * mask_west * bc_mask_w

    # --- NORTH FACE ---
    D_face_n = 0.5 * (D + D_north)
    dy_face_n = 0.5 * (dy + dy_north)
    grad_n = (state_north - state) / dy_face_n
    flux_north = -D_face_n * grad_n * face_width * mask * mask_north * bc_mask_n

    # --- SOUTH FACE ---
    D_face_s = 0.5 * (D_south + D)
    dy_face_s = 0.5 * (dy_south + dy)
    grad_s = (state - state_south) / dy_face_s
    flux_south = -D_face_s * grad_s * face_width * mask * mask_south * bc_mask_s

    return flux_east, flux_west, flux_north, flux_south


# =============================================================================
# MAIN TRANSPORT FUNCTION
# =============================================================================


@functional(
    name="phys:transport_tendency",
    backend="jax",
    core_dims={
        "state": ["Y", "X"],
        "u": ["Y", "X"],
        "v": ["Y", "X"],
        "D": ["Y", "X"],
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
    D: jnp.ndarray,
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
        D: Diffusion coefficient [m²/s] (Y, X)
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
        >>> D = jnp.full((ny, nx), 100.0)  # 100 m²/s
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
