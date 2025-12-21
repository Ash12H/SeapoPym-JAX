"""New unified transport architecture using finite volume method.

This module implements the refactored transport functions with:
- Shared data preparation
- Separate flux computation (advection and diffusion)
- Unified orchestrators for Xarray and Numba
"""

from typing import Any

import numpy as np
import xarray as xr

from seapopym.standard.coordinates import Coordinates, GridPosition
from seapopym.transport.boundary import BoundaryConditions, BoundaryType, get_neighbors_with_bc

# =============================================================================
# SHARED HELPERS
# =============================================================================


def _prepare_transport_data(
    state: xr.DataArray,
    u: xr.DataArray,
    v: xr.DataArray,
    D: xr.DataArray | float,
    dx: xr.DataArray | float,
    dy: xr.DataArray | float,
    cell_areas: xr.DataArray,
    face_areas_ew: xr.DataArray,
    face_areas_ns: xr.DataArray,
    boundary_conditions: BoundaryConditions,
    mask: xr.DataArray | None = None,
) -> dict[str, Any]:
    """Prepare all shared data for transport computation.

    This function performs all data preparation steps that are common to both
    advection and diffusion calculations:
    - Clean inputs (replace NaN with 0)
    - Get neighbor values
    - Interpolate velocities and diffusivity to faces
    - Compute face masks
    - Extract face areas

    Args:
        state: Concentration field
        u: Zonal velocity
        v: Meridional velocity
        D: Diffusion coefficient
        dx: Grid spacing in x
        dy: Grid spacing in y
        cell_areas: Cell areas
        face_areas_ew: East-West face areas
        face_areas_ns: North-South face areas
        boundary_conditions: Boundary conditions
        mask: Ocean/land mask (1=ocean, 0=land)

    Returns:
        Dictionary with all prepared data:
        - state_clean: Cleaned state
        - state_neighbors: {'east', 'west', 'north', 'south'}
        - u_face: {'east', 'west', 'north', 'south'}
        - v_face: {'east', 'west', 'north', 'south'}
        - D_face: {'east', 'west', 'north', 'south'}
        - dx_face: {'east', 'west', 'north', 'south'}
        - dy_face: {'east', 'west', 'north', 'south'}
        - face_areas: {'east', 'west', 'north', 'south'}
        - face_masks: {'east', 'west', 'north', 'south'}
        - cell_areas: Cell areas
        - mask: Ocean/land mask
    """
    dim_y = Coordinates.Y.value
    dim_x = Coordinates.X.value

    # 1. CLEAN INPUTS (replace NaN with 0 to prevent propagation)
    state_clean = state.fillna(0.0)
    u_clean = u.fillna(0.0)
    v_clean = v.fillna(0.0)

    # 2. GET NEIGHBORS
    state_west, state_east, state_south, state_north = get_neighbors_with_bc(
        state_clean, boundary_conditions
    )
    u_west, u_east, u_south, u_north = get_neighbors_with_bc(u_clean, boundary_conditions)
    v_west, v_east, v_south, v_north = get_neighbors_with_bc(v_clean, boundary_conditions)

    # 3. INTERPOLATE TO FACES
    # Velocities (always DataArrays)
    u_face_east = 0.5 * (u_clean + u_east)
    u_face_west = 0.5 * (u_clean + u_west)
    v_face_north = 0.5 * (v_clean + v_north)
    v_face_south = 0.5 * (v_clean + v_south)

    # Diffusivity (can be scalar or DataArray)
    if isinstance(D, xr.DataArray):
        D_west, D_east, D_south, D_north = get_neighbors_with_bc(D, boundary_conditions)
        D_face_east = 0.5 * (D + D_east)
        D_face_west = 0.5 * (D + D_west)
        D_face_north = 0.5 * (D + D_north)
        D_face_south = 0.5 * (D + D_south)
    else:
        D_face_east = D
        D_face_west = D
        D_face_north = D
        D_face_south = D

    # Grid spacing (can be scalar or DataArray)
    if isinstance(dx, xr.DataArray):
        dx_west, dx_east, _, _ = get_neighbors_with_bc(dx, boundary_conditions)
        dx_face_east = 0.5 * (dx + dx_east)
        dx_face_west = 0.5 * (dx + dx_west)
    else:
        dx_face_east = dx
        dx_face_west = dx

    if isinstance(dy, xr.DataArray):
        _, _, dy_south, dy_north = get_neighbors_with_bc(dy, boundary_conditions)
        dy_face_north = 0.5 * (dy + dy_north)
        dy_face_south = 0.5 * (dy + dy_south)
    else:
        dy_face_north = dy
        dy_face_south = dy

    # 4. FACE MASKS
    # Start with all faces open
    face_mask_east = xr.ones_like(state_clean)
    face_mask_west = xr.ones_like(state_clean)
    face_mask_north = xr.ones_like(state_clean)
    face_mask_south = xr.ones_like(state_clean)

    # Apply land masking if provided
    if mask is not None:
        mask_west, mask_east, mask_south, mask_north = get_neighbors_with_bc(
            mask, boundary_conditions
        )
        # Face is open only if both adjacent cells are ocean
        face_mask_east = face_mask_east * mask * mask_east
        face_mask_west = face_mask_west * mask * mask_west
        face_mask_north = face_mask_north * mask * mask_north
        face_mask_south = face_mask_south * mask * mask_south

    # Apply CLOSED boundary conditions
    if boundary_conditions.east == BoundaryType.CLOSED:
        face_mask_east = face_mask_east.where(
            face_mask_east[dim_x] != face_mask_east[dim_x][-1], 0.0
        )
    if boundary_conditions.west == BoundaryType.CLOSED:
        face_mask_west = face_mask_west.where(
            face_mask_west[dim_x] != face_mask_west[dim_x][0], 0.0
        )
    if boundary_conditions.north == BoundaryType.CLOSED:
        face_mask_north = face_mask_north.where(
            face_mask_north[dim_y] != face_mask_north[dim_y][-1], 0.0
        )
    if boundary_conditions.south == BoundaryType.CLOSED:
        face_mask_south = face_mask_south.where(
            face_mask_south[dim_y] != face_mask_south[dim_y][0], 0.0
        )

    # 5. EXTRACT FACE AREAS
    x_face_dim = GridPosition.get_face_dim(Coordinates.X, GridPosition.LEFT)
    y_face_dim = GridPosition.get_face_dim(Coordinates.Y, GridPosition.LEFT)

    area_east = xr.DataArray(
        face_areas_ew.isel({x_face_dim: slice(1, None)}).values, dims=(dim_y, dim_x)
    )
    area_west = xr.DataArray(
        face_areas_ew.isel({x_face_dim: slice(None, -1)}).values, dims=(dim_y, dim_x)
    )
    area_north = xr.DataArray(
        face_areas_ns.isel({y_face_dim: slice(1, None)}).values, dims=(dim_y, dim_x)
    )
    area_south = xr.DataArray(
        face_areas_ns.isel({y_face_dim: slice(None, -1)}).values, dims=(dim_y, dim_x)
    )

    return {
        "state_clean": state_clean,
        "state_neighbors": {
            "east": state_east,
            "west": state_west,
            "north": state_north,
            "south": state_south,
        },
        "u_face": {
            "east": u_face_east,
            "west": u_face_west,
        },
        "v_face": {
            "north": v_face_north,
            "south": v_face_south,
        },
        "D_face": {
            "east": D_face_east,
            "west": D_face_west,
            "north": D_face_north,
            "south": D_face_south,
        },
        "dx_face": {
            "east": dx_face_east,
            "west": dx_face_west,
        },
        "dy_face": {
            "north": dy_face_north,
            "south": dy_face_south,
        },
        "face_areas": {
            "east": area_east,
            "west": area_west,
            "north": area_north,
            "south": area_south,
        },
        "face_masks": {
            "east": face_mask_east,
            "west": face_mask_west,
            "north": face_mask_north,
            "south": face_mask_south,
        },
        "cell_areas": cell_areas,
        "mask": mask,
    }


def _compute_divergence(
    fluxes: dict[str, xr.DataArray],
    cell_areas: xr.DataArray,
) -> xr.DataArray:
    """Compute flux divergence.

    Args:
        fluxes: Dictionary with 'flux_east', 'flux_west', 'flux_north', 'flux_south'
        cell_areas: Cell areas

    Returns:
        Flux divergence
    """
    return (
        fluxes["flux_east"] - fluxes["flux_west"] + fluxes["flux_north"] - fluxes["flux_south"]
    ) / cell_areas


# =============================================================================
# FLUX COMPUTATION (XARRAY)
# =============================================================================


def _compute_advection_flux_xarray(
    state_center: xr.DataArray,
    state_neighbors: dict[str, xr.DataArray],
    u_face: dict[str, xr.DataArray],
    v_face: dict[str, xr.DataArray],
    face_areas: dict[str, xr.DataArray],
    face_masks: dict[str, xr.DataArray],
) -> dict[str, xr.DataArray]:
    """Compute advection fluxes using upwind scheme (Xarray implementation).

    Args:
        state_center: Concentration at cell centers
        state_neighbors: Neighbor concentrations {'east', 'west', 'north', 'south'}
        u_face: Zonal velocities at faces
        v_face: Meridional velocities at faces
        face_areas: Face areas
        face_masks: Face masks

    Returns:
        Dictionary with 'flux_east', 'flux_west', 'flux_north', 'flux_south'
    """
    # UPWIND SELECTION
    # East face: if u > 0, flow is West→East, take current cell
    state_face_east = xr.where(u_face["east"] > 0, state_center, state_neighbors["east"])
    # West face: if u > 0, flow is West→East, take west neighbor
    state_face_west = xr.where(u_face["west"] > 0, state_neighbors["west"], state_center)

    # North face: if v > 0, flow is South→North, take current cell
    state_face_north = xr.where(v_face["north"] > 0, state_center, state_neighbors["north"])
    # South face: if v > 0, flow is South→North, take south neighbor
    state_face_south = xr.where(v_face["south"] > 0, state_neighbors["south"], state_center)

    # FLUXES: velocity × concentration × face_area × face_mask
    flux_east = u_face["east"] * state_face_east * face_areas["east"] * face_masks["east"]
    flux_west = u_face["west"] * state_face_west * face_areas["west"] * face_masks["west"]
    flux_north = v_face["north"] * state_face_north * face_areas["north"] * face_masks["north"]
    flux_south = v_face["south"] * state_face_south * face_areas["south"] * face_masks["south"]

    return {
        "flux_east": flux_east,
        "flux_west": flux_west,
        "flux_north": flux_north,
        "flux_south": flux_south,
    }


def _compute_diffusion_flux_xarray(
    state_center: xr.DataArray,
    state_neighbors: dict[str, xr.DataArray],
    D_face: dict[str, xr.DataArray | float],
    dx_face: dict[str, xr.DataArray | float],
    dy_face: dict[str, xr.DataArray | float],
    face_areas: dict[str, xr.DataArray],
    face_masks: dict[str, xr.DataArray],
) -> dict[str, xr.DataArray]:
    """Compute diffusion fluxes using gradient method (Xarray implementation).

    Args:
        state_center: Concentration at cell centers
        state_neighbors: Neighbor concentrations
        D_face: Diffusivity at faces
        dx_face: Grid spacing x at faces
        dy_face: Grid spacing y at faces
        face_areas: Face areas
        face_masks: Face masks

    Returns:
        Dictionary with 'flux_east', 'flux_west', 'flux_north', 'flux_south'
    """
    # GRADIENTS
    grad_x_east = (state_neighbors["east"] - state_center) / dx_face["east"]
    grad_x_west = (state_center - state_neighbors["west"]) / dx_face["west"]
    grad_y_north = (state_neighbors["north"] - state_center) / dy_face["north"]
    grad_y_south = (state_center - state_neighbors["south"]) / dy_face["south"]

    # FLUXES: -D × gradient × face_area × face_mask
    flux_east = -D_face["east"] * grad_x_east * face_areas["east"] * face_masks["east"]
    flux_west = -D_face["west"] * grad_x_west * face_areas["west"] * face_masks["west"]
    flux_north = -D_face["north"] * grad_y_north * face_areas["north"] * face_masks["north"]
    flux_south = -D_face["south"] * grad_y_south * face_areas["south"] * face_masks["south"]

    return {
        "flux_east": flux_east,
        "flux_west": flux_west,
        "flux_north": flux_north,
        "flux_south": flux_south,
    }


# =============================================================================
# ORCHESTRATORS
# =============================================================================


def compute_transport_xarray(
    state: xr.DataArray,
    u: xr.DataArray,
    v: xr.DataArray,
    D: xr.DataArray | float,
    dx: xr.DataArray | float,
    dy: xr.DataArray | float,
    cell_areas: xr.DataArray,
    face_areas_ew: xr.DataArray,
    face_areas_ns: xr.DataArray,
    boundary_conditions: BoundaryConditions,
    mask: xr.DataArray | None = None,
) -> dict[str, xr.DataArray]:
    """Compute transport tendencies using Xarray implementation.

    This function computes both advection and diffusion tendencies using a unified
    finite volume approach that guarantees mass conservation.

    Args:
        state: Concentration field [Units], shape (..., lat, lon)
        u: Zonal velocity [m/s], shape (..., lat, lon)
        v: Meridional velocity [m/s], shape (..., lat, lon)
        D: Diffusion coefficient [m²/s], scalar or DataArray
        dx: Grid spacing in X [m], scalar or DataArray
        dy: Grid spacing in Y [m], scalar or DataArray
        cell_areas: Cell areas [m²], shape (lat, lon)
        face_areas_ew: East/West face areas [m], shape (lat, lon+1)
        face_areas_ns: North/South face areas [m], shape (lat+1, lon)
        boundary_conditions: Boundary conditions
        mask: Ocean/land mask (1=ocean, 0=land), shape (lat, lon)

    Returns:
        Dictionary with:
        - advection_rate: Advection tendency [Units/s]
        - diffusion_rate: Diffusion tendency [Units/s]

    Example:
        >>> result = compute_transport_xarray(...)
        >>> total_tendency = result['advection_rate'] + result['diffusion_rate']
        >>> biomass_new = biomass + total_tendency * dt
    """
    # 1. PREPARE (shared data preparation)
    prepared = _prepare_transport_data(
        state, u, v, D, dx, dy, cell_areas, face_areas_ew, face_areas_ns, boundary_conditions, mask
    )

    # 2. COMPUTE FLUXES (Xarray)
    flux_adv = _compute_advection_flux_xarray(
        prepared["state_clean"],
        prepared["state_neighbors"],
        prepared["u_face"],
        prepared["v_face"],
        prepared["face_areas"],
        prepared["face_masks"],
    )

    flux_diff = _compute_diffusion_flux_xarray(
        prepared["state_clean"],
        prepared["state_neighbors"],
        prepared["D_face"],
        prepared["dx_face"],
        prepared["dy_face"],
        prepared["face_areas"],
        prepared["face_masks"],
    )

    # 3. DIVERGENCES (shared)
    div_adv = _compute_divergence(flux_adv, prepared["cell_areas"])
    div_diff = _compute_divergence(flux_diff, prepared["cell_areas"])

    # 4. TENDENCIES
    advection_rate = -div_adv
    diffusion_rate = -div_diff

    # Apply mask to final results
    if mask is not None:
        advection_rate = advection_rate * mask
        diffusion_rate = diffusion_rate * mask

    # Restore original dimension order
    advection_rate = advection_rate.transpose(*state.dims)
    diffusion_rate = diffusion_rate.transpose(*state.dims)

    return {
        "advection_rate": advection_rate,
        "diffusion_rate": diffusion_rate,
    }


# =============================================================================
# FLUX COMPUTATION (NUMBA)
# =============================================================================


def _compute_advection_flux_numba(prepared: dict) -> dict[str, xr.DataArray]:
    """Compute advection fluxes using Numba-accelerated kernel.

    Args:
        prepared: Prepared data dictionary from _prepare_transport_data()

    Returns:
        Dictionary with 'flux_east', 'flux_west', 'flux_north', 'flux_south'
    """
    try:
        from seapopym.transport.numba_kernels import advection_flux_numba
    except ImportError as err:
        raise ImportError("Numba is required for Numba implementation.") from err

    dim_y = Coordinates.Y.value
    dim_x = Coordinates.X.value

    # Prepare boundary conditions encoding
    bc_vals = np.array(
        [
            0 if prepared["boundary_conditions"].north == BoundaryType.CLOSED else 1,
            0 if prepared["boundary_conditions"].south == BoundaryType.CLOSED else 1,
            0 if prepared["boundary_conditions"].east == BoundaryType.CLOSED else 1,
            0 if prepared["boundary_conditions"].west == BoundaryType.CLOSED else 1,
        ],
        dtype=np.int32,
    )
    bc_da = xr.DataArray(bc_vals, dims="boundary_params")

    # Get mask (create if None)
    mask = prepared["mask"]
    if mask is None:
        mask = xr.ones_like(prepared["state_clean"])

    # Call kernel via apply_ufunc
    flux_e, flux_w, flux_n, flux_s = xr.apply_ufunc(
        advection_flux_numba,
        prepared["state_clean"],
        prepared["u_clean"],
        prepared["v_clean"],
        prepared["face_areas"]["east"],  # ew_area
        prepared["face_areas"]["north"],  # ns_area
        mask,
        bc_da,
        input_core_dims=[
            [dim_y, dim_x],  # state
            [dim_y, dim_x],  # u
            [dim_y, dim_x],  # v
            [dim_y, dim_x],  # ew_area
            [dim_y, dim_x],  # ns_area
            [dim_y, dim_x],  # mask
            ["boundary_params"],  # bc
        ],
        output_core_dims=[
            [dim_y, dim_x],  # flux_east
            [dim_y, dim_x],  # flux_west
            [dim_y, dim_x],  # flux_north
            [dim_y, dim_x],  # flux_south
        ],
        dask="parallelized",
        output_dtypes=[prepared["state_clean"].dtype] * 4,
    )

    return {
        "flux_east": flux_e,
        "flux_west": flux_w,
        "flux_north": flux_n,
        "flux_south": flux_s,
    }


def _compute_diffusion_flux_numba(prepared: dict) -> dict[str, xr.DataArray]:
    """Compute diffusion fluxes using Numba-accelerated kernel.

    Args:
        prepared: Prepared data dictionary from _prepare_transport_data()

    Returns:
        Dictionary with 'flux_east', 'flux_west', 'flux_north', 'flux_south'
    """
    try:
        from seapopym.transport.numba_kernels import diffusion_flux_numba
    except ImportError as err:
        raise ImportError("Numba is required for Numba implementation.") from err

    dim_y = Coordinates.Y.value
    dim_x = Coordinates.X.value

    # Ensure D, dx, dy are DataArrays
    def _ensure_da(val: xr.DataArray | float, template: xr.DataArray) -> xr.DataArray:
        if isinstance(val, int | float):
            return xr.full_like(template, val)
        return val

    state = prepared["state_clean"]
    D_da = _ensure_da(prepared["D"], state)
    dx_da = _ensure_da(prepared["dx"], state)
    dy_da = _ensure_da(prepared["dy"], state)

    # Prepare boundary conditions
    bc_vals = np.array(
        [
            0 if prepared["boundary_conditions"].north == BoundaryType.CLOSED else 1,
            0 if prepared["boundary_conditions"].south == BoundaryType.CLOSED else 1,
            0 if prepared["boundary_conditions"].east == BoundaryType.CLOSED else 1,
            0 if prepared["boundary_conditions"].west == BoundaryType.CLOSED else 1,
        ],
        dtype=np.int32,
    )
    bc_da = xr.DataArray(bc_vals, dims="boundary_params")

    # Get mask
    mask = prepared["mask"]
    if mask is None:
        mask = xr.ones_like(state)

    # Call kernel
    flux_e, flux_w, flux_n, flux_s = xr.apply_ufunc(
        diffusion_flux_numba,
        state,
        D_da,
        dx_da,
        dy_da,
        prepared["face_areas"]["east"],  # ew_area
        prepared["face_areas"]["north"],  # ns_area
        mask,
        bc_da,
        input_core_dims=[
            [dim_y, dim_x],  # state
            [dim_y, dim_x],  # D
            [dim_y, dim_x],  # dx
            [dim_y, dim_x],  # dy
            [dim_y, dim_x],  # ew_area
            [dim_y, dim_x],  # ns_area
            [dim_y, dim_x],  # mask
            ["boundary_params"],  # bc
        ],
        output_core_dims=[
            [dim_y, dim_x],  # flux_east
            [dim_y, dim_x],  # flux_west
            [dim_y, dim_x],  # flux_north
            [dim_y, dim_x],  # flux_south
        ],
        dask="parallelized",
        output_dtypes=[state.dtype] * 4,
    )

    return {
        "flux_east": flux_e,
        "flux_west": flux_w,
        "flux_north": flux_n,
        "flux_south": flux_s,
    }


def compute_transport_numba(
    state: xr.DataArray,
    u: xr.DataArray,
    v: xr.DataArray,
    D: xr.DataArray | float,
    dx: xr.DataArray | float,
    dy: xr.DataArray | float,
    cell_areas: xr.DataArray,
    face_areas_ew: xr.DataArray,
    face_areas_ns: xr.DataArray,
    boundary_conditions: BoundaryConditions,
    mask: xr.DataArray | None = None,
) -> dict[str, xr.DataArray]:
    """Compute transport tendencies using Numba-accelerated implementation.

    This function computes both advection and diffusion tendencies using Numba-accelerated
    kernels for optimal performance while maintaining mass conservation.

    Args:
        state: Concentration field [Units], shape (..., lat, lon)
        u: Zonal velocity [m/s], shape (..., lat, lon)
        v: Meridional velocity [m/s], shape (..., lat, lon)
        D: Diffusion coefficient [m²/s], scalar or DataArray
        dx: Grid spacing in X [m], scalar or DataArray
        dy: Grid spacing in Y [m], scalar or DataArray
        cell_areas: Cell areas [m²], shape (lat, lon)
        face_areas_ew: East/West face areas [m], shape (lat, lon+1)
        face_areas_ns: North/South face areas [m], shape (lat+1, lon)
        boundary_conditions: Boundary conditions
        mask: Ocean/land mask (1=ocean, 0=land), shape (lat, lon)

    Returns:
        Dictionary with:
        - advection_rate: Advection tendency [Units/s]
        - diffusion_rate: Diffusion tendency [Units/s]

    Example:
        >>> result = compute_transport_numba(...)
        >>> total_tendency = result['advection_rate'] + result['diffusion_rate']
        >>> biomass_new = biomass + total_tendency * dt
    """
    # 1. PREPARE (shared data preparation)
    prepared = _prepare_transport_data(
        state, u, v, D, dx, dy, cell_areas, face_areas_ew, face_areas_ns, boundary_conditions, mask
    )

    # Store additional info needed by Numba wrappers
    prepared["boundary_conditions"] = boundary_conditions
    prepared["D"] = D
    prepared["dx"] = dx
    prepared["dy"] = dy
    prepared["u_clean"] = u.fillna(0.0)
    prepared["v_clean"] = v.fillna(0.0)

    # 2. COMPUTE FLUXES (Numba)
    flux_adv = _compute_advection_flux_numba(prepared)
    flux_diff = _compute_diffusion_flux_numba(prepared)

    # 3. DIVERGENCES (shared)
    div_adv = _compute_divergence(flux_adv, prepared["cell_areas"])
    div_diff = _compute_divergence(flux_diff, prepared["cell_areas"])

    # 4. TENDENCIES
    advection_rate = -div_adv
    diffusion_rate = -div_diff

    # Apply mask to final results
    if mask is not None:
        advection_rate = advection_rate * mask
        diffusion_rate = diffusion_rate * mask

    # Restore original dimension order
    advection_rate = advection_rate.transpose(*state.dims)
    diffusion_rate = diffusion_rate.transpose(*state.dims)

    return {
        "advection_rate": advection_rate,
        "diffusion_rate": diffusion_rate,
    }
