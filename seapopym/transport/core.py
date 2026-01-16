"""New unified transport architecture using finite volume method.

This module implements the refactored transport functions with:
- Shared data preparation
- Separate flux computation (advection and diffusion)
- Unified orchestrators for Xarray and Numba
"""

import logging
import warnings

import numpy as np
import xarray as xr

from seapopym.standard.coordinates import Coordinates, GridPosition

logger = logging.getLogger(__name__)


# =============================================================================
# CUSTOM WARNINGS
# =============================================================================


class PerformanceWarning(UserWarning):
    """Warning category for performance-related issues."""

    pass


# =============================================================================
# SHARED HELPERS
# =============================================================================


def _validate_chunking_for_transport(state: xr.DataArray) -> None:
    """Validate that chunking is optimal for transport computation.

    This function checks if the state DataArray has chunked Dask arrays, and if so,
    verifies that the core dimensions (Y and X) are NOT chunked. Chunking along
    core dimensions triggers expensive rechunking operations that degrade performance.

    Args:
        state: The state DataArray to validate

    Warnings:
        Issues a performance warning if Y or X dimensions are chunked.
    """
    # Only validate if it's a Dask array
    if not hasattr(state.data, "chunks") or state.data.chunks is None:
        return

    dim_y = Coordinates.Y.value
    dim_x = Coordinates.X.value

    # Check if Y and X are in the dimensions
    if dim_y not in state.dims or dim_x not in state.dims:
        return

    # Get the chunk structure for Y and X
    dim_y_idx = state.dims.index(dim_y)
    dim_x_idx = state.dims.index(dim_x)

    y_chunks = state.data.chunks[dim_y_idx]
    x_chunks = state.data.chunks[dim_x_idx]

    # Check if Y or X are chunked (more than one chunk)
    y_is_chunked = len(y_chunks) > 1
    x_is_chunked = len(x_chunks) > 1

    if y_is_chunked or x_is_chunked:
        # Build detailed warning message
        chunk_info = []
        if y_is_chunked:
            chunk_info.append(f"  - {dim_y}: {len(y_chunks)} chunks {y_chunks}")
        if x_is_chunked:
            chunk_info.append(f"  - {dim_x}: {len(x_chunks)} chunks {x_chunks}")

        # Identify non-core dimensions that SHOULD be chunked
        non_core_dims = [d for d in state.dims if d not in [dim_y, dim_x]]
        suggested_chunks = {dim_y: -1, dim_x: -1}
        if non_core_dims:
            # Suggest chunking along first non-core dimension (typically 'cohort')
            suggested_chunks[non_core_dims[0]] = 1

        warnings.warn(
            "Transport computation: Core dimensions (Y, X) are chunked, which will trigger expensive rechunking.\n"
            "Current chunking:\n" + "\n".join(chunk_info) + "\n\n"
            "Performance impact:\n"
            "  - Rechunking causes data shuffling between workers\n"
            "  - Increased memory usage and communication overhead\n"
            "  - Can significantly slow down transport operations\n\n",
            PerformanceWarning,
            stacklevel=4,
        )
        logger.warning(
            f"Transport: Core dimensions are chunked. Y: {len(y_chunks)} chunks, X: {len(x_chunks)} chunks. "
            f"This will trigger rechunking."
        )


def _ensure_dataarray(value: xr.DataArray | float, template: xr.DataArray) -> xr.DataArray:
    """Convert scalar and non-spatial DataArrays to DataArray with spatial dimensions.

    Args:
        value: Scalar or DataArray to ensure has correct spatial dimensions
        template: Template DataArray for spatial shape/dims

    Returns:
        DataArray with at least the same spatial dimensions as template
    """
    dim_y = Coordinates.Y.value
    dim_x = Coordinates.X.value

    if not isinstance(value, xr.DataArray):
        # Bare float/int
        return xr.full_like(template, value)

    # It's a DataArray. Check if it has the required spatial dimensions.
    if dim_y not in value.dims or dim_x not in value.dims:
        # 0D DataArray or DataArray with other dims (e.g. time) but no space.
        # Broadcast to template's spatial grid.
        # We use a simple addition with zeros to handle broadcasting.
        # This preserves 'value' attributes and extra dimensions (like time).
        return value + xr.zeros_like(template)

    return value


def _encode_boundary_conditions(
    boundary_north: int | float,
    boundary_south: int | float,
    boundary_east: int | float,
    boundary_west: int | float,
) -> xr.DataArray:
    """Encode boundary conditions as integer array for Numba kernels.

    Args:
        boundary_north: Northern boundary condition (0=CLOSED, 1=OPEN, 2=PERIODIC)
        boundary_south: Southern boundary condition (0=CLOSED, 1=OPEN, 2=PERIODIC)
        boundary_east: Eastern boundary condition (0=CLOSED, 1=OPEN, 2=PERIODIC)
        boundary_west: Western boundary condition (0=CLOSED, 1=OPEN, 2=PERIODIC)

    Returns:
        DataArray with shape (4,) encoding [north, south, east, west] as 0=CLOSED, 1=PERIODIC
    """
    bc_vals = np.array(
        [int(boundary_north), int(boundary_south), int(boundary_east), int(boundary_west)],
        dtype=np.int32,
    )
    return xr.DataArray(bc_vals, dims="boundary_params")


# =============================================================================
# FLUX COMPUTATION (NUMBA)
# =============================================================================


def compute_transport_fv(
    state: xr.DataArray,
    u: xr.DataArray,
    v: xr.DataArray,
    D: xr.DataArray | float,
    dx: xr.DataArray | float,
    dy: xr.DataArray | float,
    cell_areas: xr.DataArray,
    face_areas_ew: xr.DataArray,
    face_areas_ns: xr.DataArray,
    mask: xr.DataArray | None = None,
    boundary_north: int | float = 0,
    boundary_south: int | float = 0,
    boundary_east: int | float = 0,
    boundary_west: int | float = 0,
) -> dict[str, xr.DataArray]:
    """Compute transport tendencies using Numba-accelerated finite volume method.

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
        mask: Ocean/land mask (1=ocean, 0=land), shape (lat, lon)
        boundary_north: Northern boundary condition (0=CLOSED, 1=OPEN, 2=PERIODIC)
        boundary_south: Southern boundary condition (0=CLOSED, 1=OPEN, 2=PERIODIC)
        boundary_east: Eastern boundary condition (0=CLOSED, 1=OPEN, 2=PERIODIC)
        boundary_west: Western boundary condition (0=CLOSED, 1=OPEN, 2=PERIODIC)

    Returns:
        Dictionary with:
        - advection_rate: Advection tendency [Units/s]
        - diffusion_rate: Diffusion tendency [Units/s]
    """
    try:
        from seapopym.transport.numba_kernels import advection_flux_numba, diffusion_flux_numba
    except ImportError as err:
        raise ImportError("Numba is required for Numba implementation.") from err

    # --- 0. VALIDATE CHUNKING ---
    # Check if chunking is optimal for transport (warn if Y or X are chunked)
    _validate_chunking_for_transport(state)

    dim_y = Coordinates.Y.value
    dim_x = Coordinates.X.value
    x_face_dim = GridPosition.get_face_dim(Coordinates.X, GridPosition.LEFT)
    y_face_dim = GridPosition.get_face_dim(Coordinates.Y, GridPosition.LEFT)

    # --- 1. DATA PREPARATION ---

    # Encode boundary conditions for Numba kernels
    bc_da = _encode_boundary_conditions(
        boundary_north, boundary_south, boundary_east, boundary_west
    )

    # Clean inputs (replace NaN with 0 to prevent propagation)
    state_clean = state.fillna(0.0)
    u_clean = u.fillna(0.0)
    v_clean = v.fillna(0.0)

    # Ensure scalar parameters are converted to DataArrays
    D_da = _ensure_dataarray(D, state_clean)
    dx_da = _ensure_dataarray(dx, state_clean)
    dy_da = _ensure_dataarray(dy, state_clean)

    # Default mask to all ocean if not provided
    if mask is None:
        mask = xr.ones_like(state_clean)

    # Extract face areas aligned with cell centers
    # Drop coordinates to prevent alignment errors (face coords != center coords)
    ew_area = (
        face_areas_ew.isel({x_face_dim: slice(1, None)})
        .rename({x_face_dim: dim_x})
        .drop_vars([dim_x, dim_y], errors="ignore")
    )
    ns_area = (
        face_areas_ns.isel({y_face_dim: slice(1, None)})
        .rename({y_face_dim: dim_y})
        .drop_vars([dim_x, dim_y], errors="ignore")
    )

    # --- 2. ADVECTION FLUXES ---
    flux_adv_e, flux_adv_w, flux_adv_n, flux_adv_s = xr.apply_ufunc(
        advection_flux_numba,
        state_clean,
        u_clean,
        v_clean,
        ew_area,
        ns_area,
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
            [dim_y, dim_x],
            [dim_y, dim_x],
            [dim_y, dim_x],
            [dim_y, dim_x],
        ],
        dask="parallelized",
        output_dtypes=[state_clean.dtype] * 4,
        dask_gufunc_kwargs={"allow_rechunk": False},
    )

    # --- 3. DIFFUSION FLUXES ---
    flux_diff_e, flux_diff_w, flux_diff_n, flux_diff_s = xr.apply_ufunc(
        diffusion_flux_numba,
        state_clean,
        D_da,
        dx_da,
        dy_da,
        ew_area,
        ns_area,
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
            [dim_y, dim_x],
            [dim_y, dim_x],
            [dim_y, dim_x],
            [dim_y, dim_x],
        ],
        dask="parallelized",
        output_dtypes=[state_clean.dtype] * 4,
        dask_gufunc_kwargs={"allow_rechunk": False},
    )

    # --- 4. COMPUTE TENDENCIES ---

    # Compute flux divergence (inline for minimal overhead)
    div_adv = (flux_adv_e - flux_adv_w + flux_adv_n - flux_adv_s) / cell_areas
    div_diff = (flux_diff_e - flux_diff_w + flux_diff_n - flux_diff_s) / cell_areas

    # Tendencies are negative divergence (conservation form)
    advection_rate = -div_adv
    diffusion_rate = -div_diff

    # Mask land cells in final result
    if mask is not None:
        advection_rate = advection_rate * mask
        diffusion_rate = diffusion_rate * mask

    return {
        "advection_rate": advection_rate.transpose(*state.dims),
        "diffusion_rate": diffusion_rate.transpose(*state.dims),
    }
