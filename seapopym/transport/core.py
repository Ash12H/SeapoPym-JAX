"""Transport processes (Advection and Diffusion) adapted for the Blueprint architecture.

These functions compute tendencies (rates of change) for advection and diffusion
using flux-based finite volume methods that conserve mass on spherical grids.

Key features:
- Upwind advection scheme for stability
- Explicit diffusion with varying dx(lat)
- Proper handling of boundary conditions
- Land/ocean masking support
- Mass conservation through flux-based approach

References:
    - Original implementation: IA/transport/advection.py, IA/transport/diffusion.py
    - Flux conservation: IA/TRANSPORT_ANALYSIS.md (Section 8)
    - Spherical geometry: IA/Diffusion-euler-explicite-description.md
"""

from typing import Any

import xarray as xr

from seapopym.standard.coordinates import Coordinates, GridPosition
from seapopym.transport.boundary import BoundaryConditions, BoundaryType, get_neighbors_with_bc


def compute_advection_tendency(
    state: xr.DataArray,
    u: xr.DataArray,
    v: xr.DataArray,
    cell_areas: xr.DataArray,
    face_areas_ew: xr.DataArray,
    face_areas_ns: xr.DataArray,
    boundary_conditions: BoundaryConditions,
    mask: xr.DataArray | None = None,
) -> dict[str, Any]:
    """Compute advection tendency using upwind flux scheme.

    This function implements a finite volume advection scheme that conserves mass:
        dC/dt = -∇·(u⃗C) = -(∂(uC)/∂x + ∂(vC)/∂y)

    Discretized as:
        tendency = -(flux_divergence)
        flux_divergence = (flux_east - flux_west + flux_north - flux_south) / cell_area

    where fluxes are computed using an upwind scheme:
        flux_east = u_face_east × C_upwind × face_area_east

    The upwind choice ensures stability by taking concentration from the upwind cell:
        C_upwind = C_current if u > 0 else C_neighbor

    Boundary conditions:
    - CLOSED: No flux through boundary (face_mask = 0)
    - PERIODIC: Wrap around (handled by get_neighbors_with_bc)
    - OPEN: Allow flux through boundary

    Land masking:
    - Ocean-ocean faces: Normal flux
    - Ocean-land faces: Zero flux (face_mask = 0)
    - Land-land faces: Zero flux

    Args:
        state: Concentration field [Units], shape (..., lat, lon)
        u: Zonal velocity (positive East) [m/s], shape (..., lat, lon)
        v: Meridional velocity (positive North) [m/s], shape (..., lat, lon)
        cell_areas: Cell areas [m²], shape (lat, lon)
        face_areas_ew: East/West face areas [m], shape (lat, lon+1) or (lat, lon)
        face_areas_ns: North/South face areas [m], shape (lat+1, lon) or (lat, lon)
        boundary_conditions: BoundaryConditions object specifying edge behavior
        mask: Optional ocean/land mask (1=ocean, 0=land), shape (lat, lon)

    Returns:
        Dictionary with:
        - advection_rate: Advection tendency [Units/s], same shape as state

    Example:
        >>> from seapopym.transport import compute_advection_tendency
        >>> from seapopym.transport.boundary import BoundaryConditions, BoundaryType
        >>> from seapopym.transport.grid import compute_spherical_cell_areas, ...
        >>>
        >>> # Setup grid
        >>> cell_areas = compute_spherical_cell_areas(lats, lons)
        >>> face_areas_ew = compute_spherical_face_areas_ew(lats, lons)
        >>> face_areas_ns = compute_spherical_face_areas_ns(lats, lons)
        >>> bc = BoundaryConditions(
        ...     north=BoundaryType.CLOSED,
        ...     south=BoundaryType.CLOSED,
        ...     east=BoundaryType.PERIODIC,
        ...     west=BoundaryType.PERIODIC,
        ... )
        >>>
        >>> # Compute tendency
        >>> result = compute_advection_tendency(
        ...     state=biomass,
        ...     u=current_u,
        ...     v=current_v,
        ...     cell_areas=cell_areas,
        ...     face_areas_ew=face_areas_ew,
        ...     face_areas_ns=face_areas_ns,
        ...     boundary_conditions=bc,
        ...     mask=ocean_mask,
        ... )
        >>> tendency = result["advection_rate"]

    Reference:
        IA/transport/advection.py: advection_upwind_flux()
    """
    # Get neighbor values for state
    state_west, state_east, state_south, state_north = get_neighbors_with_bc(
        state, boundary_conditions
    )

    # Get neighbor values for velocities
    u_west, u_east, u_south, u_north = get_neighbors_with_bc(u, boundary_conditions)
    v_west, v_east, v_south, v_north = get_neighbors_with_bc(v, boundary_conditions)

    # Clean velocities (replace NaN with 0)
    u_clean = u.fillna(0.0)
    v_clean = v.fillna(0.0)

    # --- VELOCITY INTERPOLATION TO FACES ---
    # Velocities are defined at cell centers, but we need them at faces
    # Use simple averaging (can be improved with flux reconstruction)

    # East face (i+1/2): average of current and east neighbor
    u_face_east = 0.5 * (u_clean + u_east.fillna(0.0))
    # West face (i-1/2): average of current and west neighbor
    u_face_west = 0.5 * (u_clean + u_west.fillna(0.0))

    # North face (j+1/2): average of current and north neighbor
    v_face_north = 0.5 * (v_clean + v_north.fillna(0.0))
    # South face (j-1/2): average of current and south neighbor
    v_face_south = 0.5 * (v_clean + v_south.fillna(0.0))

    # --- UPWIND CONCENTRATION SELECTION ---
    # Choose concentration from upwind cell based on velocity direction

    # East face: if u_face_east > 0, flow is West→East, take current cell
    state_face_east = xr.where(u_face_east > 0, state, state_east)
    # West face: if u_face_west > 0, flow is West→East, take west neighbor
    state_face_west = xr.where(u_face_west > 0, state_west, state)

    # North face: if v_face_north > 0, flow is South→North, take current cell
    state_face_north = xr.where(v_face_north > 0, state, state_north)
    # South face: if v_face_south > 0, flow is South→North, take south neighbor
    state_face_south = xr.where(v_face_south > 0, state_south, state)

    # --- FACE MASKING ---
    # Create face masks to handle land boundaries and closed boundaries

    # Start with all faces open (mask = 1)
    face_mask_east = xr.ones_like(state)
    face_mask_west = xr.ones_like(state)
    face_mask_north = xr.ones_like(state)
    face_mask_south = xr.ones_like(state)

    # Apply land masking if provided
    if mask is not None:
        # Ocean mask: 1 = ocean, 0 = land
        # A face is open only if both adjacent cells are ocean
        mask_west, mask_east, mask_south, mask_north = get_neighbors_with_bc(
            mask, boundary_conditions
        )

        # East face: open if current cell AND east neighbor are ocean
        face_mask_east = face_mask_east * mask * mask_east
        # West face: open if current cell AND west neighbor are ocean
        face_mask_west = face_mask_west * mask * mask_west
        # North face: open if current cell AND north neighbor are ocean
        face_mask_north = face_mask_north * mask * mask_north
        # South face: open if current cell AND south neighbor are ocean
        face_mask_south = face_mask_south * mask * mask_south

    # Apply CLOSED boundary conditions
    # At closed boundaries, set face mask to 0 (no flux)

    # Use standardized coordinate names from Coordinates enum
    dim_y = Coordinates.Y.value  # "y"
    dim_x = Coordinates.X.value  # "x"

    if boundary_conditions.east == BoundaryType.CLOSED:
        # Close east boundary: mask cells at last column (no flux through east face)
        # Using where() with coordinate comparison to set boundary values to 0
        face_mask_east = face_mask_east.where(
            face_mask_east[dim_x] != face_mask_east[dim_x][-1], 0.0
        )

    if boundary_conditions.west == BoundaryType.CLOSED:
        # Close west boundary: mask cells at first column (no flux through west face)
        face_mask_west = face_mask_west.where(
            face_mask_west[dim_x] != face_mask_west[dim_x][0], 0.0
        )

    if boundary_conditions.north == BoundaryType.CLOSED:
        # Close north boundary: mask cells at last row (no flux through north face)
        face_mask_north = face_mask_north.where(
            face_mask_north[dim_y] != face_mask_north[dim_y][-1], 0.0
        )

    if boundary_conditions.south == BoundaryType.CLOSED:
        # Close south boundary: mask cells at first row (no flux through south face)
        face_mask_south = face_mask_south.where(
            face_mask_south[dim_y] != face_mask_south[dim_y][0], 0.0
        )

    # --- FLUX CALCULATION ---
    # Flux = velocity × concentration × face_area × face_mask
    # Units: [m/s] × [Units] × [m] × [1] = [Units×m²/s]

    # Extract face areas from staggered arrays using dimensional slicing
    # face_areas_ew: dims (y, x_left) - shape (nlat, nlon+1)
    # face_areas_ns: dims (y_left, x) - shape (nlat+1, nlon)
    #
    # Staggered grid layout (Xgcm convention):
    # - x_left: face positions at west edges (nlon+1 faces for nlon cells)
    # - y_left: face positions at south edges (nlat+1 faces for nlat cells)
    #
    # For each cell, we need:
    # - East face: x_left[i+1] (index 1:)
    # - West face: x_left[i] (index :-1)
    # - North face: y_left[j+1] (index 1:)
    # - South face: y_left[j] (index :-1)

    # Get dimension names following Xgcm convention
    x_face_dim = GridPosition.get_face_dim(Coordinates.X, GridPosition.LEFT)  # "x_left"
    y_face_dim = GridPosition.get_face_dim(Coordinates.Y, GridPosition.LEFT)  # "y_left"

    # Extract face areas using dimensional slicing
    area_east = face_areas_ew.isel({x_face_dim: slice(1, None)}).values
    area_west = face_areas_ew.isel({x_face_dim: slice(None, -1)}).values
    area_north = face_areas_ns.isel({y_face_dim: slice(1, None)}).values
    area_south = face_areas_ns.isel({y_face_dim: slice(None, -1)}).values

    # Identify extra dimensions beyond (y, x) to loop over them
    dim_y = Coordinates.Y.value
    dim_x = Coordinates.X.value
    core_dims = [dim_y, dim_x]

    # Define inner function that computes flux for 2D slices
    def compute_flux_2d(
        u_face_val: float, state_face_val: float, area_val: float, mask_val: float
    ) -> float:
        """Compute flux for a 2D (y, x) slice."""
        return u_face_val * state_face_val * area_val * mask_val

    # Use apply_ufunc to loop over extra dimensions automatically
    # This applies the 2D flux computation to each slice along extra dims
    flux_east = xr.apply_ufunc(
        compute_flux_2d,
        u_face_east,
        state_face_east,
        xr.DataArray(area_east, dims=core_dims),
        face_mask_east,
        input_core_dims=[core_dims, core_dims, core_dims, core_dims],
        output_core_dims=[core_dims],
        vectorize=True,
    )
    flux_west = xr.apply_ufunc(
        compute_flux_2d,
        u_face_west,
        state_face_west,
        xr.DataArray(area_west, dims=core_dims),
        face_mask_west,
        input_core_dims=[core_dims, core_dims, core_dims, core_dims],
        output_core_dims=[core_dims],
        vectorize=True,
    )
    flux_north = xr.apply_ufunc(
        compute_flux_2d,
        v_face_north,
        state_face_north,
        xr.DataArray(area_north, dims=core_dims),
        face_mask_north,
        input_core_dims=[core_dims, core_dims, core_dims, core_dims],
        output_core_dims=[core_dims],
        vectorize=True,
    )
    flux_south = xr.apply_ufunc(
        compute_flux_2d,
        v_face_south,
        state_face_south,
        xr.DataArray(area_south, dims=core_dims),
        face_mask_south,
        input_core_dims=[core_dims, core_dims, core_dims, core_dims],
        output_core_dims=[core_dims],
        vectorize=True,
    )

    # --- FLUX DIVERGENCE ---
    # Divergence = (flux_in - flux_out) / cell_volume
    # For 2D: divergence = (flux_east - flux_west + flux_north - flux_south) / cell_area

    # Note: The sign convention is:
    # - Positive flux_east = flow leaving through east face
    # - Negative flux_west = flow entering through west face
    # Net outflow = flux_east - flux_west + flux_north - flux_south

    flux_divergence = (flux_east - flux_west + flux_north - flux_south) / cell_areas

    # --- TENDENCY ---
    # Advection tendency = -divergence(flux)
    advection_rate = -flux_divergence

    # Apply mask to final result (ensure land stays at zero)
    if mask is not None:
        advection_rate = advection_rate * mask

    # Restore original dimension order (apply_ufunc may reorder them)
    advection_rate = advection_rate.transpose(*state.dims)

    return {"advection_rate": advection_rate}


def compute_diffusion_tendency(
    state: xr.DataArray,
    D: xr.DataArray | float,
    dx: xr.DataArray | float,
    dy: xr.DataArray | float,
    boundary_conditions: BoundaryConditions,
    mask: xr.DataArray | None = None,
) -> dict[str, Any]:
    """Compute diffusion tendency using explicit Laplacian with varying dx.

    This function implements the diffusion equation:
        dC/dt = D × ∇²C = D × (∂²C/∂x² + ∂²C/∂y²)

    The Laplacian is approximated using centered finite differences:
        ∂²C/∂x² ≈ (C_east - 2C + C_west) / dx²
        ∂²C/∂y² ≈ (C_north - 2C + C_south) / dy²

    For spherical grids, dx varies with latitude:
        dx(lat) = R × cos(lat) × dλ

    This variation must be accounted for in the discretization.

    Boundary conditions:
    - CLOSED/OPEN: Zero-gradient (Neumann BC: ∂C/∂n = 0)
      Implemented by copying edge values to ghost cells
    - PERIODIC: Wrap around (handled by get_neighbors_with_bc)

    Land masking:
    - Ocean cells: Normal diffusion
    - Land cells: Set to zero
    - Ocean-land boundaries: Zero-gradient BC (copy ocean value to land neighbor)
      This prevents artificial gradients and mass loss

    Stability constraint:
        dt ≤ min(dx², dy²) / (4 × D)

    Use check_diffusion_stability() from stability.py to verify before running.

    Args:
        state: Concentration field [Units], shape (..., lat, lon)
        D: Diffusion coefficient [m²/s], scalar or DataArray
        dx: Grid spacing in X direction [m], shape (lat, lon) or scalar
        dy: Grid spacing in Y direction [m], shape (lat, lon) or scalar
        boundary_conditions: BoundaryConditions object specifying edge behavior
        mask: Optional ocean/land mask (1=ocean, 0=land), shape (lat, lon)

    Returns:
        Dictionary with:
        - diffusion_rate: Diffusion tendency [Units/s], same shape as state

    Example:
        >>> from seapopym.transport import compute_diffusion_tendency
        >>> from seapopym.transport.boundary import BoundaryConditions, BoundaryType
        >>> from seapopym.transport.grid import compute_spherical_dx, compute_spherical_dy
        >>> from seapopym.transport.stability import check_diffusion_stability
        >>>
        >>> # Setup grid
        >>> dx = compute_spherical_dx(lats, lons)
        >>> dy = compute_spherical_dy(lats, lons)
        >>> bc = BoundaryConditions(
        ...     north=BoundaryType.CLOSED,
        ...     south=BoundaryType.CLOSED,
        ...     east=BoundaryType.PERIODIC,
        ...     west=BoundaryType.PERIODIC,
        ... )
        >>>
        >>> # Check stability
        >>> stability = check_diffusion_stability(D=1000.0, dx=dx, dy=dy, dt=3600.0)
        >>> if not stability["is_stable"]:
        ...     raise ValueError(f"Unstable! Reduce dt to {stability['dt_max']:.1f} s")
        >>>
        >>> # Compute tendency
        >>> result = compute_diffusion_tendency(
        ...     state=biomass,
        ...     D=1000.0,
        ...     dx=dx,
        ...     dy=dy,
        ...     boundary_conditions=bc,
        ...     mask=ocean_mask,
        ... )
        >>> tendency = result["diffusion_rate"]

    Reference:
        IA/transport/diffusion.py: diffusion_explicit_spherical()
    """
    # Get neighbor values for state
    state_west, state_east, state_south, state_north = get_neighbors_with_bc(
        state, boundary_conditions
    )

    # Apply land masking to state and neighbors (Neumann BC at ocean-land boundaries)
    if mask is not None:
        # Ocean mask: 1 = ocean, 0 = land
        # Apply mask to current state (set land to zero)
        state_clean = state * mask

        # Get neighbor masks
        mask_west, mask_east, mask_south, mask_north = get_neighbors_with_bc(
            mask, boundary_conditions
        )

        # For ocean-land boundaries, use zero-gradient BC:
        # If neighbor is land (mask=0), replace neighbor value with current cell value
        # This ensures ∂C/∂n = 0 at the boundary

        # East neighbor: if land, use current value
        state_east = xr.where(mask_east == 0, state_clean, state_east)
        # West neighbor: if land, use current value
        state_west = xr.where(mask_west == 0, state_clean, state_west)
        # North neighbor: if land, use current value
        state_north = xr.where(mask_north == 0, state_clean, state_north)
        # South neighbor: if land, use current value
        state_south = xr.where(mask_south == 0, state_clean, state_south)

        # Use masked state for computation
        state = state_clean
    else:
        # No masking needed
        pass

    # --- LAPLACIAN CALCULATION ---
    # Compute second derivatives using centered differences

    # Compute squared grid spacings (works for both scalar and DataArray)
    dx_sq = dx**2
    dy_sq = dy**2

    # Second derivative in x direction (longitude)
    # ∂²C/∂x² ≈ (C_east - 2C + C_west) / dx²
    d2C_dx2 = (state_east - 2 * state + state_west) / dx_sq

    # Second derivative in y direction (latitude)
    # ∂²C/∂y² ≈ (C_north - 2C + C_south) / dy²
    d2C_dy2 = (state_north - 2 * state + state_south) / dy_sq

    # Laplacian: ∇²C = ∂²C/∂x² + ∂²C/∂y²
    laplacian = d2C_dx2 + d2C_dy2

    # --- DIFFUSION TENDENCY ---
    # dC/dt = D × ∇²C
    diffusion_rate = D * laplacian

    # Apply mask to final result (ensure land stays at zero)
    if mask is not None:
        diffusion_rate = diffusion_rate * mask

    return {"diffusion_rate": diffusion_rate}
