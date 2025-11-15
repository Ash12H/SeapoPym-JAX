"""Advection scheme for transport computations.

This module implements flux-based upwind advection using the finite volume method.
This approach guarantees mass conservation by construction.

References:
    - Upwind method (volumes finis): IA/Advection-upwind-description.md
    - Comparison with gradients approach: IA/TRANSPORT_ANALYSIS.md (Section 8)
"""

import jax.numpy as jnp

from seapopym_message.transport.boundary import BoundaryConditions, get_neighbors_with_bc
from seapopym_message.transport.grid import Grid


def advection_upwind_flux(
    biomass: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    dt: float,
    grid: Grid,
    boundary: BoundaryConditions,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute advection using upwind finite volume method.

    This function implements the flux-based upwind scheme described in
    IA/Advection-upwind-description.md. The principle is:

        dC/dt = -(F_east - F_west + F_north - F_south) / Volume
        F_face = u_face × C_upwind × Area_face

    where C_upwind is chosen based on flow direction (upwind scheme).

    Mass conservation is guaranteed by construction: total change equals
    sum of boundary fluxes (net flux through domain edges).

    Args:
        biomass: Concentration field [kg/m²], shape (nlat, nlon)
        u: Velocity in x/longitude direction [m/s], shape (nlat, nlon)
           Positive = eastward
        v: Velocity in y/latitude direction [m/s], shape (nlat, nlon)
           Positive = northward
        dt: Time step [s]
        grid: Grid object providing cell areas and face areas
        boundary: Boundary conditions for domain edges
        mask: Optional land/sea mask (1=ocean, 0=land, NaN=land), shape (nlat, nlon)
              If provided, sets u=0, v=0 at land cells to enforce zero flux

    Returns:
        Updated biomass field [kg/m²], shape (nlat, nlon)

    Mathematical formulation:
        For cell (i,j), the flux through each face is:
        - East face: F_e = u_e × C_upwind_e × A_e
          where C_upwind_e = C[i,j] if u_e > 0, else C[i,j+1]
        - West face: F_w = u_w × C_upwind_w × A_w
        - North face: F_n = v_n × C_upwind_n × A_n
        - South face: F_s = v_s × C_upwind_s × A_s

        Net flux divergence:
        div_F = (F_e - F_w + F_n - F_s) / Volume[i,j]

        Update:
        C^{n+1}[i,j] = C^n[i,j] - dt × div_F

    Land masking (IA/Advection-upwind-description.md, lines 99-109):
        Setting u=0, v=0 at land/ocean interfaces ensures zero flux.
        This is simpler than modifying the upwind choice.

    Example:
        >>> grid = SphericalGrid(-60, 60, 0, 360, 120, 360)
        >>> biomass = jnp.ones((120, 360))
        >>> u = jnp.ones((120, 360)) * 0.1  # 0.1 m/s eastward
        >>> v = jnp.zeros((120, 360))
        >>> bc = BoundaryConditions(CLOSED, CLOSED, PERIODIC, PERIODIC)
        >>> biomass_new = advection_upwind_flux(biomass, u, v, 3600.0, grid, bc)
    """
    nlat, nlon = biomass.shape

    # Apply land mask if provided
    if mask is not None:
        # Create ocean mask (1 where ocean, 0 where land)
        ocean_mask = jnp.where(jnp.isnan(mask), 0.0, mask)
        # Set velocities to zero on land
        u = u * ocean_mask
        v = v * ocean_mask

    # Get cell areas and volumes (for 2D, volume = area)
    cell_areas = grid.cell_areas()  # (nlat, nlon)
    face_areas_ew = grid.face_areas_ew()  # (nlat, nlon+1)
    face_areas_ns = grid.face_areas_ns()  # (nlat+1, nlon)

    # Get neighbor biomass values with boundary conditions
    biomass_west, biomass_east, biomass_south, biomass_north = get_neighbors_with_bc(
        biomass, boundary
    )

    # Compute velocities at cell faces (average of adjacent cells)
    # For East/West faces (longitude direction)
    # u_face_ew[i, j] is velocity at face between cells [i, j-1] and [i, j]
    # Shape: (nlat, nlon+1)

    # Get neighbor velocities with BC
    u_west, u_east, _, _ = get_neighbors_with_bc(u, boundary)

    # East face of cell (i,j): average of u[i,j] and u[i,j+1]
    u_face_east = (u + u_east) / 2  # (nlat, nlon)
    # West face of cell (i,j): average of u[i,j-1] and u[i,j]
    u_face_west = (u_west + u) / 2  # (nlat, nlon)

    # For North/South faces (latitude direction)
    _, _, v_south, v_north = get_neighbors_with_bc(v, boundary)

    # North face of cell (i,j): average of v[i,j] and v[i+1,j]
    v_face_north = (v + v_north) / 2  # (nlat, nlon)
    # South face of cell (i,j): average of v[i-1,j] and v[i,j]
    v_face_south = (v_south + v) / 2  # (nlat, nlon)

    # Upwind choice for biomass at faces
    # East face: if u_e > 0, use C[i,j], else use C[i,j+1]
    biomass_face_east = jnp.where(u_face_east > 0, biomass, biomass_east)
    # West face: if u_w > 0, use C[i,j-1], else use C[i,j]
    biomass_face_west = jnp.where(u_face_west > 0, biomass_west, biomass)

    # North face: if v_n > 0, use C[i,j], else use C[i+1,j]
    biomass_face_north = jnp.where(v_face_north > 0, biomass, biomass_north)
    # South face: if v_s > 0, use C[i-1,j], else use C[i,j]
    biomass_face_south = jnp.where(v_face_south > 0, biomass_south, biomass)

    # Compute fluxes at faces [kg/s]
    # Flux = velocity × concentration × area
    # Need to extract correct face areas for each cell

    # For E/W faces, we need areas at i and i+1
    # face_areas_ew has shape (nlat, nlon+1)
    # East face of cell (i,j) is at index j+1 in the face array
    # West face of cell (i,j) is at index j in the face array

    # Extract face areas for each cell
    area_east = face_areas_ew[:, 1:]  # (nlat, nlon) - east faces
    area_west = face_areas_ew[:, :-1]  # (nlat, nlon) - west faces

    flux_east = u_face_east * biomass_face_east * area_east
    flux_west = u_face_west * biomass_face_west * area_west

    # For N/S faces
    # face_areas_ns has shape (nlat+1, nlon)
    # North face of cell (i,j) is at index i+1 in the face array
    # South face of cell (i,j) is at index i in the face array

    area_north = face_areas_ns[1:, :]  # (nlat, nlon) - north faces
    area_south = face_areas_ns[:-1, :]  # (nlat, nlon) - south faces

    flux_north = v_face_north * biomass_face_north * area_north
    flux_south = v_face_south * biomass_face_south * area_south

    # Compute flux divergence [kg/(m²·s)]
    # div(F) = (F_out - F_in) / Volume
    # Positive flux = outward, negative = inward
    # Net flux out = F_east - F_west + F_north - F_south
    flux_divergence = (flux_east - flux_west + flux_north - flux_south) / cell_areas

    # Update biomass (Euler forward)
    # dC/dt = -div(F)
    biomass_new = biomass - dt * flux_divergence

    # Apply mask to final result (ensure land stays at zero if masked)
    if mask is not None:
        biomass_new = biomass_new * ocean_mask

    return biomass_new


def compute_advection_diagnostics(
    biomass: jnp.ndarray,
    biomass_new: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    dt: float,
    grid: Grid,
    mask: jnp.ndarray | None = None,
) -> dict:
    """Compute diagnostics for advection step.

    Args:
        biomass: Initial biomass [kg/m²]
        biomass_new: Final biomass after advection [kg/m²]
        u, v: Velocity fields [m/s]
        dt: Time step [s]
        grid: Grid object
        mask: Optional land/sea mask

    Returns:
        Dictionary with:
        - total_mass_before: Total mass before [kg]
        - total_mass_after: Total mass after [kg]
        - mass_change: Absolute mass change [kg]
        - conservation_fraction: Fraction of mass conserved (should be ~1.0)
        - max_velocity: Maximum velocity magnitude [m/s]
        - cfl_number: Maximum CFL number (u*dt/dx + v*dt/dy)
    """
    cell_areas = grid.cell_areas()

    # Apply mask if provided
    if mask is not None:
        ocean_mask = jnp.where(jnp.isnan(mask), 0.0, mask)
    else:
        ocean_mask = jnp.ones_like(biomass)

    # Total mass [kg]
    total_before = jnp.sum(biomass * cell_areas * ocean_mask)
    total_after = jnp.sum(biomass_new * cell_areas * ocean_mask)

    mass_change = jnp.abs(total_after - total_before)
    conservation = total_after / (total_before + 1e-10)  # Avoid division by zero

    # Velocity magnitude
    vel_magnitude = jnp.sqrt(u**2 + v**2)
    max_vel = jnp.max(vel_magnitude)

    # CFL number
    dx = grid.dx()  # May be array (nlat,) for spherical or scalar for plane
    dy = grid.dy()  # Scalar

    # For spherical grid, dx is array, so we need to handle both cases
    nlat, nlon = biomass.shape
    dx_2d = dx[:, None] * jnp.ones((nlat, nlon)) if isinstance(dx, jnp.ndarray) else dx

    cfl_x = jnp.abs(u) * dt / dx_2d
    cfl_y = jnp.abs(v) * dt / dy
    cfl_max = jnp.max(cfl_x + cfl_y)

    return {
        "total_mass_before": float(total_before),
        "total_mass_after": float(total_after),
        "mass_change": float(mass_change),
        "conservation_fraction": float(conservation),
        "max_velocity": float(max_vel),
        "cfl_number": float(cfl_max),
    }
