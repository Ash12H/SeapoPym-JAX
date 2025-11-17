"""Advection scheme for transport computations.

This module implements flux-based upwind advection using the finite volume method.
This approach guarantees mass conservation by construction.

References:
    - Upwind method (volumes finis): IA/Advection-upwind-description.md
    - Comparison with gradients approach: IA/TRANSPORT_ANALYSIS.md (Section 8)
"""

import jax.numpy as jnp

from seapopym_message.transport.boundary import (
    BoundaryConditions,
    BoundaryType,
    get_neighbors_with_bc,
)
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

    # --- FIX 1 & 2 : Nettoyage des vitesses et des masques ---

    # Vitesses "propres" (u_clean, v_clean)
    u_clean = u
    v_clean = v

    # Appliquer le masque si fourni
    if mask is not None:
        # Créer le masque océan (1=ocean, 0=terre)
        ocean_mask = jnp.where(jnp.isnan(mask), 0.0, mask)

        # FIX 2: Remplacer les NaN dans les vitesses par 0.0
        # Empêche (NaN * 0.0 = NaN) et (v_ocean + NaN = NaN)
        u_clean = jnp.nan_to_num(u, nan=0.0)
        v_clean = jnp.nan_to_num(v, nan=0.0)

        # FIX 1: SUPPRESSION des lignes `u = u * ocean_mask`
        # Nous ne modifions PAS les vitesses au centre des cellules.

    else:
        # S'il n'y a pas de masque, tout est océan
        ocean_mask = jnp.ones_like(biomass)

    # --- Fin des FIX 1 & 2 ---

    # Obtenir les géométries (inchangé)
    cell_areas = grid.cell_areas()
    face_areas_ew = grid.face_areas_ew()
    face_areas_ns = grid.face_areas_ns()

    # Obtenir les voisins de la biomasse (inchangé)
    biomass_west, biomass_east, biomass_south, biomass_north = get_neighbors_with_bc(
        biomass, boundary
    )

    # Obtenir les masques des voisins (logique inchangée)
    if mask is not None:
        mask_west, mask_east, mask_south, mask_north = get_neighbors_with_bc(ocean_mask, boundary)
        face_mask_east = ocean_mask * mask_east
        face_mask_west = ocean_mask * mask_west
        face_mask_north = ocean_mask * mask_north
        face_mask_south = ocean_mask * mask_south
    else:
        face_mask_east = jnp.ones_like(biomass)
        face_mask_west = jnp.ones_like(biomass)
        face_mask_north = jnp.ones_like(biomass)
        face_mask_south = jnp.ones_like(biomass)

    # --- FIX 3 : Forcer les frontières 'CLOSED' à être des murs ---
    # Corrige le problème de création de masse dû à `get_neighbors_with_bc`
    # qui copie la valeur intérieure (gradient nul) au lieu de la mettre à 0.

    if boundary.north == BoundaryType.CLOSED:
        face_mask_north = face_mask_north.at[nlat - 1, :].set(0.0)
    if boundary.south == BoundaryType.CLOSED:
        face_mask_south = face_mask_south.at[0, :].set(0.0)
    if boundary.east == BoundaryType.CLOSED:
        face_mask_east = face_mask_east.at[:, nlon - 1].set(0.0)
    if boundary.west == BoundaryType.CLOSED:
        face_mask_west = face_mask_west.at[:, 0].set(0.0)

    # --- Fin du FIX 3 ---

    # Calcul des vitesses aux faces
    # Utilise u_clean et v_clean (du FIX 2)

    # Obtenir les voisins des vitesses PROPRES
    u_west, u_east, _, _ = get_neighbors_with_bc(u_clean, boundary)
    _, _, v_south, v_north = get_neighbors_with_bc(v_clean, boundary)

    # Vitesse aux faces (utilise les vitesses propres ET les masques de face corrigés)
    u_face_east = (u_clean + u_east) / 2 * face_mask_east
    u_face_west = (u_west + u_clean) / 2 * face_mask_west
    v_face_north = (v_clean + v_north) / 2 * face_mask_north
    v_face_south = (v_south + v_clean) / 2 * face_mask_south

    # Choix Upwind (inchangé)
    biomass_face_east = jnp.where(u_face_east > 0, biomass, biomass_east)
    biomass_face_west = jnp.where(u_face_west > 0, biomass_west, biomass)
    biomass_face_north = jnp.where(v_face_north > 0, biomass, biomass_north)
    biomass_face_south = jnp.where(v_face_south > 0, biomass_south, biomass)

    # Calcul des flux (inchangé)
    area_east = face_areas_ew[:, 1:]
    area_west = face_areas_ew[:, :-1]
    flux_east = u_face_east * biomass_face_east * area_east
    flux_west = u_face_west * biomass_face_west * area_west

    area_north = face_areas_ns[1:, :]
    area_south = face_areas_ns[:-1, :]
    flux_north = v_face_north * biomass_face_north * area_north
    flux_south = v_face_south * biomass_face_south * area_south

    # Calcul de la divergence (inchangé)
    flux_divergence = (flux_east - flux_west + flux_north - flux_south) / cell_areas

    # Mise à jour de la biomasse (inchangé)
    biomass_new = biomass - dt * flux_divergence

    # Appliquer le masque final (inchangé, c'est correct)
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
