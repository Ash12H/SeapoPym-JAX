"""Diffusion scheme for transport computations.

This module implements explicit Euler diffusion on spherical grids with
varying cell width dx(lat). Uses centered finite differences for the Laplacian.

References:
    - Euler explicit method: IA/Diffusion-euler-explicite-description.md
    - Spherical grid dx(lat): IA/TRANSPORT_ANALYSIS.md (Section 8)
"""

import jax.numpy as jnp

from seapopym_message.transport.boundary import BoundaryConditions, get_neighbors_with_bc
from seapopym_message.transport.grid import Grid


def diffusion_explicit_spherical(
    biomass: jnp.ndarray,
    D: float,
    dt: float,
    grid: Grid,
    boundary: BoundaryConditions,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute diffusion using explicit Euler method on spherical grid.

    This function implements the centered finite difference scheme described in
    IA/Diffusion-euler-explicite-description.md. The principle is:

        dC/dt = D × ∇²C
        ∇²C = ∂²C/∂x² + ∂²C/∂y²

    where the Laplacian is approximated using centered differences:
        ∂²C/∂x² ≈ (C[i+1,j] - 2C[i,j] + C[i-1,j]) / dx(j)²
        ∂²C/∂y² ≈ (C[i,j+1] - 2C[i,j] + C[i,j-1]) / dy²

    IMPORTANT (line 26 of description):
        dx(j) depends on latitude j for spherical grids:
        dx(j) = R × cos(lat[j]) × dλ

    Stability constraint (line 41 of description):
        dt ≤ min(dx²)/(4·D)
        For spherical grids, dx is minimal at poles, which limits dt!

    Args:
        biomass: Concentration field [kg/m²], shape (nlat, nlon)
        D: Diffusion coefficient [m²/s], also written K_h in theory
        dt: Time step [s]
        grid: Grid object providing dx(lat), dy
        boundary: Boundary conditions for domain edges
        mask: Optional land/sea mask (1=ocean, 0=land, NaN=land), shape (nlat, nlon)
              If provided, treats land cells as zero-flux boundaries (Neumann BC)

    Returns:
        Updated biomass field [kg/m²], shape (nlat, nlon)

    Mathematical formulation:
        For cell (i,j):
        C^{n+1}[i,j] = C^n[i,j] + dt × D × (
            (C_east - 2C + C_west) / dx[j]² +
            (C_north - 2C + C_south) / dy²
        )

    Land masking (lines 88-103 of description, Cas 3):
        For ocean cells adjacent to land, the land neighbor is treated
        as having the same value as the ocean cell (zero-flux BC).
        This is handled automatically by setting biomass=0 on land
        and using the boundary condition logic.

    Example:
        >>> grid = SphericalGrid(-60, 60, 0, 360, 120, 360)
        >>> biomass = jnp.ones((120, 360))
        >>> D = 1000.0  # 1000 m²/s horizontal diffusion
        >>> bc = BoundaryConditions(CLOSED, CLOSED, PERIODIC, PERIODIC)
        >>> biomass_new = diffusion_explicit_spherical(biomass, D, dt, grid, bc)
    """
    nlat, nlon = biomass.shape

    # Apply land mask if provided (set biomass=0 on land)
    if mask is not None:
        ocean_mask = jnp.where(jnp.isnan(mask), 0.0, mask)
        biomass = biomass * ocean_mask

    # Get grid spacing
    dx = grid.dx()  # Array (nlat,) for spherical, scalar for plane
    dy = grid.dy()  # Scalar

    # Get neighbor values with boundary conditions
    biomass_west, biomass_east, biomass_south, biomass_north = get_neighbors_with_bc(
        biomass, boundary
    )

    # Apply zero-flux boundary condition at ocean-land interfaces
    # If neighbor is land, use current cell value (Neumann BC: dC/dn = 0)
    # This prevents artificial gradients and mass loss at boundaries
    if mask is not None:
        mask_west, mask_east, mask_south, mask_north = get_neighbors_with_bc(ocean_mask, boundary)
        # Replace land neighbor values with current cell value
        biomass_east = jnp.where(mask_east == 0, biomass, biomass_east)
        biomass_west = jnp.where(mask_west == 0, biomass, biomass_west)
        biomass_north = jnp.where(mask_north == 0, biomass, biomass_north)
        biomass_south = jnp.where(mask_south == 0, biomass, biomass_south)

    # Compute Laplacian using centered differences
    # ∂²C/∂x² = (C_east - 2C + C_west) / dx²
    # ∂²C/∂y² = (C_north - 2C + C_south) / dy²

    # For spherical grid, dx varies with latitude
    dx_2d = dx[:, None] * jnp.ones((nlat, nlon)) if isinstance(dx, jnp.ndarray) else dx

    # Second derivative in x (longitude)
    d2C_dx2 = (biomass_east - 2 * biomass + biomass_west) / (dx_2d**2)

    # Second derivative in y (latitude)
    d2C_dy2 = (biomass_north - 2 * biomass + biomass_south) / (dy**2)

    # Laplacian
    laplacian = d2C_dx2 + d2C_dy2

    # Update biomass (Euler explicit)
    # dC/dt = D × ∇²C
    biomass_new = biomass + dt * D * laplacian

    # Apply mask to final result (ensure land stays at zero)
    if mask is not None:
        biomass_new = biomass_new * ocean_mask

    return biomass_new


def check_diffusion_stability(
    dt: float,
    D: float,
    grid: Grid,
) -> dict:
    """Check stability criterion for explicit diffusion.

    The explicit Euler scheme for diffusion is stable if:
        dt ≤ min(dx², dy²) / (4 × D)

    For spherical grids, dx decreases toward poles, so the minimum dx
    (at the poles) limits the timestep.

    Args:
        dt: Proposed time step [s]
        D: Diffusion coefficient [m²/s]
        grid: Grid object

    Returns:
        Dictionary with:
        - is_stable: bool, whether dt satisfies stability criterion
        - dt_max: Maximum stable dt [s]
        - dx_min: Minimum grid spacing [m]
        - dy: Grid spacing in y [m]
        - cfl_diffusion: Actual CFL number (should be ≤ 0.25)

    Reference:
        IA/Diffusion-euler-explicite-description.md, line 41
    """
    dx = grid.dx()
    dy = grid.dy()

    # Find minimum dx
    dx_min = float(jnp.min(dx)) if isinstance(dx, jnp.ndarray) else float(dx)

    # Stability criterion: dt ≤ min(dx², dy²) / (4D)
    dt_max = min(dx_min**2, dy**2) / (4 * D)

    # CFL number for diffusion: (D × dt) / dx²
    cfl = (D * dt) / min(dx_min**2, dy**2)

    is_stable = dt <= dt_max

    return {
        "is_stable": is_stable,
        "dt_max": dt_max,
        "dx_min": dx_min,
        "dy": dy,
        "cfl_diffusion": float(cfl),
    }


def compute_diffusion_diagnostics(
    biomass: jnp.ndarray,
    biomass_new: jnp.ndarray,
    D: float,
    dt: float,
    grid: Grid,
    mask: jnp.ndarray | None = None,
) -> dict:
    """Compute diagnostics for diffusion step.

    Args:
        biomass: Initial biomass [kg/m²]
        biomass_new: Final biomass after diffusion [kg/m²]
        D: Diffusion coefficient [m²/s]
        dt: Time step [s]
        grid: Grid object
        mask: Optional land/sea mask

    Returns:
        Dictionary with:
        - total_mass_before: Total mass before [kg]
        - total_mass_after: Total mass after [kg]
        - mass_change: Absolute mass change [kg]
        - conservation_fraction: Fraction of mass conserved (should be ~1.0)
        - max_gradient: Maximum spatial gradient [kg/m³]
        - stability: Stability check results
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

    # Estimate maximum gradient (for diagnostics)
    dx = grid.dx()
    dy = grid.dy()

    dx_min = float(jnp.min(dx)) if isinstance(dx, jnp.ndarray) else float(dx)

    # Simple gradient estimate
    grad_x = jnp.max(jnp.abs(jnp.diff(biomass, axis=1))) / dx_min
    grad_y = jnp.max(jnp.abs(jnp.diff(biomass, axis=0))) / dy
    max_gradient = float(max(grad_x, grad_y))

    # Stability check
    stability = check_diffusion_stability(dt, D, grid)

    return {
        "total_mass_before": float(total_before),
        "total_mass_after": float(total_after),
        "mass_change": float(mass_change),
        "conservation_fraction": float(conservation),
        "max_gradient": max_gradient,
        "stability": stability,
    }
