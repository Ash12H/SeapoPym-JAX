"""Transport process Units for spatial coupling.

This module provides Units for physical transport processes:
- Diffusion: Random dispersion of biomass (Laplacian operator)
- Advection: Directional movement (not yet implemented)

These Units have scope='global' and require halo exchange for boundary conditions.
"""

import jax.numpy as jnp

from seapopym_message.core.unit import unit


@unit(
    name="diffusion_2d",
    inputs=["biomass"],
    outputs=["biomass"],
    scope="global",
    compiled=True,
)
def compute_diffusion_2d(
    biomass: jnp.ndarray,
    dt: float,
    params: dict,
    halo_north: dict | None = None,
    halo_south: dict | None = None,
    halo_east: dict | None = None,
    halo_west: dict | None = None,
) -> jnp.ndarray:
    """Apply 2D diffusion using finite differences.

    Solves: ∂B/∂t = D * (∂²B/∂x² + ∂²B/∂y²)
    where D is the diffusion coefficient.

    Uses centered finite differences for the Laplacian:
    ∇²B[i,j] ≈ (B[i-1,j] + B[i+1,j] + B[i,j-1] + B[i,j+1] - 4*B[i,j]) / dx²

    Boundary conditions:
    - With halo data: uses neighbor values
    - Without halo data: Neumann (zero flux) boundary conditions

    Args:
        biomass: Current biomass distribution (nlat, nlon).
        dt: Time step.
        params: Dictionary containing:
            - 'D': Diffusion coefficient [m²/s]
            - 'dx': Grid spacing in x-direction [m]
            - 'dy': Grid spacing in y-direction [m] (optional, defaults to dx)
        halo_north: Boundary data from northern neighbor {'biomass': array}
        halo_south: Boundary data from southern neighbor {'biomass': array}
        halo_east: Boundary data from eastern neighbor {'biomass': array}
        halo_west: Boundary data from western neighbor {'biomass': array}

    Returns:
        Updated biomass after diffusion.

    Example:
        >>> biomass = jnp.array([[10., 50., 10.],
        ...                       [10., 50., 10.],
        ...                       [10., 50., 10.]])
        >>> params = {'D': 100.0, 'dx': 1000.0}  # 1 km grid
        >>> dt = 100.0  # 100 seconds
        >>> biomass_new = compute_diffusion_2d(biomass=biomass, dt=dt, params=params)
        >>> # Peak at center will diffuse outward
    """
    nlat, nlon = biomass.shape
    D = params["D"]
    dx = params["dx"]
    dy = params.get("dy", dx)  # Assume square cells if dy not provided

    # Build extended array with halos
    # Shape will be (nlat+2, nlon+2) to include ghost cells
    extended = jnp.zeros((nlat + 2, nlon + 2))
    extended = extended.at[1:-1, 1:-1].set(biomass)

    # Fill halos (boundary conditions)
    # North boundary (top row, i=0)
    if halo_north is not None and "biomass" in halo_north:
        extended = extended.at[0, 1:-1].set(halo_north["biomass"])
    else:
        # Neumann BC: ∂B/∂n = 0 => B[ghost] = B[boundary]
        extended = extended.at[0, 1:-1].set(biomass[0, :])

    # South boundary (bottom row, i=nlat+1)
    if halo_south is not None and "biomass" in halo_south:
        extended = extended.at[-1, 1:-1].set(halo_south["biomass"])
    else:
        extended = extended.at[-1, 1:-1].set(biomass[-1, :])

    # West boundary (left column, j=0)
    if halo_west is not None and "biomass" in halo_west:
        extended = extended.at[1:-1, 0].set(halo_west["biomass"])
    else:
        extended = extended.at[1:-1, 0].set(biomass[:, 0])

    # East boundary (right column, j=nlon+1)
    if halo_east is not None and "biomass" in halo_east:
        extended = extended.at[1:-1, -1].set(halo_east["biomass"])
    else:
        extended = extended.at[1:-1, -1].set(biomass[:, -1])

    # Fill corners (not used in 5-point stencil, but for completeness)
    extended = extended.at[0, 0].set(extended[1, 1])
    extended = extended.at[0, -1].set(extended[1, -2])
    extended = extended.at[-1, 0].set(extended[-2, 1])
    extended = extended.at[-1, -1].set(extended[-2, -2])

    # Compute Laplacian using 5-point stencil
    # ∇²B = (B[i-1,j] + B[i+1,j]) / dy² + (B[i,j-1] + B[i,j+1]) / dx² - 2*B[i,j]*(1/dx² + 1/dy²)
    center = extended[1:-1, 1:-1]
    north = extended[:-2, 1:-1]
    south = extended[2:, 1:-1]
    west = extended[1:-1, :-2]
    east = extended[1:-1, 2:]

    laplacian = (
        (north + south) / dy**2 + (west + east) / dx**2 - center * 2 * (1 / dx**2 + 1 / dy**2)
    )

    # Forward Euler integration: B_new = B + D * ∇²B * dt
    biomass_new = biomass + D * laplacian * dt

    # Ensure non-negative (biomass can't be negative)
    biomass_new = jnp.maximum(biomass_new, 0.0)

    return biomass_new


@unit(
    name="diffusion_simple",
    inputs=["biomass"],
    outputs=["biomass"],
    scope="local",
    compiled=True,
)
def compute_diffusion_simple(biomass: jnp.ndarray, dt: float, params: dict) -> jnp.ndarray:
    """Simple local diffusion without halo exchange (for testing).

    This is a simplified version that applies Neumann boundary conditions
    on all sides. Useful for testing local kernels without distributed setup.

    Args:
        biomass: Current biomass distribution.
        dt: Time step.
        params: Dictionary with 'D', 'dx', 'dy' (optional).

    Returns:
        Updated biomass.
    """
    # Call the underlying function, not the Unit wrapper
    return compute_diffusion_2d.func(
        biomass=biomass,
        dt=dt,
        params=params,
        halo_north=None,
        halo_south=None,
        halo_east=None,
        halo_west=None,
    )
