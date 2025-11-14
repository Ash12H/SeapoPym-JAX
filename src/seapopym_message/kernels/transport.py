"""Transport process Units for spatial coupling.

This module provides Units for physical transport processes:
- Diffusion: Random dispersion of biomass (Laplacian operator)
- Advection: Directional movement with currents (upwind scheme)

These Units have scope='global' and require halo exchange for boundary conditions.
"""

import warnings

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


@unit(
    name="advection_2d",
    inputs=["biomass"],
    outputs=["biomass"],
    scope="global",
    compiled=True,
    forcings=["u", "v"],
)
def compute_advection_2d(
    biomass: jnp.ndarray,
    dt: float,
    params: dict,
    forcings: dict,
    halo_north: dict | None = None,
    halo_south: dict | None = None,
    halo_east: dict | None = None,
    halo_west: dict | None = None,
) -> jnp.ndarray:
    """Apply 2D advection using upwind scheme.

    Solves: ∂B/∂t = -u*∂B/∂x - v*∂B/∂y
    where u, v are velocity components from forcings.

    Uses 1st order upwind finite differences:
    - If u > 0: backward difference in x (westward)
    - If u < 0: forward difference in x (eastward)
    - If v > 0: backward difference in y (southward)
    - If v < 0: forward difference in y (northward)

    The scheme is stable for CFL ≤ 1, where:
        CFL = max(|u|, |v|) * dt / min(dx, dy)

    Islands/land cells can be masked using the 'mask' parameter.

    Boundary conditions:
    - With halo data: uses neighbor values
    - Without halo data: Neumann (zero flux) boundary conditions

    Args:
        biomass: Current biomass distribution (nlat, nlon).
        dt: Time step.
        params: Dictionary containing:
            - 'dx': Grid spacing in x-direction [m]
            - 'dy': Grid spacing in y-direction [m] (optional, defaults to dx)
            - 'mask': Optional boolean array (nlat, nlon) where True = ocean, False = land
        forcings: Dictionary containing:
            - 'u': Zonal velocity (east-west) [m/s], shape (nlat, nlon)
            - 'v': Meridional velocity (north-south) [m/s], shape (nlat, nlon)
        halo_north: Boundary data from northern neighbor {'biomass': array}
        halo_south: Boundary data from southern neighbor {'biomass': array}
        halo_east: Boundary data from eastern neighbor {'biomass': array}
        halo_west: Boundary data from western neighbor {'biomass': array}

    Returns:
        Updated biomass after advection.

    Note:
        This function assumes CFL condition is satisfied (checked upstream).
        Use check_cfl_condition() to verify stability before running.
        Land cells (mask = False) are set to zero in the output.

    Example:
        >>> biomass = jnp.ones((10, 10)) * 100.0
        >>> u = jnp.ones((10, 10)) * 0.5  # 0.5 m/s eastward
        >>> v = jnp.zeros((10, 10))
        >>> params = {'dx': 1000.0, 'dy': 1000.0}
        >>> forcings = {'u': u, 'v': v}
        >>> dt = 100.0
        >>> biomass_new = compute_advection_2d(biomass=biomass, dt=dt,
        ...                                     params=params, forcings=forcings)
        >>> # Biomass will be advected eastward
    """
    nlat, nlon = biomass.shape
    dx = params["dx"]
    dy = params.get("dy", dx)

    # Get velocities from forcings
    u = forcings["u"]
    v = forcings["v"]

    # Build extended arrays with halos
    biomass_ext = jnp.zeros((nlat + 2, nlon + 2))
    u_ext = jnp.zeros((nlat + 2, nlon + 2))
    v_ext = jnp.zeros((nlat + 2, nlon + 2))

    biomass_ext = biomass_ext.at[1:-1, 1:-1].set(biomass)
    u_ext = u_ext.at[1:-1, 1:-1].set(u)
    v_ext = v_ext.at[1:-1, 1:-1].set(v)

    # Fill halos for biomass
    if halo_north is not None and "biomass" in halo_north:
        biomass_ext = biomass_ext.at[0, 1:-1].set(halo_north["biomass"])
    else:
        # Neumann BC: zero flux
        biomass_ext = biomass_ext.at[0, 1:-1].set(biomass[0, :])

    if halo_south is not None and "biomass" in halo_south:
        biomass_ext = biomass_ext.at[-1, 1:-1].set(halo_south["biomass"])
    else:
        biomass_ext = biomass_ext.at[-1, 1:-1].set(biomass[-1, :])

    if halo_west is not None and "biomass" in halo_west:
        biomass_ext = biomass_ext.at[1:-1, 0].set(halo_west["biomass"])
    else:
        biomass_ext = biomass_ext.at[1:-1, 0].set(biomass[:, 0])

    if halo_east is not None and "biomass" in halo_east:
        biomass_ext = biomass_ext.at[1:-1, -1].set(halo_east["biomass"])
    else:
        biomass_ext = biomass_ext.at[1:-1, -1].set(biomass[:, -1])

    # Fill halos for u and v (replicate boundary values)
    u_ext = u_ext.at[0, 1:-1].set(u[0, :])
    u_ext = u_ext.at[-1, 1:-1].set(u[-1, :])
    u_ext = u_ext.at[1:-1, 0].set(u[:, 0])
    u_ext = u_ext.at[1:-1, -1].set(u[:, -1])

    v_ext = v_ext.at[0, 1:-1].set(v[0, :])
    v_ext = v_ext.at[-1, 1:-1].set(v[-1, :])
    v_ext = v_ext.at[1:-1, 0].set(v[:, 0])
    v_ext = v_ext.at[1:-1, -1].set(v[:, -1])

    # Extract center and neighbor values
    B_c = biomass_ext[1:-1, 1:-1]
    u_c = u_ext[1:-1, 1:-1]
    v_c = v_ext[1:-1, 1:-1]

    B_north = biomass_ext[:-2, 1:-1]
    B_south = biomass_ext[2:, 1:-1]
    B_west = biomass_ext[1:-1, :-2]
    B_east = biomass_ext[1:-1, 2:]

    # Upwind scheme for x-direction (longitude)
    # u > 0: flow from west to east, use backward difference
    # u < 0: flow from east to west, use forward difference
    dB_dx_upwind = jnp.where(
        u_c >= 0,
        (B_c - B_west) / dx,
        (B_east - B_c) / dx,
    )

    # Upwind scheme for y-direction (latitude)
    # v > 0: flow from north to south, use backward difference
    # v < 0: flow from south to north, use forward difference
    dB_dy_upwind = jnp.where(
        v_c >= 0,
        (B_c - B_north) / dy,
        (B_south - B_c) / dy,
    )

    # Advection equation: ∂B/∂t = -u*∂B/∂x - v*∂B/∂y
    dB_dt = -u_c * dB_dx_upwind - v_c * dB_dy_upwind

    # Forward Euler integration
    biomass_new = biomass + dB_dt * dt

    # Apply mask if provided
    if "mask" in params:
        mask = params["mask"]
        # mask: True = ocean, False = land
        biomass_new = jnp.where(mask, biomass_new, 0.0)

    # Ensure non-negative (biomass can't be negative)
    biomass_new = jnp.maximum(biomass_new, 0.0)

    return biomass_new


@unit(
    name="advection_simple",
    inputs=["biomass"],
    outputs=["biomass"],
    scope="local",
    compiled=True,
    forcings=["u", "v"],
)
def compute_advection_simple(
    biomass: jnp.ndarray, dt: float, params: dict, forcings: dict
) -> jnp.ndarray:
    """Simple local advection without halo exchange (for testing).

    This is a simplified version that applies Neumann boundary conditions
    on all sides. Useful for testing local kernels without distributed setup.

    Args:
        biomass: Current biomass distribution.
        dt: Time step.
        params: Dictionary with 'dx', 'dy' (optional), 'mask' (optional).
        forcings: Dictionary with 'u', 'v' velocity fields.

    Returns:
        Updated biomass.
    """
    # Call the underlying function, not the Unit wrapper
    return compute_advection_2d.func(
        biomass=biomass,
        dt=dt,
        params=params,
        forcings=forcings,
        halo_north=None,
        halo_south=None,
        halo_east=None,
        halo_west=None,
    )


def check_cfl_condition(u: jnp.ndarray, v: jnp.ndarray, dt: float, dx: float, dy: float) -> dict:
    """Check CFL (Courant-Friedrichs-Lewy) stability condition for advection.

    The CFL condition for stability is:
        CFL = max(|u|, |v|) * dt / min(dx, dy) ≤ 1

    Args:
        u: Zonal velocity field [m/s].
        v: Meridional velocity field [m/s].
        dt: Time step [s].
        dx: Grid spacing in x-direction [m].
        dy: Grid spacing in y-direction [m].

    Returns:
        Dictionary with:
            - 'cfl': Maximum CFL number
            - 'stable': True if CFL ≤ 1, False otherwise
            - 'max_u': Maximum absolute u velocity
            - 'max_v': Maximum absolute v velocity
            - 'max_dt_stable': Maximum stable time step

    Example:
        >>> u = jnp.ones((10, 10)) * 0.5
        >>> v = jnp.zeros((10, 10))
        >>> result = check_cfl_condition(u, v, dt=100.0, dx=1000.0, dy=1000.0)
        >>> if not result['stable']:
        ...     warnings.warn(f"CFL condition violated: {result['cfl']:.3f} > 1.0")
    """
    max_u = float(jnp.max(jnp.abs(u)))
    max_v = float(jnp.max(jnp.abs(v)))

    max_velocity = max(max_u, max_v)
    min_spacing = min(dx, dy)

    cfl = max_velocity * dt / min_spacing
    stable = cfl <= 1.0

    # Maximum stable timestep
    max_dt_stable = min_spacing / max_velocity if max_velocity > 0 else float("inf")

    result = {
        "cfl": cfl,
        "stable": stable,
        "max_u": max_u,
        "max_v": max_v,
        "max_dt_stable": max_dt_stable,
    }

    # Issue warning if unstable
    if not stable:
        warnings.warn(
            f"CFL condition violated: CFL = {cfl:.3f} > 1.0. "
            f"Consider reducing dt below {max_dt_stable:.1f} s or increasing grid spacing.",
            UserWarning,
            stacklevel=2,
        )

    return result
