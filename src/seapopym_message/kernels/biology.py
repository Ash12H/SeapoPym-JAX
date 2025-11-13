"""Biological process Units for population dynamics.

This module provides pre-defined Units for common biological processes:
- Recruitment (R): Input of new biomass into the system
- Mortality (λB): Loss of biomass due to natural death
- Growth (dB/dt = R - M): Net change in biomass

These Units implement a simple 0D model: dB/dt = R - λB
where B is biomass, R is constant recruitment, and λ is mortality rate.
"""

import jax.numpy as jnp

from seapopym_message.core.unit import unit


@unit(name="recruitment", inputs=[], outputs=["R"], scope="local", compiled=True)
def compute_recruitment(params: dict) -> jnp.ndarray:
    """Compute recruitment (constant input).

    This is the simplest recruitment model: constant input independent
    of existing biomass or environmental conditions.

    Args:
        params: Dictionary containing 'R' (recruitment rate).

    Returns:
        Array with constant recruitment value.

    Example:
        >>> params = {'R': 5.0}
        >>> R = compute_recruitment(params=params)
        >>> # R will be a scalar 5.0
    """
    return jnp.array(params["R"])


@unit(
    name="mortality", inputs=["biomass"], outputs=["mortality_rate"], scope="local", compiled=True
)
def compute_mortality(biomass: jnp.ndarray, params: dict) -> jnp.ndarray:
    """Compute linear mortality rate.

    Mortality is proportional to biomass: M = λ * B
    where λ is the mortality coefficient.

    Args:
        biomass: Current biomass distribution.
        params: Dictionary containing 'lambda' (mortality coefficient).

    Returns:
        Mortality rate array (same shape as biomass).

    Example:
        >>> biomass = jnp.array([[10., 20.], [30., 40.]])
        >>> params = {'lambda': 0.1}
        >>> M = compute_mortality(biomass=biomass, params=params)
        >>> # M = [[1., 2.], [3., 4.]]
    """
    return params["lambda"] * biomass


@unit(
    name="growth",
    inputs=["biomass", "R", "mortality_rate"],
    outputs=["biomass"],
    scope="local",
    compiled=True,
)
def compute_growth(
    biomass: jnp.ndarray, R: jnp.ndarray, mortality_rate: jnp.ndarray, dt: float
) -> jnp.ndarray:
    """Update biomass based on recruitment and mortality.

    Integrates the ODE: dB/dt = R - M
    using forward Euler: B(t+dt) = B(t) + (R - M) * dt

    At equilibrium (dB/dt = 0): B_eq = R / λ

    Args:
        biomass: Current biomass.
        R: Recruitment rate.
        mortality_rate: Mortality rate (M = λ * B).
        dt: Time step.

    Returns:
        Updated biomass.

    Example:
        >>> biomass = jnp.array([[10., 20.]])
        >>> R = jnp.array([[5., 5.]])
        >>> M = jnp.array([[1., 2.]])
        >>> dt = 0.1
        >>> biomass_new = compute_growth(biomass=biomass, R=R, mortality_rate=M, dt=dt)
        >>> # biomass_new = [[10.4, 20.3]]  (biomass + (R - M) * dt)
    """
    return biomass + (R - mortality_rate) * dt


@unit(
    name="recruitment_2d",
    inputs=[],
    outputs=["R"],
    scope="local",
    compiled=False,  # Can't JIT with dynamic shape argument
)
def compute_recruitment_2d(params: dict, grid_shape: tuple[int, int]) -> jnp.ndarray:
    """Compute spatially constant recruitment for 2D grids.

    Args:
        params: Dictionary containing 'R' (recruitment rate).
        grid_shape: Tuple (nlat, nlon) specifying grid dimensions.

    Returns:
        2D array with constant recruitment value.

    Example:
        >>> params = {'R': 5.0}
        >>> grid_shape = (10, 20)
        >>> R = compute_recruitment_2d(params=params, grid_shape=grid_shape)
        >>> R.shape  # (10, 20)
    """
    return jnp.full(grid_shape, params["R"])
