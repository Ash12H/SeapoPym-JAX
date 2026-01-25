"""Biological functions for ecosystem models.

This module provides @functional-decorated functions for biological processes
such as growth, predation, mortality, and aging.
"""

from __future__ import annotations

import jax.numpy as jnp

from seapopym.blueprint import functional


@functional(
    name="biol:simple_growth",
    backend="jax",
    units={
        "biomass": "g",
        "rate": "1/d",
        "temp": "degC",
        "return": "g/d",
    },
)
def simple_growth(
    biomass: jnp.ndarray,
    rate: float | jnp.ndarray,
    temp: jnp.ndarray,
) -> jnp.ndarray:
    """Simple exponential growth modulated by temperature.

    Computes growth tendency as: biomass * rate * (temp / 20.0)

    This is a toy function for testing the Blueprint system.

    Args:
        biomass: Current biomass values.
        rate: Growth rate (per day).
        temp: Temperature values (degrees Celsius).

    Returns:
        Growth tendency (g/d).
    """
    return biomass * rate * (temp / 20.0)


@functional(
    name="biol:growth",
    backend="jax",
    core_dims={"biomass": ["C"]},
    out_dims=["C"],
    units={
        "biomass": "g",
        "rate": "1/d",
        "temp": "degC",
        "return": "g/d",
    },
)
def growth(
    biomass: jnp.ndarray,
    rate: float | jnp.ndarray,
    temp: jnp.ndarray,
) -> jnp.ndarray:
    """Exponential growth with temperature dependence.

    Computes growth tendency with Arrhenius-like temperature response.

    Args:
        biomass: Current biomass values with cohort dimension.
        rate: Growth rate (per day).
        temp: Temperature values (degrees Celsius).

    Returns:
        Growth tendency (g/d).
    """
    # Expand temp to broadcast with cohort dimension
    temp_expanded = temp[..., jnp.newaxis]
    return biomass * rate * jnp.exp(temp_expanded / 10.0)


@functional(
    name="biol:predation",
    backend="jax",
    outputs=["prey_loss", "predator_gain"],
    units={
        "prey_biomass": "g",
        "predator_biomass": "g",
        "rate": "1/d",
        "prey_loss": "g/d",
        "predator_gain": "g/d",
    },
)
def predation(
    prey_biomass: jnp.ndarray,
    predator_biomass: jnp.ndarray,
    rate: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Predation interaction between two populations.

    Computes mass transfer from prey to predator following Lotka-Volterra dynamics.

    Args:
        prey_biomass: Prey population biomass.
        predator_biomass: Predator population biomass.
        rate: Predation rate (per day).

    Returns:
        Tuple of (prey_loss, predator_gain) - note prey_loss is negative.
    """
    flux = rate * prey_biomass * predator_biomass
    return -flux, +flux


@functional(
    name="biol:gillooly_transform",
    backend="jax",
    units={
        "temp": "degC",
        "return": "dimensionless",
    },
)
def gillooly_transform(temp: jnp.ndarray) -> jnp.ndarray:
    """Gillooly temperature transformation for metabolic scaling.

    Applies the Arrhenius-based temperature correction factor from
    Gillooly et al. (2001) for metabolic rate scaling.

    Args:
        temp: Temperature in degrees Celsius.

    Returns:
        Dimensionless temperature correction factor.
    """
    T_ref = 15.0  # Reference temperature (°C)
    E_a = 0.63  # Activation energy (eV)
    k_B = 8.617e-5  # Boltzmann constant (eV/K)

    # Convert to Kelvin
    temp_K = temp + 273.15
    T_ref_K = T_ref + 273.15

    return jnp.exp(-E_a / k_B * (1.0 / temp_K - 1.0 / T_ref_K))
