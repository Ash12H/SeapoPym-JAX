"""LMTL (Low/Mid Trophic Level) biological functions.

This module provides JAX-compatible functions for the LMTL ecosystem model,
implementing zooplankton dynamics with cohort-based production and biomass.

Processes:
1. Temperature normalization (Gillooly transform)
2. Recruitment age calculation
3. NPP injection into cohort 0
4. Aging flow between cohorts
5. Recruitment flow from cohorts to biomass
6. Natural mortality

All functions use the @functional decorator for Blueprint integration
and support automatic differentiation via JAX.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from seapopym.blueprint import functional

# Sigmoid steepness for smooth recruitment transition.
# Transition from ~1% to ~99% over ±1 day (DVM timescale).
# k = ln(99) / (1 day in seconds) ≈ 5.32e-5 s⁻¹
_RECRUITMENT_TRANSITION_DAYS = 1.0
_K_SIGMOID = float(jnp.log(99.0)) / (_RECRUITMENT_TRANSITION_DAYS * 86400.0)

# =============================================================================
# ENVIRONMENT FUNCTIONS
# =============================================================================


@functional(
    name="lmtl:day_length",
    units={
        "latitude": "degrees",
        "day_of_year": "dimensionless",
        "return": "dimensionless",
    },
)
def day_length(latitude: jnp.ndarray, day_of_year: jnp.ndarray) -> jnp.ndarray:
    """Compute day length fraction based on latitude and day of year.

    Uses the CBM model (Forsythe et al., 1995) to calculate the fraction
    of the day with sunlight, used for diel vertical migration weighting.

    Args:
        latitude: Latitude in degrees [-90, 90].
        day_of_year: Day of year [1-366].

    Returns:
        Day length fraction [0-1], where 0 = polar night, 1 = midnight sun.

    Reference:
        Forsythe et al. (1995) "A model comparison for daylength as a
        function of latitude and day of year"
    """
    # Convert latitude to radians
    phi = jnp.deg2rad(latitude)

    # Solar declination (Brock, 1981)
    declination = 23.45 * jnp.sin(jnp.deg2rad(360.0 * (284.0 + day_of_year) / 365.0))
    declination_rad = jnp.deg2rad(declination)

    # Sunset hour angle
    # cos(omega) = -tan(phi) * tan(delta)
    arg = -jnp.tan(phi) * jnp.tan(declination_rad)
    # Clamp to [-1, 1] for polar days/nights
    arg = jnp.clip(arg, -1.0, 1.0)
    hour_angle = jnp.arccos(arg)

    # Day length as fraction of 24 hours
    # day_length_hours = 24 * hour_angle / pi
    return hour_angle / jnp.pi


@functional(
    name="lmtl:layer_weighted_mean",
    core_dims={"forcing": ["Z"]},
    units={
        "day_length": "dimensionless",
        "day_layer": "dimensionless",
        "night_layer": "dimensionless",
    },
)
def layer_weighted_mean(
    forcing: jnp.ndarray,
    day_length: jnp.ndarray,
    day_layer: int,
    night_layer: int,
) -> jnp.ndarray:
    """Compute day/night weighted mean of a forcing field across depth layers.

    Used to compute the mean environmental conditions experienced by
    vertically migrating organisms (diel vertical migration - DVM).

    With vmap, this function receives:
    - forcing: (Z,) - depth dimension only
    - day_length: scalar (for each spatial point)
    - day_layer, night_layer: int parameters

    Args:
        forcing: Forcing field with depth dimension (Z,).
        day_length: Day length fraction [0-1].
        day_layer: Index of the depth layer occupied during daytime.
        night_layer: Index of the depth layer occupied during nighttime.

    Returns:
        Weighted mean: forcing[day] * day_length + forcing[night] * (1 - day_length)
    """
    val_day = forcing[day_layer]
    val_night = forcing[night_layer]
    return val_day * day_length + val_night * (1.0 - day_length)


# =============================================================================
# TEMPERATURE FUNCTIONS
# =============================================================================


@functional(
    name="lmtl:threshold_temperature",
    units={"temp": "degC", "min_temp": "degC", "return": "degC"},
)
def threshold_temperature(temp: jnp.ndarray, min_temp: float) -> jnp.ndarray:
    """Threshold temperature values below a minimum.

    Args:
        temp: Input temperature [degC].
        min_temp: Minimum allowed temperature [degC].

    Returns:
        Thresholded temperature: max(temp, min_temp).
    """
    return jnp.maximum(temp, min_temp)


@functional(
    name="lmtl:gillooly_temperature",
    units={"temp": "degC", "return": "degC"},
)
def gillooly_temperature(temp: jnp.ndarray) -> jnp.ndarray:
    """Normalize temperature using Gillooly et al. (2001).

    This transformation is used in metabolic scaling models based on the
    Arrhenius equation. It normalizes temperature from Celsius to a form
    suitable for metabolic rate calculations.

    T_normalized = T / (1 + T/273)

    Args:
        temp: Input temperature [degC].

    Returns:
        Normalized temperature [degC].

    Reference:
        Gillooly et al. (2001) "Effects of size and temperature on metabolic rate"
    """
    return temp / (1.0 + temp / 273.0)


# =============================================================================
# DERIVED QUANTITIES
# =============================================================================


@functional(
    name="lmtl:recruitment_age",
    units={
        "temp": "degC",
        "tau_r_0": "s",
        "gamma": "1/delta_degC",
        "t_ref": "degC",
        "return": "s",
    },
)
def recruitment_age(
    temp: jnp.ndarray,
    tau_r_0: float,
    gamma: float,
    t_ref: float,
) -> jnp.ndarray:
    """Compute recruitment age (time to recruitment).

    tau_r = tau_r_0 * exp(-gamma * (T - T_ref))

    Args:
        temp: Temperature [degC] (typically Gillooly-normalized).
        tau_r_0: Base recruitment age [s] at reference temperature.
        gamma: Thermal sensitivity coefficient [1/degC].
        t_ref: Reference temperature [degC].

    Returns:
        Recruitment age [s].
    """
    return tau_r_0 * jnp.exp(-gamma * (temp - t_ref))


# =============================================================================
# MORTALITY
# =============================================================================


@functional(
    name="lmtl:mortality",
    units={
        "biomass": "g/m^2",
        "temp": "degC",
        "lambda_0": "1/s",
        "gamma": "1/delta_degC",
        "t_ref": "degC",
        "return": "g/m^2/s",
    },
)
def mortality_tendency(
    biomass: jnp.ndarray,
    temp: jnp.ndarray,
    lambda_0: float,
    gamma: float,
    t_ref: float,
) -> jnp.ndarray:
    """Compute mortality loss for biomass.

    rate = lambda_0 * exp(gamma * (T - T_ref))
    tendency = -rate * biomass

    Args:
        biomass: Current biomass [g/m^2].
        temp: Temperature [degC] (typically Gillooly-normalized).
        lambda_0: Base mortality rate [1/s] at reference temperature.
        gamma: Thermal sensitivity coefficient [1/degC].
        t_ref: Reference temperature [degC].

    Returns:
        Mortality tendency [g/m^2/s] (negative, representing loss).
    """
    rate = lambda_0 * jnp.exp(gamma * (temp - t_ref))
    return -rate * biomass


# =============================================================================
# HELPERS
# =============================================================================


def _cohort_durations(cohort_ages: jnp.ndarray) -> jnp.ndarray:
    """Compute cohort time durations from age boundaries.

    Last cohort (plus-group) reuses the duration of the penultimate cohort.
    """
    d_tau_raw = cohort_ages[1:] - cohort_ages[:-1]
    return jnp.concatenate([d_tau_raw, d_tau_raw[-1:]])


# =============================================================================
# PRODUCTION DYNAMICS
# =============================================================================


@functional(
    name="lmtl:npp_injection",
    core_dims={"production": ["C"]},
    out_dims=["C"],
    units={
        "npp": "g/m^2/s",
        "efficiency": "dimensionless",
        "production": "g/m^2",
        "return": "g/m^2/s",
    },
)
def npp_injection(
    npp: jnp.ndarray,
    efficiency: float,
    production: jnp.ndarray,
) -> jnp.ndarray:
    """Inject Primary Production into the first cohort (0).

    With vmap, this function receives:
    - npp: scalar (for each spatial point)
    - efficiency: scalar
    - production: (C,) - cohort dimension only

    Args:
        npp: Net Primary Production [g/m^2/s].
        efficiency: Transfer efficiency from NPP [dimensionless].
        production: Current production state [g/m^2] with shape (C,).

    Returns:
        Production tendency [g/m^2/s] with shape (C,).
        Only cohort 0 receives flux (efficiency * NPP), others are zero.
    """
    source_flux = npp * efficiency
    tendency = jnp.zeros_like(production)
    tendency = tendency.at[0].set(source_flux)
    return tendency


@functional(
    name="lmtl:aging_flow",
    core_dims={"production": ["C"], "cohort_ages": ["C"]},
    out_dims=["C"],
    units={
        "production": "g/m^2",
        "cohort_ages": "s",
        "rec_age": "s",
        "return": "g/m^2/s",
    },
)
def aging_flow(
    production: jnp.ndarray,
    cohort_ages: jnp.ndarray,
    rec_age: jnp.ndarray,
) -> jnp.ndarray:
    """Compute aging flux (transfer from cohort C to C+1).

    With vmap, this function receives:
    - production: (C,) - cohort dimension only
    - cohort_ages: (C,)
    - rec_age: scalar (for each spatial point)

    Logic:
    - Calculates flow based on cohort duration.
    - A smooth sigmoid determines the recruitment fraction per cohort
      (0 = not recruited, 1 = fully recruited). The non-recruited fraction
      flows to the next cohort (aging).
    - The last cohort does NOT flow out (accumulation/plus group).

    Args:
        production: Current production [g/m^2] with shape (C,).
        cohort_ages: Age of each cohort [s] with shape (C,).
        rec_age: Recruitment age threshold [s].

    Returns:
        Aging tendency [g/m^2/s] with shape (C,).
    """
    d_tau = _cohort_durations(cohort_ages)

    # Aging rate
    aging_coef = 1.0 / d_tau

    # Base Outflow
    base_outflow = production * aging_coef

    # Smooth recruitment fraction: 0 (young) -> 1 (old enough)
    # Sigmoid with half-saturation at rec_age, transition over ±1 day
    recruit_fraction = jax.nn.sigmoid(_K_SIGMOID * (cohort_ages - rec_age))

    # Aging gets the non-recruited fraction
    aging_outflow = (1.0 - recruit_fraction) * base_outflow

    # Last Cohort: no outflow (plus group)
    aging_outflow = aging_outflow.at[-1].set(0.0)

    # Balance (Loss + Gain from prev)
    loss = -aging_outflow

    # Gain for cohort i comes from outflow of i-1
    # Cohort 0 has no aging input
    gain = jnp.concatenate([jnp.zeros(1), aging_outflow[:-1]])

    return loss + gain


@functional(
    name="lmtl:recruitment_flow",
    core_dims={"production": ["C"], "cohort_ages": ["C"]},
    out_dims=["C"],
    outputs=["prod_loss", "biomass_gain"],
    units={
        "production": "g/m^2",
        "cohort_ages": "s",
        "rec_age": "s",
        "prod_loss": "g/m^2/s",
        "biomass_gain": "g/m^2/s",
    },
)
def recruitment_flow(
    production: jnp.ndarray,
    cohort_ages: jnp.ndarray,
    rec_age: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute recruitment flux (transfer from cohorts to Biomass).

    With vmap, this function receives:
    - production: (C,) - cohort dimension only
    - cohort_ages: (C,)
    - rec_age: scalar (for each spatial point)

    Uses a smooth sigmoid to determine the recruited fraction per cohort,
    ensuring differentiability w.r.t. rec_age (and thus tau_r_0, gamma_tau_r).

    Args:
        production: Current production [g/m^2] with shape (C,).
        cohort_ages: Age of each cohort [s] with shape (C,).
        rec_age: Recruitment age threshold [s].

    Returns:
        Tuple of:
        - prod_loss: Production loss tendency [g/m^2/s] with shape (C,).
        - biomass_gain: Biomass gain tendency [g/m^2/s] (scalar).
    """
    d_tau = _cohort_durations(cohort_ages)

    # Aging coefficient
    aging_coef = 1.0 / d_tau
    base_outflow = production * aging_coef

    # Smooth recruitment fraction
    recruit_fraction = jax.nn.sigmoid(_K_SIGMOID * (cohort_ages - rec_age))
    flux_to_biomass = recruit_fraction * base_outflow

    # 1. Loss from Production (C,)
    prod_loss = -flux_to_biomass

    # 2. Gain to Biomass (scalar - sum over all cohorts)
    biomass_gain = jnp.sum(flux_to_biomass)

    return prod_loss, biomass_gain
