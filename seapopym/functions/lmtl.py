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

Ageing and recruitment
----------------------
The cohorts are the cells of an age axis along which production is advected at unit speed. What leaves a cohort
is the density reconstructed at its older face by MUSCL with the van Leer limiter, on the actual cohort widths,
which may differ. With a time step shorter than the cohorts, as transport imposes, the first-order outflow
``P_i / d_i`` makes each cohort a compartment and spreads the delay, its variance the sum of the squared widths
passed; the reconstruction keeps most of that spread away. Like every tendency here, it does not see the step.

Recruitment splits what leaves the cohorts around the recruitment age ``r`` so that the mean delay is ``r``
whatever the widths: cohort ``i`` is left at the mean age ``E_i = a_i + d_i``, and the share
``(E_i + d_(i+1) - r) / d_(i+1)``, clipped to [0, 1], of what leaves the cohort whose exit is the last before
``r`` is recruited, the rest of it at the next exit. The cohorts that recruit give their first-order outflow,
since the cohort after them has been emptied by the recruitment and a reconstruction reading it would hold the
mass back. The last cohort recruits everything it holds.
"""

from __future__ import annotations

import jax.numpy as jnp

from seapopym.blueprint import functional

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
    It is the formula of the original SeapoPym (``function/day_length.py``)
    with its default angle between the sun and the horizon, p = 0: the day
    starts and ends when the centre of the sun crosses the horizon, with no
    twilight and no refraction.

    Args:
        latitude: Latitude in degrees [-90, 90].
        day_of_year: Day of year [1-366].

    Returns:
        Day length fraction [0-1], where 0 = polar night, 1 = midnight sun.

    Reference:
        Forsythe, W. C., Rykiel, E. J., Stahl, R. S., Wu, H., Schoolfield, R. M.
        (1995) "A model comparison for daylength as a function of latitude and
        day of year", Ecological Modelling 80, 87-95.
    """
    phi = jnp.deg2rad(latitude)

    # Revolution angle of the Earth on its orbit, then the sun's declination (CBM)
    theta = 0.2163108 + 2.0 * jnp.arctan(0.9671396 * jnp.tan(0.00860 * (day_of_year - 186.0)))
    declination = jnp.arcsin(0.39795 * jnp.cos(theta))

    # With p = 0 the sin(p) term of CBM vanishes. Clamp to [-1, 1] for polar days and nights.
    arg = jnp.clip(jnp.sin(phi) * jnp.sin(declination) / (jnp.cos(phi) * jnp.cos(declination)), -1.0, 1.0)
    return 1.0 - jnp.arccos(arg) / jnp.pi


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


def _recruited_share(cohort_ages: jnp.ndarray, rec_age: jnp.ndarray) -> jnp.ndarray:
    """Return the share of what leaves each cohort that is recruited.

    The mean delay is ``rec_age`` whatever the widths: the share of the cohort whose exit is the last before
    ``rec_age`` is interpolated between its exit and the next one, the cohorts after it recruit everything, and
    the last cohort always does.
    """
    d_tau = _cohort_durations(cohort_ages)
    exits = cohort_ages + d_tau
    following = jnp.concatenate([d_tau[1:], d_tau[-1:]])
    share = jnp.clip((exits + following - rec_age) / following, 0.0, 1.0)
    return share.at[-1].set(1.0)


def _outflow(production: jnp.ndarray, cohort_ages: jnp.ndarray, share: jnp.ndarray) -> jnp.ndarray:
    """Return what leaves each cohort per second, by MUSCL with the van Leer limiter along the age axis.

    The content of a cohort is the density ``P_i / d_i`` moving at unit speed. The flux through its older face is
    the density there, reconstructed from the cohort and its two neighbours, the slopes taken over the distances
    between cohort centres. The youngest cohort sees a flat profile behind it. The cohorts that recruit
    (``share`` above zero) and the last one give their own density, the first-order outflow.
    """
    d_tau = _cohort_durations(cohort_ages)
    density = production / d_tau
    younger = jnp.concatenate([density[:1], density[:-1]])
    older = jnp.concatenate([density[1:], density[-1:]])
    d_younger = jnp.concatenate([d_tau[:1], d_tau[:-1]])
    d_older = jnp.concatenate([d_tau[1:], d_tau[-1:]])
    back = (density - younger) / (0.5 * (d_tau + d_younger))
    ahead = (older - density) / (0.5 * (d_tau + d_older))
    product = back * ahead
    same_sign = product > 0
    slope = jnp.where(same_sign, 2.0 * product / jnp.where(same_sign, back + ahead, 1.0), 0.0)
    face = jnp.where(share > 0, density, density + 0.5 * d_tau * slope)
    return face.at[-1].set(density[-1])


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
    - What leaves a cohort is reconstructed by MUSCL along the age axis (see the module docstring).
    - The share of it that is not recruited (``_recruited_share``) flows to the next cohort.
    - The last cohort recruits everything, so nothing ages out of it.

    Args:
        production: Current production [g/m^2] with shape (C,).
        cohort_ages: Age of each cohort [s] with shape (C,).
        rec_age: Recruitment age threshold [s].

    Returns:
        Aging tendency [g/m^2/s] with shape (C,).
    """
    share = _recruited_share(cohort_ages, rec_age)
    aging_outflow = (1.0 - share) * _outflow(production, cohort_ages, share)

    # Last Cohort: no outflow (it recruits everything)
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

    The recruited share of what leaves each cohort (``_recruited_share``) puts the mean delay at ``rec_age``
    whatever the widths. It is piecewise linear in ``rec_age``, hence differentiable except at the exits.

    Args:
        production: Current production [g/m^2] with shape (C,).
        cohort_ages: Age of each cohort [s] with shape (C,).
        rec_age: Recruitment age threshold [s].

    Returns:
        Tuple of:
        - prod_loss: Production loss tendency [g/m^2/s] with shape (C,).
        - biomass_gain: Biomass gain tendency [g/m^2/s] (scalar).
    """
    share = _recruited_share(cohort_ages, rec_age)
    flux_to_biomass = share * _outflow(production, cohort_ages, share)

    # 1. Loss from Production (C,)
    prod_loss = -flux_to_biomass

    # 2. Gain to Biomass (scalar - sum over all cohorts)
    biomass_gain = jnp.sum(flux_to_biomass)

    return prod_loss, biomass_gain
