"""Zooplankton model from SEAPODYM-LMTL.

This module implements a 2-compartment zooplankton model:
1. Adult biomass B (age-independent): dB/dt = R - λB
2. Juvenile production p(τ) (age-structured): dp/dt + dp/dτ = -μp

Key features:
- Temperature-dependent mortality: λ(T) = λ₀ exp(γ_λ (T - T_ref))
- Production source from NPP: p(τ=0) = E × NPP
- Recruitment by total absorption (α→∞) in window [τ_r(T), τ_r0]
- Both compartments are transported (advection + diffusion)

Unit execution order:
    CRITICAL: The biological units must be executed in this specific order:

    1. compute_recruitment(production, tau_r) -> recruitment
       Calculates recruitment from production[age-1] BEFORE aging occurs

    2. age_production(production, tau_r) -> production
       Ages production and absorbs (sets to 0) recruited age classes

    3. update_biomass(biomass, recruitment, mortality) -> biomass
       Updates adult biomass with the calculated recruitment

    The order is critical because compute_recruitment must read production
    values before age_production modifies them. When creating a Kernel,
    pass units in this order to ensure correct execution.

    Example:
        >>> from seapopym_message.core.kernel import Kernel
        >>> kernel = Kernel([
        ...     compute_recruitment,  # MUST be first
        ...     age_production,       # MUST be second
        ...     update_biomass        # MUST be third
        ... ])

Mathematical formulation:
    See: Annexe A – Formulation du modèle sans transport.md
    Implementation plan: IA/ZOOPLANKTON_IMPLEMENTATION_PLAN.md

References:
    - SEAPODYM-LMTL model (Lehodey et al.)
    - McKendrick-Von Foerster age-structured equation
"""

import jax.numpy as jnp

from seapopym_message.core.unit import unit
from seapopym_message.forcing import derived_forcing

# =============================================================================
# Physiology Units (Temperature -> Parameters)
# =============================================================================


@unit(
    name="compute_tau_r",
    inputs=["temperature"],
    outputs=["tau_r"],
    scope="local",
    forcings=[],
)
def compute_tau_r_unit(
    temperature: jnp.ndarray,
    dt: float,  # noqa: ARG001
    params: dict,
    forcings: dict,  # noqa: ARG001
) -> jnp.ndarray:
    """Calculate minimum recruitment age as function of temperature.

    Equation:
        τ_r(T) = τ_r0 × exp(-γ_τr × (T - T_ref))

    Args:
        temperature: Temperature field [°C]
        dt: Time step (unused).
        params: Dictionary with keys:
            - tau_r0: Maximum recruitment age at T_ref [days]
            - gamma_tau_r: Thermal sensitivity coefficient [°C⁻¹]
            - T_ref: Reference temperature [°C]
        forcings: Empty dict

    Returns:
        Minimum recruitment age [days]
    """
    tau_r0 = params["tau_r0"]
    gamma_tau_r = params["gamma_tau_r"]
    T_ref = params["T_ref"]

    return tau_r0 * jnp.exp(-gamma_tau_r * (temperature - T_ref))


@unit(
    name="compute_mortality",
    inputs=["temperature"],
    outputs=["mortality"],
    scope="local",
    forcings=[],
)
def compute_mortality_unit(
    temperature: jnp.ndarray,
    dt: float,  # noqa: ARG001
    params: dict,
    forcings: dict,  # noqa: ARG001
) -> jnp.ndarray:
    """Calculate temperature-dependent mortality rate.

    Equation:
        λ(T) = λ₀ × exp(γ_λ × (T - T_ref))

    with temperature clipping: T_effective = max(T, T_ref)

    Args:
        temperature: Temperature field [°C]
        dt: Time step (unused).
        params: Dictionary with keys:
            - lambda_0: Baseline mortality at T_ref [day⁻¹]
            - gamma_lambda: Thermal sensitivity coefficient [°C⁻¹]
            - T_ref: Reference temperature [°C]
        forcings: Empty dict

    Returns:
        Mortality rate [day⁻¹]
    """
    lambda_0 = params["lambda_0"]
    gamma_lambda = params["gamma_lambda"]
    T_ref = params["T_ref"]

    # Temperature clipping (SEAPODYM-LMTL formulation)
    T_effective = jnp.maximum(temperature, T_ref)

    return lambda_0 * jnp.exp(gamma_lambda * (T_effective - T_ref))


# =============================================================================
# Biological Units (Dynamics)
# =============================================================================


@unit(
    name="age_production",
    inputs=["production", "tau_r"],
    outputs=["production"],
    scope="local",
    forcings=["npp"],
)
def age_production(
    production: jnp.ndarray,
    tau_r: jnp.ndarray,
    dt: float,  # noqa: ARG001
    params: dict,
    forcings: dict,
) -> jnp.ndarray:
    """Age production with NPP source and total absorption at recruitment.

    Algorithm (with α→∞ limit):
    1. production[0] ← E × NPP (new generation from primary production)
    2. For age=1..n_ages-1:
       - If age < τ_r: production[age] ← production[age-1] (aging)
       - If age ≥ τ_r: production[age] ← 0 (absorbed → recruited to biomass)

    Args:
        production: Production by age class, shape (n_ages, nlat, nlon) [kg/m²]
        tau_r: Minimum recruitment age [days], shape (nlat, nlon)
        dt: Time step [days] (not used)
        params: Model parameters dict with keys:
            - n_ages: Number of age classes
            - E: Transfer efficiency from NPP to production
        forcings: Forcing fields dict with keys:
            - npp: Net primary production [kg/m²/day], shape (nlat, nlon)

    Returns:
        Updated production field, shape (n_ages, nlat, nlon) [kg/m²]
    """
    npp = forcings["npp"]
    n_ages = params["n_ages"]
    E = params["E"]

    # Initialize new production array
    production_new = jnp.zeros_like(production)

    # Age class 0: source from NPP
    production_new = production_new.at[0].set(E * npp)

    # Age classes 1 to n_ages-1: aging with absorption
    for age in range(1, n_ages):
        # Survival mask: 1 if age < τ_r (survives), 0 if age ≥ τ_r (recruited)
        survives = jnp.where(age < tau_r, 1.0, 0.0)

        # Aging: move from age-1 to age, with absorption at recruitment
        production_new = production_new.at[age].set(production[age - 1] * survives)

    return production_new


@unit(
    name="compute_recruitment",
    inputs=["production", "tau_r"],
    outputs=["recruitment"],
    scope="local",
    forcings=[],
)
def compute_recruitment(
    production: jnp.ndarray,
    tau_r: jnp.ndarray,
    dt: float,  # noqa: ARG001
    params: dict,
    forcings: dict,  # noqa: ARG001
) -> jnp.ndarray:
    """Calculate recruitment from absorbed production.

    Equation:
        R = Σ_{age=τ_r}^{τ_r0} p(age)

    Args:
        production: Production by age class, shape (n_ages, nlat, nlon) [kg/m²]
        tau_r: Minimum recruitment age [days], shape (nlat, nlon)
        dt: Time step [days] (not used)
        params: Model parameters dict with keys:
            - n_ages: Number of age classes
        forcings: Empty dict

    Returns:
        Recruitment flux [kg/m²/day], shape (nlat, nlon)
    """
    n_ages = params["n_ages"]

    # Sum production that ages into recruitment window (and gets absorbed)
    # R_age = p_{age-1} if age >= τ_r (production aging from age-1 to age gets recruited)
    recruitment = jnp.zeros_like(production[0])

    for age in range(1, n_ages):  # Start at 1, not 0 (age 0 is newly produced)
        # Recruitment mask: 1 if age ≥ τ_r (recruited), 0 otherwise
        is_recruited = jnp.where(age >= tau_r, 1.0, 0.0)
        # Recruitment is production from previous age class that gets absorbed
        recruitment += production[age - 1] * is_recruited

    return recruitment


@unit(
    name="update_biomass",
    inputs=["biomass", "recruitment", "mortality"],
    outputs=["biomass"],
    scope="local",
    forcings=[],
)
def update_biomass(
    biomass: jnp.ndarray,
    recruitment: jnp.ndarray,
    mortality: jnp.ndarray,
    dt: float,
    params: dict,  # noqa: ARG001
    forcings: dict,  # noqa: ARG001
) -> jnp.ndarray:
    """Update adult biomass using implicit Euler scheme.

    Equation (from Eq. 6 in Annexe A):
        B^{n+1} = (B^n + Δt × R) / (1 + Δt × λ)

    Args:
        biomass: Adult biomass [kg/m²], shape (nlat, nlon)
        recruitment: Recruitment flux [kg/m²/day], shape (nlat, nlon)
        mortality: Temperature-dependent mortality rate [day⁻¹], shape (nlat, nlon)
        dt: Time step [days]
        params: Model parameters dict (not used for this unit)
        forcings: Empty dict

    Returns:
        Updated biomass [kg/m²], shape (nlat, nlon)
    """
    # Implicit Euler update (unconditionally stable)
    biomass_new = (biomass + dt * recruitment) / (1.0 + dt * mortality)

    return biomass_new


# =============================================================================
# Legacy / Derived Forcings (Kept for reference, but Units preferred)
# =============================================================================


@derived_forcing(
    name="tau_r",
    inputs=["temperature"],
    params=["tau_r0", "gamma_tau_r", "T_ref"],
)
def compute_tau_r_forcing(
    temperature: jnp.ndarray, tau_r0: float, gamma_tau_r: float, T_ref: float
) -> jnp.ndarray:
    """Compute minimum recruitment age as derived forcing."""
    return tau_r0 * jnp.exp(-gamma_tau_r * (temperature - T_ref))


@derived_forcing(
    name="mortality",
    inputs=["temperature"],
    params=["lambda_0", "gamma_lambda", "T_ref"],
)
def compute_mortality_forcing(
    temperature: jnp.ndarray, lambda_0: float, gamma_lambda: float, T_ref: float
) -> jnp.ndarray:
    """Compute temperature-dependent mortality as derived forcing."""
    T_effective = jnp.maximum(temperature, T_ref)
    return lambda_0 * jnp.exp(gamma_lambda * (T_effective - T_ref))
