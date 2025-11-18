"""Zooplankton model from SEAPODYM-LMTL.

This module implements a 2-compartment zooplankton model:
1. Adult biomass B (age-independent): dB/dt = R - λB
2. Juvenile production p(τ) (age-structured): dp/dt + dp/dτ = -μp

Key features:
- Temperature-dependent mortality: λ(T) = λ₀ exp(γ_λ (T - T_ref))
- Production source from NPP: p(τ=0) = E × NPP
- Recruitment by total absorption (α→∞) in window [τ_r(T), τ_r0]
- Both compartments are transported (advection + diffusion)

Mathematical formulation:
    See: Annexe A – Formulation du modèle sans transport.md
    Implementation plan: IA/ZOOPLANKTON_IMPLEMENTATION_PLAN.md

References:
    - SEAPODYM-LMTL model (Lehodey et al.)
    - McKendrick-Von Foerster age-structured equation
"""

import jax.numpy as jnp

from seapopym_message.core.unit import unit


def compute_tau_r(temperature: jnp.ndarray, params: dict) -> jnp.ndarray:
    """Calculate minimum recruitment age as function of temperature.

    Equation:
        τ_r(T) = τ_r0 × exp(-γ_τr × (T - T_ref))

    Lower temperatures → longer development time (higher τ_r)
    Higher temperatures → faster development (lower τ_r)

    Args:
        temperature: Temperature field [°C], shape (nlat, nlon)
        params: Dictionary with keys:
            - tau_r0: Maximum recruitment age at T_ref [days]
            - gamma_tau_r: Thermal sensitivity coefficient [°C⁻¹]
            - T_ref: Reference temperature [°C]

    Returns:
        Minimum recruitment age [days], shape (nlat, nlon)

    Example:
        >>> import jax.numpy as jnp
        >>> params = {"tau_r0": 10.38, "gamma_tau_r": 0.11, "T_ref": 0.0}
        >>> T = jnp.array([[0.0, 10.0], [20.0, 30.0]])
        >>> tau_r = compute_tau_r(T, params)
        >>> tau_r[0, 0]  # At T_ref
        Array(10.38, dtype=float32)
        >>> tau_r[1, 0] < tau_r[0, 0]  # Warmer water → shorter development
        Array(True, dtype=bool)
    """
    tau_r0 = params["tau_r0"]
    gamma_tau_r = params["gamma_tau_r"]
    T_ref = params["T_ref"]

    tau_r = tau_r0 * jnp.exp(-gamma_tau_r * (temperature - T_ref))

    return tau_r


def compute_mortality(temperature: jnp.ndarray, params: dict) -> jnp.ndarray:
    """Calculate temperature-dependent mortality rate.

    Equation:
        λ(T) = λ₀ × exp(γ_λ × (T - T_ref))

    with temperature clipping: T_effective = max(T, T_ref)
    (as in SEAPODYM-LMTL original formulation)

    Lower temperatures → lower mortality
    Higher temperatures → higher mortality

    Args:
        temperature: Temperature field [°C], shape (nlat, nlon)
        params: Dictionary with keys:
            - lambda_0: Baseline mortality at T_ref [day⁻¹]
            - gamma_lambda: Thermal sensitivity coefficient [°C⁻¹]
            - T_ref: Reference temperature [°C]

    Returns:
        Mortality rate [day⁻¹], shape (nlat, nlon)

    Example:
        >>> import jax.numpy as jnp
        >>> params = {"lambda_0": 1/150, "gamma_lambda": 0.15, "T_ref": 0.0}
        >>> T = jnp.array([[0.0, 10.0], [20.0, 30.0]])
        >>> lambda_T = compute_mortality(T, params)
        >>> lambda_T[0, 0]  # At T_ref
        Array(0.00666667, dtype=float32)
        >>> lambda_T[1, 0] > lambda_T[0, 0]  # Warmer water → higher mortality
        Array(True, dtype=bool)
    """
    lambda_0 = params["lambda_0"]
    gamma_lambda = params["gamma_lambda"]
    T_ref = params["T_ref"]

    # Temperature clipping (SEAPODYM-LMTL formulation)
    T_effective = jnp.maximum(temperature, T_ref)

    mortality = lambda_0 * jnp.exp(gamma_lambda * (T_effective - T_ref))

    return mortality


@unit(
    name="age_production",
    inputs=["production"],
    outputs=["production"],
    scope="local",
    forcings=["npp", "temperature"],
)
def age_production(
    production: jnp.ndarray, _dt: float, params: dict, forcings: dict
) -> jnp.ndarray:
    """Age production with NPP source and total absorption at recruitment.

    Algorithm (with α→∞ limit):
    1. production[0] ← E × NPP (new generation from primary production)
    2. For age=1..n_ages-1:
       - If age < τ_r(T): production[age] ← production[age-1] (aging)
       - If age ≥ τ_r(T): production[age] ← 0 (absorbed → recruited to biomass)

    The total absorption (α→∞) means all production reaching recruitment age
    τ_r is immediately transferred to biomass B, simplifying the original
    McKendrick-Von Foerster equation.

    Args:
        production: Production by age class, shape (n_ages, nlat, nlon) [kg/m²]
        dt: Time step [days]
        params: Model parameters dict with keys:
            - n_ages: Number of age classes
            - E: Transfer efficiency from NPP to production
            - tau_r0, gamma_tau_r, T_ref: recruitment age parameters
        forcings: Forcing fields dict with keys:
            - npp: Net primary production [kg/m²/day], shape (nlat, nlon)
            - temperature: Sea temperature [°C], shape (nlat, nlon)

    Returns:
        Updated production field, shape (n_ages, nlat, nlon) [kg/m²]

    Note:
        This unit must be executed BEFORE compute_recruitment in the kernel
        to ensure proper mass conservation.

    Example:
        >>> import jax.numpy as jnp
        >>> production = jnp.zeros((11, 10, 10))  # 11 age classes
        >>> forcings = {
        ...     "npp": jnp.ones((10, 10)) * 5.0,
        ...     "temperature": jnp.ones((10, 10)) * 15.0
        ... }
        >>> params = {
        ...     "n_ages": 11, "E": 0.1668,
        ...     "tau_r0": 10.38, "gamma_tau_r": 0.11, "T_ref": 0.0
        ... }
        >>> prod_new = age_production(production, 1.0, params, forcings)
        >>> prod_new[0, 0, 0]  # New production from NPP
        Array(0.834, dtype=float32)
    """
    npp = forcings["npp"]
    temperature = forcings["temperature"]

    n_ages = params["n_ages"]
    E = params["E"]

    # Calculate minimum recruitment age (days) - depends on temperature
    tau_r = compute_tau_r(temperature, params)

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
    inputs=["production"],
    outputs=["recruitment"],
    scope="local",
    forcings=["temperature"],
)
def compute_recruitment(
    production: jnp.ndarray, _dt: float, params: dict, forcings: dict
) -> jnp.ndarray:
    """Calculate recruitment from absorbed production.

    With total absorption (α→∞), all production reaching the recruitment
    window [τ_r(T), τ_r0] is immediately recruited to adult biomass.

    Equation:
        R = Σ_{age=τ_r}^{τ_r0} p(age)

    where τ_r(T) is temperature-dependent minimum recruitment age.

    Args:
        production: Production by age class, shape (n_ages, nlat, nlon) [kg/m²]
        dt: Time step [days] (not used but required by unit signature)
        params: Model parameters dict with keys:
            - n_ages: Number of age classes
            - tau_r0, gamma_tau_r, T_ref: recruitment age parameters
        forcings: Forcing fields dict with keys:
            - temperature: Sea temperature [°C], shape (nlat, nlon)

    Returns:
        Recruitment flux [kg/m²/day], shape (nlat, nlon)

    Note:
        After this unit, the recruited production is already set to 0 by
        age_production unit, ensuring mass conservation.

    Example:
        >>> import jax.numpy as jnp
        >>> # Production with some in recruitment window
        >>> production = jnp.zeros((11, 10, 10))
        >>> production = production.at[5].set(jnp.ones((10, 10)) * 2.0)
        >>> production = production.at[8].set(jnp.ones((10, 10)) * 1.0)
        >>> forcings = {"temperature": jnp.ones((10, 10)) * 10.0}
        >>> params = {"n_ages": 11, "tau_r0": 10.38, "gamma_tau_r": 0.11, "T_ref": 0.0}
        >>> R = compute_recruitment(production, 1.0, params, forcings)
        >>> R[0, 0]  # Sum of recruited age classes
        Array(3.0, dtype=float32)
    """
    temperature = forcings["temperature"]
    n_ages = params["n_ages"]

    # Calculate minimum recruitment age
    tau_r = compute_tau_r(temperature, params)

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
    inputs=["biomass", "recruitment"],
    outputs=["biomass"],
    scope="local",
    forcings=["temperature"],
)
def update_biomass(
    biomass: jnp.ndarray, recruitment: jnp.ndarray, dt: float, params: dict, forcings: dict
) -> jnp.ndarray:
    """Update adult biomass using implicit Euler scheme.

    Equation (from Eq. 6 in Annexe A):
        B^{n+1} = (B^n + Δt × R) / (1 + Δt × λ(T))

    where:
    - R is recruitment from juvenile production [kg/m²/day]
    - λ(T) is temperature-dependent mortality [day⁻¹]

    The implicit Euler scheme is unconditionally stable and ensures
    biomass remains positive.

    Args:
        biomass: Adult biomass [kg/m²], shape (nlat, nlon)
        recruitment: Recruitment flux [kg/m²/day], shape (nlat, nlon)
        dt: Time step [days]
        params: Model parameters dict with keys:
            - lambda_0, gamma_lambda, T_ref: mortality parameters
        forcings: Forcing fields dict with keys:
            - temperature: Sea temperature [°C], shape (nlat, nlon)

    Returns:
        Updated biomass [kg/m²], shape (nlat, nlon)

    Note:
        At steady state without transport: R ≈ λB

    Example:
        >>> import jax.numpy as jnp
        >>> biomass = jnp.ones((10, 10)) * 100.0
        >>> recruitment = jnp.ones((10, 10)) * 5.0
        >>> forcings = {"temperature": jnp.ones((10, 10)) * 15.0}
        >>> params = {"lambda_0": 1/150, "gamma_lambda": 0.15, "T_ref": 0.0}
        >>> B_new = update_biomass(biomass, recruitment, 1.0, params, forcings)
        >>> B_new[0, 0] > biomass[0, 0]  # Biomass increases with recruitment
        Array(True, dtype=bool)
    """
    temperature = forcings["temperature"]

    # Calculate temperature-dependent mortality
    mortality = compute_mortality(temperature, params)

    # Implicit Euler update (unconditionally stable)
    biomass_new = (biomass + dt * recruitment) / (1.0 + dt * mortality)

    return biomass_new
