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

    1. compute_recruitment(production) -> recruitment
       Calculates recruitment from production[age-1] BEFORE aging occurs

    2. age_production(production) -> production
       Ages production and absorbs (sets to 0) recruited age classes

    3. update_biomass(biomass, recruitment) -> biomass
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
# Legacy Helper Functions (DEPRECATED - Use derived_forcing instead)
# =============================================================================
# These functions are kept for backward compatibility but should not be used
# in new code. Use the @derived_forcing versions (compute_tau_r_forcing,
# compute_mortality_forcing) registered with ForcingManager instead.


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


# =============================================================================
# Derived Forcings (computed once per timestep by ForcingManager)
# =============================================================================


@derived_forcing(
    name="tau_r",
    inputs=["temperature"],
    params=["tau_r0", "gamma_tau_r", "T_ref"],
)
def compute_tau_r_forcing(
    temperature: jnp.ndarray, tau_r0: float, gamma_tau_r: float, T_ref: float
) -> jnp.ndarray:
    """Compute minimum recruitment age as derived forcing.

    This derived forcing transforms temperature into tau_r field, computed
    once per timestep by ForcingManager and distributed to all workers.

    Equation:
        τ_r(T) = τ_r0 × exp(-γ_τr × (T - T_ref))

    Args:
        temperature: Temperature field [°C], shape (nlat, nlon)
        tau_r0: Maximum recruitment age at T_ref [days]
        gamma_tau_r: Thermal sensitivity coefficient [°C⁻¹]
        T_ref: Reference temperature [°C]

    Returns:
        Minimum recruitment age [days], shape (nlat, nlon)
    """
    return tau_r0 * jnp.exp(-gamma_tau_r * (temperature - T_ref))


@derived_forcing(
    name="mortality",
    inputs=["temperature"],
    params=["lambda_0", "gamma_lambda", "T_ref"],
)
def compute_mortality_forcing(
    temperature: jnp.ndarray, lambda_0: float, gamma_lambda: float, T_ref: float
) -> jnp.ndarray:
    """Compute temperature-dependent mortality as derived forcing.

    This derived forcing transforms temperature into mortality rate field,
    computed once per timestep by ForcingManager and distributed to all workers.

    Equation:
        λ(T) = λ₀ × exp(γ_λ × (T - T_ref))

    with temperature clipping: T_effective = max(T, T_ref)

    Args:
        temperature: Temperature field [°C], shape (nlat, nlon)
        lambda_0: Baseline mortality at T_ref [day⁻¹]
        gamma_lambda: Thermal sensitivity coefficient [°C⁻¹]
        T_ref: Reference temperature [°C]

    Returns:
        Mortality rate [day⁻¹], shape (nlat, nlon)
    """
    T_effective = jnp.maximum(temperature, T_ref)
    return lambda_0 * jnp.exp(gamma_lambda * (T_effective - T_ref))


# =============================================================================
# Units (biological model logic, executed by workers)
# =============================================================================


@unit(
    name="age_production",
    inputs=["production"],
    outputs=["production"],
    scope="local",
    forcings=["npp", "tau_r"],
)
def age_production(production: jnp.ndarray, dt: float, params: dict, forcings: dict) -> jnp.ndarray:  # noqa: ARG001
    """Age production with NPP source and total absorption at recruitment.

    Algorithm (with α→∞ limit):
    1. production[0] ← E × NPP (new generation from primary production)
    2. For age=1..n_ages-1:
       - If age < τ_r: production[age] ← production[age-1] (aging)
       - If age ≥ τ_r: production[age] ← 0 (absorbed → recruited to biomass)

    The total absorption (α→∞) means all production reaching recruitment age
    τ_r is immediately transferred to biomass B, simplifying the original
    McKendrick-Von Foerster equation.

    Args:
        production: Production by age class, shape (n_ages, nlat, nlon) [kg/m²]
        dt: Time step [days] (not used, kept for unit signature compatibility)
        params: Model parameters dict with keys:
            - n_ages: Number of age classes
            - E: Transfer efficiency from NPP to production
        forcings: Forcing fields dict with keys:
            - npp: Net primary production [kg/m²/day], shape (nlat, nlon)
            - tau_r: Minimum recruitment age [days], shape (nlat, nlon)
                    (computed as derived forcing from temperature)

    Returns:
        Updated production field, shape (n_ages, nlat, nlon) [kg/m²]

    Note:
        This unit must be executed AFTER compute_recruitment in the kernel
        to ensure proper mass conservation. The recruitment is calculated from
        production[age-1] before aging occurs, then aging absorbs (sets to 0)
        the recruited production.

    Example:
        >>> import jax.numpy as jnp
        >>> production = jnp.zeros((11, 10, 10))  # 11 age classes
        >>> forcings = {
        ...     "npp": jnp.ones((10, 10)) * 5.0,
        ...     "tau_r": jnp.ones((10, 10)) * 3.45  # Pre-computed from temperature
        ... }
        >>> params = {"n_ages": 11, "E": 0.1668}
        >>> prod_new = age_production(production, 1.0, params, forcings)
        >>> prod_new[0, 0, 0]  # New production from NPP
        Array(0.834, dtype=float32)
    """
    npp = forcings["npp"]
    tau_r = forcings["tau_r"]

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
    inputs=["production"],
    outputs=["recruitment"],
    scope="local",
    forcings=["tau_r"],
)
def compute_recruitment(
    production: jnp.ndarray,
    dt: float,  # noqa: ARG001
    params: dict,
    forcings: dict,
) -> jnp.ndarray:
    """Calculate recruitment from absorbed production.

    With total absorption (α→∞), all production reaching the recruitment
    window [τ_r, τ_r0] is immediately recruited to adult biomass.

    Equation:
        R = Σ_{age=τ_r}^{τ_r0} p(age)

    where τ_r is the minimum recruitment age (temperature-dependent,
    pre-computed as derived forcing).

    Args:
        production: Production by age class, shape (n_ages, nlat, nlon) [kg/m²]
        dt: Time step [days] (not used, kept for unit signature compatibility)
        params: Model parameters dict with keys:
            - n_ages: Number of age classes
        forcings: Forcing fields dict with keys:
            - tau_r: Minimum recruitment age [days], shape (nlat, nlon)
                    (computed as derived forcing from temperature)

    Returns:
        Recruitment flux [kg/m²/day], shape (nlat, nlon)

    Note:
        This unit must be executed BEFORE age_production in the kernel.
        It calculates recruitment from production[age-1] before the aging
        step absorbs (sets to 0) the recruited production.

    Example:
        >>> import jax.numpy as jnp
        >>> # Production with some in recruitment window
        >>> production = jnp.zeros((11, 10, 10))
        >>> production = production.at[5].set(jnp.ones((10, 10)) * 2.0)
        >>> production = production.at[8].set(jnp.ones((10, 10)) * 1.0)
        >>> forcings = {"tau_r": jnp.ones((10, 10)) * 3.45}  # Pre-computed
        >>> params = {"n_ages": 11}
        >>> R = compute_recruitment(production, 1.0, params, forcings)
        >>> R[0, 0]  # Sum of recruited age classes
        Array(3.0, dtype=float32)
    """
    tau_r = forcings["tau_r"]
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
    inputs=["biomass", "recruitment"],
    outputs=["biomass"],
    scope="local",
    forcings=["mortality"],
)
def update_biomass(
    biomass: jnp.ndarray,
    recruitment: jnp.ndarray,
    dt: float,
    params: dict,  # noqa: ARG001
    forcings: dict,
) -> jnp.ndarray:
    """Update adult biomass using implicit Euler scheme.

    Equation (from Eq. 6 in Annexe A):
        B^{n+1} = (B^n + Δt × R) / (1 + Δt × λ)

    where:
    - R is recruitment from juvenile production [kg/m²/day]
    - λ is temperature-dependent mortality [day⁻¹]
      (pre-computed as derived forcing from temperature)

    The implicit Euler scheme is unconditionally stable and ensures
    biomass remains positive.

    Args:
        biomass: Adult biomass [kg/m²], shape (nlat, nlon)
        recruitment: Recruitment flux [kg/m²/day], shape (nlat, nlon)
        dt: Time step [days]
        params: Model parameters dict (not used for this unit)
        forcings: Forcing fields dict with keys:
            - mortality: Temperature-dependent mortality rate [day⁻¹], shape (nlat, nlon)
                        (computed as derived forcing from temperature)

    Returns:
        Updated biomass [kg/m²], shape (nlat, nlon)

    Note:
        At steady state without transport: R ≈ λB

    Example:
        >>> import jax.numpy as jnp
        >>> biomass = jnp.ones((10, 10)) * 100.0
        >>> recruitment = jnp.ones((10, 10)) * 5.0
        >>> forcings = {"mortality": jnp.ones((10, 10)) * 0.01}  # Pre-computed
        >>> params = {}
        >>> B_new = update_biomass(biomass, recruitment, 1.0, params, forcings)
        >>> B_new[0, 0] > biomass[0, 0]  # Biomass increases with recruitment
        Array(True, dtype=bool)
    """
    mortality = forcings["mortality"]

    # Implicit Euler update (unconditionally stable)
    biomass_new = (biomass + dt * recruitment) / (1.0 + dt * mortality)

    return biomass_new
