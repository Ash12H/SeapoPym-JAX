"""Unit tests for zooplankton model."""

import jax.numpy as jnp

from seapopym_message.kernels.zooplankton import (
    age_production as age_production_unit,
)
from seapopym_message.kernels.zooplankton import (
    compute_mortality,
    compute_mortality_forcing,
    compute_tau_r,
    compute_tau_r_forcing,
)
from seapopym_message.kernels.zooplankton import (
    compute_recruitment as compute_recruitment_unit,
)
from seapopym_message.kernels.zooplankton import (
    update_biomass as update_biomass_unit,
)

# Extract underlying functions from Unit decorators
age_production = age_production_unit.func
compute_recruitment = compute_recruitment_unit.func
update_biomass = update_biomass_unit.func


class TestComputeTauR:
    """Tests for compute_tau_r function."""

    def test_tau_r_at_tref(self):
        """τ_r at T_ref should equal τ_r0."""
        params = {"tau_r0": 10.38, "gamma_tau_r": 0.11, "T_ref": 0.0}
        temperature = jnp.ones((5, 5)) * 0.0  # T = T_ref

        tau_r = compute_tau_r(temperature, params)

        assert jnp.allclose(tau_r, 10.38, rtol=1e-5)

    def test_tau_r_decreases_with_temperature(self):
        """τ_r should decrease as temperature increases."""
        params = {"tau_r0": 10.38, "gamma_tau_r": 0.11, "T_ref": 0.0}

        T_cold = jnp.ones((5, 5)) * 0.0
        T_warm = jnp.ones((5, 5)) * 20.0

        tau_r_cold = compute_tau_r(T_cold, params)
        tau_r_warm = compute_tau_r(T_warm, params)

        # Warmer water → faster development → lower τ_r
        assert jnp.all(tau_r_warm < tau_r_cold)

    def test_tau_r_warm_water_value(self):
        """τ_r at 20°C should be approximately 1.16 days."""
        params = {"tau_r0": 10.38, "gamma_tau_r": 0.11, "T_ref": 0.0}
        temperature = jnp.ones((5, 5)) * 20.0

        tau_r = compute_tau_r(temperature, params)

        # τ_r(20°C) = 10.38 × exp(-0.11 × 20) ≈ 1.16
        expected = 10.38 * jnp.exp(-0.11 * 20)
        assert jnp.allclose(tau_r, expected, rtol=1e-3)

    def test_tau_r_shape(self):
        """τ_r should have same shape as temperature."""
        params = {"tau_r0": 10.38, "gamma_tau_r": 0.11, "T_ref": 0.0}
        temperature = jnp.ones((10, 20)) * 15.0

        tau_r = compute_tau_r(temperature, params)

        assert tau_r.shape == (10, 20)


class TestComputeMortality:
    """Tests for compute_mortality function."""

    def test_mortality_at_tref(self):
        """λ at T_ref should equal λ₀."""
        params = {"lambda_0": 1 / 150, "gamma_lambda": 0.15, "T_ref": 0.0}
        temperature = jnp.ones((5, 5)) * 0.0

        mortality = compute_mortality(temperature, params)

        assert jnp.allclose(mortality, 1 / 150, rtol=1e-5)

    def test_mortality_increases_with_temperature(self):
        """λ should increase as temperature increases."""
        params = {"lambda_0": 1 / 150, "gamma_lambda": 0.15, "T_ref": 0.0}

        T_cold = jnp.ones((5, 5)) * 0.0
        T_warm = jnp.ones((5, 5)) * 20.0

        lambda_cold = compute_mortality(T_cold, params)
        lambda_warm = compute_mortality(T_warm, params)

        # Warmer water → higher mortality
        assert jnp.all(lambda_warm > lambda_cold)

    def test_mortality_warm_water_value(self):
        """λ at 20°C should be approximately 0.134 day⁻¹."""
        params = {"lambda_0": 1 / 150, "gamma_lambda": 0.15, "T_ref": 0.0}
        temperature = jnp.ones((5, 5)) * 20.0

        mortality = compute_mortality(temperature, params)

        # λ(20°C) = (1/150) × exp(0.15 × 20) ≈ 0.134
        expected = (1 / 150) * jnp.exp(0.15 * 20)
        assert jnp.allclose(mortality, expected, rtol=1e-3)

    def test_mortality_temperature_clipping(self):
        """λ should use max(T, T_ref) for negative temperatures."""
        params = {"lambda_0": 1 / 150, "gamma_lambda": 0.15, "T_ref": 0.0}

        T_negative = jnp.ones((5, 5)) * -10.0
        T_at_ref = jnp.ones((5, 5)) * 0.0

        lambda_negative = compute_mortality(T_negative, params)
        lambda_at_ref = compute_mortality(T_at_ref, params)

        # With clipping, both should give same result
        assert jnp.allclose(lambda_negative, lambda_at_ref, rtol=1e-5)

    def test_mortality_shape(self):
        """λ should have same shape as temperature."""
        params = {"lambda_0": 1 / 150, "gamma_lambda": 0.15, "T_ref": 0.0}
        temperature = jnp.ones((10, 20)) * 15.0

        mortality = compute_mortality(temperature, params)

        assert mortality.shape == (10, 20)


class TestAgeProduction:
    """Tests for age_production unit."""

    def test_source_from_npp(self):
        """production[0] should be E × NPP."""
        n_ages = 11
        nlat, nlon = 5, 5

        production = jnp.zeros((n_ages, nlat, nlon))

        # Compute tau_r as derived forcing
        temperature = jnp.ones((nlat, nlon)) * 15.0
        tau_r = compute_tau_r_forcing.func(
            temperature=temperature, tau_r0=10.38, gamma_tau_r=0.11, T_ref=0.0
        )

        forcings = {
            "npp": jnp.ones((nlat, nlon)) * 10.0,
            "tau_r": tau_r,
        }
        params = {"n_ages": n_ages, "E": 0.1668}

        production_new = age_production(production, 1.0, params, forcings)

        # production[0] = E × NPP = 0.1668 × 10 = 1.668
        assert jnp.allclose(production_new[0], 1.668, rtol=1e-4)

    def test_aging_before_recruitment(self):
        """Production should age correctly before reaching τ_r."""
        n_ages = 11
        nlat, nlon = 5, 5

        # Set production at age 2
        production = jnp.zeros((n_ages, nlat, nlon))
        production = production.at[2].set(jnp.ones((nlat, nlon)) * 5.0)

        # Low temperature → high τ_r → age 3 not yet recruited
        temperature = jnp.ones((nlat, nlon)) * 0.0
        tau_r = compute_tau_r_forcing.func(
            temperature=temperature, tau_r0=10.38, gamma_tau_r=0.11, T_ref=0.0
        )

        forcings = {"npp": jnp.zeros((nlat, nlon)), "tau_r": tau_r}
        params = {"n_ages": n_ages, "E": 0.1668}

        production_new = age_production(production, 1.0, params, forcings)

        # At T=0°C, τ_r = 10.38, so age 3 should survive
        # production[3] should be 5.0 (aged from production[2])
        assert jnp.allclose(production_new[3], 5.0, rtol=1e-4)
        # production[2] should be 0 (aged out)
        assert jnp.allclose(production_new[2], 0.0, rtol=1e-4)

    def test_absorption_at_recruitment(self):
        """Production ≥ τ_r should be absorbed (set to 0)."""
        n_ages = 11
        nlat, nlon = 5, 5

        # Set production at age 5
        production = jnp.zeros((n_ages, nlat, nlon))
        production = production.at[5].set(jnp.ones((nlat, nlon)) * 10.0)

        # High temperature → low τ_r → age 6 will be recruited
        # At T=20°C: τ_r ≈ 1.16, so age 6 > τ_r → absorbed
        temperature = jnp.ones((nlat, nlon)) * 20.0
        tau_r = compute_tau_r_forcing.func(
            temperature=temperature, tau_r0=10.38, gamma_tau_r=0.11, T_ref=0.0
        )

        forcings = {"npp": jnp.zeros((nlat, nlon)), "tau_r": tau_r}
        params = {"n_ages": n_ages, "E": 0.1668}

        production_new = age_production(production, 1.0, params, forcings)

        # production[6] should be 0 (absorbed, age ≥ τ_r)
        assert jnp.allclose(production_new[6], 0.0, rtol=1e-4)

    def test_output_shape(self):
        """Output should have same shape as input."""
        n_ages = 11
        nlat, nlon = 10, 20

        production = jnp.zeros((n_ages, nlat, nlon))

        temperature = jnp.ones((nlat, nlon)) * 15.0
        tau_r = compute_tau_r_forcing.func(
            temperature=temperature, tau_r0=10.38, gamma_tau_r=0.11, T_ref=0.0
        )

        forcings = {"npp": jnp.ones((nlat, nlon)), "tau_r": tau_r}
        params = {"n_ages": n_ages, "E": 0.1668}

        production_new = age_production(production, 1.0, params, forcings)

        assert production_new.shape == (n_ages, nlat, nlon)


class TestComputeRecruitment:
    """Tests for compute_recruitment unit."""

    def test_recruitment_sum(self):
        """R should be sum of production in recruitment window."""
        n_ages = 11
        nlat, nlon = 5, 5

        # Set production at various ages
        production = jnp.zeros((n_ages, nlat, nlon))
        production = production.at[5].set(jnp.ones((nlat, nlon)) * 2.0)
        production = production.at[8].set(jnp.ones((nlat, nlon)) * 3.0)
        production = production.at[10].set(jnp.ones((nlat, nlon)) * 1.0)

        # At T=5°C: τ_r = 10.38 × exp(-0.11 × 5) ≈ 5.99
        # So ages >= 6 will be recruited
        temperature = jnp.ones((nlat, nlon)) * 5.0
        tau_r = compute_tau_r_forcing.func(
            temperature=temperature, tau_r0=10.38, gamma_tau_r=0.11, T_ref=0.0
        )

        forcings = {"tau_r": tau_r}
        params = {"n_ages": n_ages}

        recruitment = compute_recruitment(production, 1.0, params, forcings)

        # Recruitment happens when age >= tau_r (5.99)
        # age=6 recruits production[5] = 2.0
        # age=7 recruits production[6] = 0
        # age=8 recruits production[7] = 0
        # age=9 recruits production[8] = 3.0
        # age=10 recruits production[9] = 0
        # Total: 2.0 + 3.0 = 5.0
        expected_R = 2.0 + 3.0  # production at ages 5 and 8
        assert jnp.allclose(recruitment, expected_R, rtol=0.1)

    def test_recruitment_zero_when_no_production(self):
        """R should be 0 when no production exists."""
        n_ages = 11
        nlat, nlon = 5, 5

        production = jnp.zeros((n_ages, nlat, nlon))

        temperature = jnp.ones((nlat, nlon)) * 15.0
        tau_r = compute_tau_r_forcing.func(
            temperature=temperature, tau_r0=10.38, gamma_tau_r=0.11, T_ref=0.0
        )

        forcings = {"tau_r": tau_r}
        params = {"n_ages": n_ages}

        recruitment = compute_recruitment(production, 1.0, params, forcings)

        assert jnp.allclose(recruitment, 0.0, rtol=1e-5)

    def test_recruitment_output_shape(self):
        """R should have shape (nlat, nlon)."""
        n_ages = 11
        nlat, nlon = 10, 20

        production = jnp.zeros((n_ages, nlat, nlon))

        temperature = jnp.ones((nlat, nlon)) * 15.0
        tau_r = compute_tau_r_forcing.func(
            temperature=temperature, tau_r0=10.38, gamma_tau_r=0.11, T_ref=0.0
        )

        forcings = {"tau_r": tau_r}
        params = {"n_ages": n_ages}

        recruitment = compute_recruitment(production, 1.0, params, forcings)

        assert recruitment.shape == (nlat, nlon)


class TestUpdateBiomass:
    """Tests for update_biomass unit."""

    def test_biomass_increases_with_recruitment(self):
        """Biomass should increase when R > λB."""
        nlat, nlon = 5, 5

        biomass = jnp.ones((nlat, nlon)) * 10.0
        recruitment = jnp.ones((nlat, nlon)) * 5.0

        # Low mortality at T=0°C
        temperature = jnp.ones((nlat, nlon)) * 0.0
        mortality = compute_mortality_forcing.func(
            temperature=temperature, lambda_0=1 / 150, gamma_lambda=0.15, T_ref=0.0
        )

        forcings = {"mortality": mortality}
        params = {}

        biomass_new = update_biomass(biomass, recruitment, 1.0, params, forcings)

        # R=5, λ=1/150, λB ≈ 0.067 → R >> λB → biomass increases
        assert jnp.all(biomass_new > biomass)

    def test_biomass_decreases_without_recruitment(self):
        """Biomass should decrease when R=0."""
        nlat, nlon = 5, 5

        biomass = jnp.ones((nlat, nlon)) * 100.0
        recruitment = jnp.zeros((nlat, nlon))

        temperature = jnp.ones((nlat, nlon)) * 15.0
        mortality = compute_mortality_forcing.func(
            temperature=temperature, lambda_0=1 / 150, gamma_lambda=0.15, T_ref=0.0
        )

        forcings = {"mortality": mortality}
        params = {}

        biomass_new = update_biomass(biomass, recruitment, 1.0, params, forcings)

        # No recruitment → biomass decays
        assert jnp.all(biomass_new < biomass)

    def test_biomass_steady_state(self):
        """At steady state: B ≈ R/λ."""
        nlat, nlon = 5, 5

        # Set R and λ such that R = λB
        recruitment = jnp.ones((nlat, nlon)) * 0.1  # kg/m²/day
        lambda_0 = 1 / 150

        temperature = jnp.ones((nlat, nlon)) * 0.0  # λ = 1/150
        mortality = compute_mortality_forcing.func(
            temperature=temperature, lambda_0=lambda_0, gamma_lambda=0.15, T_ref=0.0
        )

        forcings = {"mortality": mortality}
        params = {}

        # Steady state: B = R/λ = 0.1 / (1/150) = 15
        biomass_ss = recruitment / lambda_0

        # Start from steady state
        biomass = biomass_ss.copy()

        biomass_new = update_biomass(biomass, recruitment, 1.0, params, forcings)

        # Should remain approximately constant
        assert jnp.allclose(biomass_new, biomass, rtol=0.01)

    def test_biomass_remains_positive(self):
        """Biomass should always remain positive (implicit Euler)."""
        nlat, nlon = 5, 5

        biomass = jnp.ones((nlat, nlon)) * 0.01  # Very low biomass
        recruitment = jnp.zeros((nlat, nlon))  # No recruitment

        # High mortality at warm temperature
        temperature = jnp.ones((nlat, nlon)) * 30.0
        mortality = compute_mortality_forcing.func(
            temperature=temperature, lambda_0=1 / 150, gamma_lambda=0.15, T_ref=0.0
        )

        forcings = {"mortality": mortality}
        params = {}

        biomass_new = update_biomass(biomass, recruitment, 1.0, params, forcings)

        # Implicit Euler ensures positivity
        assert jnp.all(biomass_new >= 0.0)

    def test_biomass_output_shape(self):
        """Biomass output should have same shape as input."""
        nlat, nlon = 10, 20

        biomass = jnp.ones((nlat, nlon)) * 50.0
        recruitment = jnp.ones((nlat, nlon)) * 2.0

        temperature = jnp.ones((nlat, nlon)) * 15.0
        mortality = compute_mortality_forcing.func(
            temperature=temperature, lambda_0=1 / 150, gamma_lambda=0.15, T_ref=0.0
        )

        forcings = {"mortality": mortality}
        params = {}

        biomass_new = update_biomass(biomass, recruitment, 1.0, params, forcings)

        assert biomass_new.shape == (nlat, nlon)


class TestMassConservation:
    """Tests for overall mass conservation."""

    def test_conservation_without_transport(self):
        """Total mass should evolve as: dM/dt = NPP_in - λB."""
        n_ages = 11
        nlat, nlon = 5, 5

        # Initial state (start with zero production for clean conservation test)
        biomass = jnp.ones((nlat, nlon)) * 50.0
        production = jnp.zeros((n_ages, nlat, nlon))

        # Forcings (base)
        npp = jnp.ones((nlat, nlon)) * 3.0
        temperature = jnp.ones((nlat, nlon)) * 10.0

        # Compute derived forcings
        tau_r = compute_tau_r_forcing.func(
            temperature=temperature, tau_r0=10.38, gamma_tau_r=0.11, T_ref=0.0
        )

        mortality = compute_mortality_forcing.func(
            temperature=temperature, lambda_0=1 / 150, gamma_lambda=0.15, T_ref=0.0
        )

        # Prepare forcings for each unit
        forcings_prod = {"npp": npp, "tau_r": tau_r}
        forcings_rec = {"tau_r": tau_r}
        forcings_bio = {"mortality": mortality}

        params_prod = {"n_ages": n_ages, "E": 0.1668}
        params_rec = {"n_ages": n_ages}
        params_bio = {}

        # Calculate initial total mass
        mass_init = float(jnp.sum(biomass) + jnp.sum(production))

        # One timestep (correct order: recruit BEFORE aging)
        recruitment = compute_recruitment(production, 1.0, params_rec, forcings_rec)
        production_new = age_production(production, 1.0, params_prod, forcings_prod)
        biomass_new = update_biomass(biomass, recruitment, 1.0, params_bio, forcings_bio)

        # Calculate final total mass
        mass_final = float(jnp.sum(biomass_new) + jnp.sum(production_new))

        # Calculate expected change
        # Mass in = E × NPP × cells × dt = 0.1668 × 3.0 × 25 × 1.0
        mass_in = float(params_prod["E"] * jnp.sum(npp) * 1.0)

        # Mass out = λ × B × cells × dt / (1 + λ × dt)
        # (Implicit Euler reduces effective mortality by factor 1/(1+λdt))
        dt = 1.0
        mass_out = float(jnp.sum(mortality * biomass * dt / (1.0 + mortality * dt)))

        expected_change = mass_in - mass_out
        actual_change = mass_final - mass_init

        # Should match within numerical precision
        assert jnp.abs(actual_change - expected_change) / jnp.abs(expected_change) < 0.01
