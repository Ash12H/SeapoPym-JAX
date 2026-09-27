"""Tests for LMTL biological functions."""

import jax
import jax.numpy as jnp
import pytest

from seapopym.functions.lmtl import (
    _cohort_durations,
    aging_flow,
    day_length,
    gillooly_temperature,
    layer_weighted_mean,
    mortality_tendency,
    npp_injection,
    recruitment_age,
    recruitment_flow,
    threshold_temperature,
)

# =============================================================================
# HELPERS
# =============================================================================


class TestCohortDurations:
    """Tests for _cohort_durations helper."""

    def test_uniform_spacing(self):
        """Uniform age grid should give uniform durations."""
        ages = jnp.array([0.0, 1.0, 2.0, 3.0])
        d = _cohort_durations(ages)
        assert jnp.allclose(d, jnp.ones(4))

    def test_last_cohort_reuses_penultimate(self):
        """Last (plus-group) duration should equal the penultimate one."""
        ages = jnp.array([0.0, 1.0, 3.0])
        d = _cohort_durations(ages)
        assert d[-1] == d[-2]


# =============================================================================
# ENVIRONMENT FUNCTIONS
# =============================================================================


class TestDayLength:
    """Tests for day_length function."""

    def test_equinox_gives_half(self):
        """At equinox (day ~80), day fraction should be ~0.5 at equator."""
        frac = day_length(jnp.array(0.0), jnp.array(80.0))
        assert jnp.isclose(frac, 0.5, atol=0.02)

    def test_polar_summer_near_one(self):
        """At high latitude in summer, day fraction should approach 1."""
        frac = day_length(jnp.array(80.0), jnp.array(172.0))
        assert frac > 0.9

    def test_polar_winter_near_zero(self):
        """At high latitude in winter, day fraction should approach 0."""
        frac = day_length(jnp.array(80.0), jnp.array(355.0))
        assert frac < 0.1

    def test_southern_hemisphere_opposite(self):
        """Southern hemisphere should have opposite day length to northern."""
        north = day_length(jnp.array(60.0), jnp.array(172.0))
        south = day_length(jnp.array(-60.0), jnp.array(172.0))
        assert jnp.isclose(north + south, 1.0, atol=0.02)


class TestLayerWeightedMean:
    """Tests for layer_weighted_mean function."""

    def test_full_night(self):
        """day_length=0 should return night layer value."""
        forcing = jnp.array([10.0, 20.0, 30.0])
        result = layer_weighted_mean(forcing, jnp.array(0.0), 0, 2)
        assert jnp.isclose(result, 30.0)

    def test_full_day(self):
        """day_length=1 should return day layer value."""
        forcing = jnp.array([10.0, 20.0, 30.0])
        result = layer_weighted_mean(forcing, jnp.array(1.0), 0, 2)
        assert jnp.isclose(result, 10.0)

    def test_half_day(self):
        """day_length=0.5 should return equal-weight mean of day and night layers."""
        forcing = jnp.array([10.0, 20.0, 30.0])
        result = layer_weighted_mean(forcing, jnp.array(0.5), 0, 2)
        assert jnp.isclose(result, 20.0)


# =============================================================================
# TEMPERATURE FUNCTIONS
# =============================================================================


class TestThresholdTemperature:
    """Tests for threshold_temperature function."""

    def test_above_threshold_passthrough(self):
        """Temperatures above threshold should pass through unchanged."""
        result = threshold_temperature(jnp.array(15.0), 0.0)
        assert jnp.isclose(result, 15.0)

    def test_below_threshold_clamped(self):
        """Temperatures below threshold should be clamped to the threshold."""
        result = threshold_temperature(jnp.array(-5.0), 0.0)
        assert jnp.isclose(result, 0.0)

    def test_at_threshold(self):
        """Temperature exactly at threshold should return threshold."""
        result = threshold_temperature(jnp.array(2.0), 2.0)
        assert jnp.isclose(result, 2.0)


class TestGilloolyTemperature:
    """Tests for gillooly_temperature function."""

    def test_zero_returns_zero(self):
        """T=0 should give 0 normalized temperature."""
        result = gillooly_temperature(jnp.array(0.0))
        assert jnp.isclose(result, 0.0)

    def test_known_value(self):
        """T=15 should give 15/(1+15/273) = 15*273/288 ≈ 14.22."""
        result = gillooly_temperature(jnp.array(15.0))
        expected = 15.0 * 273.0 / 288.0
        assert jnp.isclose(result, expected, atol=1e-4)

    def test_monotonically_increasing(self):
        """Gillooly transform should be monotonically increasing for T > 0."""
        temps = jnp.array([5.0, 10.0, 15.0, 20.0, 25.0])
        results = gillooly_temperature(temps)
        assert jnp.all(jnp.diff(results) > 0)


# =============================================================================
# DERIVED QUANTITIES
# =============================================================================


class TestRecruitmentAge:
    """Tests for recruitment_age function."""

    def test_at_reference_temp(self):
        """At reference temperature, recruitment age should equal tau_r_0."""
        result = recruitment_age(jnp.array(10.0), tau_r_0=100.0, gamma=0.1, t_ref=10.0)
        assert jnp.isclose(result, 100.0)

    def test_increases_when_temp_decreases(self):
        """Colder temperatures should give longer recruitment age."""
        warm = recruitment_age(jnp.array(20.0), tau_r_0=100.0, gamma=0.1, t_ref=10.0)
        cold = recruitment_age(jnp.array(5.0), tau_r_0=100.0, gamma=0.1, t_ref=10.0)
        assert cold > warm

    def test_always_positive(self):
        """Recruitment age should always be positive."""
        temps = jnp.array([-5.0, 0.0, 10.0, 30.0])
        results = recruitment_age(temps, tau_r_0=100.0, gamma=0.1, t_ref=10.0)
        assert jnp.all(results > 0)


# =============================================================================
# MORTALITY
# =============================================================================


class TestMortalityTendency:
    """Tests for mortality_tendency function."""

    def test_sign_is_negative(self):
        """Mortality tendency should be negative (loss)."""
        result = mortality_tendency(jnp.array(10.0), jnp.array(15.0), lambda_0=1e-6, gamma=0.1, t_ref=10.0)
        assert result < 0

    def test_increases_with_temperature(self):
        """Higher temperature should give greater mortality magnitude."""
        cold = mortality_tendency(jnp.array(10.0), jnp.array(5.0), lambda_0=1e-6, gamma=0.1, t_ref=10.0)
        warm = mortality_tendency(jnp.array(10.0), jnp.array(20.0), lambda_0=1e-6, gamma=0.1, t_ref=10.0)
        # More negative = greater loss
        assert warm < cold

    def test_zero_biomass_zero_mortality(self):
        """Zero biomass should give zero mortality tendency."""
        result = mortality_tendency(jnp.array(0.0), jnp.array(15.0), lambda_0=1e-6, gamma=0.1, t_ref=10.0)
        assert jnp.isclose(result, 0.0)


# =============================================================================
# PRODUCTION DYNAMICS
# =============================================================================


class TestNppInjection:
    """Tests for npp_injection function."""

    def test_only_first_cohort_receives(self):
        """Only cohort 0 should receive the NPP flux."""
        production = jnp.zeros(5)
        result = npp_injection(jnp.array(10.0), 0.5, production)
        assert jnp.isclose(result[0], 5.0)
        assert jnp.allclose(result[1:], 0.0)

    def test_zero_npp_gives_zero(self):
        """Zero NPP should give zero tendency everywhere."""
        production = jnp.ones(5)
        result = npp_injection(jnp.array(0.0), 0.5, production)
        assert jnp.allclose(result, 0.0)

    def test_shape_preserved(self):
        """Output should have the same shape as production."""
        production = jnp.zeros(8)
        result = npp_injection(jnp.array(10.0), 0.5, production)
        assert result.shape == production.shape


class TestAgingFlow:
    """Tests for aging_flow function."""

    @pytest.fixture
    def cohort_setup(self):
        """Standard cohort configuration for testing."""
        cohort_ages = jnp.array([0.0, 86400.0, 172800.0, 259200.0, 345600.0])  # 5 cohorts, 1-day spacing
        production = jnp.array([10.0, 8.0, 6.0, 4.0, 2.0])
        rec_age = jnp.array(1e9)  # Very large -> no recruitment
        return production, cohort_ages, rec_age

    def test_mass_conservation(self, cohort_setup):
        """Total aging tendency should sum to approximately zero (mass conserved)."""
        production, cohort_ages, rec_age = cohort_setup
        result = aging_flow(production, cohort_ages, rec_age)
        assert jnp.isclose(jnp.sum(result), 0.0, atol=1e-6)

    def test_last_cohort_no_outflow(self, cohort_setup):
        """Last cohort should have no outflow (plus-group accumulates)."""
        production, cohort_ages, rec_age = cohort_setup
        result = aging_flow(production, cohort_ages, rec_age)
        # Last cohort only gains from prev, never loses its own aging outflow:
        # it recruits everything it holds, so aging_outflow[-1] = 0
        assert result[-1] >= 0

    def test_shape_preserved(self, cohort_setup):
        """Output shape should match production shape."""
        production, cohort_ages, rec_age = cohort_setup
        result = aging_flow(production, cohort_ages, rec_age)
        assert result.shape == production.shape


class TestRecruitmentFlow:
    """Tests for recruitment_flow function."""

    @pytest.fixture
    def cohort_setup(self):
        """Standard cohort configuration for testing."""
        cohort_ages = jnp.array([0.0, 86400.0, 172800.0, 259200.0, 345600.0])
        production = jnp.array([10.0, 8.0, 6.0, 4.0, 2.0])
        rec_age = jnp.array(172800.0)  # Recruitment at cohort 2 boundary
        return production, cohort_ages, rec_age

    def test_prod_loss_negative(self, cohort_setup):
        """Production loss from recruitment should be negative."""
        production, cohort_ages, rec_age = cohort_setup
        prod_loss, _ = recruitment_flow(production, cohort_ages, rec_age)
        assert jnp.all(prod_loss <= 0)

    def test_biomass_gain_positive(self, cohort_setup):
        """Biomass gain from recruitment should be positive."""
        production, cohort_ages, rec_age = cohort_setup
        _, biomass_gain = recruitment_flow(production, cohort_ages, rec_age)
        assert biomass_gain > 0

    def test_shapes(self, cohort_setup):
        """prod_loss should be (C,) and biomass_gain should be scalar."""
        production, cohort_ages, rec_age = cohort_setup
        prod_loss, biomass_gain = recruitment_flow(production, cohort_ages, rec_age)
        assert prod_loss.shape == production.shape
        assert biomass_gain.shape == ()

    def test_flux_balance(self, cohort_setup):
        """Sum of production losses should equal biomass gain (mass conservation)."""
        production, cohort_ages, rec_age = cohort_setup
        prod_loss, biomass_gain = recruitment_flow(production, cohort_ages, rec_age)
        assert jnp.isclose(-jnp.sum(prod_loss), biomass_gain, atol=1e-6)


# =============================================================================
# JAX DIFFERENTIABILITY
# =============================================================================


class TestLmtlDifferentiability:
    """Tests for JAX differentiability of LMTL functions."""

    def test_recruitment_flow_grad_wrt_rec_age(self):
        """recruitment_flow should be differentiable w.r.t. rec_age."""
        cohort_ages = jnp.array([0.0, 86400.0, 172800.0, 259200.0])
        production = jnp.array([10.0, 8.0, 6.0, 4.0])

        def loss_fn(rec_age):
            _, biomass_gain = recruitment_flow(production, cohort_ages, rec_age)
            return biomass_gain

        # Inside a cohort: the recruited share is linear there, and kinked at the exits.
        grad = jax.grad(loss_fn)(jnp.array(129600.0))
        assert not jnp.isnan(grad)
        # Higher rec_age -> less recruitment -> negative gradient
        assert grad < 0

    def test_aging_flow_grad_wrt_production(self):
        """aging_flow should be differentiable w.r.t. production."""
        cohort_ages = jnp.array([0.0, 86400.0, 172800.0, 259200.0])
        rec_age = jnp.array(1e9)

        def loss_fn(production):
            return jnp.sum(aging_flow(production, cohort_ages, rec_age) ** 2)

        grad = jax.grad(loss_fn)(jnp.array([10.0, 8.0, 6.0, 4.0]))
        assert grad.shape == (4,)
        assert not jnp.any(jnp.isnan(grad))


# =============================================================================
# AGEING ALONG THE AGE AXIS
# =============================================================================


def _pulse(cohort_ages, rec_age, days=None, dt_days=0.01):
    """Age a unit of production from cohort 0 without mortality; return times [d] and biomass."""
    days = days or float(cohort_ages[-1] / 86400.0) * 2 + 10
    dt = dt_days * 86400.0
    steps = int(days / dt_days)

    def step(production, _):
        aging = aging_flow(production, cohort_ages, rec_age)
        prod_loss, gain = recruitment_flow(production, cohort_ages, rec_age)
        production = production + dt * (aging + prod_loss)
        return production, (gain * dt, jnp.min(production))

    start = jnp.zeros(cohort_ages.shape[0]).at[0].set(1.0)
    _, (recruited, lowest) = jax.lax.scan(step, start, None, length=steps)
    times = (jnp.arange(steps) + 1) * dt_days
    return times, recruited, lowest


class TestAgeingAlongTheAgeAxis:
    """Tests of the MUSCL ageing and of the recruitment at the exact age."""

    @pytest.mark.parametrize("rec_days", [4.5, 7.5, 12.25])
    def test_mean_delay_is_the_recruitment_age(self, rec_days):
        """A pulse is recruited, on average, at the recruitment age, even inside a cohort."""
        cohort_ages = jnp.arange(0.0, 21.0) * 86400.0
        times, recruited, _ = _pulse(cohort_ages, jnp.array(rec_days * 86400.0))
        assert jnp.isclose(jnp.sum(recruited), 1.0, atol=1e-4)
        mean = jnp.sum(times * recruited) / jnp.sum(recruited)
        assert abs(float(mean) - rec_days) < 0.1

    def test_muscl_spreads_less_than_first_order(self):
        """Ten cohorts before recruitment spread the delay far less than a chain of compartments."""
        cohort_ages = jnp.arange(0.0, 21.0) * 86400.0
        times, recruited, _ = _pulse(cohort_ages, jnp.array(10.0 * 86400.0))
        mean = jnp.sum(times * recruited) / jnp.sum(recruited)
        spread = jnp.sqrt(jnp.sum((times - mean) ** 2 * recruited) / jnp.sum(recruited))
        # A chain of ten compartments of one day gives 1 / sqrt(10) = 0.32; MUSCL gives about 0.20.
        assert float(spread / mean) < 0.25

    def test_positive_with_growing_widths(self):
        """Cohorts that widen with age keep the production positive."""
        widths = 0.5 * 1.3 ** jnp.arange(12)
        cohort_ages = jnp.concatenate([jnp.zeros(1), jnp.cumsum(widths)]) * 86400.0
        _, recruited, lowest = _pulse(cohort_ages, jnp.array(20.0 * 86400.0), days=120.0)
        assert float(jnp.min(lowest)) > -1e-9
        assert jnp.isclose(jnp.sum(recruited), 1.0, atol=1e-3)

    def test_mass_is_conserved_with_uneven_widths(self):
        """Ageing moves production without loss, and recruitment gives the biomass what it takes."""
        cohort_ages = jnp.array([0.0, 1.0, 2.5, 4.5, 7.0, 10.0]) * 86400.0
        production = jnp.array([3.0, 5.0, 2.0, 4.0, 1.0, 0.5])
        rec_age = jnp.array(5.2 * 86400.0)
        aging = aging_flow(production, cohort_ages, rec_age)
        prod_loss, gain = recruitment_flow(production, cohort_ages, rec_age)
        assert jnp.isclose(jnp.sum(aging), 0.0, atol=1e-9)
        assert jnp.isclose(-jnp.sum(prod_loss), gain)

    def test_last_cohort_recruits_everything(self):
        """Production past the ladder is recruited rather than kept."""
        cohort_ages = jnp.arange(0.0, 5.0) * 86400.0
        production = jnp.array([0.0, 0.0, 0.0, 0.0, 2.0])
        prod_loss, gain = recruitment_flow(production, cohort_ages, jnp.array(1e9))
        assert jnp.isclose(prod_loss[-1], -2.0 / 86400.0)
        assert jnp.isclose(gain, 2.0 / 86400.0)

    def test_recruiting_cohorts_give_their_first_order_outflow(self):
        """Where a cohort recruits, what leaves it is its own density, not a reconstruction."""
        cohort_ages = jnp.arange(0.0, 8.0) * 86400.0
        production = jnp.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        rec_age = jnp.array(3.4 * 86400.0)
        prod_loss, _ = recruitment_flow(production, cohort_ages, rec_age)
        # Exits at 3 and 4 days: 60 % of cohort 2's outflow, all of cohort 3's.
        assert jnp.isclose(prod_loss[2], -0.6 * 6.0 / 86400.0)
        assert jnp.isclose(prod_loss[3], -5.0 / 86400.0)
        assert jnp.all(prod_loss[:2] == 0.0)
