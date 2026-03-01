"""Tests for prior distributions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from seapopym.optimization.prior import (
    HalfNormal,
    LogNormal,
    Normal,
    PriorSet,
    TruncatedNormal,
    Uniform,
)


class TestUniform:
    def test_log_prob_inside(self):
        prior = Uniform(0.0, 1.0)
        lp = prior.log_prob(jnp.array(0.5))
        assert jnp.isfinite(lp)
        assert float(lp) == pytest.approx(0.0, abs=1e-5)  # log(1/(1-0)) = 0

    def test_log_prob_outside(self):
        prior = Uniform(0.0, 1.0)
        lp = prior.log_prob(jnp.array(1.5))
        assert float(lp) == -jnp.inf

    def test_bounds(self):
        prior = Uniform(2.0, 5.0)
        assert prior.bounds == (2.0, 5.0)

    def test_sample_in_bounds(self):
        prior = Uniform(0.0, 1.0)
        key = jax.random.key(0)
        s = prior.sample(key)
        assert 0.0 <= float(s) <= 1.0


class TestNormal:
    def test_log_prob_at_mean(self):
        prior = Normal(loc=0.0, scale=1.0)
        lp_mean = prior.log_prob(jnp.array(0.0))
        lp_far = prior.log_prob(jnp.array(5.0))
        assert float(lp_mean) > float(lp_far)

    def test_bounds_finite(self):
        prior = Normal(loc=0.0, scale=1.0)
        low, high = prior.bounds
        assert jnp.isfinite(low)
        assert jnp.isfinite(high)
        assert low < 0.0 < high

    def test_bounds_symmetric(self):
        prior = Normal(loc=0.0, scale=1.0)
        low, high = prior.bounds
        assert float(low) == pytest.approx(-float(high), abs=1e-5)

    def test_sample(self):
        prior = Normal(loc=0.0, scale=1.0)
        key = jax.random.key(42)
        s = prior.sample(key)
        assert jnp.isfinite(s)


class TestLogNormal:
    def test_log_prob_positive(self):
        prior = LogNormal(mu=0.0, sigma=0.5)
        lp = prior.log_prob(jnp.array(1.0))
        assert jnp.isfinite(lp)

    def test_log_prob_at_mode(self):
        prior = LogNormal(mu=0.0, sigma=0.5)
        # Mode of lognormal = exp(mu - sigma^2)
        mode = jnp.exp(0.0 - 0.5**2)
        lp_mode = prior.log_prob(mode)
        lp_far = prior.log_prob(jnp.array(10.0))
        assert float(lp_mode) > float(lp_far)

    def test_bounds_positive(self):
        prior = LogNormal(mu=0.0, sigma=0.5)
        low, high = prior.bounds
        assert low > 0.0
        assert high > low

    def test_sample_positive(self):
        prior = LogNormal(mu=0.0, sigma=0.5)
        key = jax.random.key(0)
        s = prior.sample(key)
        assert float(s) > 0.0


class TestHalfNormal:
    def test_log_prob_positive(self):
        prior = HalfNormal(scale=1.0)
        lp = prior.log_prob(jnp.array(0.5))
        assert jnp.isfinite(lp)

    def test_log_prob_negative_is_neginf(self):
        prior = HalfNormal(scale=1.0)
        lp = prior.log_prob(jnp.array(-0.5))
        assert float(lp) == -jnp.inf

    def test_bounds(self):
        prior = HalfNormal(scale=1.0)
        low, high = prior.bounds
        assert low == 0.0
        assert high > 0.0

    def test_sample_positive(self):
        prior = HalfNormal(scale=1.0)
        key = jax.random.key(0)
        s = prior.sample(key)
        assert float(s) >= 0.0


class TestTruncatedNormal:
    def test_log_prob_inside(self):
        prior = TruncatedNormal(loc=0.5, scale=0.1, low=0.0, high=1.0)
        lp = prior.log_prob(jnp.array(0.5))
        assert jnp.isfinite(lp)

    def test_log_prob_outside(self):
        prior = TruncatedNormal(loc=0.5, scale=0.1, low=0.0, high=1.0)
        lp = prior.log_prob(jnp.array(1.5))
        assert float(lp) == -jnp.inf

    def test_bounds(self):
        prior = TruncatedNormal(loc=0.5, scale=0.1, low=0.0, high=1.0)
        assert prior.bounds == (0.0, 1.0)

    def test_sample_in_bounds(self):
        prior = TruncatedNormal(loc=0.5, scale=0.1, low=0.0, high=1.0)
        key = jax.random.key(0)
        s = prior.sample(key)
        assert 0.0 <= float(s) <= 1.0


class TestPriorSet:
    def setup_method(self):
        self.prior_set = PriorSet({
            "alpha": Uniform(0.0, 1.0),
            "beta": Normal(loc=0.0, scale=1.0),
        })
        self.params = {
            "alpha": jnp.array(0.5),
            "beta": jnp.array(0.0),
        }

    def test_log_prob_is_sum(self):
        joint_lp = self.prior_set.log_prob(self.params)
        individual_sum = (
            Uniform(0.0, 1.0).log_prob(jnp.array(0.5))
            + Normal(loc=0.0, scale=1.0).log_prob(jnp.array(0.0))
        )
        assert float(joint_lp) == pytest.approx(float(individual_sum), abs=1e-5)

    def test_log_prob_ignores_extra_params(self):
        params_extra = {**self.params, "gamma": jnp.array(99.0)}
        lp1 = self.prior_set.log_prob(self.params)
        lp2 = self.prior_set.log_prob(params_extra)
        assert float(lp1) == pytest.approx(float(lp2), abs=1e-5)

    def test_get_bounds(self):
        bounds = self.prior_set.get_bounds()
        assert "alpha" in bounds
        assert "beta" in bounds
        assert bounds["alpha"] == (0.0, 1.0)

    def test_sample(self):
        key = jax.random.key(0)
        samples = self.prior_set.sample(key)
        assert "alpha" in samples
        assert "beta" in samples
        assert 0.0 <= float(samples["alpha"]) <= 1.0

    def test_jit_compatible(self):
        log_prob_jit = jax.jit(self.prior_set.log_prob)
        lp = log_prob_jit(self.params)
        assert jnp.isfinite(lp)

    def test_grad_compatible(self):
        grad_fn = jax.grad(self.prior_set.log_prob)
        grads = grad_fn(self.params)
        assert "alpha" in grads
        assert "beta" in grads
        assert jnp.isfinite(grads["beta"])

    def test_to_unit_from_unit_roundtrip(self):
        """from_unit(to_unit(x)) should return x."""
        ps = PriorSet({"x": Uniform(2.0, 8.0), "y": Normal(loc=0.0, scale=1.0)})
        params = {"x": jnp.array(5.0), "y": jnp.array(0.0)}
        roundtrip = ps.from_unit(ps.to_unit(params))
        assert float(roundtrip["x"]) == pytest.approx(5.0, abs=1e-5)
        assert float(roundtrip["y"]) == pytest.approx(0.0, abs=1e-5)

    def test_to_unit_at_bounds(self):
        """to_unit at bounds should give 0 and 1."""
        ps = PriorSet({"x": Uniform(2.0, 8.0)})
        assert float(ps.to_unit({"x": jnp.array(2.0)})["x"]) == pytest.approx(0.0, abs=1e-6)
        assert float(ps.to_unit({"x": jnp.array(8.0)})["x"]) == pytest.approx(1.0, abs=1e-6)

    def test_log_det_jacobian_known_value(self):
        """For Uniform(0, 10), log_det_jacobian should be log(10)."""
        ps = PriorSet({"x": Uniform(0.0, 10.0)})
        ldj = ps.log_det_jacobian()
        assert float(ldj) == pytest.approx(float(jnp.log(10.0)), abs=1e-5)

    def test_log_prob_missing_param(self):
        """log_prob should ignore priors whose key is absent from params."""
        ps = PriorSet({"x": Uniform(0.0, 1.0), "y": Normal(loc=0.0, scale=1.0)})
        # Only pass "x", "y" is missing
        lp_partial = ps.log_prob({"x": jnp.array(0.5)})
        lp_x_only = Uniform(0.0, 1.0).log_prob(jnp.array(0.5))
        assert float(lp_partial) == pytest.approx(float(lp_x_only), abs=1e-5)
