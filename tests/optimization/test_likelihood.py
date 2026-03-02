"""Tests for likelihood and log-posterior construction."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from seapopym.optimization.likelihood import GaussianLikelihood, reparameterize_log_posterior
from seapopym.optimization.prior import PriorSet, Uniform


class TestGaussianLikelihood:
    def test_log_likelihood_fixed_sigma(self):
        lik = GaussianLikelihood(sigma=1.0)
        pred = jnp.array([1.0, 2.0, 3.0])
        obs = jnp.array([1.0, 2.0, 3.0])
        ll = lik.log_likelihood(pred, obs)
        # Perfect predictions: ll = -N/2 * log(2*pi*1) - 0 = -N/2 * log(2*pi)
        expected = -3.0 / 2.0 * jnp.log(2.0 * jnp.pi)
        assert float(ll) == pytest.approx(float(expected), abs=1e-5)

    def test_log_likelihood_with_residuals(self):
        lik = GaussianLikelihood(sigma=1.0)
        pred = jnp.array([1.0, 2.0, 3.0])
        obs = jnp.array([1.5, 2.5, 3.5])
        ll = lik.log_likelihood(pred, obs)
        # Residuals = [-0.5, -0.5, -0.5], sum(r^2) = 0.75
        expected = -3.0 / 2.0 * jnp.log(2.0 * jnp.pi) - 0.75 / 2.0
        assert float(ll) == pytest.approx(float(expected), abs=1e-5)

    def test_log_likelihood_free_sigma(self):
        lik = GaussianLikelihood()  # sigma free
        pred = jnp.array([1.0, 2.0, 3.0])
        obs = jnp.array([1.0, 2.0, 3.0])
        sigma = jnp.array(0.5)
        ll = lik.log_likelihood(pred, obs, sigma=sigma)
        expected = -3.0 / 2.0 * jnp.log(2.0 * jnp.pi * 0.25)
        assert float(ll) == pytest.approx(float(expected), abs=1e-5)

    def test_log_likelihood_free_sigma_requires_sigma_arg(self):
        lik = GaussianLikelihood()  # sigma free
        with pytest.raises(ValueError, match="sigma must be provided"):
            lik.log_likelihood(jnp.array([1.0]), jnp.array([1.0]))

    def test_smaller_sigma_sharper_likelihood(self):
        pred = jnp.array([1.0, 2.0])
        obs = jnp.array([1.1, 2.1])
        ll_wide = GaussianLikelihood(sigma=1.0).log_likelihood(pred, obs)
        ll_narrow = GaussianLikelihood(sigma=0.01).log_likelihood(pred, obs)
        # With small residuals, narrow sigma gives higher LL near truth
        # But with nonzero residuals, very narrow sigma penalizes heavily
        # Here residuals are 0.1, sigma=0.01 means (0.1/0.01)^2 = 100 per point
        assert float(ll_wide) > float(ll_narrow)


class TestReparameterizeLogPosterior:
    def test_reparameterize_finite_in_unit_space(self):
        prior_set = PriorSet({"x": Uniform(0.0, 10.0)})

        def log_post(p):
            return -(p["x"] - 5.0) ** 2 + prior_set.log_prob(p)

        log_post_unit = reparameterize_log_posterior(log_post, prior_set)
        # Unit-space params in [0, 1] should give finite log-posterior
        result = log_post_unit({"x": jnp.array(0.5)})
        assert jnp.isfinite(result)

    def test_jacobian_correction(self):
        prior_set = PriorSet({"x": Uniform(0.0, 10.0)})

        def log_post(p):
            return -(p["x"] - 5.0) ** 2 + prior_set.log_prob(p)

        log_post_unit = reparameterize_log_posterior(log_post, prior_set)
        # Unit 0.5 maps to physical 5.0; Jacobian correction = log(10 - 0) = log(10)
        lp_unit = float(log_post_unit({"x": jnp.array(0.5)}))
        lp_phys = float(log_post({"x": jnp.array(5.0)}))
        expected = lp_phys + float(jnp.log(jnp.array(10.0)))
        assert lp_unit == pytest.approx(expected, abs=1e-5)
