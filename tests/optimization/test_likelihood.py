"""Tests for likelihood and log-posterior construction."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from seapopym.optimization.likelihood import GaussianLikelihood, make_log_posterior
from seapopym.optimization.prior import HalfNormal, Normal, PriorSet, Uniform


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


class TestMakeLogPosteriorMode1:
    """Mode 1: -loss_fn + log_prior (MAP proxy)."""

    def setup_method(self):
        self.prior_set = PriorSet({"x": Uniform(0.0, 10.0)})

    def test_basic(self):
        def loss_fn(params):
            return (params["x"] - 3.0) ** 2

        log_post = make_log_posterior(loss_fn, self.prior_set)
        # At x=3: log_post = -0 + log_prior(3) = Uniform.log_prob(3)
        lp = log_post({"x": jnp.array(3.0)})
        expected = Uniform(0.0, 10.0).log_prob(jnp.array(3.0))
        assert float(lp) == pytest.approx(float(expected), abs=1e-5)

    def test_jit_compatible(self):
        def loss_fn(params):
            return (params["x"] - 3.0) ** 2

        log_post = jax.jit(make_log_posterior(loss_fn, self.prior_set))
        lp = log_post({"x": jnp.array(3.0)})
        assert jnp.isfinite(lp)

    def test_grad_compatible(self):
        def loss_fn(params):
            return (params["x"] - 3.0) ** 2

        log_post = make_log_posterior(loss_fn, self.prior_set)
        grad_fn = jax.grad(log_post)
        grads = grad_fn({"x": jnp.array(3.0)})
        assert "x" in grads
        # At x=3 (minimum of loss), grad of -loss is 0, grad of uniform log_prob is 0
        assert float(grads["x"]) == pytest.approx(0.0, abs=1e-5)


class TestMakeLogPosteriorMode2:
    """Mode 2: Full Gaussian likelihood with sigma."""

    def setup_method(self):
        self.prior_set = PriorSet({"a": Normal(loc=2.0, scale=1.0)})
        self.sigma_prior = HalfNormal(scale=1.0)
        self.likelihood = GaussianLikelihood()  # sigma free
        self.obs = jnp.array([1.0, 2.0, 3.0])

    def _make_predict_fn(self):
        def predict_fn(params):
            # Simple model: predictions = a * [1, 2, 3] (scaling)
            return params["a"] * jnp.array([0.5, 1.0, 1.5])
        return predict_fn

    def test_basic(self):
        predict_fn = self._make_predict_fn()
        log_post = make_log_posterior(
            loss_fn=None,
            prior_set=self.prior_set,
            likelihood=self.likelihood,
            sigma_prior=self.sigma_prior,
            observations_for_likelihood=(predict_fn, self.obs),
        )
        params = {"a": jnp.array(2.0), "sigma": jnp.array(0.5)}
        lp = log_post(params)
        assert jnp.isfinite(lp)

    def test_jit_compatible(self):
        predict_fn = self._make_predict_fn()
        log_post = jax.jit(make_log_posterior(
            loss_fn=None,
            prior_set=self.prior_set,
            likelihood=self.likelihood,
            sigma_prior=self.sigma_prior,
            observations_for_likelihood=(predict_fn, self.obs),
        ))
        params = {"a": jnp.array(2.0), "sigma": jnp.array(0.5)}
        lp = log_post(params)
        assert jnp.isfinite(lp)

    def test_grad_compatible(self):
        predict_fn = self._make_predict_fn()
        log_post = make_log_posterior(
            loss_fn=None,
            prior_set=self.prior_set,
            likelihood=self.likelihood,
            sigma_prior=self.sigma_prior,
            observations_for_likelihood=(predict_fn, self.obs),
        )
        grad_fn = jax.grad(log_post)
        params = {"a": jnp.array(2.0), "sigma": jnp.array(0.5)}
        grads = grad_fn(params)
        assert "a" in grads
        assert "sigma" in grads
        assert jnp.isfinite(grads["a"])
        assert jnp.isfinite(grads["sigma"])

    def test_fixed_sigma(self):
        predict_fn = self._make_predict_fn()
        likelihood_fixed = GaussianLikelihood(sigma=0.5)
        log_post = make_log_posterior(
            loss_fn=None,
            prior_set=self.prior_set,
            likelihood=likelihood_fixed,
            observations_for_likelihood=(predict_fn, self.obs),
        )
        # No sigma in params needed
        params = {"a": jnp.array(2.0)}
        lp = log_post(params)
        assert jnp.isfinite(lp)

    def test_better_params_higher_posterior(self):
        predict_fn = self._make_predict_fn()
        log_post = make_log_posterior(
            loss_fn=None,
            prior_set=self.prior_set,
            likelihood=self.likelihood,
            sigma_prior=self.sigma_prior,
            observations_for_likelihood=(predict_fn, self.obs),
        )
        # True a=2: predictions = [1, 2, 3] = obs
        good_params = {"a": jnp.array(2.0), "sigma": jnp.array(0.5)}
        bad_params = {"a": jnp.array(5.0), "sigma": jnp.array(0.5)}
        assert float(log_post(good_params)) > float(log_post(bad_params))
