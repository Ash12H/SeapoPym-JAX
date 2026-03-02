"""Tests for the high-level Sampler orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock

import jax.numpy as jnp

from seapopym.optimization.likelihood import GaussianLikelihood
from seapopym.optimization.objective import Objective, PreparedObjective
from seapopym.optimization.prior import HalfNormal, PriorSet, Uniform
from seapopym.optimization.runner import CalibrationRunner
from seapopym.optimization.sampler import Sampler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_model(fixed_params: dict, run_outputs: dict):
    """Create a mock model that returns controlled outputs."""
    model = MagicMock()
    model.parameters = fixed_params
    model.run_with_params = MagicMock(return_value=({}, run_outputs))
    return model


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestSamplerInit:
    def test_defaults(self):
        runner = CalibrationRunner.standard()
        priors = PriorSet({"x": Uniform(0.0, 1.0)})
        objectives = [Objective(observations=jnp.zeros(3), transform=lambda o: o["x"][:3])]
        sampler = Sampler(runner, priors, objectives)

        assert isinstance(sampler.likelihood, GaussianLikelihood)
        assert sampler.likelihood.sigma is None  # Free sigma by default
        assert isinstance(sampler.sigma_prior, HalfNormal)
        assert sampler.reparameterize is True

    def test_fixed_sigma(self):
        sampler = Sampler(
            runner=CalibrationRunner.standard(),
            priors=PriorSet({"x": Uniform(0.0, 1.0)}),
            objectives=[Objective(observations=jnp.zeros(1), transform=lambda o: o["x"][:1])],
            likelihood=GaussianLikelihood(sigma=0.5),
        )
        assert sampler.likelihood.sigma == 0.5

    def test_reparameterize_false(self):
        sampler = Sampler(
            runner=CalibrationRunner.standard(),
            priors=PriorSet({"x": Uniform(0.0, 1.0)}),
            objectives=[Objective(observations=jnp.zeros(1), transform=lambda o: o["x"][:1])],
            reparameterize=False,
        )
        assert sampler.reparameterize is False


# ---------------------------------------------------------------------------
# _build_log_posterior
# ---------------------------------------------------------------------------


class TestBuildLogPosterior:
    def test_log_posterior_is_callable(self):
        runner = CalibrationRunner.standard()
        priors = PriorSet({"x": Uniform(0.0, 10.0)})
        sampler = Sampler(
            runner, priors,
            objectives=[Objective(observations=jnp.zeros(1), transform=lambda o: o["x"][:1])],
            likelihood=GaussianLikelihood(sigma=1.0),
        )
        model = MagicMock()

        prepared = [
            PreparedObjective(
                extract_fn=lambda o: o["x"][:1],
                obs_array=jnp.zeros(1),
            )
        ]

        log_post = sampler._build_log_posterior(model, prepared)
        assert callable(log_post)

    def test_log_posterior_fixed_sigma(self):
        """With fixed sigma, log_posterior doesn't need 'sigma' in params."""
        runner = CalibrationRunner.standard()
        priors = PriorSet({"x": Uniform(0.0, 10.0)})
        sampler = Sampler(
            runner, priors,
            objectives=[Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])],
            likelihood=GaussianLikelihood(sigma=1.0),
        )

        # Mock model returns predictions matching obs
        model = _mock_model(
            {"x": jnp.array(5.0)},
            {"out": jnp.array([0.0])},
        )

        prepared = [
            PreparedObjective(
                extract_fn=lambda o: o["out"],
                obs_array=jnp.array([0.0]),
            )
        ]

        log_post = sampler._build_log_posterior(model, prepared)
        lp = log_post({"x": jnp.array(5.0)})

        assert jnp.isfinite(lp)

    def test_log_posterior_free_sigma(self):
        """With free sigma, log_posterior requires 'sigma' in params."""
        runner = CalibrationRunner.standard()
        priors = PriorSet({"x": Uniform(0.0, 10.0)})
        sampler = Sampler(
            runner, priors,
            objectives=[Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])],
            likelihood=GaussianLikelihood(),  # sigma free
        )

        model = _mock_model(
            {"x": jnp.array(5.0)},
            {"out": jnp.array([0.0])},
        )

        prepared = [
            PreparedObjective(
                extract_fn=lambda o: o["out"],
                obs_array=jnp.array([0.0]),
            )
        ]

        log_post = sampler._build_log_posterior(model, prepared)
        lp = log_post({"x": jnp.array(5.0), "sigma": jnp.array(1.0)})

        assert jnp.isfinite(lp)

    def test_better_predictions_higher_posterior(self):
        """Better-matching predictions should yield higher log-posterior."""
        runner = CalibrationRunner.standard()
        priors = PriorSet({"x": Uniform(0.0, 10.0)})
        sampler = Sampler(
            runner, priors,
            objectives=[Objective(observations=jnp.array([5.0]), transform=lambda o: o["out"])],
            likelihood=GaussianLikelihood(sigma=1.0),
        )

        # Good: prediction matches obs
        model_good = _mock_model({"x": jnp.array(5.0)}, {"out": jnp.array([5.0])})
        prepared = [
            PreparedObjective(
                extract_fn=lambda o: o["out"],
                obs_array=jnp.array([5.0]),
            )
        ]
        log_post_good = sampler._build_log_posterior(model_good, prepared)
        lp_good = float(log_post_good({"x": jnp.array(5.0)}))

        # Bad: prediction far from obs
        model_bad = _mock_model({"x": jnp.array(5.0)}, {"out": jnp.array([0.0])})
        log_post_bad = sampler._build_log_posterior(model_bad, prepared)
        lp_bad = float(log_post_bad({"x": jnp.array(5.0)}))

        assert lp_good > lp_bad

    def test_multiple_objectives_summed(self):
        """Log-likelihoods from multiple objectives are summed."""
        runner = CalibrationRunner.standard()
        priors = PriorSet({"x": Uniform(0.0, 10.0)})

        sampler = Sampler(
            runner, priors,
            objectives=[
                Objective(observations=jnp.zeros(1), transform=lambda o: o["a"]),
                Objective(observations=jnp.zeros(1), transform=lambda o: o["b"]),
            ],
            likelihood=GaussianLikelihood(sigma=1.0),
        )

        model = _mock_model(
            {"x": jnp.array(5.0)},
            {"a": jnp.zeros(1), "b": jnp.zeros(1)},
        )

        prepared = [
            PreparedObjective(extract_fn=lambda o: o["a"], obs_array=jnp.zeros(1)),
            PreparedObjective(extract_fn=lambda o: o["b"], obs_array=jnp.zeros(1)),
        ]

        log_post = sampler._build_log_posterior(model, prepared)
        lp = log_post({"x": jnp.array(5.0)})

        assert jnp.isfinite(lp)
