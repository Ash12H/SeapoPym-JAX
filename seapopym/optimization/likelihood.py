"""Log-likelihood and log-posterior construction for Bayesian inference.

Provides tools to build a differentiable log-posterior function from:
- A model run function (params -> predictions)
- Sparse observations
- A PriorSet (from prior.py)
- A likelihood model (Gaussian by default)

The resulting log_posterior(params) is compatible with jax.grad() and NUTS.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp

from seapopym.optimization.prior import HalfNormal, PriorSet
from seapopym.types import Array, Params


@dataclass(frozen=True)
class GaussianLikelihood:
    """Gaussian log-likelihood: errors ~ N(0, sigma^2).

    When sigma is None (default), it is treated as a free parameter
    expected in params["sigma"], estimated jointly by NUTS.

    When sigma is a float, it is fixed.

    Example:
        >>> lik = GaussianLikelihood()            # sigma free
        >>> lik = GaussianLikelihood(sigma=0.1)   # sigma fixed
    """

    sigma: float | None = None

    def log_likelihood(self, predictions: Array, observations: Array, sigma: Array | None = None) -> Array:
        """Compute Gaussian log-likelihood.

        Args:
            predictions: Model predictions at observation locations.
            observations: Observed values.
            sigma: Observation noise std. Required if self.sigma is None.

        Returns:
            Scalar log-likelihood value.
        """
        if sigma is None:
            if self.sigma is None:
                msg = "sigma must be provided as argument when GaussianLikelihood.sigma is None (free parameter)"
                raise ValueError(msg)
            s = jnp.array(self.sigma)
        else:
            s = sigma

        n = jnp.array(predictions.size, dtype=jnp.float32)
        residuals = predictions - observations
        return -n / 2.0 * jnp.log(2.0 * jnp.pi * s**2) - jnp.sum(residuals**2) / (2.0 * s**2)


def make_log_posterior(
    loss_fn: Callable[[Params], Array],
    prior_set: PriorSet,
    likelihood: GaussianLikelihood | None = None,
    sigma_prior: HalfNormal | None = None,
    observations_for_likelihood: tuple[Callable[[Params], Array], Array] | None = None,
) -> Callable[[Params], Array]:
    """Build a differentiable log-posterior function.

    Two modes of operation:

    **Mode 1 — From existing loss_fn (simple):**
    Converts a loss function (e.g., MSE) into a log-posterior by treating
    -loss as a proxy log-likelihood and adding log-prior.
    Use this when you don't need a proper probabilistic model (e.g., MAP estimation).

        >>> log_post = make_log_posterior(loss_fn, prior_set)

    **Mode 2 — Full Gaussian likelihood (for NUTS):**
    Builds a proper log-posterior with Gaussian log-likelihood and priors.
    sigma can be free (estimated by NUTS) or fixed.

        >>> log_post = make_log_posterior(
        ...     loss_fn=None,
        ...     prior_set=prior_set,
        ...     likelihood=GaussianLikelihood(),  # sigma free
        ...     sigma_prior=HalfNormal(scale=1.0),
        ...     observations_for_likelihood=(predict_fn, obs_values),
        ... )

    Args:
        loss_fn: Function mapping params -> scalar loss. Used in Mode 1,
            or ignored if observations_for_likelihood is provided.
        prior_set: PriorSet with priors for model parameters.
        likelihood: Likelihood model. If None, uses -loss_fn as proxy.
        sigma_prior: Prior on sigma (when sigma is free). Ignored if sigma is fixed.
        observations_for_likelihood: Tuple of (predict_fn, obs_values) where
            predict_fn(params) -> predictions at observation locations.
            Required for Mode 2.

    Returns:
        Function mapping params -> scalar log-posterior, suitable for jax.grad().
    """
    if likelihood is not None and observations_for_likelihood is not None:
        predict_fn, obs_values = observations_for_likelihood
        _likelihood = likelihood  # local binding for closure type narrowing

        def log_posterior(params: Params) -> Array:
            # Extract sigma from params if free
            if _likelihood.sigma is None:
                sigma = params.get("sigma")
                if sigma is None:
                    msg = "params must contain 'sigma' when likelihood.sigma is None"
                    raise KeyError(msg)
            else:
                sigma = None

            # Log-likelihood
            predictions = predict_fn(params)
            ll = _likelihood.log_likelihood(predictions, obs_values, sigma=sigma)

            # Log-prior on model parameters
            lp = prior_set.log_prob(params)

            # Log-prior on sigma (if free)
            if _likelihood.sigma is None and sigma_prior is not None:
                lp = lp + sigma_prior.log_prob(params["sigma"])

            return ll + lp

    else:

        def log_posterior(params: Params) -> Array:
            # Mode 1: -loss + log_prior
            return -loss_fn(params) + prior_set.log_prob(params)

    return log_posterior
