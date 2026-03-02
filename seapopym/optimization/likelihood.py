"""Gaussian likelihood and log-posterior reparameterization for Bayesian inference.

Provides:
- :class:`GaussianLikelihood` — Gaussian log-likelihood with fixed or free sigma
- :func:`reparameterize_log_posterior` — unit-space transform for NUTS sampling
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp

from seapopym.optimization.prior import PriorSet
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


def reparameterize_log_posterior(
    log_posterior_fn: Callable[[Params], Array],
    prior_set: PriorSet,
) -> Callable[[Params], Array]:
    """Wrap a log-posterior to operate in unit space [0, 1].

    NUTS works best when all parameters have similar scales. This function
    reparameterizes the log-posterior so that the sampler works in [0, 1]^d
    (normalized via prior bounds), while the model receives physical values.

    The Jacobian correction (log|det J|) is included so that the sampling
    distribution is correct.

    Args:
        log_posterior_fn: Log-posterior in physical parameter space.
        prior_set: PriorSet providing bounds for normalization.

    Returns:
        Log-posterior in unit space, suitable for run_nuts().

    Example:
        >>> log_post_unit = reparameterize_log_posterior(log_post, prior_set)
        >>> init_unit = prior_set.to_unit(initial_params)
        >>> result = run_nuts(log_post_unit, init_unit, ...)
        >>> samples_phys = prior_set.from_unit(result.samples)
    """
    # Precompute bounds eagerly (outside JIT) to avoid ConcretizationTypeError
    # when priors like HalfNormal compute bounds via jstats.norm.ppf + float().
    bounds = prior_set.get_bounds()
    log_det_jac = prior_set.log_det_jacobian()

    def log_posterior_unit(params_unit: Params) -> Array:
        # Inline from_unit using precomputed bounds (JIT-safe)
        params_phys = {}
        for name in bounds:
            if name in params_unit:
                low, high = bounds[name]
                params_phys[name] = params_unit[name] * (high - low) + low
        return log_posterior_fn(params_phys) + log_det_jac

    return log_posterior_unit
