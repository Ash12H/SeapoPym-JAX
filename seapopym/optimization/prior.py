"""Prior distributions for Bayesian parameter estimation.

Provides a unified system for defining parameter priors that serves both:
- CMA-ES / gradient optimization: bounds extraction, optional MAP regularization
- NUTS (Bayesian inference): log_prob for log-posterior construction

All distributions are JAX-compatible (jit, grad).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats

from seapopym.types import Array, Params


class Prior(Protocol):
    """Protocol for prior distributions."""

    def log_prob(self, value: Array) -> Array:
        """Log-probability density at value."""
        ...

    @property
    def bounds(self) -> tuple[float, float]:
        """Parameter bounds (finite, suitable for CMA-ES)."""
        ...

    def sample(self, key: Array) -> Array:
        """Draw a random sample."""
        ...


@dataclass(frozen=True)
class Uniform:
    """Uniform prior on [low, high].

    Example:
        >>> prior = Uniform(0.0, 1.0)
        >>> prior.log_prob(jnp.array(0.5))
        Array(0., dtype=float32)
        >>> prior.bounds
        (0.0, 1.0)
    """

    low: float
    high: float

    def log_prob(self, value: Array) -> Array:
        return jstats.uniform.logpdf(value, loc=self.low, scale=self.high - self.low)

    @property
    def bounds(self) -> tuple[float, float]:
        return (self.low, self.high)

    def sample(self, key: Array) -> Array:
        return jax.random.uniform(key, minval=self.low, maxval=self.high)


@dataclass(frozen=True)
class Normal:
    """Normal (Gaussian) prior.

    Bounds are derived from quantiles for use with bounded optimizers.

    Example:
        >>> prior = Normal(loc=0.1, scale=0.02)
        >>> prior.bounds  # ~(0.038, 0.162) at 0.1% quantiles
    """

    loc: float
    scale: float
    bounds_quantile: float = 0.001

    def log_prob(self, value: Array) -> Array:
        return jstats.norm.logpdf(value, loc=self.loc, scale=self.scale)

    @property
    def bounds(self) -> tuple[float, float]:
        z = float(jstats.norm.ppf(self.bounds_quantile))
        return (self.loc + z * self.scale, self.loc - z * self.scale)

    def sample(self, key: Array) -> Array:
        return self.loc + self.scale * jax.random.normal(key)


@dataclass(frozen=True)
class LogNormal:
    """Log-normal prior (value > 0, log(value) ~ Normal).

    Args:
        mu: Mean of the underlying normal distribution (in log-space).
        sigma: Std of the underlying normal distribution (in log-space).

    Example:
        >>> prior = LogNormal(mu=0.0, sigma=0.5)
        >>> prior.bounds  # positive bounds from quantiles
    """

    mu: float
    sigma: float
    bounds_quantile: float = 0.001

    def log_prob(self, value: Array) -> Array:
        # log p(x) = -log(x) - log(sigma) - 0.5*((log(x)-mu)/sigma)^2 - 0.5*log(2*pi)
        log_value = jnp.log(value)
        return jstats.norm.logpdf(log_value, loc=self.mu, scale=self.sigma) - log_value

    @property
    def bounds(self) -> tuple[float, float]:
        z = float(jstats.norm.ppf(self.bounds_quantile))
        return (float(jnp.exp(self.mu + z * self.sigma)), float(jnp.exp(self.mu - z * self.sigma)))

    def sample(self, key: Array) -> Array:
        return jnp.exp(self.mu + self.sigma * jax.random.normal(key))


@dataclass(frozen=True)
class HalfNormal:
    """Half-normal prior (value >= 0).

    A normal distribution folded at zero: p(x) = 2 * Normal(0, scale).pdf(x) for x >= 0.

    Example:
        >>> prior = HalfNormal(scale=1.0)
        >>> prior.bounds  # (0.0, quantile)
    """

    scale: float
    bounds_quantile: float = 0.001

    def log_prob(self, value: Array) -> Array:
        # log(2) + Normal(0, scale).logpdf(value), valid for value >= 0
        # For value < 0, return -inf
        lp = jnp.log(2.0) + jstats.norm.logpdf(value, loc=0.0, scale=self.scale)
        return jnp.where(value >= 0.0, lp, -jnp.inf)

    @property
    def bounds(self) -> tuple[float, float]:
        # Upper bound from quantile of the half-normal
        # P(X < x) = 2*Phi(x/scale) - 1, so x = scale * Phi_inv((1+q)/2)
        upper_q = (1.0 + (1.0 - self.bounds_quantile)) / 2.0
        z = float(jstats.norm.ppf(upper_q))
        return (0.0, z * self.scale)

    def sample(self, key: Array) -> Array:
        return jnp.abs(self.scale * jax.random.normal(key))


@dataclass(frozen=True)
class TruncatedNormal:
    """Truncated normal prior on [low, high].

    Example:
        >>> prior = TruncatedNormal(loc=0.5, scale=0.1, low=0.0, high=1.0)
        >>> prior.bounds
        (0.0, 1.0)
    """

    loc: float
    scale: float
    low: float
    high: float

    def log_prob(self, value: Array) -> Array:
        # Convert to standard truncnorm parameterization: a, b in standard units
        a = (self.low - self.loc) / self.scale
        b = (self.high - self.loc) / self.scale
        return jstats.truncnorm.logpdf(value, a, b, loc=self.loc, scale=self.scale)  # type: ignore[arg-type]

    @property
    def bounds(self) -> tuple[float, float]:
        return (self.low, self.high)

    def sample(self, key: Array) -> Array:
        # Rejection sampling from normal, clipped to bounds
        raw = self.loc + self.scale * jax.random.normal(key)
        return jnp.clip(raw, self.low, self.high)


@dataclass(frozen=True)
class PriorSet:
    """Collection of priors for a set of parameters.

    Provides a unified interface for computing joint log-probability,
    extracting bounds, and sampling all parameters.

    Example:
        >>> priors = PriorSet({
        ...     "lambda_0": Normal(loc=0.1, scale=0.02),
        ...     "gamma": Uniform(0.0, 5.0),
        ... })
        >>> priors.log_prob({"lambda_0": jnp.array(0.1), "gamma": jnp.array(2.5)})
        >>> priors.get_bounds()
        {'lambda_0': (0.038, 0.162), 'gamma': (0.0, 5.0)}
    """

    priors: dict[str, Prior]

    def log_prob(self, params: Params) -> Array:
        """Compute joint log-prior: sum of individual log-probabilities.

        Args:
            params: Parameter values. Only parameters with a prior are evaluated.

        Returns:
            Scalar log-probability.
        """
        total = jnp.array(0.0)
        for name, prior in self.priors.items():
            if name in params:
                total = total + jnp.sum(prior.log_prob(params[name]))
        return total

    def get_bounds(self) -> dict[str, tuple[float, float]]:
        """Extract bounds for all parameters.

        Returns:
            Dict of {param_name: (lower, upper)} suitable for Optimizer/EvolutionaryOptimizer.
        """
        return {name: prior.bounds for name, prior in self.priors.items()}

    def sample(self, key: Array) -> Params:
        """Sample all parameters from their priors.

        Args:
            key: JAX random key.

        Returns:
            Dict of sampled parameter values.
        """
        keys = jax.random.split(key, len(self.priors))
        result = {}
        for (name, prior), subkey in zip(self.priors.items(), keys, strict=True):
            result[name] = prior.sample(subkey)
        return result

    def to_unit(self, params: Params) -> Params:
        """Map physical parameters to unit space [0, 1] using prior bounds.

        Args:
            params: Parameter values in physical space.

        Returns:
            Parameter values normalized to [0, 1].
        """
        result = {}
        for name, prior in self.priors.items():
            if name in params:
                low, high = prior.bounds
                result[name] = (params[name] - low) / (high - low)
        return result

    def from_unit(self, params_unit: Params) -> Params:
        """Map unit-space [0, 1] parameters back to physical space.

        Args:
            params_unit: Parameter values in [0, 1].

        Returns:
            Parameter values in physical space.
        """
        result = {}
        for name, prior in self.priors.items():
            if name in params_unit:
                low, high = prior.bounds
                result[name] = params_unit[name] * (high - low) + low
        return result

    def log_det_jacobian(self) -> Array:
        """Log-determinant of the Jacobian for the unit -> physical transform.

        Returns:
            Scalar: sum of log(high - low) for all priors.
        """
        total = 0.0
        for prior in self.priors.values():
            low, high = prior.bounds
            total += float(jnp.log(high - low))
        return jnp.array(total)
