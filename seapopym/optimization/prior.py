"""Prior distributions for parameter initialization.

Provides distributions for sampling initial parameter values in
multi-start optimization. Each distribution defines bounds and
a sampling method compatible with JAX (jit, vmap).
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

    @property
    def bounds(self) -> tuple[float, float]:
        """Parameter bounds (finite, suitable for optimizers)."""
        ...

    def sample(self, key: Array, shape: tuple[int, ...] = ()) -> Array:
        """Draw random sample(s).

        Args:
            key: JAX random key.
            shape: Output shape. ``()`` returns a scalar.
        """
        ...


@dataclass(frozen=True)
class Uniform:
    """Uniform distribution on [low, high].

    Example:
        >>> prior = Uniform(0.0, 1.0)
        >>> prior.bounds
        (0.0, 1.0)
    """

    low: float
    high: float

    @property
    def bounds(self) -> tuple[float, float]:
        return (self.low, self.high)

    def sample(self, key: Array, shape: tuple[int, ...] = ()) -> Array:
        return jax.random.uniform(key, shape=shape, minval=self.low, maxval=self.high)


@dataclass(frozen=True)
class Normal:
    """Normal (Gaussian) distribution.

    Bounds are derived from quantiles for use with bounded optimizers.

    Example:
        >>> prior = Normal(loc=0.1, scale=0.02)
        >>> prior.bounds  # ~(0.038, 0.162) at 0.1% quantiles
    """

    loc: float
    scale: float
    bounds_quantile: float = 0.001

    @property
    def bounds(self) -> tuple[float, float]:
        z = float(jstats.norm.ppf(self.bounds_quantile))
        return (self.loc + z * self.scale, self.loc - z * self.scale)

    def sample(self, key: Array, shape: tuple[int, ...] = ()) -> Array:
        return self.loc + self.scale * jax.random.normal(key, shape=shape)


@dataclass(frozen=True)
class LogNormal:
    """Log-normal distribution (value > 0, log(value) ~ Normal).

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

    @property
    def bounds(self) -> tuple[float, float]:
        z = float(jstats.norm.ppf(self.bounds_quantile))
        return (float(jnp.exp(self.mu + z * self.sigma)), float(jnp.exp(self.mu - z * self.sigma)))

    def sample(self, key: Array, shape: tuple[int, ...] = ()) -> Array:
        return jnp.exp(self.mu + self.sigma * jax.random.normal(key, shape=shape))


@dataclass(frozen=True)
class HalfNormal:
    """Half-normal distribution (value >= 0).

    A normal distribution folded at zero: p(x) = 2 * Normal(0, scale).pdf(x) for x >= 0.

    Example:
        >>> prior = HalfNormal(scale=1.0)
        >>> prior.bounds  # (0.0, quantile)
    """

    scale: float
    bounds_quantile: float = 0.001

    @property
    def bounds(self) -> tuple[float, float]:
        upper_q = (1.0 + (1.0 - self.bounds_quantile)) / 2.0
        z = float(jstats.norm.ppf(upper_q))
        return (0.0, z * self.scale)

    def sample(self, key: Array, shape: tuple[int, ...] = ()) -> Array:
        return jnp.abs(self.scale * jax.random.normal(key, shape=shape))


@dataclass(frozen=True)
class TruncatedNormal:
    """Truncated normal distribution on [low, high].

    Example:
        >>> prior = TruncatedNormal(loc=0.5, scale=0.1, low=0.0, high=1.0)
        >>> prior.bounds
        (0.0, 1.0)
    """

    loc: float
    scale: float
    low: float
    high: float

    @property
    def bounds(self) -> tuple[float, float]:
        return (self.low, self.high)

    def sample(self, key: Array, shape: tuple[int, ...] = ()) -> Array:
        u = jax.random.uniform(key, shape=shape)
        a = (self.low - self.loc) / self.scale
        b = (self.high - self.loc) / self.scale
        cdf_a = jstats.norm.cdf(a)
        cdf_b = jstats.norm.cdf(b)
        return self.loc + self.scale * jstats.norm.ppf(cdf_a + u * (cdf_b - cdf_a))


@dataclass(frozen=True)
class PriorSet:
    """Collection of priors for a set of parameters.

    Provides a unified interface for extracting bounds and sampling
    all parameters.

    Example:
        >>> priors = PriorSet({
        ...     "lambda_0": Normal(loc=0.1, scale=0.02),
        ...     "gamma": Uniform(0.0, 5.0),
        ... })
        >>> priors.get_bounds()
        {'lambda_0': (0.038, 0.162), 'gamma': (0.0, 5.0)}
    """

    priors: dict[str, Prior]

    def get_bounds(self) -> dict[str, tuple[float, float]]:
        """Extract bounds for all parameters.

        Returns:
            Dict of {param_name: (lower, upper)} suitable for optimizers.
        """
        return {name: prior.bounds for name, prior in self.priors.items()}

    def sample(self, key: Array, shapes: dict[str, tuple[int, ...]] | None = None) -> Params:
        """Sample all parameters from their distributions.

        Args:
            key: JAX random key.
            shapes: Per-parameter output shapes. ``None`` samples scalars.
                Use ``{name: (n_starts, *param_shape)}`` for batched sampling.

        Returns:
            Dict of sampled parameter values.
        """
        keys = jax.random.split(key, len(self.priors))
        result = {}
        for (name, prior), subkey in zip(self.priors.items(), keys, strict=True):
            shape = shapes[name] if shapes is not None and name in shapes else ()
            result[name] = prior.sample(subkey, shape=shape)
        return result
