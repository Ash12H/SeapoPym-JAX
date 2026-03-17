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
    def test_bounds(self):
        prior = Uniform(2.0, 5.0)
        assert prior.bounds == (2.0, 5.0)

    def test_sample_in_bounds(self):
        prior = Uniform(0.0, 1.0)
        key = jax.random.key(0)
        s = prior.sample(key)
        assert 0.0 <= float(s) <= 1.0

    def test_sample_shape(self):
        prior = Uniform(0.0, 1.0)
        key = jax.random.key(0)
        s = prior.sample(key, shape=(5, 3))
        assert s.shape == (5, 3)
        assert jnp.all(s >= 0.0) and jnp.all(s <= 1.0)


class TestNormal:
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

    def test_sample_shape(self):
        prior = Normal(loc=0.0, scale=1.0)
        key = jax.random.key(0)
        s = prior.sample(key, shape=(8,))
        assert s.shape == (8,)


class TestLogNormal:
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

    def test_sample_shape(self):
        prior = LogNormal(mu=0.0, sigma=0.5)
        key = jax.random.key(0)
        s = prior.sample(key, shape=(4,))
        assert s.shape == (4,)
        assert jnp.all(s > 0.0)


class TestHalfNormal:
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

    def test_sample_shape(self):
        prior = HalfNormal(scale=1.0)
        key = jax.random.key(0)
        s = prior.sample(key, shape=(10,))
        assert s.shape == (10,)
        assert jnp.all(s >= 0.0)


class TestTruncatedNormal:
    def test_bounds(self):
        prior = TruncatedNormal(loc=0.5, scale=0.1, low=0.0, high=1.0)
        assert prior.bounds == (0.0, 1.0)

    def test_sample_in_bounds(self):
        prior = TruncatedNormal(loc=0.5, scale=0.1, low=0.0, high=1.0)
        key = jax.random.key(0)
        s = prior.sample(key)
        assert 0.0 <= float(s) <= 1.0

    def test_sample_shape(self):
        prior = TruncatedNormal(loc=0.5, scale=0.1, low=0.0, high=1.0)
        key = jax.random.key(0)
        s = prior.sample(key, shape=(6,))
        assert s.shape == (6,)
        assert jnp.all(s >= 0.0) and jnp.all(s <= 1.0)


class TestPriorSet:
    def setup_method(self):
        self.prior_set = PriorSet(
            {
                "alpha": Uniform(0.0, 1.0),
                "beta": Normal(loc=0.0, scale=1.0),
            }
        )

    def test_get_bounds(self):
        bounds = self.prior_set.get_bounds()
        assert "alpha" in bounds
        assert "beta" in bounds
        assert bounds["alpha"] == (0.0, 1.0)

    def test_sample_scalar(self):
        key = jax.random.key(0)
        samples = self.prior_set.sample(key)
        assert "alpha" in samples
        assert "beta" in samples
        assert 0.0 <= float(samples["alpha"]) <= 1.0

    def test_sample_with_shapes(self):
        key = jax.random.key(0)
        shapes = {"alpha": (8,), "beta": (8,)}
        samples = self.prior_set.sample(key, shapes=shapes)
        assert samples["alpha"].shape == (8,)
        assert samples["beta"].shape == (8,)
        assert jnp.all(samples["alpha"] >= 0.0) and jnp.all(samples["alpha"] <= 1.0)

    def test_sample_mixed_shapes(self):
        """Sample with different shapes per parameter."""
        key = jax.random.key(0)
        shapes = {"alpha": (4, 3), "beta": (4,)}
        samples = self.prior_set.sample(key, shapes=shapes)
        assert samples["alpha"].shape == (4, 3)
        assert samples["beta"].shape == (4,)
