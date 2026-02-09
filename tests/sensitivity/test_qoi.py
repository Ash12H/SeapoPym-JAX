"""Tests for QoI (Quantity of Interest) functions."""

import jax.numpy as jnp
import pytest

from seapopym.sensitivity.qoi import available_qoi, compute_qoi


class TestIndividualQoI:
    """Tests for individual QoI functions on known time series."""

    def setup_method(self):
        """Create a known time series: (batch=2, T=10, n_points=3)."""
        # Simple ramp: values 0..9 along time axis
        t = jnp.arange(10, dtype=jnp.float32)
        # batch=2, n_points=3, all same ramp
        self.ts = jnp.broadcast_to(t[None, :, None], (2, 10, 3))

    def test_mean(self):
        result = compute_qoi(self.ts, ["mean"])
        expected = 4.5  # mean of 0..9
        assert jnp.allclose(result["mean"], expected)

    def test_var(self):
        result = compute_qoi(self.ts, ["var"])
        expected = jnp.var(jnp.arange(10, dtype=jnp.float32))
        assert jnp.allclose(result["var"], expected)

    def test_std(self):
        result = compute_qoi(self.ts, ["std"])
        expected = jnp.std(jnp.arange(10, dtype=jnp.float32))
        assert jnp.allclose(result["std"], expected)

    def test_min(self):
        result = compute_qoi(self.ts, ["min"])
        assert jnp.allclose(result["min"], 0.0)

    def test_max(self):
        result = compute_qoi(self.ts, ["max"])
        assert jnp.allclose(result["max"], 9.0)

    def test_median(self):
        result = compute_qoi(self.ts, ["median"])
        expected = 4.5
        assert jnp.allclose(result["median"], expected)

    def test_argmax(self):
        result = compute_qoi(self.ts, ["argmax"])
        assert jnp.allclose(result["argmax"], 9.0)


class TestComputeQoi:
    """Tests for compute_qoi orchestrator."""

    def test_multiple_qoi(self):
        """Should compute all requested QoI in one call."""
        ts = jnp.ones((4, 20, 2))
        result = compute_qoi(ts, ["mean", "var", "max"])
        assert len(result) == 3
        assert result["mean"].shape == (4, 2)
        assert result["var"].shape == (4, 2)
        assert result["max"].shape == (4, 2)

    def test_unknown_qoi_raises(self):
        ts = jnp.ones((1, 5, 1))
        with pytest.raises(ValueError, match="Unknown QoI"):
            compute_qoi(ts, ["nonexistent"])

    def test_output_shapes(self):
        """All QoI should produce (batch, n_points) from (batch, T, n_points)."""
        ts = jnp.ones((8, 100, 5))
        result = compute_qoi(ts, available_qoi())
        for name, arr in result.items():
            assert arr.shape == (8, 5), f"QoI '{name}' has wrong shape: {arr.shape}"

    def test_argmax_with_peak(self):
        """Argmax should find the correct peak position."""
        # Create series with peak at different positions per point
        ts = jnp.zeros((1, 10, 2))
        ts = ts.at[0, 3, 0].set(10.0)  # peak at t=3 for point 0
        ts = ts.at[0, 7, 1].set(10.0)  # peak at t=7 for point 1
        result = compute_qoi(ts, ["argmax"])
        assert result["argmax"][0, 0] == 3.0
        assert result["argmax"][0, 1] == 7.0


class TestAvailableQoi:
    """Tests for QoI registry."""

    def test_returns_list(self):
        result = available_qoi()
        assert isinstance(result, list)

    def test_contains_expected(self):
        result = available_qoi()
        for name in ["mean", "var", "std", "min", "max", "median", "argmax"]:
            assert name in result, f"Missing QoI: {name}"
