"""Tests for optimization loss functions."""

import jax.numpy as jnp
import pytest

from seapopym.optimization.loss import mse, nrmse, rmse


class TestRMSE:
    """Tests for the rmse function."""

    def test_rmse_perfect_match(self):
        """RMSE should be 0 for identical arrays."""
        pred = jnp.array([1.0, 2.0, 3.0])
        obs = jnp.array([1.0, 2.0, 3.0])
        result = rmse(pred, obs)
        assert float(result) == pytest.approx(0.0, abs=1e-6)

    def test_rmse_known_value(self):
        """RMSE should match manual calculation."""
        pred = jnp.array([1.0, 2.0, 3.0])
        obs = jnp.array([2.0, 3.0, 4.0])
        # Differences: [1, 1, 1], squared: [1, 1, 1], mean: 1, sqrt: 1
        result = rmse(pred, obs)
        assert float(result) == pytest.approx(1.0, abs=1e-6)

    def test_rmse_with_mask(self):
        """RMSE should only consider masked points."""
        pred = jnp.array([1.0, 2.0, 3.0, 100.0])  # Last point is outlier
        obs = jnp.array([1.0, 2.0, 3.0, 0.0])
        mask = jnp.array([True, True, True, False])  # Ignore outlier
        result = rmse(pred, obs, mask)
        assert float(result) == pytest.approx(0.0, abs=1e-6)

    def test_rmse_partial_mask(self):
        """RMSE with partial mask should compute correctly."""
        pred = jnp.array([1.0, 3.0, 5.0])
        obs = jnp.array([2.0, 3.0, 4.0])
        # Differences: [1, 0, 1]
        mask = jnp.array([True, False, True])  # Only use indices 0 and 2
        # Masked differences: [1, 1], squared: [1, 1], mean: 1, sqrt: 1
        result = rmse(pred, obs, mask)
        assert float(result) == pytest.approx(1.0, abs=1e-6)

    def test_rmse_single_value(self):
        """RMSE should work with single values."""
        pred = jnp.array([5.0])
        obs = jnp.array([3.0])
        result = rmse(pred, obs)
        assert float(result) == pytest.approx(2.0, abs=1e-6)


class TestMSE:
    """Tests for the mse function."""

    def test_mse_perfect_match(self):
        """MSE should be 0 for identical arrays."""
        pred = jnp.array([1.0, 2.0, 3.0])
        obs = jnp.array([1.0, 2.0, 3.0])
        result = mse(pred, obs)
        assert float(result) == pytest.approx(0.0, abs=1e-6)

    def test_mse_known_value(self):
        """MSE should match manual calculation."""
        pred = jnp.array([1.0, 2.0, 3.0])
        obs = jnp.array([2.0, 3.0, 4.0])
        # Differences: [1, 1, 1], squared: [1, 1, 1], mean: 1
        result = mse(pred, obs)
        assert float(result) == pytest.approx(1.0, abs=1e-6)

    def test_mse_with_mask(self):
        """MSE should only consider masked points."""
        pred = jnp.array([1.0, 2.0, 100.0])
        obs = jnp.array([1.0, 2.0, 0.0])
        mask = jnp.array([True, True, False])
        result = mse(pred, obs, mask)
        assert float(result) == pytest.approx(0.0, abs=1e-6)


class TestNRMSE:
    """Tests for the nrmse function."""

    def test_nrmse_std_mode(self):
        """NRMSE with std normalization."""
        pred = jnp.array([1.0, 2.0, 3.0, 4.0])
        obs = jnp.array([1.5, 2.5, 3.5, 4.5])
        # RMSE = 0.5, std(obs) = 1.118..., NRMSE = 0.5 / 1.118 = 0.447
        result = nrmse(pred, obs, mode="std")
        assert float(result) == pytest.approx(0.447, abs=0.01)

    def test_nrmse_mean_mode(self):
        """NRMSE with mean normalization."""
        pred = jnp.array([10.0, 20.0, 30.0])
        obs = jnp.array([11.0, 21.0, 31.0])
        # RMSE = 1.0, mean(obs) = 21.0, NRMSE = 1/21 = 0.0476
        result = nrmse(pred, obs, mode="mean")
        assert float(result) == pytest.approx(0.0476, abs=0.001)

    def test_nrmse_minmax_mode(self):
        """NRMSE with minmax normalization."""
        pred = jnp.array([0.0, 5.0, 10.0])
        obs = jnp.array([1.0, 5.0, 9.0])
        # RMSE = sqrt((1+0+1)/3) = 0.816, range = 9-1 = 8, NRMSE = 0.816/8 = 0.102
        result = nrmse(pred, obs, mode="minmax")
        assert float(result) == pytest.approx(0.102, abs=0.01)

    def test_nrmse_with_mask(self):
        """NRMSE should respect mask for both RMSE and normalization."""
        pred = jnp.array([1.0, 2.0, 100.0])
        obs = jnp.array([1.0, 2.0, 0.0])
        mask = jnp.array([True, True, False])
        # Only first two points: RMSE = 0, so NRMSE = 0
        result = nrmse(pred, obs, mask, mode="std")
        assert float(result) == pytest.approx(0.0, abs=1e-6)

    def test_nrmse_invalid_mode(self):
        """NRMSE should raise error for invalid mode."""
        pred = jnp.array([1.0, 2.0])
        obs = jnp.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Unknown normalization mode"):
            nrmse(pred, obs, mode="invalid")  # type: ignore[arg-type]


class TestGradients:
    """Test that loss functions are differentiable."""

    def test_rmse_gradient(self):
        """RMSE should be differentiable."""
        import jax

        def loss(pred):
            obs = jnp.array([1.0, 2.0, 3.0])
            return rmse(pred, obs)

        pred = jnp.array([1.5, 2.5, 3.5])
        grad = jax.grad(loss)(pred)
        assert grad.shape == pred.shape
        assert not jnp.any(jnp.isnan(grad))

    def test_mse_gradient(self):
        """MSE should be differentiable."""
        import jax

        def loss(pred):
            obs = jnp.array([1.0, 2.0, 3.0])
            return mse(pred, obs)

        pred = jnp.array([1.5, 2.5, 3.5])
        grad = jax.grad(loss)(pred)
        assert grad.shape == pred.shape
        assert not jnp.any(jnp.isnan(grad))

    def test_nrmse_gradient(self):
        """NRMSE should be differentiable."""
        import jax

        def loss(pred):
            obs = jnp.array([1.0, 2.0, 3.0, 4.0])
            return nrmse(pred, obs, mode="std")

        pred = jnp.array([1.5, 2.5, 3.5, 4.5])
        grad = jax.grad(loss)(pred)
        assert grad.shape == pred.shape
        assert not jnp.any(jnp.isnan(grad))
