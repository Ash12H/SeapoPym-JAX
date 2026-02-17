"""Tests for NUTS sampler integration."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from seapopym.optimization.nuts import NUTSResult, run_nuts


class TestRunNuts:
    """Test NUTS on a simple 2D Gaussian: N([3, -1], I)."""

    @pytest.fixture()
    def gaussian_result(self):
        def log_posterior(params):
            x = params["x"]
            y = params["y"]
            return -0.5 * (x - 3.0) ** 2 + -0.5 * (y + 1.0) ** 2

        initial = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        return run_nuts(log_posterior, initial, n_warmup=500, n_samples=1000, seed=42)

    def test_returns_nuts_result(self, gaussian_result):
        assert isinstance(gaussian_result, NUTSResult)

    def test_sample_shapes(self, gaussian_result):
        assert gaussian_result.samples["x"].shape == (1000,)
        assert gaussian_result.samples["y"].shape == (1000,)

    def test_recovers_mean(self, gaussian_result):
        mean_x = float(jnp.mean(gaussian_result.samples["x"]))
        mean_y = float(jnp.mean(gaussian_result.samples["y"]))
        assert mean_x == pytest.approx(3.0, abs=0.3)
        assert mean_y == pytest.approx(-1.0, abs=0.3)

    def test_recovers_std(self, gaussian_result):
        std_x = float(jnp.std(gaussian_result.samples["x"]))
        std_y = float(jnp.std(gaussian_result.samples["y"]))
        assert std_x == pytest.approx(1.0, abs=0.3)
        assert std_y == pytest.approx(1.0, abs=0.3)

    def test_log_posterior_values(self, gaussian_result):
        assert gaussian_result.log_posterior_values.shape == (1000,)
        assert jnp.all(jnp.isfinite(gaussian_result.log_posterior_values))

    def test_no_divergences(self, gaussian_result):
        # Simple Gaussian should have no divergences
        assert int(jnp.sum(gaussian_result.divergences)) == 0

    def test_acceptance_rate(self, gaussian_result):
        assert 0.5 < gaussian_result.acceptance_rate < 1.0

    def test_metadata(self, gaussian_result):
        assert gaussian_result.n_warmup == 500
        assert gaussian_result.n_samples == 1000

    def test_kernel_params_returned(self, gaussian_result):
        kp = gaussian_result.kernel_params
        assert kp is not None
        assert "step_size" in kp
        assert "inverse_mass_matrix" in kp


class TestSharedWarmup:
    """Test warmup sharing: run warmup once, reuse kernel_params for a second chain."""

    def test_shared_warmup_produces_valid_samples(self):
        def log_posterior(params):
            x = params["x"]
            return -0.5 * (x - 5.0) ** 2

        initial = {"x": jnp.array(0.0)}

        # Chain 1: full warmup
        result1 = run_nuts(log_posterior, initial, n_warmup=500, n_samples=500, seed=0)
        assert result1.kernel_params is not None
        assert result1.n_warmup == 500

        # Chain 2: skip warmup, reuse kernel_params
        result2 = run_nuts(
            log_posterior,
            initial,
            n_samples=500,
            seed=1,
            kernel_params=result1.kernel_params,
        )

        assert result2.n_warmup == 0
        assert result2.kernel_params is not None
        assert result2.samples["x"].shape == (500,)
        mean_x = float(jnp.mean(result2.samples["x"]))
        assert mean_x == pytest.approx(5.0, abs=0.5)
