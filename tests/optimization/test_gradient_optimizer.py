"""Tests for the GradientOptimizer class."""

import jax.numpy as jnp
import pytest

from seapopym.optimization._common import GradientStepResult, OptimizeResult
from seapopym.optimization.gradient_optimizer import GradientOptimizer


class TestGradientOptimizerInit:
    def test_invalid_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unknown algorithm 'bad'"):
            GradientOptimizer(
                bounds={"x": (0.0, 10.0)},
                initial_params={"x": jnp.array(1.0)},
                algorithm="bad",  # type: ignore[reportArgumentType]
            )

    def test_bounds_scaling_without_bounds_raises(self):
        with pytest.raises(ValueError, match="scaling='bounds' requires bounds"):
            GradientOptimizer(
                bounds={},
                initial_params={"x": jnp.array(1.0)},
                scaling="bounds",
            )


class TestGradientStep:
    def test_step_returns_gradient_step_result(self):
        opt = GradientOptimizer(
            bounds={"x": (0.0, 10.0)},
            initial_params={"x": jnp.array(5.0)},
            algorithm="adam",
            learning_rate=0.1,
        )
        result = opt.step(lambda p: (p["x"] - 3.0) ** 2)
        assert isinstance(result, GradientStepResult)
        assert result.step == 0
        assert result.loss > 0
        assert result.grad_norm > 0

    def test_step_increments(self):
        opt = GradientOptimizer(
            bounds={"x": (0.0, 10.0)},
            initial_params={"x": jnp.array(5.0)},
            algorithm="adam",
            learning_rate=0.1,
        )

        def loss_fn(p):
            return (p["x"] - 3.0) ** 2

        r0 = opt.step(loss_fn)
        r1 = opt.step(loss_fn)
        assert r0.step == 0
        assert r1.step == 1

    def test_step_grad_norm_is_finite(self):
        opt = GradientOptimizer(
            bounds={"x": (0.0, 10.0)},
            initial_params={"x": jnp.array(5.0)},
            algorithm="adam",
            learning_rate=0.01,
        )
        result = opt.step(lambda p: p["x"] ** 2)
        assert jnp.isfinite(result.grad_norm)


class TestGradientRun:
    def test_minimizes_quadratic_adam(self):
        opt = GradientOptimizer(
            bounds={"x": (0.0, 10.0)},
            initial_params={"x": jnp.array(0.0)},
            algorithm="adam",
            learning_rate=0.1,
        )
        result = opt.run(lambda p: (p["x"] - 3.0) ** 2, max_steps=200)

        assert isinstance(result, OptimizeResult)
        assert float(result.params["x"]) == pytest.approx(3.0, abs=0.5)
        assert result.loss < 1.0

    def test_scaling_bounds(self):
        opt = GradientOptimizer(
            bounds={"x": (0.0, 10.0)},
            initial_params={"x": jnp.array(1.0)},
            algorithm="adam",
            learning_rate=0.05,
            scaling="bounds",
        )
        result = opt.run(lambda p: (p["x"] - 5.0) ** 2, max_steps=200)

        assert float(result.params["x"]) == pytest.approx(5.0, abs=1.0)

    def test_scaling_none_clips_to_bounds(self):
        opt = GradientOptimizer(
            bounds={"x": (0.0, 10.0)},
            initial_params={"x": jnp.array(5.0)},
            algorithm="adam",
            learning_rate=0.5,
        )
        # Loss that wants to push x far negative
        result = opt.run(lambda p: p["x"], max_steps=100)

        assert float(result.params["x"]) >= -1e-6

    def test_scaling_log(self):
        opt = GradientOptimizer(
            bounds={"x": (0.1, 10.0)},
            initial_params={"x": jnp.array(1.0)},
            algorithm="adam",
            learning_rate=0.05,
            scaling="log",
        )
        result = opt.run(lambda p: (p["x"] - 2.0) ** 2, max_steps=200)

        assert float(result.params["x"]) == pytest.approx(2.0, abs=0.5)

    def test_convergence_detection(self):
        opt = GradientOptimizer(
            bounds={"x": (0.0, 10.0)},
            initial_params={"x": jnp.array(1.0)},
            algorithm="adam",
            learning_rate=0.01,
        )
        result = opt.run(lambda p: jnp.array(1.0), max_steps=100, tolerance=1e-3)

        assert result.converged is True
        assert result.n_iterations < 100

    def test_loss_history_length(self):
        opt = GradientOptimizer(
            bounds={"x": (0.0, 10.0)},
            initial_params={"x": jnp.array(5.0)},
            algorithm="adam",
            learning_rate=0.01,
        )
        result = opt.run(lambda p: (p["x"] - 1.0) ** 2, max_steps=50, tolerance=0.0)

        assert len(result.loss_history) == 50
        assert result.n_iterations == 50
