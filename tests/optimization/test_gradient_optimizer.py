"""Tests for the GradientOptimizer class."""

import jax.numpy as jnp
import pytest

from seapopym.optimization.gradient_optimizer import GradientOptimizer, OptimizeResult
from seapopym.optimization.objective import Objective


class TestGradientOptimizerInit:
    def test_invalid_algorithm_raises(self):
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        with pytest.raises(ValueError, match="Unknown algorithm 'bad'"):
            GradientOptimizer([(obj, "mse", 1.0)], algorithm="bad")

    def test_bounds_scaling_without_bounds_raises(self):
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        with pytest.raises(ValueError, match="scaling='bounds' requires bounds"):
            GradientOptimizer([(obj, "mse", 1.0)], scaling="bounds")


class TestGradientOptimizerRunLossFn:
    def test_minimizes_quadratic_adam(self):
        """GradientOptimizer should minimize a simple quadratic via _run_loss_fn."""

        def loss_fn(params):
            x = params["x"]
            return (x - 3.0) ** 2

        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        opt = GradientOptimizer(
            [(obj, "mse", 1.0)],
            bounds={"x": (0.0, 10.0)},
            algorithm="adam",
            learning_rate=0.1,
            scaling="none",
        )

        result = opt._run_loss_fn(loss_fn, {"x": jnp.array(0.0)}, n_steps=200)

        assert isinstance(result, OptimizeResult)
        assert float(result.params["x"]) == pytest.approx(3.0, abs=0.5)
        assert result.loss < 1.0

    def test_scaling_bounds_normalizes(self):
        """With scaling='bounds', optimizer should normalize to [0,1]."""

        def loss_fn(params):
            x = params["x"]
            return (x - 5.0) ** 2

        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        opt = GradientOptimizer(
            [(obj, "mse", 1.0)],
            bounds={"x": (0.0, 10.0)},
            algorithm="adam",
            learning_rate=0.05,
            scaling="bounds",
        )

        result = opt._run_loss_fn(loss_fn, {"x": jnp.array(1.0)}, n_steps=200)

        assert float(result.params["x"]) == pytest.approx(5.0, abs=1.0)

    def test_scaling_none_clips_to_bounds(self):
        """With scaling='none' and bounds, params should be clipped."""

        def loss_fn(params):
            # Loss that wants to push x far negative
            return params["x"]

        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        opt = GradientOptimizer(
            [(obj, "mse", 1.0)],
            bounds={"x": (0.0, 10.0)},
            algorithm="adam",
            learning_rate=0.5,
            scaling="none",
        )

        result = opt._run_loss_fn(loss_fn, {"x": jnp.array(5.0)}, n_steps=100)

        assert float(result.params["x"]) >= -1e-6  # clipped at lower bound

    def test_scaling_log(self):
        """With scaling='log', optimizer should work with positive params."""

        def loss_fn(params):
            x = params["x"]
            return (x - 2.0) ** 2

        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        opt = GradientOptimizer(
            [(obj, "mse", 1.0)],
            algorithm="adam",
            learning_rate=0.05,
            scaling="log",
        )

        result = opt._run_loss_fn(loss_fn, {"x": jnp.array(1.0)}, n_steps=200)

        assert float(result.params["x"]) == pytest.approx(2.0, abs=0.5)

    def test_convergence_detection(self):
        """Optimizer should detect convergence when loss change < tolerance."""

        def loss_fn(params):
            return jnp.array(1.0)  # constant → converges immediately

        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        opt = GradientOptimizer(
            [(obj, "mse", 1.0)],
            algorithm="adam",
            learning_rate=0.01,
            scaling="none",
        )

        result = opt._run_loss_fn(loss_fn, {"x": jnp.array(1.0)}, n_steps=100, tolerance=1e-3)

        assert result.converged is True
        assert result.n_iterations < 100

    def test_loss_history_length(self):
        """Loss history should have one entry per iteration."""

        def loss_fn(params):
            return (params["x"] - 1.0) ** 2

        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        opt = GradientOptimizer(
            [(obj, "mse", 1.0)],
            algorithm="adam",
            learning_rate=0.01,
            scaling="none",
        )

        result = opt._run_loss_fn(loss_fn, {"x": jnp.array(5.0)}, n_steps=50, tolerance=0.0)

        assert len(result.loss_history) == 50
        assert result.n_iterations == 50
