"""Tests for the Optimizer class."""

import jax.numpy as jnp
import pytest

from seapopym.optimization.optimizer import Optimizer, OptimizeResult


class TestOptimizerInit:
    """Tests for Optimizer initialization."""

    def test_default_init(self):
        """Optimizer should initialize with default values."""
        opt = Optimizer()
        assert opt.algorithm == "adam"
        assert opt.learning_rate == 0.01
        assert opt.bounds == {}

    def test_custom_algorithm(self):
        """Optimizer should accept different algorithms."""
        for algo in ("adam", "sgd", "rmsprop", "adagrad"):
            opt = Optimizer(algorithm=algo)  # type: ignore[arg-type]
            assert opt.algorithm == algo

    def test_invalid_algorithm(self):
        """Optimizer should reject invalid algorithms."""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            Optimizer(algorithm="invalid")  # type: ignore[arg-type]

    def test_with_bounds(self):
        """Optimizer should store bounds."""
        bounds = {"x": (0.0, 1.0), "y": (-1.0, 1.0)}
        opt = Optimizer(bounds=bounds)
        assert opt.bounds == bounds


class TestOptimizerStep:
    """Tests for single optimization step."""

    def test_step_reduces_simple_loss(self):
        """A step should move params in gradient direction."""
        opt = Optimizer(algorithm="sgd", learning_rate=0.1)
        params = {"x": jnp.array(5.0)}
        grads = {"x": jnp.array(1.0)}  # Gradient pointing to increase x

        new_params = opt.step(params, grads)

        # SGD: x_new = x - lr * grad = 5 - 0.1 * 1 = 4.9
        assert float(new_params["x"]) == pytest.approx(4.9, abs=1e-6)

    def test_step_with_bounds(self):
        """Step should clip parameters to bounds."""
        opt = Optimizer(algorithm="sgd", learning_rate=1.0, bounds={"x": (0.0, 10.0)})
        params = {"x": jnp.array(1.0)}
        grads = {"x": jnp.array(5.0)}  # Large gradient

        new_params = opt.step(params, grads)

        # Without bounds: 1 - 1.0 * 5 = -4, but bounded to [0, 10]
        assert float(new_params["x"]) == pytest.approx(0.0, abs=1e-6)

    def test_step_multiple_params(self):
        """Step should handle multiple parameters."""
        opt = Optimizer(algorithm="sgd", learning_rate=0.1)
        params = {"x": jnp.array(1.0), "y": jnp.array(2.0)}
        grads = {"x": jnp.array(1.0), "y": jnp.array(-1.0)}

        new_params = opt.step(params, grads)

        assert float(new_params["x"]) == pytest.approx(0.9, abs=1e-6)
        assert float(new_params["y"]) == pytest.approx(2.1, abs=1e-6)


class TestOptimizerRun:
    """Tests for full optimization run."""

    def test_run_minimizes_quadratic(self):
        """Optimizer should minimize a simple quadratic."""

        def loss_fn(params):
            x = params["x"]
            return (x - 3.0) ** 2  # Minimum at x=3

        opt = Optimizer(algorithm="adam", learning_rate=0.5)
        initial_params = {"x": jnp.array(0.0)}

        result = opt.run(loss_fn, initial_params, n_steps=100)

        assert isinstance(result, OptimizeResult)
        assert float(result.params["x"]) == pytest.approx(3.0, abs=0.1)
        assert result.loss < 0.01
        assert len(result.loss_history) <= 100

    def test_run_respects_n_steps(self):
        """Optimizer should stop after n_steps."""

        def loss_fn(params):
            return params["x"] ** 2

        opt = Optimizer(learning_rate=0.001)  # Small LR, won't converge
        initial_params = {"x": jnp.array(10.0)}

        result = opt.run(loss_fn, initial_params, n_steps=20)

        assert result.n_iterations == 20
        assert not result.converged

    def test_run_converges_early(self):
        """Optimizer should converge early if tolerance reached."""

        def loss_fn(params):
            return jnp.array(0.0)  # Already at minimum

        opt = Optimizer()
        initial_params = {"x": jnp.array(0.0)}

        result = opt.run(loss_fn, initial_params, n_steps=100, tolerance=1e-6)

        assert result.converged
        assert result.n_iterations < 100

    def test_run_with_bounds(self):
        """Optimizer should respect bounds during run."""

        def loss_fn(params):
            x = params["x"]
            return -(x**2)  # Wants to maximize |x|, unbounded would go to infinity

        opt = Optimizer(
            algorithm="sgd",
            learning_rate=0.1,
            bounds={"x": (-1.0, 1.0)},
        )
        initial_params = {"x": jnp.array(0.5)}

        result = opt.run(loss_fn, initial_params, n_steps=100)

        # Should hit the bound
        assert abs(float(result.params["x"])) == pytest.approx(1.0, abs=0.01)

    def test_run_multivariate(self):
        """Optimizer should handle multiple parameters."""

        def loss_fn(params):
            x = params["x"]
            y = params["y"]
            return (x - 1.0) ** 2 + (y - 2.0) ** 2  # Minimum at (1, 2)

        opt = Optimizer(algorithm="adam", learning_rate=0.3)
        initial_params = {"x": jnp.array(0.0), "y": jnp.array(0.0)}

        result = opt.run(loss_fn, initial_params, n_steps=200)

        assert float(result.params["x"]) == pytest.approx(1.0, abs=0.1)
        assert float(result.params["y"]) == pytest.approx(2.0, abs=0.1)


class TestOptimizeResult:
    """Tests for OptimizeResult dataclass."""

    def test_optimize_result_creation(self):
        """OptimizeResult should store all fields."""
        result = OptimizeResult(
            params={"x": jnp.array(1.0)},
            loss=0.5,
            loss_history=[1.0, 0.7, 0.5],
            n_iterations=3,
            converged=True,
            message="Converged",
        )

        assert result.params == {"x": jnp.array(1.0)}
        assert result.loss == 0.5
        assert result.loss_history == [1.0, 0.7, 0.5]
        assert result.n_iterations == 3
        assert result.converged is True
        assert result.message == "Converged"
