"""Tests for the GradientOptimizer class."""

import jax.numpy as jnp
import pytest

from seapopym.optimization.optimizer import GradientOptimizer, OptimizeResult


class TestGradientOptimizerInit:
    """Tests for GradientOptimizer initialization."""

    def test_default_init(self):
        """GradientOptimizer should initialize with default values."""
        opt = GradientOptimizer()
        assert opt.algorithm == "adam"
        assert opt.learning_rate == 0.01
        assert opt.bounds == {}

    def test_custom_algorithm(self):
        """GradientOptimizer should accept different algorithms."""
        for algo in ("adam", "sgd", "rmsprop", "adagrad"):
            opt = GradientOptimizer(algorithm=algo)  # type: ignore[arg-type]
            assert opt.algorithm == algo

    def test_invalid_algorithm(self):
        """GradientOptimizer should reject invalid algorithms."""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            GradientOptimizer(algorithm="invalid")  # type: ignore[arg-type]

    def test_with_bounds(self):
        """GradientOptimizer should store bounds."""
        bounds = {"x": (0.0, 1.0), "y": (-1.0, 1.0)}
        opt = GradientOptimizer(bounds=bounds)
        assert opt.bounds == bounds


class TestGradientOptimizerStep:
    """Tests for single optimization step."""

    def test_step_reduces_simple_loss(self):
        """A step should move params in gradient direction."""
        opt = GradientOptimizer(algorithm="sgd", learning_rate=0.1)
        params = {"x": jnp.array(5.0)}
        grads = {"x": jnp.array(1.0)}  # Gradient pointing to increase x

        new_params = opt.step(params, grads)

        # SGD: x_new = x - lr * grad = 5 - 0.1 * 1 = 4.9
        assert float(new_params["x"]) == pytest.approx(4.9, abs=1e-6)

    def test_step_with_bounds(self):
        """Step should clip parameters to bounds."""
        opt = GradientOptimizer(algorithm="sgd", learning_rate=1.0, bounds={"x": (0.0, 10.0)})
        params = {"x": jnp.array(1.0)}
        grads = {"x": jnp.array(5.0)}  # Large gradient

        new_params = opt.step(params, grads)

        # Without bounds: 1 - 1.0 * 5 = -4, but bounded to [0, 10]
        assert float(new_params["x"]) == pytest.approx(0.0, abs=1e-6)

    def test_step_multiple_params(self):
        """Step should handle multiple parameters."""
        opt = GradientOptimizer(algorithm="sgd", learning_rate=0.1)
        params = {"x": jnp.array(1.0), "y": jnp.array(2.0)}
        grads = {"x": jnp.array(1.0), "y": jnp.array(-1.0)}

        new_params = opt.step(params, grads)

        assert float(new_params["x"]) == pytest.approx(0.9, abs=1e-6)
        assert float(new_params["y"]) == pytest.approx(2.1, abs=1e-6)


class TestGradientOptimizerRun:
    """Tests for full optimization run."""

    def test_run_minimizes_quadratic(self):
        """GradientOptimizer should minimize a simple quadratic."""

        def loss_fn(params):
            x = params["x"]
            return (x - 3.0) ** 2  # Minimum at x=3

        opt = GradientOptimizer(algorithm="adam", learning_rate=0.5)
        initial_params = {"x": jnp.array(0.0)}

        result = opt.run(loss_fn, initial_params, n_steps=100)

        assert isinstance(result, OptimizeResult)
        assert float(result.params["x"]) == pytest.approx(3.0, abs=0.1)
        assert result.loss < 0.01
        assert len(result.loss_history) <= 100

    def test_run_respects_n_steps(self):
        """GradientOptimizer should stop after n_steps."""

        def loss_fn(params):
            return params["x"] ** 2

        opt = GradientOptimizer(learning_rate=0.001)  # Small LR, won't converge
        initial_params = {"x": jnp.array(10.0)}

        result = opt.run(loss_fn, initial_params, n_steps=20)

        assert result.n_iterations == 20
        assert not result.converged

    def test_run_converges_early(self):
        """GradientOptimizer should converge early if tolerance reached."""

        def loss_fn(params):
            return jnp.array(0.0)  # Already at minimum

        opt = GradientOptimizer()
        initial_params = {"x": jnp.array(0.0)}

        result = opt.run(loss_fn, initial_params, n_steps=100, tolerance=1e-6)

        assert result.converged
        assert result.n_iterations < 100

    def test_run_with_bounds(self):
        """GradientOptimizer should respect bounds during run."""

        def loss_fn(params):
            x = params["x"]
            return -(x**2)  # Wants to maximize |x|, unbounded would go to infinity

        opt = GradientOptimizer(
            algorithm="sgd",
            learning_rate=0.1,
            bounds={"x": (-1.0, 1.0)},
        )
        initial_params = {"x": jnp.array(0.5)}

        result = opt.run(loss_fn, initial_params, n_steps=100)

        # Should hit the bound
        assert abs(float(result.params["x"])) == pytest.approx(1.0, abs=0.01)

    def test_run_multivariate(self):
        """GradientOptimizer should handle multiple parameters."""

        def loss_fn(params):
            x = params["x"]
            y = params["y"]
            return (x - 1.0) ** 2 + (y - 2.0) ** 2  # Minimum at (1, 2)

        opt = GradientOptimizer(algorithm="adam", learning_rate=0.3)
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


class TestScaling:
    """Tests for parameter scaling functionality."""

    def test_scaling_none_is_default(self):
        """Default scaling should be 'none'."""
        opt = GradientOptimizer()
        assert opt.scaling == "none"

    def test_scaling_bounds_requires_bounds(self):
        """scaling='bounds' should require bounds parameter."""
        with pytest.raises(ValueError, match="requires bounds"):
            GradientOptimizer(scaling="bounds")

    def test_scaling_bounds_accepted_with_bounds(self):
        """scaling='bounds' should work when bounds provided."""
        opt = GradientOptimizer(scaling="bounds", bounds={"x": (0.0, 1.0)})
        assert opt.scaling == "bounds"

    def test_normalize_bounds(self):
        """_normalize should map to [0, 1] with bounds scaling."""
        opt = GradientOptimizer(scaling="bounds", bounds={"x": (0.0, 10.0)})
        params = {"x": jnp.array(5.0)}

        normalized = opt._normalize(params)

        assert float(normalized["x"]) == pytest.approx(0.5, abs=1e-6)

    def test_denormalize_bounds(self):
        """_denormalize should map from [0, 1] with bounds scaling."""
        opt = GradientOptimizer(scaling="bounds", bounds={"x": (0.0, 10.0)})
        params_norm = {"x": jnp.array(0.5)}

        denormalized = opt._denormalize(params_norm)

        assert float(denormalized["x"]) == pytest.approx(5.0, abs=1e-6)

    def test_normalize_denormalize_roundtrip(self):
        """normalize then denormalize should return original value."""
        opt = GradientOptimizer(scaling="bounds", bounds={"x": (2.0, 8.0), "y": (-1.0, 1.0)})
        params = {"x": jnp.array(4.0), "y": jnp.array(0.5)}

        roundtrip = opt._denormalize(opt._normalize(params))

        assert float(roundtrip["x"]) == pytest.approx(4.0, abs=1e-6)
        assert float(roundtrip["y"]) == pytest.approx(0.5, abs=1e-6)

    def test_scaling_log(self):
        """Log scaling should use log/exp transforms."""
        opt = GradientOptimizer(scaling="log")
        params = {"x": jnp.array(10.0)}

        normalized = opt._normalize(params)
        assert float(normalized["x"]) == pytest.approx(jnp.log(10.0), abs=1e-6)

        denormalized = opt._denormalize(normalized)
        assert float(denormalized["x"]) == pytest.approx(10.0, abs=1e-6)

    def test_run_with_bounds_scaling(self):
        """GradientOptimizer should converge with bounds scaling."""

        def loss_fn(params):
            x = params["x"]
            return (x - 5.0) ** 2  # Minimum at x=5

        # Use bounds that include the minimum
        opt = GradientOptimizer(
            algorithm="adam",
            learning_rate=0.1,  # Normal LR works with scaling
            bounds={"x": (0.0, 10.0)},
            scaling="bounds",
        )
        initial_params = {"x": jnp.array(1.0)}

        result = opt.run(loss_fn, initial_params, n_steps=100)

        assert float(result.params["x"]) == pytest.approx(5.0, abs=0.1)

    def test_scaling_reduces_loss(self):
        """Bounds scaling should reduce loss even with small params."""

        def loss_fn(params):
            rate = params["rate"]
            target = 5e-6
            return (rate - target) ** 2

        opt = GradientOptimizer(
            algorithm="adam",
            learning_rate=0.1,
            bounds={"rate": (1e-7, 1e-5)},
            scaling="bounds",
        )
        initial_params = {"rate": jnp.array(1e-6)}
        initial_loss = float(loss_fn(initial_params))

        result = opt.run(loss_fn, initial_params, n_steps=50)

        # Loss should decrease
        assert result.loss < initial_loss

    def test_callback_receives_denormalized_params(self):
        """Callback should receive params in original space."""
        received_params = []

        def callback(_i, params, _loss):
            received_params.append(float(params["x"]))

        def loss_fn(params):
            return (params["x"] - 5.0) ** 2

        opt = GradientOptimizer(
            algorithm="adam",
            learning_rate=0.1,
            bounds={"x": (0.0, 10.0)},
            scaling="bounds",
        )
        initial_params = {"x": jnp.array(2.0)}

        opt.run(loss_fn, initial_params, n_steps=5, callback=callback)

        # All received values should be in original space [0, 10], not [0, 1]
        for val in received_params:
            assert 0.0 <= val <= 10.0

    def test_mixed_bounded_unbounded_params(self):
        """Params with bounds on some keys should pass through unbounded ones."""

        def loss_fn(params):
            return (params["x"] - 5.0) ** 2 + (params["y"] - 3.0) ** 2

        opt = GradientOptimizer(
            algorithm="adam",
            learning_rate=0.3,
            bounds={"x": (0.0, 10.0)},
            scaling="bounds",
        )
        initial_params = {"x": jnp.array(1.0), "y": jnp.array(0.0)}

        result = opt.run(loss_fn, initial_params, n_steps=200)

        assert float(result.params["x"]) == pytest.approx(5.0, abs=0.2)
        assert float(result.params["y"]) == pytest.approx(3.0, abs=0.2)
