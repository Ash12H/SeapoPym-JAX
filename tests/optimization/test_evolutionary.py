"""Tests for the EvolutionaryOptimizer class."""

import jax.numpy as jnp
import pytest

from seapopym.optimization.evolutionary import EvolutionaryOptimizer
from seapopym.optimization.gradient_optimizer import OptimizeResult


class TestEvolutionaryOptimizerInit:
    """Tests for EvolutionaryOptimizer initialization."""

    def test_default_init(self):
        """EvolutionaryOptimizer should initialize with default values."""
        opt = EvolutionaryOptimizer()
        assert opt.strategy_name == "cma_es"
        assert opt.popsize == 32
        assert opt.bounds == {}
        assert opt.seed == 0

    def test_custom_popsize(self):
        """EvolutionaryOptimizer should accept custom population size."""
        opt = EvolutionaryOptimizer(popsize=64)
        assert opt.popsize == 64

    def test_odd_popsize_rounded_up(self):
        """Odd popsize should be rounded up to even (CMA-ES requirement)."""
        opt = EvolutionaryOptimizer(popsize=31)
        assert opt.popsize == 32

    def test_invalid_strategy(self):
        """EvolutionaryOptimizer should reject invalid strategies."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            EvolutionaryOptimizer(strategy="invalid")  # type: ignore[arg-type]

    def test_with_bounds(self):
        """EvolutionaryOptimizer should store bounds."""
        bounds = {"x": (0.0, 1.0), "y": (-1.0, 1.0)}
        opt = EvolutionaryOptimizer(bounds=bounds)
        assert opt.bounds == bounds

    def test_custom_seed(self):
        """EvolutionaryOptimizer should accept custom seed."""
        opt = EvolutionaryOptimizer(seed=42)
        assert opt.seed == 42


class TestFlattenUnflatten:
    """Tests for parameter flattening/unflattening."""

    def test_flatten_single_scalar(self):
        """Flatten should handle single scalar parameter."""
        opt = EvolutionaryOptimizer()
        params = {"x": jnp.array(5.0)}

        keys, flat = opt._flatten(params)

        assert keys == ["x"]
        assert flat.shape == (1,)
        assert float(flat[0]) == pytest.approx(5.0)

    def test_flatten_multiple_params(self):
        """Flatten should concatenate multiple parameters sorted by key."""
        opt = EvolutionaryOptimizer()
        params = {"a": jnp.array(1.0), "b": jnp.array(2.0)}

        keys, flat = opt._flatten(params)

        assert keys == ["a", "b"]
        assert flat.shape == (2,)
        assert float(flat[0]) == pytest.approx(1.0)
        assert float(flat[1]) == pytest.approx(2.0)

    def test_build_bounds_arrays(self):
        """_build_bounds_arrays should return lower/upper matching flat vector."""
        opt = EvolutionaryOptimizer(bounds={"x": (0.0, 10.0)})
        params = {"x": jnp.array(5.0)}

        keys, _ = opt._flatten(params)
        lower, upper = opt._build_bounds_arrays(keys, params)

        assert float(lower[0]) == pytest.approx(0.0)
        assert float(upper[0]) == pytest.approx(10.0)

    def test_unflatten_preserves_scalar(self):
        """Unflatten should preserve scalar type."""
        opt = EvolutionaryOptimizer()
        original = {"x": jnp.array(5.0)}

        result = opt._unflatten(["x"], jnp.array([5.0]), {"x": ()}, original)

        # Should be 0-dim array, not 1-dim
        assert jnp.ndim(result["x"]) == 0

    def test_roundtrip_flatten_unflatten(self):
        """Flatten then unflatten should return original values."""
        opt = EvolutionaryOptimizer()
        params = {"x": jnp.array(3.0), "y": jnp.array(-2.0)}

        keys, flat = opt._flatten(params)
        shapes = {k: jnp.atleast_1d(params[k]).shape for k in keys}
        result = opt._unflatten(keys, flat, shapes, params)

        assert float(result["x"]) == pytest.approx(3.0)
        assert float(result["y"]) == pytest.approx(-2.0)


class TestNormalization:
    """Tests for normalize/denormalize."""

    def test_normalize_maps_to_unit_interval(self):
        """_normalize should map [lower, upper] to [0, 1]."""
        opt = EvolutionaryOptimizer()
        lower = jnp.array([0.0])
        upper = jnp.array([10.0])

        assert float(opt._normalize(jnp.array([0.0]), lower, upper)[0]) == pytest.approx(0.0)
        assert float(opt._normalize(jnp.array([5.0]), lower, upper)[0]) == pytest.approx(0.5)
        assert float(opt._normalize(jnp.array([10.0]), lower, upper)[0]) == pytest.approx(1.0)

    def test_denormalize_roundtrip(self):
        """Normalize then denormalize should return original value."""
        opt = EvolutionaryOptimizer()
        lower = jnp.array([-5.0, 0.0])
        upper = jnp.array([5.0, 100.0])
        original = jnp.array([2.5, 75.0])

        normed = opt._normalize(original, lower, upper)
        restored = opt._denormalize(normed, lower, upper)

        assert jnp.allclose(restored, original)


class TestEvolutionaryOptimizerRun:
    """Tests for full evolutionary optimization run."""

    def test_run_minimizes_quadratic(self):
        """CMA-ES should minimize a simple quadratic."""

        def loss_fn(params):
            x = params["x"]
            return (x - 3.0) ** 2  # Minimum at x=3

        opt = EvolutionaryOptimizer(popsize=16, bounds={"x": (-5.0, 10.0)}, seed=42)
        initial_params = {"x": jnp.array(0.0)}

        result = opt.run(loss_fn, initial_params, n_generations=50)

        assert isinstance(result, OptimizeResult)
        assert float(result.params["x"]) == pytest.approx(3.0, abs=0.5)
        assert result.loss < 0.5

    def test_run_respects_n_generations(self):
        """Optimizer should run for specified number of generations."""

        def loss_fn(params):
            return params["x"] ** 2

        opt = EvolutionaryOptimizer(popsize=8, seed=42)
        initial_params = {"x": jnp.array(10.0)}

        result = opt.run(loss_fn, initial_params, n_generations=20)

        assert result.n_iterations == 20
        assert len(result.loss_history) == 20

    def test_run_with_bounds(self):
        """CMA-ES should respect bounds during optimization."""

        def loss_fn(params):
            x = params["x"]
            return -(x**2)  # Wants to maximize |x|

        opt = EvolutionaryOptimizer(
            popsize=16,
            bounds={"x": (-1.0, 1.0)},
            seed=42,
        )
        initial_params = {"x": jnp.array(0.0)}

        result = opt.run(loss_fn, initial_params, n_generations=30)

        # Should be at or near a bound
        assert abs(float(result.params["x"])) <= 1.0 + 1e-6

    def test_run_multivariate(self):
        """CMA-ES should handle multiple parameters."""

        def loss_fn(params):
            x = params["x"]
            y = params["y"]
            return (x - 1.0) ** 2 + (y - 2.0) ** 2  # Minimum at (1, 2)

        opt = EvolutionaryOptimizer(popsize=32, bounds={"x": (-5.0, 5.0), "y": (-5.0, 5.0)}, seed=42)
        initial_params = {"x": jnp.array(0.0), "y": jnp.array(0.0)}

        result = opt.run(loss_fn, initial_params, n_generations=100)

        assert float(result.params["x"]) == pytest.approx(1.0, abs=0.5)
        assert float(result.params["y"]) == pytest.approx(2.0, abs=0.5)

    def test_run_returns_best_params(self):
        """run should return the best parameters found, not latest."""

        def loss_fn(params):
            x = params["x"]
            return (x - 5.0) ** 2

        opt = EvolutionaryOptimizer(popsize=8, seed=42)
        initial_params = {"x": jnp.array(0.0)}

        result = opt.run(loss_fn, initial_params, n_generations=20)

        # Best loss should match the best params
        expected_loss = (float(result.params["x"]) - 5.0) ** 2
        assert result.loss == pytest.approx(expected_loss, abs=1e-5)

    def test_reproducible_with_seed(self):
        """Same seed should give same results."""

        def loss_fn(params):
            return params["x"] ** 2

        opt1 = EvolutionaryOptimizer(popsize=8, seed=123)
        opt2 = EvolutionaryOptimizer(popsize=8, seed=123)
        initial_params = {"x": jnp.array(5.0)}

        result1 = opt1.run(loss_fn, initial_params, n_generations=10)
        result2 = opt2.run(loss_fn, initial_params, n_generations=10)

        assert float(result1.params["x"]) == pytest.approx(float(result2.params["x"]))
        assert result1.loss == pytest.approx(result2.loss)

    def test_loss_history_decreasing(self):
        """Loss history should generally decrease."""

        def loss_fn(params):
            return params["x"] ** 2

        opt = EvolutionaryOptimizer(popsize=32, seed=42)
        initial_params = {"x": jnp.array(10.0)}

        result = opt.run(loss_fn, initial_params, n_generations=30)

        # First loss should be higher than last
        assert result.loss_history[0] > result.loss_history[-1]
