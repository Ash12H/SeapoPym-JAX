"""Tests for the HybridOptimizer class."""

import jax.numpy as jnp
import pytest

from seapopym.optimization.hybrid import HybridOptimizer
from seapopym.optimization.optimizer import OptimizeResult


class TestHybridOptimizerInit:
    """Tests for HybridOptimizer initialization."""

    def test_default_init(self):
        """HybridOptimizer should initialize with default values."""
        opt = HybridOptimizer()
        assert opt.popsize == 32
        assert opt.top_k == 5
        assert opt.bounds == {}
        assert opt.gradient_steps == 50
        assert opt.gradient_lr == 0.1
        assert opt.seed == 0

    def test_custom_popsize(self):
        """HybridOptimizer should accept custom population size."""
        opt = HybridOptimizer(popsize=64)
        assert opt.popsize == 64

    def test_custom_top_k(self):
        """HybridOptimizer should accept custom top_k."""
        opt = HybridOptimizer(top_k=10)
        assert opt.top_k == 10

    def test_with_bounds(self):
        """HybridOptimizer should store bounds."""
        bounds = {"x": (0.0, 1.0), "y": (-1.0, 1.0)}
        opt = HybridOptimizer(bounds=bounds)
        assert opt.bounds == bounds

    def test_custom_gradient_params(self):
        """HybridOptimizer should accept gradient parameters."""
        opt = HybridOptimizer(gradient_steps=100, gradient_lr=0.05)
        assert opt.gradient_steps == 100
        assert opt.gradient_lr == 0.05

    def test_parallel_gradients_default(self):
        """parallel_gradients should default to top_k."""
        opt = HybridOptimizer(top_k=7)
        assert opt.parallel_gradients == 7

    def test_parallel_gradients_custom(self):
        """parallel_gradients should accept custom value."""
        opt = HybridOptimizer(top_k=10, parallel_gradients=3)
        assert opt.parallel_gradients == 3


class TestHybridOptimizerRun:
    """Tests for full hybrid optimization run."""

    def test_run_minimizes_quadratic(self):
        """Hybrid optimizer should minimize a simple quadratic."""

        def loss_fn(params):
            x = params["x"]
            return (x - 3.0) ** 2  # Minimum at x=3

        opt = HybridOptimizer(
            popsize=16,
            top_k=3,
            bounds={"x": (-5.0, 10.0)},
            gradient_steps=20,
            seed=42,
        )
        initial_params = {"x": jnp.array(0.0)}

        result = opt.run(loss_fn, initial_params, n_generations=30)

        assert isinstance(result, OptimizeResult)
        assert float(result.params["x"]) == pytest.approx(3.0, abs=0.5)
        assert result.loss < 0.5

    def test_run_with_bounds(self):
        """Hybrid optimizer should respect bounds."""

        def loss_fn(params):
            x = params["x"]
            return (x - 5.0) ** 2  # Minimum at x=5

        opt = HybridOptimizer(
            popsize=16,
            top_k=3,
            bounds={"x": (0.0, 10.0)},
            gradient_steps=20,
            seed=42,
        )
        initial_params = {"x": jnp.array(0.0)}

        result = opt.run(loss_fn, initial_params, n_generations=30)

        # Result should be within bounds
        assert 0.0 <= float(result.params["x"]) <= 10.0
        # And close to optimal
        assert float(result.params["x"]) == pytest.approx(5.0, abs=0.5)

    def test_run_multivariate(self):
        """Hybrid optimizer should handle multiple parameters."""

        def loss_fn(params):
            x = params["x"]
            y = params["y"]
            return (x - 1.0) ** 2 + (y - 2.0) ** 2  # Minimum at (1, 2)

        opt = HybridOptimizer(
            popsize=32,
            top_k=5,
            gradient_steps=30,
            seed=42,
        )
        initial_params = {"x": jnp.array(0.0), "y": jnp.array(0.0)}

        result = opt.run(loss_fn, initial_params, n_generations=50)

        assert float(result.params["x"]) == pytest.approx(1.0, abs=0.2)
        assert float(result.params["y"]) == pytest.approx(2.0, abs=0.2)

    def test_run_returns_optimize_result(self):
        """run should return a proper OptimizeResult."""

        def loss_fn(params):
            return params["x"] ** 2

        opt = HybridOptimizer(popsize=8, top_k=2, gradient_steps=10, seed=42)
        initial_params = {"x": jnp.array(5.0)}

        result = opt.run(loss_fn, initial_params, n_generations=10)

        assert isinstance(result, OptimizeResult)
        assert result.params is not None
        assert result.loss is not None
        assert result.loss_history is not None
        assert result.n_iterations is not None
        assert result.message is not None

    def test_loss_history_combined(self):
        """Loss history should include both CMA-ES and gradient phases."""

        def loss_fn(params):
            return params["x"] ** 2

        opt = HybridOptimizer(
            popsize=8,
            top_k=2,
            gradient_steps=10,
            seed=42,
        )
        initial_params = {"x": jnp.array(5.0)}

        n_generations = 15
        result = opt.run(loss_fn, initial_params, n_generations=n_generations)

        # History should have CMA-ES entries + gradient entries
        # CMA-ES: n_generations entries
        # Gradient: up to gradient_steps entries (may converge early)
        assert len(result.loss_history) >= n_generations

    def test_n_iterations_combined(self):
        """n_iterations should reflect both phases."""

        def loss_fn(params):
            return params["x"] ** 2

        opt = HybridOptimizer(
            popsize=8,
            top_k=2,
            gradient_steps=20,
            seed=42,
        )
        initial_params = {"x": jnp.array(5.0)}

        n_generations = 25
        result = opt.run(loss_fn, initial_params, n_generations=n_generations)

        assert result.n_iterations == n_generations + opt.gradient_steps

    def test_message_describes_phases(self):
        """Message should describe both phases."""

        def loss_fn(params):
            return params["x"] ** 2

        opt = HybridOptimizer(popsize=8, top_k=2, gradient_steps=10, seed=42)
        initial_params = {"x": jnp.array(5.0)}

        result = opt.run(loss_fn, initial_params, n_generations=15)

        assert "CMA-ES" in result.message
        assert "15 gen" in result.message
        assert "Gradient" in result.message
        assert "10 steps" in result.message

    def test_reproducible_with_seed(self):
        """Same seed should give same results."""

        def loss_fn(params):
            return params["x"] ** 2

        opt1 = HybridOptimizer(popsize=8, top_k=2, gradient_steps=10, seed=123)
        opt2 = HybridOptimizer(popsize=8, top_k=2, gradient_steps=10, seed=123)
        initial_params = {"x": jnp.array(5.0)}

        result1 = opt1.run(loss_fn, initial_params, n_generations=10)
        result2 = opt2.run(loss_fn, initial_params, n_generations=10)

        assert float(result1.params["x"]) == pytest.approx(float(result2.params["x"]))
        assert result1.loss == pytest.approx(result2.loss)

    def test_hybrid_achieves_good_result(self):
        """Hybrid should achieve a good result combining both methods."""

        def loss_fn(params):
            x = params["x"]
            return (x - 3.0) ** 2

        hybrid_opt = HybridOptimizer(
            popsize=16,
            top_k=3,
            bounds={"x": (-5.0, 10.0)},
            gradient_steps=30,
            seed=42,
        )
        initial_params = {"x": jnp.array(0.0)}
        result = hybrid_opt.run(loss_fn, initial_params, n_generations=20)

        # Hybrid should achieve good convergence
        assert result.loss < 0.5
        assert float(result.params["x"]) == pytest.approx(3.0, abs=0.5)


class TestHybridTopKSelection:
    """Tests for top K candidate selection."""

    def test_top_k_selects_best(self):
        """Top K should select the best candidates from population."""

        def loss_fn(params):
            return params["x"] ** 2

        # Use small population so we can verify selection
        opt = HybridOptimizer(
            popsize=8,
            top_k=3,
            gradient_steps=5,
            seed=42,
        )
        initial_params = {"x": jnp.array(10.0)}

        result = opt.run(loss_fn, initial_params, n_generations=5)

        # Should still converge reasonably (validates top K worked)
        assert result.loss < 100  # Started at 100, should improve

    def test_top_k_larger_than_popsize_clamped(self):
        """top_k larger than popsize should work (effectively uses whole pop)."""

        def loss_fn(params):
            return params["x"] ** 2

        opt = HybridOptimizer(
            popsize=8,
            top_k=20,  # Larger than popsize
            gradient_steps=5,
            seed=42,
        )
        initial_params = {"x": jnp.array(5.0)}

        # Should not raise, just use available candidates
        result = opt.run(loss_fn, initial_params, n_generations=5)
        assert result.loss is not None
