"""Tests for the CMAESOptimizer class."""

import jax.numpy as jnp
import numpy as np
import pytest

from seapopym.optimization._common import GenerationResult, OptimizeResult
from seapopym.optimization.cmaes import CMAESOptimizer


class TestCMAESOptimizerInit:
    def test_default_init(self):
        opt = CMAESOptimizer(
            bounds={"x": (0.0, 10.0)},
            initial_params={"x": jnp.array(5.0)},
        )
        assert opt.popsize == 32

    def test_odd_popsize_rounded_up(self):
        opt = CMAESOptimizer(
            bounds={"x": (0.0, 10.0)},
            initial_params={"x": jnp.array(5.0)},
            popsize=31,
        )
        assert opt.popsize == 32


class TestCMAESStep:
    def test_step_returns_generation_result(self):
        opt = CMAESOptimizer(
            bounds={"x": (-5.0, 5.0)},
            initial_params={"x": jnp.array(0.0)},
            popsize=8,
        )
        gen = opt.step(lambda p: p["x"] ** 2)
        assert isinstance(gen, GenerationResult)
        assert gen.gen == 0
        assert gen.n_valid == 8
        assert len(gen.population_params) == 8
        assert len(gen.population_fitness) == 8

    def test_step_increments_gen(self):
        opt = CMAESOptimizer(
            bounds={"x": (-5.0, 5.0)},
            initial_params={"x": jnp.array(0.0)},
            popsize=8,
        )

        def loss_fn(p):
            return p["x"] ** 2

        gen0 = opt.step(loss_fn)
        gen1 = opt.step(loss_fn)
        assert gen0.gen == 0
        assert gen1.gen == 1

    def test_step_nan_penalty(self):
        """Non-finite losses should be replaced by nan_penalty."""

        def bad_loss(params):
            return jnp.where(params["x"] > 0.5, jnp.inf, params["x"] ** 2)

        opt = CMAESOptimizer(
            bounds={"x": (0.0, 1.0)},
            initial_params={"x": jnp.array(0.5)},
            popsize=16,
            nan_penalty=999.0,
        )
        gen = opt.step(bad_loss)
        assert gen.n_valid <= 16  # some may have hit penalty
        assert np.all(np.isfinite(gen.population_fitness))


class TestCMAESRun:
    def test_minimizes_quadratic(self):
        opt = CMAESOptimizer(
            bounds={"x": (-5.0, 10.0)},
            initial_params={"x": jnp.array(0.0)},
            popsize=16,
            seed=42,
        )
        result = opt.run(lambda p: (p["x"] - 3.0) ** 2, max_gen=50)

        assert isinstance(result, OptimizeResult)
        assert float(result.params["x"]) == pytest.approx(3.0, abs=0.5)
        assert result.loss < 0.5

    def test_respects_max_gen(self):
        opt = CMAESOptimizer(
            bounds={"x": (0.0, 10.0)},
            initial_params={"x": jnp.array(10.0)},
            popsize=8,
            seed=42,
        )
        result = opt.run(lambda p: p["x"] ** 2, max_gen=20, patience=999)

        assert result.n_iterations == 20
        assert len(result.loss_history) == 20

    def test_multivariate(self):
        opt = CMAESOptimizer(
            bounds={"x": (-5.0, 5.0), "y": (-5.0, 5.0)},
            initial_params={"x": jnp.array(0.0), "y": jnp.array(0.0)},
            popsize=32,
            seed=42,
        )
        result = opt.run(
            lambda p: (p["x"] - 1.0) ** 2 + (p["y"] - 2.0) ** 2,
            max_gen=100,
        )

        assert float(result.params["x"]) == pytest.approx(1.0, abs=0.5)
        assert float(result.params["y"]) == pytest.approx(2.0, abs=0.5)

    def test_reproducible_with_seed(self):
        def loss_fn(params):
            return params["x"] ** 2

        opt1 = CMAESOptimizer(bounds={"x": (0.0, 10.0)}, initial_params={"x": jnp.array(5.0)}, popsize=8, seed=123)
        opt2 = CMAESOptimizer(bounds={"x": (0.0, 10.0)}, initial_params={"x": jnp.array(5.0)}, popsize=8, seed=123)

        result1 = opt1.run(loss_fn, max_gen=10, patience=999)
        result2 = opt2.run(loss_fn, max_gen=10, patience=999)

        assert float(result1.params["x"]) == pytest.approx(float(result2.params["x"]))
        assert result1.loss == pytest.approx(result2.loss)

    def test_patience_convergence(self):
        opt = CMAESOptimizer(
            bounds={"x": (0.0, 10.0)},
            initial_params={"x": jnp.array(5.0)},
            popsize=8,
            seed=42,
        )
        result = opt.run(lambda p: jnp.array(1.0), max_gen=200, patience=5)

        assert result.converged is True
        assert result.n_iterations < 200

    def test_loss_history_decreasing(self):
        opt = CMAESOptimizer(
            bounds={"x": (-10.0, 20.0)},
            initial_params={"x": jnp.array(-8.0)},
            popsize=32,
            seed=42,
        )
        result = opt.run(lambda p: (p["x"] - 5.0) ** 2, max_gen=30, patience=999)

        assert result.loss_history[0] > result.loss_history[-1]
