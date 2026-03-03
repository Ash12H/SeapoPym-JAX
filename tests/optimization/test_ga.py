"""Tests for the GAOptimizer class."""

import jax.numpy as jnp
import pytest

from seapopym.optimization.ga import GAOptimizer
from seapopym.optimization.gradient_optimizer import OptimizeResult
from seapopym.optimization.objective import Objective
from seapopym.optimization.prior import PriorSet, Uniform


class _FakeRunner:
    def __call__(self, model, free_params):
        return {"out": jnp.array([0.0])}


class TestGAOptimizerInit:
    def test_default_init(self):
        runner = _FakeRunner()
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        opt = GAOptimizer(runner, [(obj, "mse", 1.0)], bounds={"x": (0.0, 10.0)})
        assert opt.popsize == 32
        assert opt.seed == 0
        assert opt.crossover_rate == 0.5
        assert opt.mutation_std == 0.1

    def test_no_default_priors(self):
        runner = _FakeRunner()
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        opt = GAOptimizer(runner, [(obj, "mse", 1.0)], bounds={"x": (0.0, 10.0)})
        assert opt.priors is None

    def test_custom_priors(self):
        runner = _FakeRunner()
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        priors = PriorSet({"x": Uniform(1.0, 5.0)})
        opt = GAOptimizer(runner, [(obj, "mse", 1.0)], bounds={"x": (0.0, 10.0)}, priors=priors)
        assert opt.priors is priors


class TestGAOptimizerRunLossFn:
    def test_minimizes_quadratic(self):
        """SimpleGA should minimize a simple quadratic via _run_loss_fn."""

        def loss_fn(params):
            x = params["x"]
            return (x - 3.0) ** 2

        runner = _FakeRunner()
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        opt = GAOptimizer(
            runner, [(obj, "mse", 1.0)],
            bounds={"x": (-5.0, 10.0)},
            popsize=64, seed=42,
        )

        result = opt._run_loss_fn(loss_fn, {"x": jnp.array(0.0)}, n_generations=100)

        assert isinstance(result, OptimizeResult)
        assert float(result.params["x"]) == pytest.approx(3.0, abs=1.0)
        assert result.loss < 2.0

    def test_respects_n_generations(self):
        def loss_fn(params):
            return params["x"] ** 2

        runner = _FakeRunner()
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        opt = GAOptimizer(runner, [(obj, "mse", 1.0)], bounds={"x": (0.0, 10.0)}, popsize=8, seed=42)

        result = opt._run_loss_fn(loss_fn, {"x": jnp.array(10.0)}, n_generations=20)

        assert result.n_iterations == 20
        assert len(result.loss_history) == 20

    def test_respects_bounds(self):
        def loss_fn(params):
            return -(params["x"] ** 2)

        runner = _FakeRunner()
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        opt = GAOptimizer(
            runner, [(obj, "mse", 1.0)],
            bounds={"x": (-1.0, 1.0)},
            popsize=32, seed=42,
        )

        result = opt._run_loss_fn(loss_fn, {"x": jnp.array(0.0)}, n_generations=50)

        assert abs(float(result.params["x"])) <= 1.0 + 1e-6

    def test_patience_convergence(self):
        """GA should set converged=True when patience is exhausted."""

        def loss_fn(params):
            return jnp.array(1.0)  # constant → never improves

        runner = _FakeRunner()
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        opt = GAOptimizer(
            runner, [(obj, "mse", 1.0)],
            bounds={"x": (0.0, 10.0)},
            popsize=8, seed=42,
        )

        result = opt._run_loss_fn(loss_fn, {"x": jnp.array(5.0)}, n_generations=200, patience=5)

        assert result.converged is True
        assert result.n_iterations < 200

    def test_reproducible_with_seed(self):
        def loss_fn(params):
            return params["x"] ** 2

        runner = _FakeRunner()
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        opt1 = GAOptimizer(runner, [(obj, "mse", 1.0)], bounds={"x": (0.0, 10.0)}, popsize=8, seed=123)
        opt2 = GAOptimizer(runner, [(obj, "mse", 1.0)], bounds={"x": (0.0, 10.0)}, popsize=8, seed=123)

        result1 = opt1._run_loss_fn(loss_fn, {"x": jnp.array(5.0)}, n_generations=10)
        result2 = opt2._run_loss_fn(loss_fn, {"x": jnp.array(5.0)}, n_generations=10)

        assert float(result1.params["x"]) == pytest.approx(float(result2.params["x"]))
        assert result1.loss == pytest.approx(result2.loss)
