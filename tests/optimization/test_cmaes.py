"""Tests for the CMAESOptimizer class."""

import jax.numpy as jnp
import pytest

from seapopym.optimization.cmaes import CMAESOptimizer
from seapopym.optimization.gradient_optimizer import OptimizeResult
from seapopym.optimization.objective import Objective
from seapopym.optimization.prior import PriorSet, Uniform


class _FakeRunner:
    """Simple callable that returns outputs from a loss-like function."""

    def __init__(self, output_fn=None):
        self.output_fn = output_fn

    def __call__(self, model, free_params):
        if self.output_fn is not None:
            return self.output_fn(free_params)
        return {"out": jnp.array([0.0])}


class _FakeModel:
    """Minimal model stub."""

    def __init__(self, params):
        self.parameters = params
        self.coords = {}


class TestCMAESOptimizerInit:
    def test_default_init(self):
        runner = _FakeRunner()
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        opt = CMAESOptimizer(runner, [(obj, "mse", 1.0)], bounds={"x": (0.0, 10.0)})
        assert opt.popsize == 32
        assert opt.seed == 0

    def test_odd_popsize_rounded_up(self):
        runner = _FakeRunner()
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        opt = CMAESOptimizer(runner, [(obj, "mse", 1.0)], bounds={"x": (0.0, 10.0)}, popsize=31)
        assert opt.popsize == 32

    def test_default_priors_from_bounds(self):
        runner = _FakeRunner()
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        opt = CMAESOptimizer(runner, [(obj, "mse", 1.0)], bounds={"x": (0.0, 10.0)})
        assert "x" in opt.priors.priors
        assert opt.priors.priors["x"].bounds == (0.0, 10.0)

    def test_custom_priors(self):
        runner = _FakeRunner()
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        priors = PriorSet({"x": Uniform(1.0, 5.0)})
        opt = CMAESOptimizer(runner, [(obj, "mse", 1.0)], bounds={"x": (0.0, 10.0)}, priors=priors)
        assert opt.priors is priors


class TestCMAESOptimizerRunLossFn:
    def test_minimizes_quadratic(self):
        """CMA-ES should minimize a simple quadratic via _run_loss_fn."""

        def loss_fn(params):
            x = params["x"]
            return (x - 3.0) ** 2

        runner = _FakeRunner()
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        opt = CMAESOptimizer(runner, [(obj, "mse", 1.0)], bounds={"x": (-5.0, 10.0)}, popsize=16, seed=42)

        result = opt._run_loss_fn(loss_fn, {"x": jnp.array(0.0)}, n_generations=50)

        assert isinstance(result, OptimizeResult)
        assert float(result.params["x"]) == pytest.approx(3.0, abs=0.5)
        assert result.loss < 0.5

    def test_respects_n_generations(self):
        def loss_fn(params):
            return params["x"] ** 2

        runner = _FakeRunner()
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        opt = CMAESOptimizer(runner, [(obj, "mse", 1.0)], bounds={"x": (0.0, 10.0)}, popsize=8, seed=42)

        result = opt._run_loss_fn(loss_fn, {"x": jnp.array(10.0)}, n_generations=20)

        assert result.n_iterations == 20
        assert len(result.loss_history) == 20

    def test_respects_bounds(self):
        def loss_fn(params):
            return -(params["x"] ** 2)

        runner = _FakeRunner()
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        opt = CMAESOptimizer(runner, [(obj, "mse", 1.0)], bounds={"x": (-1.0, 1.0)}, popsize=16, seed=42)

        result = opt._run_loss_fn(loss_fn, {"x": jnp.array(0.0)}, n_generations=30)

        assert abs(float(result.params["x"])) <= 1.0 + 1e-6

    def test_multivariate(self):
        def loss_fn(params):
            return (params["x"] - 1.0) ** 2 + (params["y"] - 2.0) ** 2

        runner = _FakeRunner()
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        opt = CMAESOptimizer(
            runner, [(obj, "mse", 1.0)],
            bounds={"x": (-5.0, 5.0), "y": (-5.0, 5.0)},
            popsize=32, seed=42,
        )

        result = opt._run_loss_fn(
            loss_fn, {"x": jnp.array(0.0), "y": jnp.array(0.0)}, n_generations=100,
        )

        assert float(result.params["x"]) == pytest.approx(1.0, abs=0.5)
        assert float(result.params["y"]) == pytest.approx(2.0, abs=0.5)

    def test_reproducible_with_seed(self):
        def loss_fn(params):
            return params["x"] ** 2

        runner = _FakeRunner()
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        opt1 = CMAESOptimizer(runner, [(obj, "mse", 1.0)], bounds={"x": (0.0, 10.0)}, popsize=8, seed=123)
        opt2 = CMAESOptimizer(runner, [(obj, "mse", 1.0)], bounds={"x": (0.0, 10.0)}, popsize=8, seed=123)

        result1 = opt1._run_loss_fn(loss_fn, {"x": jnp.array(5.0)}, n_generations=10)
        result2 = opt2._run_loss_fn(loss_fn, {"x": jnp.array(5.0)}, n_generations=10)

        assert float(result1.params["x"]) == pytest.approx(float(result2.params["x"]))
        assert result1.loss == pytest.approx(result2.loss)

    def test_patience_convergence(self):
        """CMA-ES should set converged=True when patience is exhausted."""

        def loss_fn(params):
            return jnp.array(1.0)  # constant → never improves

        runner = _FakeRunner()
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        opt = CMAESOptimizer(
            runner, [(obj, "mse", 1.0)],
            bounds={"x": (0.0, 10.0)},
            popsize=8, seed=42,
        )

        result = opt._run_loss_fn(loss_fn, {"x": jnp.array(5.0)}, n_generations=200, patience=5)

        assert result.converged is True
        assert result.n_iterations < 200

    def test_loss_history_decreasing(self):
        def loss_fn(params):
            return (params["x"] - 5.0) ** 2

        runner = _FakeRunner()
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        opt = CMAESOptimizer(runner, [(obj, "mse", 1.0)], bounds={"x": (-10.0, 20.0)}, popsize=32, seed=42)

        result = opt._run_loss_fn(loss_fn, {"x": jnp.array(-8.0)}, n_generations=30)

        assert result.loss_history[0] > result.loss_history[-1]
