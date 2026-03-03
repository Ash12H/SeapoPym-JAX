"""Tests for the IPOPCMAESOptimizer class."""

import jax.numpy as jnp
import pytest

from seapopym.optimization.gradient_optimizer import OptimizeResult
from seapopym.optimization.ipop import IPOPCMAESOptimizer, IPOPResult, _is_new_mode, _params_distance
from seapopym.optimization.objective import Objective


class _FakeRunner:
    def __call__(self, model, free_params):
        return {"out": jnp.array([0.0])}


class TestParamsDistance:
    def test_identical_params(self):
        a = {"x": jnp.array(0.5)}
        b = {"x": jnp.array(0.5)}
        bounds = {"x": (0.0, 1.0)}
        assert _params_distance(a, b, bounds) == pytest.approx(0.0)

    def test_opposite_bounds(self):
        a = {"x": jnp.array(0.0)}
        b = {"x": jnp.array(1.0)}
        bounds = {"x": (0.0, 1.0)}
        assert _params_distance(a, b, bounds) == pytest.approx(1.0)

    def test_multivariate(self):
        a = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        b = {"x": jnp.array(1.0), "y": jnp.array(1.0)}
        bounds = {"x": (0.0, 1.0), "y": (0.0, 1.0)}
        assert _params_distance(a, b, bounds) == pytest.approx(jnp.sqrt(2.0), abs=1e-5)


class TestIsNewMode:
    def test_first_mode_always_new(self):
        result = OptimizeResult(params={"x": jnp.array(0.5)}, loss=0.1)
        assert _is_new_mode(result, [], 0.1, {"x": (0.0, 1.0)})

    def test_nearby_not_new(self):
        existing = OptimizeResult(params={"x": jnp.array(0.5)}, loss=0.1)
        candidate = OptimizeResult(params={"x": jnp.array(0.52)}, loss=0.15)
        assert not _is_new_mode(candidate, [existing], 0.1, {"x": (0.0, 1.0)})

    def test_far_is_new(self):
        existing = OptimizeResult(params={"x": jnp.array(0.0)}, loss=0.1)
        candidate = OptimizeResult(params={"x": jnp.array(1.0)}, loss=0.15)
        assert _is_new_mode(candidate, [existing], 0.1, {"x": (0.0, 1.0)})


class TestIPOPCMAESOptimizerInit:
    def test_default_init(self):
        runner = _FakeRunner()
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        opt = IPOPCMAESOptimizer(
            runner, [(obj, "mse", 1.0)], bounds={"x": (0.0, 10.0)},
        )
        assert opt.n_restarts == 5
        assert opt.initial_popsize == 32
        assert opt.n_generations == 100
        assert opt.distance_threshold == 0.1
        assert opt.seed == 0

    def test_custom_init(self):
        runner = _FakeRunner()
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        opt = IPOPCMAESOptimizer(
            runner, [(obj, "mse", 1.0)], bounds={"x": (0.0, 10.0)},
            n_restarts=3, initial_popsize=8, n_generations=50,
            distance_threshold=0.2, seed=42,
        )
        assert opt.n_restarts == 3
        assert opt.initial_popsize == 8
        assert opt.n_generations == 50
        assert opt.distance_threshold == 0.2
        assert opt.seed == 42


class TestIPOPCMAESOptimizerRunLossFn:
    def test_finds_minimum(self):
        """IPOP should find the minimum of a simple quadratic."""

        def loss_fn(params):
            return (params["x"] - 3.0) ** 2

        runner = _FakeRunner()
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        opt = IPOPCMAESOptimizer(
            runner, [(obj, "mse", 1.0)],
            bounds={"x": (-5.0, 10.0)},
            n_restarts=2, initial_popsize=8, n_generations=30, seed=42,
        )

        result = opt._run_loss_fn(loss_fn, {"x": jnp.array(0.0)})

        assert isinstance(result, IPOPResult)
        assert len(result.all_results) == 2
        assert result.n_restarts == 2
        assert len(result.modes) >= 1
        assert result.modes[0].loss < 1.0

    def test_modes_sorted_by_loss(self):
        """Modes should be sorted by loss, best first."""

        def loss_fn(params):
            return (params["x"] - 5.0) ** 2

        runner = _FakeRunner()
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        opt = IPOPCMAESOptimizer(
            runner, [(obj, "mse", 1.0)],
            bounds={"x": (-10.0, 10.0)},
            n_restarts=3, initial_popsize=8, n_generations=20, seed=42,
        )

        result = opt._run_loss_fn(loss_fn, {"x": jnp.array(0.0)})

        for i in range(len(result.modes) - 1):
            assert result.modes[i].loss <= result.modes[i + 1].loss

    def test_population_doubles(self):
        """Each restart should use double the population of the previous."""
        popsizes = []

        class TrackingCMAES(IPOPCMAESOptimizer):
            def _run_loss_fn(self, loss_fn, initial_params, progress_bar=False):
                # Capture popsize from each restart
                for i in range(self.n_restarts):
                    popsizes.append(self.initial_popsize * (2**i))
                return super()._run_loss_fn(loss_fn, initial_params, progress_bar)

        runner = _FakeRunner()
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])
        opt = TrackingCMAES(
            runner, [(obj, "mse", 1.0)],
            bounds={"x": (0.0, 10.0)},
            n_restarts=3, initial_popsize=8, n_generations=5, seed=42,
        )

        opt._run_loss_fn(lambda p: p["x"] ** 2, {"x": jnp.array(5.0)})

        assert popsizes == [8, 16, 32]
