"""Tests for the IPOPCMAESOptimizer class."""

import jax.numpy as jnp
import pytest

from seapopym.optimization._common import OptimizeResult, params_distance
from seapopym.optimization.ipop import IPOPCMAESOptimizer, IPOPResult, _is_new_mode


class TestParamsDistance:
    def test_identical_params(self):
        a = {"x": jnp.array(0.5)}
        b = {"x": jnp.array(0.5)}
        bounds = {"x": (0.0, 1.0)}
        assert params_distance(a, b, bounds) == pytest.approx(0.0)

    def test_opposite_bounds(self):
        a = {"x": jnp.array(0.0)}
        b = {"x": jnp.array(1.0)}
        bounds = {"x": (0.0, 1.0)}
        assert params_distance(a, b, bounds) == pytest.approx(1.0)

    def test_multivariate(self):
        a = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        b = {"x": jnp.array(1.0), "y": jnp.array(1.0)}
        bounds = {"x": (0.0, 1.0), "y": (0.0, 1.0)}
        assert params_distance(a, b, bounds) == pytest.approx(jnp.sqrt(2.0), abs=1e-5)


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
        opt = IPOPCMAESOptimizer(
            bounds={"x": (0.0, 10.0)},
            initial_params={"x": jnp.array(5.0)},
        )
        assert opt.n_restarts == 5
        assert opt.initial_popsize == 32
        assert opt.n_generations == 100
        assert opt.distance_threshold == 0.1
        assert opt.seed == 0

    def test_custom_init(self):
        opt = IPOPCMAESOptimizer(
            bounds={"x": (0.0, 10.0)},
            initial_params={"x": jnp.array(5.0)},
            n_restarts=3,
            initial_popsize=8,
            n_generations=50,
            distance_threshold=0.2,
            seed=42,
        )
        assert opt.n_restarts == 3
        assert opt.initial_popsize == 8
        assert opt.n_generations == 50
        assert opt.distance_threshold == 0.2
        assert opt.seed == 42


class TestIPOPCMAESOptimizerRun:
    def test_finds_minimum(self):
        opt = IPOPCMAESOptimizer(
            bounds={"x": (-5.0, 10.0)},
            initial_params={"x": jnp.array(0.0)},
            n_restarts=2,
            initial_popsize=8,
            n_generations=30,
            seed=42,
        )
        result = opt.run(lambda p: (p["x"] - 3.0) ** 2)

        assert isinstance(result, IPOPResult)
        assert len(result.all_results) == 2
        assert result.n_restarts == 2
        assert len(result.modes) >= 1
        assert result.modes[0].loss < 1.0

    def test_modes_sorted_by_loss(self):
        opt = IPOPCMAESOptimizer(
            bounds={"x": (-10.0, 10.0)},
            initial_params={"x": jnp.array(0.0)},
            n_restarts=3,
            initial_popsize=8,
            n_generations=20,
            seed=42,
        )
        result = opt.run(lambda p: (p["x"] - 5.0) ** 2)

        for i in range(len(result.modes) - 1):
            assert result.modes[i].loss <= result.modes[i + 1].loss
