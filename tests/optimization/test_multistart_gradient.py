"""Tests for the MultiStartGradientOptimizer class."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from seapopym.optimization.gradient_optimizer import OptimizeResult
from seapopym.optimization.multistart_gradient import MultiStartGradientOptimizer, MultiStartResult
from seapopym.optimization.objective import Objective
from seapopym.optimization.prior import Normal, PriorSet, Uniform


class TestMultiStartResult:
    def test_best_returns_min_loss(self):
        results = [
            OptimizeResult(params={"x": jnp.array(1.0)}, loss=5.0),
            OptimizeResult(params={"x": jnp.array(2.0)}, loss=1.0),
            OptimizeResult(params={"x": jnp.array(3.0)}, loss=3.0),
        ]
        ms = MultiStartResult(all_results=results)
        assert ms.best.loss == 1.0
        assert float(ms.best.params["x"]) == 2.0

    def test_best_is_consistent(self):
        """best should always reflect current all_results."""
        results = [
            OptimizeResult(params={"x": jnp.array(1.0)}, loss=5.0),
            OptimizeResult(params={"x": jnp.array(2.0)}, loss=1.0),
        ]
        ms = MultiStartResult(all_results=results)
        assert ms.best is ms.all_results[1]


class TestMultiStartGradientOptimizer:
    def test_minimizes_quadratic(self):
        """N starts should find the minimum of a simple quadratic."""

        def loss_fn(params):
            return (params["x"] - 3.0) ** 2

        obj_dummy = _make_dummy_objective()
        opt = MultiStartGradientOptimizer(
            objectives=[(obj_dummy, "mse", 1.0)],
            bounds={"x": (0.0, 10.0)},
            n_starts=4,
            learning_rate=0.1,
            scaling="bounds",
            seed=42,
        )

        result = opt._run_batched(
            loss_fn,
            _sample_init(opt, ["x"]),
            ["x"],
            n_steps=200,
            tolerance=1e-6,
            patience=200,
            progress_bar=False,
        )

        assert isinstance(result, MultiStartResult)
        assert len(result.all_results) == 4
        assert result.best.loss < 0.1
        assert float(result.best.params["x"]) == pytest.approx(3.0, abs=0.5)

    def test_all_histories_populated(self):
        """Each start should have a non-empty loss history."""

        def loss_fn(params):
            return (params["x"] - 1.0) ** 2

        obj_dummy = _make_dummy_objective()
        opt = MultiStartGradientOptimizer(
            objectives=[(obj_dummy, "mse", 1.0)],
            bounds={"x": (0.0, 5.0)},
            n_starts=3,
            learning_rate=0.05,
            seed=0,
        )

        result = opt._run_batched(
            loss_fn,
            _sample_init(opt, ["x"]),
            ["x"],
            n_steps=50,
            tolerance=0.0,
            patience=50,
            progress_bar=False,
        )

        for r in result.all_results:
            assert len(r.loss_history) == 50

    def test_custom_priors(self):
        """Custom priors should be used for initialization."""

        def loss_fn(params):
            return (params["x"] - 5.0) ** 2

        priors = PriorSet({"x": Normal(loc=5.0, scale=0.1)})
        obj_dummy = _make_dummy_objective()
        opt = MultiStartGradientOptimizer(
            objectives=[(obj_dummy, "mse", 1.0)],
            bounds={"x": (0.0, 10.0)},
            n_starts=4,
            priors=priors,
            learning_rate=0.05,
            seed=0,
        )

        result = opt._run_batched(
            loss_fn,
            _sample_init(opt, ["x"]),
            ["x"],
            n_steps=100,
            tolerance=1e-6,
            patience=100,
            progress_bar=False,
        )

        # With priors centered at 5.0, all starts should converge quickly
        assert result.best.loss < 0.01

    def test_multivariate(self):
        """Multi-start should work with multiple parameters."""

        def loss_fn(params):
            return (params["x"] - 2.0) ** 2 + (params["y"] - 4.0) ** 2

        obj_dummy = _make_dummy_objective()
        opt = MultiStartGradientOptimizer(
            objectives=[(obj_dummy, "mse", 1.0)],
            bounds={"x": (0.0, 10.0), "y": (0.0, 10.0)},
            n_starts=4,
            learning_rate=0.1,
            seed=42,
        )

        keys = sorted(opt.bounds.keys())
        result = opt._run_batched(
            loss_fn,
            _sample_init(opt, keys),
            keys,
            n_steps=200,
            tolerance=1e-6,
            patience=200,
            progress_bar=False,
        )

        assert float(result.best.params["x"]) == pytest.approx(2.0, abs=0.5)
        assert float(result.best.params["y"]) == pytest.approx(4.0, abs=0.5)

    def test_non_scalar_params(self):
        """Parameters with shape > () should be handled correctly."""

        def loss_fn(params):
            return jnp.sum((params["x"] - jnp.array([1.0, 2.0, 3.0])) ** 2)

        obj_dummy = _make_dummy_objective()
        opt = MultiStartGradientOptimizer(
            objectives=[(obj_dummy, "mse", 1.0)],
            bounds={"x": (0.0, 5.0)},
            n_starts=3,
            learning_rate=0.05,
            seed=0,
        )

        # Manually create batched init with shape (3, 3) — 3 starts, 3-element param
        key = jax.random.key(0)
        batched_init = {"x": jax.random.uniform(key, shape=(3, 3), minval=0.0, maxval=5.0)}

        result = opt._run_batched(
            loss_fn,
            batched_init,
            ["x"],
            n_steps=200,
            tolerance=1e-6,
            patience=200,
            progress_bar=False,
        )

        np.testing.assert_allclose(
            np.asarray(result.best.params["x"]),
            np.array([1.0, 2.0, 3.0]),
            atol=0.5,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_objective():
    """Create a dummy objective (not used in _run_batched tests)."""
    return Objective(observations=jnp.zeros(1), transform=lambda o: o["out"])


def _sample_init(opt: MultiStartGradientOptimizer, keys: list[str]) -> dict[str, jnp.ndarray]:
    """Sample initial batched params from the optimizer's priors."""
    priors = opt.priors
    if priors is None:
        priors = PriorSet({name: Uniform(low, high) for name, (low, high) in opt.bounds.items()})
    key = jax.random.key(opt.seed)
    shapes = dict.fromkeys(keys, (opt.n_starts,))
    return priors.sample(key, shapes=shapes)
