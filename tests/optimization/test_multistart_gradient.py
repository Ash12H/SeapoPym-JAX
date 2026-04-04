"""Tests for the MultiStartGradientOptimizer class."""

import jax.numpy as jnp
import numpy as np
import pytest

from seapopym.optimization._common import OptimizeResult
from seapopym.optimization.multistart_gradient import MultiStartGradientOptimizer, MultiStartResult
from seapopym.optimization.prior import Normal, PriorSet


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
        results = [
            OptimizeResult(params={"x": jnp.array(1.0)}, loss=5.0),
            OptimizeResult(params={"x": jnp.array(2.0)}, loss=1.0),
        ]
        ms = MultiStartResult(all_results=results)
        assert ms.best is ms.all_results[1]


class TestMultiStartGradientOptimizer:
    def test_minimizes_quadratic(self):
        opt = MultiStartGradientOptimizer(
            bounds={"x": (0.0, 10.0)},
            n_starts=4,
            learning_rate=0.1,
            scaling="bounds",
            seed=42,
        )
        result = opt.run(
            lambda p: (p["x"] - 3.0) ** 2,
            max_steps=200,
            param_shapes={"x": ()},
        )

        assert isinstance(result, MultiStartResult)
        assert len(result.all_results) == 4
        assert result.best.loss < 0.1
        assert float(result.best.params["x"]) == pytest.approx(3.0, abs=0.5)

    def test_all_histories_populated(self):
        opt = MultiStartGradientOptimizer(
            bounds={"x": (0.0, 5.0)},
            n_starts=3,
            learning_rate=0.05,
            seed=0,
        )
        result = opt.run(
            lambda p: (p["x"] - 1.0) ** 2,
            max_steps=50,
            tolerance=0.0,
            patience=50,
            param_shapes={"x": ()},
        )

        for r in result.all_results:
            assert len(r.loss_history) == 50

    def test_custom_priors(self):
        priors = PriorSet({"x": Normal(loc=5.0, scale=0.1)})
        opt = MultiStartGradientOptimizer(
            bounds={"x": (0.0, 10.0)},
            n_starts=4,
            priors=priors,
            learning_rate=0.05,
            seed=0,
        )
        result = opt.run(
            lambda p: (p["x"] - 5.0) ** 2,
            max_steps=100,
            param_shapes={"x": ()},
        )

        assert result.best.loss < 0.01

    def test_multivariate(self):
        opt = MultiStartGradientOptimizer(
            bounds={"x": (0.0, 10.0), "y": (0.0, 10.0)},
            n_starts=4,
            learning_rate=0.1,
            seed=42,
        )
        result = opt.run(
            lambda p: (p["x"] - 2.0) ** 2 + (p["y"] - 4.0) ** 2,
            max_steps=200,
            param_shapes={"x": (), "y": ()},
        )

        assert float(result.best.params["x"]) == pytest.approx(2.0, abs=0.5)
        assert float(result.best.params["y"]) == pytest.approx(4.0, abs=0.5)

    def test_non_scalar_params(self):
        opt = MultiStartGradientOptimizer(
            bounds={"x": (0.0, 5.0)},
            n_starts=3,
            learning_rate=0.05,
            seed=0,
        )
        result = opt.run(
            lambda p: jnp.sum((p["x"] - jnp.array([1.0, 2.0, 3.0])) ** 2),
            max_steps=200,
            param_shapes={"x": (3,)},
        )

        np.testing.assert_allclose(
            np.asarray(result.best.params["x"]),
            np.array([1.0, 2.0, 3.0]),
            atol=0.5,
        )

    def test_requires_param_shapes(self):
        opt = MultiStartGradientOptimizer(
            bounds={"x": (0.0, 10.0)},
            n_starts=2,
        )
        with pytest.raises(ValueError, match="param_shapes"):
            opt.run(lambda p: p["x"] ** 2, max_steps=10)
