"""Tests for the high-level Optimizer orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest

from seapopym.engine.runner import Runner
from seapopym.optimization.objective import Objective
from seapopym.optimization.optimizer import (
    GRADIENT_STRATEGIES,
    METRICS,
    Optimizer,
    _resolve_metric,
)
from seapopym.optimization.prior import PriorSet, Uniform


class _FakeRunner:
    """Simple callable that returns fixed outputs (avoids MagicMock + JAX issues)."""

    def __init__(self, outputs: dict):
        self.outputs = outputs

    def __call__(self, model, free_params):
        return self.outputs


# ---------------------------------------------------------------------------
# _resolve_metric
# ---------------------------------------------------------------------------


class TestResolveMetric:
    def test_known_string(self):
        for name in METRICS:
            fn = _resolve_metric(name)
            assert callable(fn)

    def test_callable_passthrough(self):
        def custom(p, o):
            return jnp.sum((p - o) ** 2)

        assert _resolve_metric(custom) is custom

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            _resolve_metric("unknown_metric")


# ---------------------------------------------------------------------------
# Optimizer init
# ---------------------------------------------------------------------------


class TestOptimizerInit:
    def _runner(self):
        return Runner.optimization()

    def _priors(self):
        return PriorSet({"x": Uniform(0.0, 10.0)})

    def _objectives(self):
        return [
            (
                Objective(observations=jnp.zeros(3), transform=lambda o: o["x"][:3]),
                "mse",
                1.0,
            )
        ]

    def test_valid_gradient_strategy(self):
        for strat in GRADIENT_STRATEGIES:
            opt = Optimizer(
                self._runner(), self._priors(), self._objectives(), strategy=strat
            )
            assert opt.strategy == strat

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            Optimizer(
                self._runner(),
                self._priors(),
                self._objectives(),
                strategy="bogus",
            )

    def test_strategy_kwargs_stored(self):
        opt = Optimizer(
            self._runner(),
            self._priors(),
            self._objectives(),
            strategy="adam",
            learning_rate=0.01,
        )
        assert opt.strategy_kwargs["learning_rate"] == 0.01


# ---------------------------------------------------------------------------
# _build_loss_fn
# ---------------------------------------------------------------------------


class TestBuildLossFn:
    def test_loss_fn_zero_when_pred_equals_obs(self):
        """Loss should be minimal when predictions match observations."""
        priors = PriorSet({"x": Uniform(0.0, 10.0)})

        obs = jnp.array([1.0, 2.0, 3.0])

        def transform(o):
            return o["out"]

        obj = Objective(observations=obs, transform=transform)
        fake_runner = _FakeRunner({"out": jnp.array([1.0, 2.0, 3.0])})
        opt = Optimizer(
            Runner.optimization(), priors, [(obj, "mse", 1.0)], strategy="adam"
        )
        opt.runner = fake_runner

        prepared = obj.setup(model_coords={})
        metric_fn = _resolve_metric("mse")

        model = MagicMock()
        model.parameters = {"x": jnp.array(5.0)}

        loss_fn = opt._build_loss_fn(model, [(prepared, metric_fn, 1.0)])
        loss = loss_fn({"x": jnp.array(5.0)})

        # loss = 0.0 (mse) + penalty
        expected_prior_penalty = -float(priors.log_prob({"x": jnp.array(5.0)}))
        assert float(loss) == pytest.approx(expected_prior_penalty, abs=1e-4)

    def test_loss_fn_weights(self):
        """Weights scale the individual metric contributions."""
        priors = PriorSet({"x": Uniform(0.0, 10.0)})

        obs = jnp.array([0.0, 0.0])

        def transform(o):
            return o["out"]

        obj = Objective(observations=obs, transform=transform)
        prepared = obj.setup(model_coords={})
        metric_fn = _resolve_metric("mse")

        fake_runner = _FakeRunner({"out": jnp.array([1.0, 1.0])})

        opt = Optimizer(
            Runner.optimization(), priors, [(obj, "mse", 2.0)], strategy="adam"
        )
        opt.runner = fake_runner

        model = MagicMock()
        model.parameters = {"x": jnp.array(5.0)}

        loss_w2 = opt._build_loss_fn(model, [(prepared, metric_fn, 2.0)])
        loss_w1 = opt._build_loss_fn(model, [(prepared, metric_fn, 1.0)])

        params = {"x": jnp.array(5.0)}
        v2 = float(loss_w2(params))
        v1 = float(loss_w1(params))

        # v2 - v1 = (2 - 1) * mse = 1.0
        assert (v2 - v1) == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# _dispatch routing
# ---------------------------------------------------------------------------


class TestDispatchRouting:
    def test_dispatch_gradient(self):
        """Gradient strategies route to GradientOptimizer."""
        priors = PriorSet({"x": Uniform(0.0, 10.0)})
        runner = Runner.optimization()
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["x"][:1])
        opt = Optimizer(runner, priors, [(obj, "mse", 1.0)], strategy="adam")

        with patch("seapopym.optimization.optimizer.GradientOptimizer") as MockGO:
            mock_instance = MagicMock()
            mock_instance.run.return_value = MagicMock()
            MockGO.return_value = mock_instance

            def loss_fn(p):
                return jnp.array(0.0)

            opt._dispatch(loss_fn, {"x": jnp.array(5.0)})

            MockGO.assert_called_once()
            mock_instance.run.assert_called_once()

    def test_dispatch_evolutionary(self):
        """Evolutionary strategies route to EvolutionaryOptimizer."""
        priors = PriorSet({"x": Uniform(0.0, 10.0)})
        runner = Runner.optimization()
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["x"][:1])
        opt = Optimizer(runner, priors, [(obj, "mse", 1.0)], strategy="cma_es")

        with patch("seapopym.optimization.evolutionary.EvolutionaryOptimizer") as MockEO:
            mock_instance = MagicMock()
            mock_instance.run.return_value = MagicMock()
            MockEO.return_value = mock_instance

            def loss_fn(p):
                return jnp.array(0.0)

            opt._dispatch(loss_fn, {"x": jnp.array(5.0)})

            MockEO.assert_called_once()
