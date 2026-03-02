"""Tests for CalibrationRunner."""

from __future__ import annotations

from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np
import pytest

from seapopym.optimization.runner import CalibrationRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_model(fixed_params: dict, run_return=None):
    """Create a mock CompiledModel.

    ``run_with_params`` records the merged params it receives and returns
    ``(state, outputs)``.
    """
    model = MagicMock()
    model.parameters = fixed_params

    if run_return is None:
        run_return = ({}, {"biomass": jnp.ones(3)})

    def _run(merged):
        model._last_merged = merged
        return run_return

    model.run_with_params = MagicMock(side_effect=_run)
    return model


# ---------------------------------------------------------------------------
# Factory methods
# ---------------------------------------------------------------------------


class TestFactoryMethods:
    def test_standard_no_vmap(self):
        runner = CalibrationRunner.standard()
        assert runner.use_vmap is False

    def test_vmapped_has_vmap(self):
        runner = CalibrationRunner.vmapped()
        assert runner.use_vmap is True

    def test_frozen(self):
        runner = CalibrationRunner.standard()
        with pytest.raises(AttributeError):
            runner.use_vmap = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


class TestSingleRun:
    def test_merges_params(self):
        """Free params override fixed params; other fixed params are kept."""
        model = _mock_model({"a": jnp.array(1.0), "b": jnp.array(2.0)})
        runner = CalibrationRunner.standard()

        free_params = {"a": jnp.array(99.0)}
        runner(model, free_params)

        merged = model._last_merged
        assert float(merged["a"]) == 99.0  # Overridden
        assert float(merged["b"]) == 2.0   # Kept

    def test_returns_outputs(self):
        """Runner returns the outputs dict from run_with_params."""
        expected_outputs = {"biomass": jnp.array([1.0, 2.0])}
        model = _mock_model({}, run_return=({}, expected_outputs))
        runner = CalibrationRunner.standard()

        outputs = runner(model, {})
        np.testing.assert_array_equal(
            np.asarray(outputs["biomass"]),
            np.asarray(expected_outputs["biomass"]),
        )

    def test_free_params_override_all(self):
        """When all params are free, fixed params are fully overridden."""
        model = _mock_model({"x": jnp.array(0.0)})
        runner = CalibrationRunner.standard()

        free = {"x": jnp.array(5.0)}
        runner(model, free)

        assert float(model._last_merged["x"]) == 5.0

    def test_empty_free_params(self):
        """With no free params, all params come from model."""
        model = _mock_model({"x": jnp.array(3.0)})
        runner = CalibrationRunner.standard()

        runner(model, {})

        assert float(model._last_merged["x"]) == 3.0
