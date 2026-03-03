"""Tests for Runner.optimization() replacing CompiledModel.run_with_params."""

import jax.numpy as jnp
import numpy as np
import pytest

from seapopym.blueprint import Blueprint, Config, functional
from seapopym.blueprint.registry import REGISTRY
from seapopym.compiler import compile_model
from seapopym.engine.runner import Runner

_TEST_FUNC_NAME = "test:growth_rwp"


@pytest.fixture(autouse=True)
def setup_registry():
    """Register test function and remove only it after."""

    @functional(name=_TEST_FUNC_NAME)
    def growth(biomass, rate, temp):
        return biomass * rate * (temp / 20.0)

    yield
    REGISTRY.pop(_TEST_FUNC_NAME, None)


def _build_toy_model(n_days=10, ny=3, nx=3, growth_rate=0.0001):
    """Build a minimal compiled model for testing."""
    blueprint = Blueprint.from_dict(
        {
            "id": "toy-rwp",
            "version": "0.1.0",
            "declarations": {
                "state": {"biomass": {"units": "g", "dims": ["Y", "X"]}},
                "parameters": {"growth_rate": {"units": "1/d"}},
                "forcings": {
                    "temperature": {"units": "degC", "dims": ["T", "Y", "X"]},
                    "mask": {"dims": ["Y", "X"]},
                },
            },
            "process": [
                {
                    "func": "test:growth_rwp",
                    "inputs": {
                        "biomass": "state.biomass",
                        "rate": "parameters.growth_rate",
                        "temp": "forcings.temperature",
                    },
                    "outputs": {"tendency": "derived.growth_flux"},
                }
            ],
            "tendencies": {"biomass": [{"source": "derived.growth_flux"}]},
        }
    )

    config = Config.from_dict(
        {
            "parameters": {"growth_rate": {"value": growth_rate}},
            "forcings": {
                "temperature": np.ones((n_days, ny, nx)) * 20.0,
                "mask": np.ones((ny, nx)),
            },
            "initial_state": {"biomass": np.ones((ny, nx)) * 100.0},
            "execution": {
                "dt": "1d",
                "time_start": "2000-01-01",
                "time_end": f"2000-01-{n_days + 1:02d}",
            },
        }
    )

    return compile_model(blueprint, config)


class TestOptimizationRunnerNominal:
    """Tests for basic Runner.optimization() behavior."""

    def test_returns_outputs_dict(self):
        """Runner.optimization() should return outputs dict."""
        model = _build_toy_model()
        runner = Runner.optimization()
        outputs = runner(model, dict(model.parameters))

        assert isinstance(outputs, dict)
        assert len(outputs) > 0

    def test_outputs_contain_variables(self):
        """Outputs should contain output arrays."""
        model = _build_toy_model()
        runner = Runner.optimization()
        outputs = runner(model, dict(model.parameters))

        assert len(outputs) > 0

    def test_biomass_grows_with_positive_rate(self):
        """Biomass should increase with positive growth rate."""
        model = _build_toy_model(growth_rate=0.001)
        runner = Runner.optimization()
        outputs = runner(model, dict(model.parameters))

        # Last timestep biomass should be > initial (100.0)
        assert jnp.all(outputs["biomass"][-1] > 100.0)

    def test_different_params_give_different_results(self):
        """Different parameter values should produce different outputs."""
        model = _build_toy_model()
        runner = Runner.optimization()

        params_low = {"growth_rate": jnp.array(0.0001)}
        params_high = {"growth_rate": jnp.array(0.01)}

        outputs_low = runner(model, params_low)
        outputs_high = runner(model, params_high)

        # Higher growth rate should give higher biomass
        assert jnp.mean(outputs_high["biomass"][-1]) > jnp.mean(outputs_low["biomass"][-1])
