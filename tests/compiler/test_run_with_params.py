"""Tests for CompiledModel.run_with_params."""

import jax.numpy as jnp
import numpy as np
import pytest

from seapopym.blueprint import Blueprint, Config, functional
from seapopym.blueprint.registry import REGISTRY
from seapopym.compiler import compile_model

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


class TestRunWithParamsNominal:
    """Tests for basic run_with_params behavior."""

    def test_returns_final_state_and_outputs(self):
        """run_with_params should return (final_state, outputs) tuple."""
        model = _build_toy_model()
        result = model.run_with_params(dict(model.parameters))

        assert isinstance(result, tuple)
        assert len(result) == 2

        final_state, outputs = result
        assert isinstance(final_state, dict)
        assert isinstance(outputs, dict)

    def test_final_state_contains_variables(self):
        """Final state should contain declared state variables."""
        model = _build_toy_model()
        final_state, _ = model.run_with_params(dict(model.parameters))

        assert "biomass" in final_state

    def test_outputs_contain_variables(self):
        """Outputs should contain output arrays."""
        model = _build_toy_model()
        _, outputs = model.run_with_params(dict(model.parameters))

        assert len(outputs) > 0

    def test_biomass_grows_with_positive_rate(self):
        """Biomass should increase with positive growth rate."""
        model = _build_toy_model(growth_rate=0.001)
        final_state, _ = model.run_with_params(dict(model.parameters))

        assert jnp.all(final_state["biomass"] > 100.0)

    def test_different_params_give_different_results(self):
        """Different parameter values should produce different outputs."""
        model = _build_toy_model()

        params_low = {"growth_rate": jnp.array(0.0001)}
        params_high = {"growth_rate": jnp.array(0.01)}

        state_low, _ = model.run_with_params(params_low)
        state_high, _ = model.run_with_params(params_high)

        # Higher growth rate should give higher biomass
        assert jnp.mean(state_high["biomass"]) > jnp.mean(state_low["biomass"])


class TestRunWithParamsOverrides:
    """Tests for state and forcings overrides."""

    def test_custom_initial_state(self):
        """Custom initial state should be used instead of default."""
        model = _build_toy_model(n_days=5, ny=2, nx=2)

        custom_state = {"biomass": jnp.ones((2, 2)) * 500.0}
        final_state, _ = model.run_with_params(
            dict(model.parameters),
            initial_state=custom_state,
        )

        # Should start from 500, not 100
        assert jnp.all(final_state["biomass"] > 100.0)

    def test_default_state_when_none(self):
        """None initial_state should use model's default state."""
        model = _build_toy_model(n_days=5, ny=2, nx=2)

        result_default = model.run_with_params(dict(model.parameters))
        result_none = model.run_with_params(dict(model.parameters), initial_state=None)

        np.testing.assert_array_equal(
            np.asarray(result_default[0]["biomass"]),
            np.asarray(result_none[0]["biomass"]),
        )
