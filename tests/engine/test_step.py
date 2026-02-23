"""Tests for step function builder."""

import numpy as np
import pytest

import jax.numpy as jnp

from seapopym.blueprint import Blueprint, Config, clear_registry, functional
from seapopym.compiler import compile_model


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear registry before each test."""
    clear_registry()
    yield
    clear_registry()


@pytest.fixture
def simple_blueprint():
    """Create a simple blueprint for testing."""
    return Blueprint.from_dict(
        {
            "id": "test-model",
            "version": "0.1.0",
            "declarations": {
                "state": {
                    "biomass": {
                        "units": "g",
                        "dims": ["Y", "X"],
                    }
                },
                "parameters": {
                    "growth_rate": {
                        "units": "1/d",
                    }
                },
                "forcings": {
                    "temperature": {
                        "units": "degC",
                        "dims": ["T", "Y", "X"],
                    },
                    "mask": {
                        "dims": ["Y", "X"],
                    },
                },
            },
            "process": [
                {
                    "func": "test:growth",
                    "inputs": {
                        "biomass": "state.biomass",
                        "rate": "parameters.growth_rate",
                        "temp": "forcings.temperature",
                    },
                    "outputs": {
                        "tendency": "derived.growth_flux",
                    },
                }
            ],
            "tendencies": {
                "biomass": [{"source": "derived.growth_flux"}],
            },
        }
    )


@pytest.fixture
def simple_config():
    """Create a simple config for testing."""
    return Config.from_dict(
        {
            "parameters": {
                "growth_rate": {"value": 0.1},
            },
            "forcings": {
                "temperature": np.ones((10, 5, 5)) * 20.0,
                "mask": np.ones((5, 5)),
            },
            "initial_state": {
                "biomass": np.ones((5, 5)) * 100.0,
            },
            "execution": {
                "dt": "1d",
                "time_start": "2000-01-01",
                "time_end": "2000-01-11",
            },
        }
    )


class TestBuildStepFn:
    """Tests for build_step_fn function."""

    def test_build_step_fn(self, simple_blueprint, simple_config):
        """Test building step function."""

        # Register test function
        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        model = compile_model(simple_blueprint, simple_config)

        from seapopym.engine.step import build_step_fn

        step_fn = build_step_fn(model)

        # Test a single step
        state = {"biomass": jnp.ones((5, 5)) * 100.0}
        params = model.parameters
        forcings_t = {
            "temperature": jnp.ones((5, 5)) * 20.0,
            "mask": jnp.ones((5, 5)),
        }

        (new_state, _params), outputs = step_fn((state, params), forcings_t)

        # Check state was updated
        assert "biomass" in new_state
        assert new_state["biomass"].shape == (5, 5)

    def test_mask_application(self, simple_blueprint, simple_config):
        """Test that mask is applied correctly."""

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        model = compile_model(simple_blueprint, simple_config)

        from seapopym.engine.step import build_step_fn

        step_fn = build_step_fn(model)

        # Create mask with some zeros
        mask = jnp.ones((5, 5))
        mask = mask.at[0, :].set(0)  # First row masked

        state = {"biomass": jnp.ones((5, 5)) * 100.0}
        params = model.parameters
        forcings_t = {
            "temperature": jnp.ones((5, 5)) * 20.0,
            "mask": mask,
        }

        (new_state, _params), outputs = step_fn((state, params), forcings_t)

        # First row should be zero (masked)
        np.testing.assert_array_equal(np.asarray(new_state["biomass"][0, :]), 0.0)
        # Other rows should have values
        assert np.all(np.asarray(new_state["biomass"][1:, :]) > 0)


class TestResolveInputs:
    """Tests for input resolution helper."""

    def test_resolve_state_input(self):
        """Test resolving state variable."""
        from seapopym.engine.step import _resolve_inputs

        inputs_mapping = {"biomass": "state.biomass"}
        state = {"biomass": jnp.array([1.0, 2.0])}
        forcings = {}
        params = {}
        intermediates = {}

        result = _resolve_inputs(inputs_mapping, state, forcings, params, intermediates)

        np.testing.assert_array_equal(np.asarray(result["biomass"]), [1.0, 2.0])

    def test_resolve_forcings_input(self):
        """Test resolving forcing variable."""
        from seapopym.engine.step import _resolve_inputs

        inputs_mapping = {"temp": "forcings.temperature"}
        state = {}
        forcings = {"temperature": jnp.array([20.0, 25.0])}
        params = {}
        intermediates = {}

        result = _resolve_inputs(inputs_mapping, state, forcings, params, intermediates)

        np.testing.assert_array_equal(np.asarray(result["temp"]), [20.0, 25.0])

    def test_resolve_parameter_input(self):
        """Test resolving parameter."""
        from seapopym.engine.step import _resolve_inputs

        inputs_mapping = {"rate": "parameters.growth_rate"}
        state = {}
        forcings = {}
        params = {"growth_rate": jnp.array(0.1)}
        intermediates = {}

        result = _resolve_inputs(inputs_mapping, state, forcings, params, intermediates)

        assert float(result["rate"]) == pytest.approx(0.1)


class TestIntegrateEuler:
    """Tests for Euler integration."""

    def test_euler_integration(self):
        """Test Euler integration."""
        from seapopym.blueprint.schema import TendencySource
        from seapopym.engine.step import _integrate_euler

        state = {"x": jnp.array([10.0, 20.0])}
        intermediates = {"flux": jnp.array([1.0, 2.0])}
        tendency_map = {"x": [TendencySource(source="derived.flux")]}
        dt = 1.0

        new_state = _integrate_euler(state, intermediates, tendency_map, dt)

        np.testing.assert_array_equal(np.asarray(new_state["x"]), [11.0, 22.0])

    def test_euler_integration_multiple_tendencies(self):
        """Test Euler integration with multiple tendencies."""
        from seapopym.blueprint.schema import TendencySource
        from seapopym.engine.step import _integrate_euler

        state = {"x": jnp.array([10.0])}
        intermediates = {"flux1": jnp.array([1.0]), "flux2": jnp.array([2.0])}
        tendency_map = {
            "x": [
                TendencySource(source="derived.flux1"),
                TendencySource(source="derived.flux2"),
            ]
        }
        dt = 1.0

        new_state = _integrate_euler(state, intermediates, tendency_map, dt)

        np.testing.assert_array_equal(np.asarray(new_state["x"]), [13.0])  # 10 + (1+2)*1

    def test_euler_no_tendency(self):
        """Test Euler integration with no tendency (state unchanged)."""
        from seapopym.blueprint.schema import TendencySource
        from seapopym.engine.step import _integrate_euler

        state = {"x": jnp.array([10.0]), "y": jnp.array([5.0])}
        intermediates = {"flux": jnp.array([1.0])}
        tendency_map = {"x": [TendencySource(source="derived.flux")]}  # No tendency for y
        dt = 1.0

        new_state = _integrate_euler(state, intermediates, tendency_map, dt)

        np.testing.assert_array_equal(np.asarray(new_state["x"]), [11.0])
        np.testing.assert_array_equal(np.asarray(new_state["y"]), [5.0])  # Unchanged

    def test_euler_with_sign(self):
        """Test Euler integration with negative sign on a tendency source."""
        from seapopym.blueprint.schema import TendencySource
        from seapopym.engine.step import _integrate_euler

        state = {"x": jnp.array([10.0])}
        intermediates = {"gain": jnp.array([3.0]), "loss": jnp.array([1.0])}
        tendency_map = {
            "x": [
                TendencySource(source="derived.gain"),
                TendencySource(source="derived.loss", sign=-1.0),
            ]
        }
        dt = 1.0

        new_state = _integrate_euler(state, intermediates, tendency_map, dt)

        np.testing.assert_array_equal(np.asarray(new_state["x"]), [12.0])  # 10 + (3 - 1)*1
