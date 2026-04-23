"""Tests for step function builder."""

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from seapopym.blueprint import functional
from seapopym.compiler import compile_model


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

        (new_state, _params), outputs = step_fn((state, params), (forcings_t, {}))

        # Check state was updated
        assert "biomass" in new_state
        assert new_state["biomass"].shape == (5, 5)


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

    def test_resolve_direct_reference_state(self):
        """Test resolving a direct reference (no category prefix) found in state."""
        from seapopym.engine.step import _resolve_inputs

        result = _resolve_inputs(
            {"b": "biomass"},
            state={"biomass": jnp.array(1.0)},
            forcings_t={},
            parameters={},
            intermediates={},
        )
        assert float(result["b"]) == pytest.approx(1.0)

    def test_resolve_direct_reference_forcings(self):
        """Test resolving a direct reference found in forcings."""
        from seapopym.engine.step import _resolve_inputs

        result = _resolve_inputs(
            {"t": "temperature"},
            state={},
            forcings_t={"temperature": jnp.array(20.0)},
            parameters={},
            intermediates={},
        )
        assert float(result["t"]) == pytest.approx(20.0)

    def test_resolve_direct_reference_parameters(self):
        """Test resolving a direct reference found in parameters."""
        from seapopym.engine.step import _resolve_inputs

        result = _resolve_inputs(
            {"r": "growth_rate"},
            state={},
            forcings_t={},
            parameters={"growth_rate": jnp.array(0.5)},
            intermediates={},
        )
        assert float(result["r"]) == pytest.approx(0.5)

    def test_resolve_direct_reference_intermediates(self):
        """Test resolving a direct reference found in intermediates."""
        from seapopym.engine.step import _resolve_inputs

        result = _resolve_inputs(
            {"f": "flux"},
            state={},
            forcings_t={},
            parameters={},
            intermediates={"flux": jnp.array(3.0)},
        )
        assert float(result["f"]) == pytest.approx(3.0)

    def test_resolve_direct_reference_not_found(self):
        """Test that a direct reference not found anywhere raises KeyError."""
        from seapopym.engine.step import _resolve_inputs

        with pytest.raises(KeyError, match="Cannot resolve input"):
            _resolve_inputs(
                {"x": "nonexistent"},
                state={},
                forcings_t={},
                parameters={},
                intermediates={},
            )

    def test_resolve_unknown_category(self):
        """Test that an unknown category raises KeyError."""
        from seapopym.engine.step import _resolve_inputs

        with pytest.raises(KeyError, match="Unknown category"):
            _resolve_inputs(
                {"x": "unknown_cat.var"},
                state={},
                forcings_t={},
                parameters={},
                intermediates={},
            )


class TestTransposeVmapOutput:
    """Tests for _transpose_vmap_output."""

    def test_transpose_single_array(self):
        """Test transposing a single array."""
        from seapopym.engine.step import _transpose_vmap_output

        arr = jnp.zeros((2, 3, 4))
        result = _transpose_vmap_output(arr, (2, 0, 1))
        assert result.shape == (4, 2, 3)

    def test_transpose_tuple_of_arrays(self):
        """Test transposing a tuple of arrays."""
        from seapopym.engine.step import _transpose_vmap_output

        a = jnp.zeros((2, 3, 4))
        b = jnp.zeros((2, 3, 4))
        result = _transpose_vmap_output((a, b), (2, 0, 1))
        assert isinstance(result, tuple)
        assert result[0].shape == (4, 2, 3)
        assert result[1].shape == (4, 2, 3)

    def test_transpose_skips_mismatched_ndim(self):
        """Test that arrays with wrong ndim are returned unchanged."""
        from seapopym.engine.step import _transpose_vmap_output

        arr = jnp.zeros((2, 3))  # 2D, but axes expect 3D
        result = _transpose_vmap_output(arr, (2, 0, 1))
        assert result.shape == (2, 3)  # Unchanged

    def test_transpose_scalar_passthrough(self):
        """Test that a scalar passes through unchanged."""
        from seapopym.engine.step import _transpose_vmap_output

        result = _transpose_vmap_output(42, (1, 0))
        assert result == 42


class TestHandleComputeOutputs:
    """Tests for _handle_compute_outputs."""

    def test_single_output(self):
        """Test handling a single compute output."""
        from seapopym.engine.step import _handle_compute_outputs

        intermediates: dict = {}
        _handle_compute_outputs(
            jnp.array(1.0),
            {"tendency": "derived.growth_flux"},
            intermediates,
        )
        assert "growth_flux" in intermediates

    def test_multi_output(self):
        """Test handling multiple compute outputs."""
        from seapopym.engine.step import _handle_compute_outputs

        intermediates: dict = {}
        _handle_compute_outputs(
            (jnp.array(1.0), jnp.array(2.0)),
            {"a": "derived.flux_a", "b": "derived.flux_b"},
            intermediates,
        )
        assert float(intermediates["flux_a"]) == pytest.approx(1.0)
        assert float(intermediates["flux_b"]) == pytest.approx(2.0)

    def test_multi_output_wrong_type(self):
        """Test that a non-tuple for multi-output raises TypeError."""
        from seapopym.engine.step import _handle_compute_outputs

        with pytest.raises(TypeError, match="expected tuple"):
            _handle_compute_outputs(
                jnp.array(1.0),
                {"a": "derived.flux_a", "b": "derived.flux_b"},
                {},
            )

    def test_multi_output_wrong_count(self):
        """Test that wrong number of outputs raises ValueError."""
        from seapopym.engine.step import _handle_compute_outputs

        with pytest.raises(ValueError, match="expected 2"):
            _handle_compute_outputs(
                (jnp.array(1.0),),
                {"a": "derived.flux_a", "b": "derived.flux_b"},
                {},
            )


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

    def test_euler_no_clamp_by_default(self):
        """Test that Euler integration does NOT clamp when no clamp_map is provided."""
        from seapopym.blueprint.schema import TendencySource
        from seapopym.engine.step import _integrate_euler

        state = {"x": jnp.array([1.0])}
        intermediates = {"loss": jnp.array([10.0])}
        tendency_map = {"x": [TendencySource(source="derived.loss", sign=-1.0)]}
        dt = 1.0

        new_state = _integrate_euler(state, intermediates, tendency_map, dt)

        # 1.0 + (-10.0)*1.0 = -9.0 → no clamping
        np.testing.assert_array_equal(np.asarray(new_state["x"]), [-9.0])

    def test_euler_clamp_non_negative(self):
        """Test that Euler integration clamps state to >= 0 with clamp_map."""
        from seapopym.blueprint.schema import TendencySource
        from seapopym.engine.step import _integrate_euler

        state = {"x": jnp.array([1.0])}
        intermediates = {"loss": jnp.array([10.0])}
        tendency_map = {"x": [TendencySource(source="derived.loss", sign=-1.0)]}
        dt = 1.0
        clamp_map = {"x": (0.0, None)}

        new_state = _integrate_euler(state, intermediates, tendency_map, dt, clamp_map)

        # 1.0 + (-10.0)*1.0 = -9.0 → clamped to 0.0
        np.testing.assert_array_equal(np.asarray(new_state["x"]), [0.0])

    def test_euler_clamp_both_bounds(self):
        """Test Euler integration with both lower and upper clamp bounds."""
        from seapopym.blueprint.schema import TendencySource
        from seapopym.engine.step import _integrate_euler

        state = {"x": jnp.array([5.0])}
        intermediates = {"flux": jnp.array([100.0])}
        tendency_map = {"x": [TendencySource(source="derived.flux")]}
        dt = 1.0
        clamp_map = {"x": (-1.0, 10.0)}

        new_state = _integrate_euler(state, intermediates, tendency_map, dt, clamp_map)

        # 5.0 + 100.0*1.0 = 105.0 → clamped to 10.0
        np.testing.assert_array_equal(np.asarray(new_state["x"]), [10.0])

    def test_euler_clamp_per_variable(self):
        """Test that clamping is applied per variable independently."""
        from seapopym.blueprint.schema import TendencySource
        from seapopym.engine.step import _integrate_euler

        state = {"x": jnp.array([1.0]), "y": jnp.array([1.0])}
        intermediates = {"loss": jnp.array([10.0])}
        tendency_map = {
            "x": [TendencySource(source="derived.loss", sign=-1.0)],
            "y": [TendencySource(source="derived.loss", sign=-1.0)],
        }
        dt = 1.0
        clamp_map = {"x": (0.0, None)}  # Only x is clamped

        new_state = _integrate_euler(state, intermediates, tendency_map, dt, clamp_map)

        np.testing.assert_array_equal(np.asarray(new_state["x"]), [0.0])  # clamped
        np.testing.assert_array_equal(np.asarray(new_state["y"]), [-9.0])  # not clamped


class TestIntegrateRK2:
    """Tests for RK2 (Heun) integration via build_step_fn."""

    def test_rk2_linear_ode(self, simple_blueprint, simple_config):
        """RK2 solves dx/dt = rate * x exactly for 1 step on linear ODE."""
        from seapopym.blueprint import functional
        from seapopym.compiler import compile_model
        from seapopym.engine.step import build_step_fn

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        simple_config.execution.integrator = "rk2"
        model = compile_model(simple_blueprint, simple_config)
        assert model.integrator == "rk2"

        step_fn = build_step_fn(model)

        state = {"biomass": jnp.ones((5, 5)) * 100.0}
        params = model.parameters
        forcings_t = {"temperature": jnp.ones((5, 5)) * 20.0, "mask": jnp.ones((5, 5))}

        (new_state, _), outputs = step_fn((state, params), (forcings_t, {}))

        # RK2 on dx/dt = r*x: x_new = x * (1 + r*dt + 0.5*(r*dt)^2)
        r = 0.1
        dt = model.dt
        expected = 100.0 * (1 + r * dt + 0.5 * (r * dt) ** 2)
        np.testing.assert_allclose(np.asarray(new_state["biomass"])[0, 0], expected, rtol=1e-6)

    def test_rk2_more_accurate_than_euler(self, simple_blueprint, simple_config):
        """RK2 should be more accurate than Euler on the same ODE."""
        from seapopym.blueprint import functional
        from seapopym.compiler import compile_model
        from seapopym.engine.step import build_step_fn

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        # Use a small rate so r*dt is in convergent regime
        simple_config.parameters["growth_rate"] = xr.DataArray(1e-7)

        # Run Euler
        simple_config.execution.integrator = "euler"
        model_euler = compile_model(simple_blueprint, simple_config)
        step_euler = build_step_fn(model_euler)

        # Run RK2
        simple_config.execution.integrator = "rk2"
        model_rk2 = compile_model(simple_blueprint, simple_config)
        step_rk2 = build_step_fn(model_rk2)

        state = {"biomass": jnp.ones((5, 5)) * 100.0}
        forcings_t = {"temperature": jnp.ones((5, 5)) * 20.0, "mask": jnp.ones((5, 5))}

        (euler_state, _), _ = step_euler((state, model_euler.parameters), (forcings_t, {}))
        (rk2_state, _), _ = step_rk2((state, model_rk2.parameters), (forcings_t, {}))

        # Exact solution: x(t) = 100 * exp(r * dt)
        import math

        r = 1e-7
        exact = 100.0 * math.exp(r * model_euler.dt)

        euler_err = abs(float(euler_state["biomass"][0, 0]) - exact)
        rk2_err = abs(float(rk2_state["biomass"][0, 0]) - exact)

        assert rk2_err < euler_err

    def test_rk2_no_tendency_passthrough(self, simple_blueprint, simple_config):
        """Variables without tendencies pass through unchanged with RK2."""
        from seapopym.blueprint import functional
        from seapopym.compiler import compile_model
        from seapopym.engine.step import build_step_fn

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        simple_config.execution.integrator = "rk2"
        model = compile_model(simple_blueprint, simple_config)
        step_fn = build_step_fn(model)

        # Add extra state variable with no tendency
        state = {"biomass": jnp.ones((5, 5)) * 100.0, "tracer": jnp.ones((5, 5)) * 42.0}
        forcings_t = {"temperature": jnp.ones((5, 5)) * 20.0, "mask": jnp.ones((5, 5))}

        (new_state, _), _ = step_fn((state, model.parameters), (forcings_t, {}))

        np.testing.assert_array_equal(np.asarray(new_state["tracer"]), 42.0)


class TestIntegrateRK4:
    """Tests for RK4 (classical) integration via build_step_fn."""

    def test_rk4_linear_ode(self, simple_blueprint, simple_config):
        """RK4 solves dx/dt = rate * x with 4th-order accuracy."""
        from seapopym.blueprint import functional
        from seapopym.compiler import compile_model
        from seapopym.engine.step import build_step_fn

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        simple_config.execution.integrator = "rk4"
        model = compile_model(simple_blueprint, simple_config)
        assert model.integrator == "rk4"

        step_fn = build_step_fn(model)

        state = {"biomass": jnp.ones((5, 5)) * 100.0}
        params = model.parameters
        forcings_t = {"temperature": jnp.ones((5, 5)) * 20.0, "mask": jnp.ones((5, 5))}

        (new_state, _), outputs = step_fn((state, params), (forcings_t, {}))

        # RK4 on dx/dt = r*x: Taylor expansion to 4th order
        r = 0.1
        dt = model.dt
        rd = r * dt
        expected = 100.0 * (1 + rd + rd**2 / 2 + rd**3 / 6 + rd**4 / 24)
        np.testing.assert_allclose(np.asarray(new_state["biomass"])[0, 0], expected, rtol=1e-6)

    def test_rk4_more_accurate_than_rk2(self, simple_blueprint, simple_config):
        """RK4 should be more accurate than RK2 on the same ODE."""
        from seapopym.blueprint import functional
        from seapopym.compiler import compile_model
        from seapopym.engine.step import build_step_fn

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        # Use a small rate so r*dt is in convergent regime
        simple_config.parameters["growth_rate"] = xr.DataArray(1e-7)

        # Run RK2
        simple_config.execution.integrator = "rk2"
        model_rk2 = compile_model(simple_blueprint, simple_config)
        step_rk2 = build_step_fn(model_rk2)

        # Run RK4
        simple_config.execution.integrator = "rk4"
        model_rk4 = compile_model(simple_blueprint, simple_config)
        step_rk4 = build_step_fn(model_rk4)

        state = {"biomass": jnp.ones((5, 5)) * 100.0}
        forcings_t = {"temperature": jnp.ones((5, 5)) * 20.0, "mask": jnp.ones((5, 5))}

        (rk2_state, _), _ = step_rk2((state, model_rk2.parameters), (forcings_t, {}))
        (rk4_state, _), _ = step_rk4((state, model_rk4.parameters), (forcings_t, {}))

        import math

        r = 1e-7
        exact = 100.0 * math.exp(r * model_rk2.dt)

        rk2_err = abs(float(rk2_state["biomass"][0, 0]) - exact)
        rk4_err = abs(float(rk4_state["biomass"][0, 0]) - exact)

        assert rk4_err < rk2_err

    def test_rk4_no_tendency_passthrough(self, simple_blueprint, simple_config):
        """Variables without tendencies pass through unchanged with RK4."""
        from seapopym.blueprint import functional
        from seapopym.compiler import compile_model
        from seapopym.engine.step import build_step_fn

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        simple_config.execution.integrator = "rk4"
        model = compile_model(simple_blueprint, simple_config)
        step_fn = build_step_fn(model)

        state = {"biomass": jnp.ones((5, 5)) * 100.0, "tracer": jnp.ones((5, 5)) * 42.0}
        forcings_t = {"temperature": jnp.ones((5, 5)) * 20.0, "mask": jnp.ones((5, 5))}

        (new_state, _), _ = step_fn((state, model.parameters), (forcings_t, {}))

        np.testing.assert_array_equal(np.asarray(new_state["tracer"]), 42.0)


class TestIntegratorDefault:
    """Tests for default integrator behavior and backward compatibility."""

    def test_default_integrator_is_euler(self, simple_blueprint, simple_config):
        """Default integrator should be Euler (backward compatible)."""
        from seapopym.blueprint import functional
        from seapopym.compiler import compile_model

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        model = compile_model(simple_blueprint, simple_config)
        assert model.integrator == "euler"

    def test_euler_unchanged_behavior(self, simple_blueprint, simple_config):
        """Euler via new code path produces same result as before refactor."""
        from seapopym.blueprint import functional
        from seapopym.compiler import compile_model
        from seapopym.engine.step import build_step_fn

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        model = compile_model(simple_blueprint, simple_config)
        step_fn = build_step_fn(model)

        state = {"biomass": jnp.ones((5, 5)) * 100.0}
        forcings_t = {"temperature": jnp.ones((5, 5)) * 20.0, "mask": jnp.ones((5, 5))}

        (new_state, _), outputs = step_fn((state, model.parameters), (forcings_t, {}))

        # Euler: x_new = x + r * x * dt = 100 + 0.1 * 100 * 86400
        expected = 100.0 + 0.1 * 100.0 * model.dt
        np.testing.assert_allclose(np.asarray(new_state["biomass"])[0, 0], expected, rtol=1e-10)
