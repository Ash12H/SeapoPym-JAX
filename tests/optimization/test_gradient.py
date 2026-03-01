"""Tests for GradientRunner and gradient computation."""

import jax.numpy as jnp
import pytest

from seapopym.optimization.gradient import GradientRunner, SparseObservations


class TestSparseObservations:
    """Tests for SparseObservations dataclass."""

    def test_valid_creation(self):
        """SparseObservations should accept valid arrays."""
        obs = SparseObservations(
            variable="biomass",
            times=jnp.array([0, 1, 2]),
            y=jnp.array([5, 10, 15]),
            x=jnp.array([5, 10, 15]),
            values=jnp.array([1.0, 2.0, 3.0]),
        )
        assert obs.variable == "biomass"
        assert len(obs.times) == 3
        assert len(obs.values) == 3

    def test_mismatched_lengths(self):
        """SparseObservations should reject mismatched array lengths."""
        with pytest.raises(ValueError, match="same length"):
            SparseObservations(
                variable="biomass",
                times=jnp.array([0, 1]),  # 2 elements
                y=jnp.array([5, 10, 15]),  # 3 elements
                x=jnp.array([5, 10, 15]),
                values=jnp.array([1.0, 2.0, 3.0]),
            )

    def test_single_observation(self):
        """SparseObservations should work with single observation."""
        obs = SparseObservations(
            variable="production",
            times=jnp.array([10]),
            y=jnp.array([0]),
            x=jnp.array([0]),
            values=jnp.array([5.0]),
        )
        assert len(obs.values) == 1


class TestBuildStepFnModes:
    """Tests for step function signature."""

    def test_step_fn_carry_signature(self):
        """step_fn should have ((state, params), forcings) signature."""
        from seapopym.blueprint import Blueprint, Config, functional
        from seapopym.compiler import compile_model
        from seapopym.engine.step import build_step_fn

        @functional(name="test:scaled", units={"x": "g", "scale": "1/s", "return": "g/s"})
        def scaled(x, scale):
            return x * scale

        blueprint = Blueprint.from_dict(
            {
                "id": "test",
                "version": "1.0",
                "declarations": {
                    "state": {"x": {"units": "g", "dims": []}},
                    "parameters": {"scale": {"units": "1/s"}},
                    "forcings": {},
                },
                "process": [
                    {
                        "func": "test:scaled",
                        "inputs": {"x": "state.x", "scale": "parameters.scale"},
                        "outputs": {"return": "derived.x_flux"},
                    }
                ],
                "tendencies": {"x": [{"source": "derived.x_flux"}]},
            }
        )

        import xarray as xr

        config = Config.from_dict(
            {
                "parameters": {"scale": {"value": 0.1}},
                "forcings": {},
                "initial_state": {"x": xr.DataArray(1.0)},
                "execution": {"time_start": "2000-01-01", "time_end": "2000-01-02", "dt": "1d"},
            }
        )

        model = compile_model(blueprint, config)
        step_fn = build_step_fn(model)

        # Should accept ((state, params), forcings)
        state = {"x": jnp.array(1.0)}
        params = {"scale": jnp.array(0.1)}
        carry = (state, params)

        (new_state, new_params), outputs = step_fn(carry, {})
        assert "x" in new_state
        assert "scale" in new_params


class TestGradientComputation:
    """Tests for gradient computation through the model."""

    def test_simple_gradient(self):
        """Gradient should flow through a simple model."""
        import jax

        from seapopym.blueprint import Blueprint, Config, functional
        from seapopym.compiler import compile_model
        from seapopym.engine.step import build_step_fn

        @functional(name="test:linear", units={"x": "g", "rate": "1/s", "return": "g/s"})
        def linear_growth(x, rate):
            return rate * x

        blueprint = Blueprint.from_dict(
            {
                "id": "test",
                "version": "1.0",
                "declarations": {
                    "state": {"x": {"units": "g", "dims": []}},
                    "parameters": {"rate": {"units": "1/s"}},
                    "forcings": {},
                },
                "process": [
                    {
                        "func": "test:linear",
                        "inputs": {"x": "state.x", "rate": "parameters.rate"},
                        "outputs": {"return": "derived.x_flux"},
                    }
                ],
                "tendencies": {"x": [{"source": "derived.x_flux"}]},
            }
        )

        import xarray as xr

        config = Config.from_dict(
            {
                "parameters": {"rate": {"value": 0.1}},
                "forcings": {},
                "initial_state": {"x": xr.DataArray(1.0)},
                "execution": {"time_start": "2000-01-01", "time_end": "2000-01-02", "dt": "1d"},
            }
        )

        model = compile_model(blueprint, config)
        step_fn = build_step_fn(model)

        def loss_fn(params):
            state = {"x": jnp.array(1.0)}
            carry = (state, params)
            (new_state, _), _ = step_fn(carry, {})
            # Target: x should be 2.0
            return (new_state["x"] - 2.0) ** 2

        params = {"rate": jnp.array(0.1)}
        grad = jax.grad(loss_fn)(params)

        # Gradient should exist and not be NaN
        assert "rate" in grad
        assert not jnp.isnan(grad["rate"])

    def test_gradient_decreases_loss(self):
        """Taking a gradient step should decrease the loss."""
        import jax

        from seapopym.blueprint import Blueprint, Config, functional
        from seapopym.compiler import compile_model
        from seapopym.engine.step import build_step_fn

        @functional(name="test:linear2", units={"x": "g", "rate": "1/s", "return": "g/s"})
        def linear_growth(x, rate):
            return rate * x

        blueprint = Blueprint.from_dict(
            {
                "id": "test",
                "version": "1.0",
                "declarations": {
                    "state": {"x": {"units": "g", "dims": []}},
                    "parameters": {"rate": {"units": "1/s"}},
                    "forcings": {},
                },
                "process": [
                    {
                        "func": "test:linear2",
                        "inputs": {"x": "state.x", "rate": "parameters.rate"},
                        "outputs": {"return": "derived.x_flux"},
                    }
                ],
                "tendencies": {"x": [{"source": "derived.x_flux"}]},
            }
        )

        import xarray as xr

        config = Config.from_dict(
            {
                "parameters": {"rate": {"value": 0.1}},
                "forcings": {},
                "initial_state": {"x": xr.DataArray(1.0)},
                "execution": {"time_start": "2000-01-01", "time_end": "2000-01-02", "dt": "1d"},
            }
        )

        model = compile_model(blueprint, config)
        step_fn = build_step_fn(model)

        def loss_fn(params):
            state = {"x": jnp.array(1.0)}
            carry = (state, params)
            (new_state, _), _ = step_fn(carry, {})
            # Want final x to be 1.5 after one step
            # x_new = x + rate * x * dt = 1 + rate * dt
            # Target rate = 0.5 / dt ≈ 5.8e-6 for dt=86400
            target = 1.5
            return (new_state["x"] - target) ** 2

        # Start with wrong rate (too low)
        params = {"rate": jnp.array(1e-6)}
        initial_loss = loss_fn(params)

        # Take gradient step with appropriate learning rate
        # Gradient is O(dt^2) so lr needs to be O(1/dt^2)
        grad = jax.grad(loss_fn)(params)
        lr = 1e-12  # Very small due to dt^2 scaling
        new_params = {"rate": params["rate"] - lr * grad["rate"]}
        new_loss = loss_fn(new_params)

        # Loss should decrease (gradient points away from minimum)
        # grad["rate"] should be negative (need to increase rate to decrease loss)
        assert grad["rate"] < 0, "Gradient should be negative (need to increase rate)"
        assert new_loss < initial_loss, "Loss should decrease after gradient step"


class TestGradientRunner:
    """Tests for GradientRunner methods."""

    @staticmethod
    def _make_runner():
        """Create a GradientRunner with a simple compiled model."""
        from seapopym.blueprint import Blueprint, Config, functional
        from seapopym.compiler import compile_model

        @functional(name="test:gr_step", units={"x": "g", "scale": "1/s", "return": "g/s"})
        def gr_step(x, scale):
            return x * scale

        blueprint = Blueprint.from_dict(
            {
                "id": "test",
                "version": "1.0",
                "declarations": {
                    "state": {"x": {"units": "g", "dims": []}},
                    "parameters": {"scale": {"units": "1/s"}},
                    "forcings": {},
                },
                "process": [
                    {
                        "func": "test:gr_step",
                        "inputs": {"x": "state.x", "scale": "parameters.scale"},
                        "outputs": {"return": "derived.x_flux"},
                    }
                ],
                "tendencies": {"x": [{"source": "derived.x_flux"}]},
            }
        )

        import xarray as xr

        config = Config.from_dict(
            {
                "parameters": {"scale": {"value": 0.1}},
                "forcings": {},
                "initial_state": {"x": xr.DataArray(1.0)},
                "execution": {"time_start": "2000-01-01", "time_end": "2000-01-02", "dt": "1d"},
            }
        )

        model = compile_model(blueprint, config)
        return GradientRunner(model)

    def test_make_loss_fn_unknown_loss_type(self):
        """make_loss_fn should raise ValueError for unknown loss_type."""
        runner = self._make_runner()
        obs = SparseObservations(
            variable="x",
            times=jnp.array([0]),
            y=jnp.array([0]),
            x=jnp.array([0]),
            values=jnp.array([1.0]),
        )
        with pytest.raises(ValueError, match="Unknown loss_type"):
            runner.make_loss_fn(obs, loss_type="unknown")  # type: ignore[arg-type]

    def test_make_loss_fn_rmse_and_nrmse(self):
        """All three loss types (mse, rmse, nrmse) should produce valid scalar losses."""
        runner = self._make_runner()

        obs = SparseObservations(
            variable="x",
            times=jnp.array([0, 1]),
            y=jnp.array([0, 1]),
            x=jnp.array([0, 1]),
            values=jnp.array([1.0, 3.0]),
        )

        # Mock run_with_params to return 3D outputs matching obs structure
        runner.run_with_params = lambda params, initial_state=None, forcings=None: (  # type: ignore[assignment]
            {},
            {"x": jnp.full((2, 3, 3), 2.0)},
        )

        params = {"scale": jnp.array(0.1)}
        for loss_type in ("mse", "rmse", "nrmse"):
            loss_fn = runner.make_loss_fn(obs, loss_type=loss_type)  # type: ignore[arg-type]
            loss = loss_fn(params)
            assert loss.shape == (), f"{loss_type} should return scalar"
            assert jnp.isfinite(loss), f"{loss_type} should return finite value"
            assert loss > 0, f"{loss_type} should be positive when pred != obs"

    def test_compute_value_and_gradient(self):
        """compute_value_and_gradient should return (value, grads) tuple."""
        runner = self._make_runner()

        def loss_fn(params):
            return (params["scale"] - 1.0) ** 2

        params = {"scale": jnp.array(0.5)}
        value, grads = runner.compute_value_and_gradient(params, loss_fn)

        assert value.shape == (), "Loss value should be scalar"
        assert jnp.isfinite(value), "Loss value should be finite"
        assert "scale" in grads, "Gradients should contain 'scale'"
        assert jnp.isfinite(grads["scale"]), "Gradient should be finite"
        # Loss = (0.5 - 1.0)^2 = 0.25
        assert jnp.allclose(value, 0.25)
        # Gradient = 2 * (0.5 - 1.0) = -1.0
        assert jnp.allclose(grads["scale"], -1.0)
