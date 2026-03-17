"""Tests for run() and simulate()."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from seapopym.blueprint import Blueprint, Config, functional
from seapopym.compiler import compile_model
from seapopym.engine import WriterRaw, run, simulate
from seapopym.engine.exceptions import ChunkingError
from seapopym.engine.step import build_step_fn


class TestRun:
    """Tests for the run() function."""

    def test_run_no_chunking(self, simple_blueprint, simple_config):
        """run() without chunking returns all timesteps."""

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        model = compile_model(simple_blueprint, simple_config)
        step_fn = build_step_fn(model)
        state, outputs = run(step_fn, model, dict(model.state), dict(model.parameters))

        assert "biomass" in state
        assert "biomass" in outputs
        assert outputs["biomass"].shape[0] == 10

    def test_run_with_chunking(self, simple_blueprint, simple_config):
        """run() with chunk_size splits execution and concatenates."""

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        model = compile_model(simple_blueprint, simple_config)
        step_fn = build_step_fn(model)

        state_full, out_full = run(step_fn, model, dict(model.state), dict(model.parameters))
        state_chunked, out_chunked = run(step_fn, model, dict(model.state), dict(model.parameters), chunk_size=5)

        np.testing.assert_allclose(
            np.asarray(out_full["biomass"]),
            np.asarray(out_chunked["biomass"]),
            rtol=1e-5,
        )

    def test_run_with_writer(self, simple_blueprint, simple_config):
        """run() with explicit WriterRaw."""

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        model = compile_model(simple_blueprint, simple_config)
        step_fn = build_step_fn(model)
        writer = WriterRaw()

        state, outputs = run(step_fn, model, dict(model.state), dict(model.parameters), writer=writer)

        assert isinstance(outputs, dict)
        assert outputs["biomass"].shape[0] == 10

    def test_run_invalid_chunk_size(self, simple_blueprint, simple_config):
        """Negative chunk_size raises ChunkingError."""

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        model = compile_model(simple_blueprint, simple_config)
        step_fn = build_step_fn(model)

        with pytest.raises(ChunkingError):
            run(step_fn, model, dict(model.state), dict(model.parameters), chunk_size=-1)

    def test_run_export_variables(self, simple_blueprint, simple_config):
        """run() with export_variables filters outputs."""

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        model = compile_model(simple_blueprint, simple_config)
        step_fn = build_step_fn(model, export_variables=["biomass"])
        state, outputs = run(step_fn, model, dict(model.state), dict(model.parameters))

        assert "biomass" in outputs
        assert "growth_flux" not in outputs


class TestSimulate:
    """Tests for the simulate() function."""

    def test_simulate_memory(self, simple_blueprint, simple_config):
        """simulate() returns xarray.Dataset by default."""

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        model = compile_model(simple_blueprint, simple_config)
        state, outputs = simulate(model, chunk_size=5)

        assert "biomass" in state
        assert outputs is not None
        assert "biomass" in outputs
        assert outputs["biomass"].shape == (10, 5, 5)

    def test_simulate_disk(self, simple_blueprint, simple_config, tmp_path):
        """simulate() with output_path writes Zarr."""

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        model = compile_model(simple_blueprint, simple_config)
        state, _ = simulate(model, chunk_size=5, output_path=str(tmp_path / "output"))

        assert "biomass" in state
        import zarr

        store = zarr.open(str(tmp_path / "output"), mode="r")
        assert store["biomass"].shape[0] == 10

    def test_simulate_export_variables(self, simple_blueprint, simple_config):
        """simulate() with export_variables filters outputs."""

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        model = compile_model(simple_blueprint, simple_config)
        state, outputs = simulate(model, export_variables=["biomass"])

        assert "biomass" in outputs
        # growth_flux should not be in the xarray Dataset
        assert "growth_flux" not in outputs


class TestTimeIndexedParams:
    """Tests for time-indexed parameters (params with dim T)."""

    def test_run_with_time_indexed_param(self):
        """Time-indexed parameter is passed per-timestep and differentiable."""

        @functional(name="test:forced_growth")
        def forced_growth(biomass, force):
            return biomass * force

        blueprint = Blueprint.from_dict(
            {
                "id": "test-time-param",
                "version": "0.1.0",
                "declarations": {
                    "state": {"biomass": {"units": "g", "dims": ["Y", "X"]}},
                    "parameters": {"force": {"units": "1/d", "dims": ["T"]}},
                    "forcings": {},
                },
                "process": [
                    {
                        "func": "test:forced_growth",
                        "inputs": {
                            "biomass": "state.biomass",
                            "force": "parameters.force",
                        },
                        "outputs": {"return": "derived.growth_flux"},
                    }
                ],
                "tendencies": {"biomass": [{"source": "derived.growth_flux"}]},
            }
        )

        n_steps = 5
        force_values = np.ones(n_steps) * 0.01
        config = Config(
            parameters={"force": xr.DataArray(force_values, dims=["T"])},
            forcings={},
            initial_state={"biomass": xr.DataArray(np.ones((1, 1)) * 100.0, dims=["Y", "X"])},
            execution={"dt": "1d", "time_start": "2000-01-01", "time_end": "2000-01-06"},
        )

        model = compile_model(blueprint, config)
        assert "force" in model.time_indexed_params

        step_fn = build_step_fn(model, export_variables=["biomass"])
        state, outputs = run(step_fn, model, dict(model.state), dict(model.parameters))

        assert outputs["biomass"].shape[0] == n_steps
        # Biomass should have grown
        assert float(outputs["biomass"][-1, 0, 0]) > 100.0

    def test_time_indexed_param_gradient(self):
        """Gradient of loss w.r.t. time-indexed parameter is computable."""

        @functional(name="test:forced_growth")
        def forced_growth(biomass, force):
            return biomass * force

        blueprint = Blueprint.from_dict(
            {
                "id": "test-time-param-grad",
                "version": "0.1.0",
                "declarations": {
                    "state": {"biomass": {"units": "g", "dims": ["Y", "X"]}},
                    "parameters": {"force": {"units": "1/d", "dims": ["T"]}},
                    "forcings": {},
                },
                "process": [
                    {
                        "func": "test:forced_growth",
                        "inputs": {
                            "biomass": "state.biomass",
                            "force": "parameters.force",
                        },
                        "outputs": {"return": "derived.growth_flux"},
                    }
                ],
                "tendencies": {"biomass": [{"source": "derived.growth_flux"}]},
            }
        )

        n_steps = 5
        config = Config(
            parameters={"force": xr.DataArray(np.ones(n_steps) * 0.01, dims=["T"])},
            forcings={},
            initial_state={"biomass": xr.DataArray(np.ones((1, 1)) * 100.0, dims=["Y", "X"])},
            execution={"dt": "1d", "time_start": "2000-01-01", "time_end": "2000-01-06"},
        )

        model = compile_model(blueprint, config)
        step_fn = build_step_fn(model, export_variables=["biomass"])

        def loss_fn(params):
            _, outputs = run(step_fn, model, dict(model.state), params)
            return jnp.mean(outputs["biomass"])

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(dict(model.parameters))

        # Gradient w.r.t. force should exist and be non-zero
        assert "force" in grads
        assert grads["force"].shape == (n_steps,)
        assert jnp.any(grads["force"] != 0.0)

    def test_checkpoint_gradient_matches_no_checkpoint(self):
        """Gradients with checkpoint=True and checkpoint=False must be identical."""

        @functional(name="test:forced_growth")
        def forced_growth(biomass, force):
            return biomass * force

        blueprint = Blueprint.from_dict(
            {
                "id": "test-checkpoint-grad",
                "version": "0.1.0",
                "declarations": {
                    "state": {"biomass": {"units": "g", "dims": ["Y", "X"]}},
                    "parameters": {"force": {"units": "1/d", "dims": ["T"]}},
                    "forcings": {},
                },
                "process": [
                    {
                        "func": "test:forced_growth",
                        "inputs": {
                            "biomass": "state.biomass",
                            "force": "parameters.force",
                        },
                        "outputs": {"return": "derived.growth_flux"},
                    }
                ],
                "tendencies": {"biomass": [{"source": "derived.growth_flux"}]},
            }
        )

        n_steps = 10
        config = Config(
            parameters={"force": xr.DataArray(np.ones(n_steps) * 0.01, dims=["T"])},
            forcings={},
            initial_state={"biomass": xr.DataArray(np.ones((3, 3)) * 100.0, dims=["Y", "X"])},
            execution={"dt": "1d", "time_start": "2000-01-01", "time_end": "2000-01-11"},
        )

        model = compile_model(blueprint, config)
        step_fn = build_step_fn(model, export_variables=["biomass"])

        def loss_fn(params, checkpoint):
            _, outputs = run(step_fn, model, dict(model.state), params, checkpoint=checkpoint)
            return jnp.mean(outputs["biomass"])

        params = dict(model.parameters)
        grads_with = jax.grad(lambda p: loss_fn(p, checkpoint=True))(params)
        grads_without = jax.grad(lambda p: loss_fn(p, checkpoint=False))(params)

        np.testing.assert_allclose(
            np.asarray(grads_with["force"]),
            np.asarray(grads_without["force"]),
            rtol=1e-6,
        )
