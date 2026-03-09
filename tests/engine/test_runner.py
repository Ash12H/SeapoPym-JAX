"""Tests for run() and simulate()."""

import numpy as np
import pytest

from seapopym.blueprint import functional
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
