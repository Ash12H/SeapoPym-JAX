"""Tests for the composable Runner."""

import jax.numpy as jnp
import numpy as np
import pytest

from seapopym.blueprint import Config, functional
from seapopym.compiler import compile_model
from seapopym.engine.exceptions import ChunkingError, EngineError
from seapopym.engine.runner import Runner, RunnerConfig


class TestRunnerConfig:
    """Tests for RunnerConfig validation."""

    def test_simulation_preset(self):
        """Verify config from Runner.simulation()."""
        runner = Runner.simulation(chunk_size=100, output="memory")
        assert runner.config.chunk_size == 100
        assert runner.config.output_mode == "memory"
        assert runner.config.vmap_params is False
        assert runner.config.loop_mode == "scan"

    def test_optimization_preset(self):
        """Verify config from Runner.optimization()."""
        runner = Runner.optimization()
        assert runner.config.vmap_params is False
        assert runner.config.output_mode == "raw"
        assert runner.config.chunk_size is None

    def test_optimization_vmap_preset(self):
        """Verify config from Runner.optimization(vmap=True)."""
        runner = Runner.optimization(vmap=True)
        assert runner.config.vmap_params is True
        assert runner.config.output_mode == "raw"

    def test_invalid_disk_fori_loop(self):
        """disk output requires scan loop mode."""
        with pytest.raises(EngineError):
            RunnerConfig(output_mode="disk", loop_mode="fori_loop")

    def test_invalid_vmap_with_disk(self):
        """vmap requires raw output mode."""
        with pytest.raises(EngineError):
            RunnerConfig(vmap_params=True, output_mode="disk")

    def test_invalid_vmap_with_chunking(self):
        """vmap is incompatible with chunking."""
        with pytest.raises(EngineError):
            RunnerConfig(vmap_params=True, chunk_size=10)


class TestRunnerSimulation:
    """Tests for Runner in simulation mode."""

    @pytest.fixture
    def runner_config(self):
        """Config with 30 timesteps."""
        return Config.from_dict(
            {
                "parameters": {"growth_rate": {"value": 0.001}},
                "forcings": {
                    "temperature": np.ones((30, 5, 5)) * 20.0,
                    "mask": np.ones((5, 5)),
                },
                "initial_state": {"biomass": np.ones((5, 5)) * 100.0},
                "execution": {
                    "dt": "1d",
                    "time_start": "2000-01-01",
                    "time_end": "2000-01-31",
                },
            }
        )

    def test_simulation_run_memory(self, simple_blueprint, runner_config):
        """Test simulation producing in-memory xarray.Dataset."""

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        model = compile_model(simple_blueprint, runner_config)
        runner = Runner.simulation(chunk_size=10)
        final_state, outputs = runner.run(model)

        assert "biomass" in final_state
        assert outputs is not None
        assert "biomass" in outputs
        assert outputs["biomass"].shape == (30, 5, 5)

    def test_simulation_run_disk(self, simple_blueprint, runner_config, tmp_path):
        """Test simulation writing to disk (Zarr)."""

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        model = compile_model(simple_blueprint, runner_config)
        runner = Runner.simulation(chunk_size=10, output="disk")
        final_state, _ = runner.run(model, output_path=str(tmp_path / "output"))

        assert "biomass" in final_state
        import zarr

        store = zarr.open(str(tmp_path / "output"), mode="r")
        assert store["biomass"].shape[0] == 30  # type: ignore[union-attr]

    def test_invalid_chunk_size(self, simple_blueprint, simple_config):
        """Negative chunk_size raises ChunkingError."""

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        model = compile_model(simple_blueprint, simple_config)
        runner = Runner.simulation(chunk_size=-1)

        with pytest.raises(ChunkingError):
            runner.run(model)


class TestRunnerOptimization:
    """Tests for Runner in optimization mode."""

    def test_optimization_run_single(self, simple_blueprint, simple_config):
        """Run with free_params returns outputs dict with correct shapes."""

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        model = compile_model(simple_blueprint, simple_config)
        runner = Runner.optimization()

        outputs = runner(model, {"growth_rate": jnp.array(0.1)})

        assert isinstance(outputs, dict)
        # Outputs should have time dimension (10 timesteps)
        for key, val in outputs.items():
            assert val.shape[0] == 10

    def test_optimization_merges_params(self, simple_blueprint, simple_config):
        """free_params override model.parameters."""

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        model = compile_model(simple_blueprint, simple_config)
        runner = Runner.optimization()

        # Run with doubled growth rate
        outputs_high = runner(model, {"growth_rate": jnp.array(0.2)})
        outputs_low = runner(model, {"growth_rate": jnp.array(0.05)})

        # Higher growth rate → higher biomass
        high_final = outputs_high["biomass"][-1]
        low_final = outputs_low["biomass"][-1]
        assert jnp.all(high_final > low_final)

    def test_optimization_run_vmap(self, simple_blueprint, simple_config):
        """Vmap over N parameter sets."""

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        model = compile_model(simple_blueprint, simple_config)
        runner = Runner.optimization(vmap=True)

        # Population of 4 growth rates
        free_params = {"growth_rate": jnp.array([0.05, 0.1, 0.15, 0.2])}
        outputs = runner(model, free_params)

        # Each output should have batch dim + time + spatial
        assert outputs["biomass"].shape[0] == 4  # batch
        assert outputs["biomass"].shape[1] == 10  # time
