"""Tests for runner implementations."""

import numpy as np
import pytest

from seapopym.blueprint import Config, functional
from seapopym.compiler import compile_model
from seapopym.engine.exceptions import ChunkingError
from seapopym.engine.runners import StreamingRunner


class TestStreamingRunner:
    """Tests for StreamingRunner."""

    @pytest.fixture
    def runner_config(self):
        """Config with 30 timesteps (runner tests need more steps for chunking)."""
        return Config.from_dict(
            {
                "parameters": {
                    "growth_rate": {"value": 0.001},  # Small rate to avoid overflow
                },
                "forcings": {
                    "temperature": np.ones((30, 5, 5)) * 20.0,
                    "mask": np.ones((5, 5)),
                },
                "initial_state": {
                    "biomass": np.ones((5, 5)) * 100.0,
                },
                "execution": {
                    "dt": "1d",
                    "time_start": "2000-01-01",
                    "time_end": "2000-01-31",  # 30 days
                },
            }
        )

    def test_init(self, simple_blueprint, runner_config):
        """Test runner initialization."""

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        model = compile_model(simple_blueprint, runner_config)
        runner = StreamingRunner(model, chunk_size=10)

        assert runner.chunk_size == 10
        assert runner.model is model

    def test_chunk_size_from_config(self, simple_blueprint, runner_config):
        """Test that runner uses chunk_size from model config if chunk_size not provided."""
        # Manually set chunk_size in config before compiling
        runner_config.execution.chunk_size = 15

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        model = compile_model(simple_blueprint, runner_config)
        runner = StreamingRunner(model)

        assert runner.chunk_size == 15

    def test_run(self, simple_blueprint, runner_config, tmp_path):
        """Test running simulation."""

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        model = compile_model(simple_blueprint, runner_config)
        runner = StreamingRunner(model, chunk_size=10)

        output_path = tmp_path / "output"
        final_state, _ = runner.run(str(output_path))

        # Check final state
        assert "biomass" in final_state
        assert final_state["biomass"].shape == (5, 5)
        # Biomass should have grown
        assert np.all(np.asarray(final_state["biomass"]) >= 100.0)

    def test_chunking_exact_division(self, simple_blueprint, runner_config, tmp_path):
        """Test chunking when timesteps divide evenly."""

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        model = compile_model(simple_blueprint, runner_config)
        runner = StreamingRunner(model, chunk_size=10)  # 30 / 10 = 3 chunks

        output_path = tmp_path / "output"
        runner.run(str(output_path))

        # Check output has all timesteps
        import zarr

        store = zarr.open(str(output_path), mode="r")
        assert store["biomass"].shape[0] == 30  # type: ignore[union-attr]

    def test_chunking_with_remainder(self, simple_blueprint, tmp_path):
        """Test chunking with remainder timesteps."""
        config = Config.from_dict(
            {
                "parameters": {"growth_rate": {"value": 0.001}},
                "forcings": {
                    "temperature": np.ones((25, 5, 5)) * 20.0,  # 25 timesteps
                    "mask": np.ones((5, 5)),
                },
                "initial_state": {"biomass": np.ones((5, 5)) * 100.0},
                "execution": {
                    "dt": "1d",
                    "time_start": "2000-01-01",
                    "time_end": "2000-01-26",  # 25 days
                },
            }
        )

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        model = compile_model(simple_blueprint, config)
        runner = StreamingRunner(model, chunk_size=10)  # 25 / 10 = 2 full + 5 remainder

        output_path = tmp_path / "output"
        runner.run(str(output_path))

        import zarr

        store = zarr.open(str(output_path), mode="r")
        assert store["biomass"].shape[0] == 25  # type: ignore[union-attr]

    def test_invalid_chunk_size(self, simple_blueprint, simple_config, tmp_path):
        """Test error for invalid chunk size."""

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        model = compile_model(simple_blueprint, simple_config)
        runner = StreamingRunner(model, chunk_size=-1)

        with pytest.raises(ChunkingError):
            runner.run(str(tmp_path / "output"))

    def test_run_in_memory(self, simple_blueprint, runner_config):
        """Test running simulation with in-memory output (no path)."""

        @functional(name="test:growth")
        def test_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        model = compile_model(simple_blueprint, runner_config)
        runner = StreamingRunner(model, chunk_size=10)

        final_state, outputs = runner.run(output_path=None)

        # Check return signature
        assert "biomass" in final_state
        assert outputs is not None
        assert "biomass" in outputs

        # Check shape
        # Input has 30 timesteps. Output should correspond.
        assert outputs["biomass"].shape == (30, 5, 5)

        # Check integrity
        # Last output step should match final state (since output is state var 'biomass')
        np.testing.assert_array_equal(np.asarray(outputs["biomass"][-1]), np.asarray(final_state["biomass"]))
