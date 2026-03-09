"""End-to-end integration tests for the engine.

Tests the complete pipeline: Blueprint → Compile → Run
"""

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from seapopym.blueprint import Blueprint, Config, functional
from seapopym.compiler import compile_model
from seapopym.engine import Runner
from seapopym.engine.runner import RunnerConfig


class TestE2EBasicSimulation:
    """End-to-end tests for basic simulation workflow."""

    def test_e2e_streaming(self, tmp_path):
        """Test complete workflow: Blueprint → Compile → Runner.simulation()."""
        blueprint = Blueprint.from_dict(
            {
                "id": "toy-growth",
                "version": "0.1.0",
                "declarations": {
                    "state": {
                        "biomass": {"units": "g", "dims": ["Y", "X"]},
                    },
                    "parameters": {
                        "growth_rate": {"units": "1/d"},
                    },
                    "forcings": {
                        "temperature": {"units": "degC", "dims": ["T", "Y", "X"]},
                        "mask": {"dims": ["Y", "X"]},
                    },
                },
                "process": [
                    {
                        "func": "biol:simple_growth",
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

        @functional(name="biol:simple_growth")
        def simple_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        n_days = 20
        ny, nx = 8, 8

        config = Config(
            parameters={"growth_rate": xr.DataArray(0.0001)},
            forcings={
                "temperature": xr.DataArray(
                    np.random.uniform(15, 25, (n_days, ny, nx)), dims=["T", "Y", "X"]
                ),
                "mask": xr.DataArray(np.ones((ny, nx)), dims=["Y", "X"]),
            },
            initial_state={
                "biomass": xr.DataArray(np.ones((ny, nx)) * 100.0, dims=["Y", "X"]),
            },
            execution={
                "dt": "1d",
                "time_start": "2000-01-01",
                "time_end": "2000-01-21",
            },
        )

        model = compile_model(blueprint, config)
        runner = Runner.simulation(chunk_size=10)
        final_state, _ = runner.run(model, output_path=str(tmp_path / "output"))

        assert "biomass" in final_state
        assert jnp.all(final_state["biomass"] >= 100.0)

    def test_e2e_optimization_runner(self):
        """Test complete workflow: Blueprint → Compile → Runner.optimization()."""

        blueprint = Blueprint.from_dict(
            {
                "id": "toy-growth",
                "version": "0.1.0",
                "declarations": {
                    "state": {
                        "biomass": {"units": "g", "dims": ["Y", "X"]},
                    },
                    "parameters": {
                        "growth_rate": {"units": "1/d"},
                    },
                    "forcings": {
                        "temperature": {"units": "degC", "dims": ["T", "Y", "X"]},
                        "mask": {"dims": ["Y", "X"]},
                    },
                },
                "process": [
                    {
                        "func": "biol:simple_growth",
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

        @functional(name="biol:simple_growth")
        def simple_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        n_days = 10
        ny, nx = 5, 5

        config = Config(
            parameters={"growth_rate": xr.DataArray(0.0001)},
            forcings={
                "temperature": xr.DataArray(
                    np.ones((n_days, ny, nx)) * 20.0, dims=["T", "Y", "X"]
                ),
                "mask": xr.DataArray(np.ones((ny, nx)), dims=["Y", "X"]),
            },
            initial_state={
                "biomass": xr.DataArray(np.ones((ny, nx)) * 100.0, dims=["Y", "X"]),
            },
            execution={
                "dt": "1d",
                "time_start": "2000-01-01",
                "time_end": "2000-01-11",
            },
        )

        model = compile_model(blueprint, config)
        runner = Runner.optimization()
        outputs = runner(model, dict(model.parameters))

        assert "biomass" in outputs
        # Biomass should have grown (output is per-timestep)
        assert jnp.all(outputs["biomass"][-1] >= 100.0)


class TestE2EMaskBehavior:
    """Tests for mask handling in E2E workflow."""

    def test_mask_zeros_state(self, tmp_path):
        """Test that masked regions stay at zero."""
        blueprint = Blueprint.from_dict(
            {
                "id": "masked-model",
                "version": "0.1.0",
                "declarations": {
                    "state": {
                        "biomass": {"units": "g", "dims": ["Y", "X"]},
                    },
                    "parameters": {
                        "growth_rate": {"units": "1/d"},
                    },
                    "forcings": {
                        "temperature": {"units": "degC", "dims": ["T", "Y", "X"]},
                        "mask": {"dims": ["Y", "X"]},
                    },
                },
                "process": [
                    {
                        "func": "biol:growth",
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

        @functional(name="biol:growth")
        def growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        # Create mask with land (zeros)
        mask = np.ones((10, 10))
        mask[:3, :] = 0  # First 3 rows are land

        config = Config(
            parameters={"growth_rate": xr.DataArray(0.0001)},
            forcings={
                "temperature": xr.DataArray(
                    np.ones((20, 10, 10)) * 20.0, dims=["T", "Y", "X"]
                ),
                "mask": xr.DataArray(mask, dims=["Y", "X"]),
            },
            initial_state={
                "biomass": xr.DataArray(np.ones((10, 10)) * 100.0, dims=["Y", "X"]),
            },
            execution={
                "dt": "1d",
                "time_start": "2000-01-01",
                "time_end": "2000-01-21",
            },
        )

        model = compile_model(blueprint, config)
        runner = Runner.simulation(chunk_size=10)
        final_state, _ = runner.run(model, output_path=str(tmp_path / "output"))

        # Masked regions should be zero
        np.testing.assert_array_equal(np.asarray(final_state["biomass"][:3, :]), 0.0)
        # Unmasked regions should have grown
        assert np.all(np.asarray(final_state["biomass"][3:, :]) > 0)


class TestE2EMultiProcess:
    """Tests for models with multiple processes."""

    def test_two_processes(self, tmp_path):
        """Test model with growth and mortality."""
        blueprint = Blueprint.from_dict(
            {
                "id": "growth-mortality",
                "version": "0.1.0",
                "declarations": {
                    "state": {
                        "biomass": {"units": "g", "dims": ["Y", "X"]},
                    },
                    "parameters": {
                        "growth_rate": {"units": "1/d"},
                        "mortality_rate": {"units": "1/d"},
                    },
                    "forcings": {
                        "temperature": {"units": "degC", "dims": ["T", "Y", "X"]},
                        "mask": {"dims": ["Y", "X"]},
                    },
                },
                "process": [
                    {
                        "func": "proc:growth",
                        "inputs": {
                            "biomass": "state.biomass",
                            "rate": "parameters.growth_rate",
                            "temp": "forcings.temperature",
                        },
                        "outputs": {
                            "tendency": "derived.biomass_growth",
                        },
                    },
                    {
                        "func": "proc:mortality",
                        "inputs": {
                            "biomass": "state.biomass",
                            "rate": "parameters.mortality_rate",
                        },
                        "outputs": {
                            "tendency": "derived.biomass_mortality",
                        },
                    },
                ],
                "tendencies": {
                    "biomass": [
                        {"source": "derived.biomass_growth"},
                        {"source": "derived.biomass_mortality"},
                    ],
                },
            }
        )

        @functional(name="proc:growth")
        def growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        @functional(name="proc:mortality")
        def mortality(biomass, rate):
            return -biomass * rate  # Negative tendency

        config = Config(
            parameters={
                "growth_rate": xr.DataArray(0.0002),
                "mortality_rate": xr.DataArray(0.0001),
            },
            forcings={
                "temperature": xr.DataArray(
                    np.ones((30, 5, 5)) * 20.0, dims=["T", "Y", "X"]
                ),
                "mask": xr.DataArray(np.ones((5, 5)), dims=["Y", "X"]),
            },
            initial_state={
                "biomass": xr.DataArray(np.ones((5, 5)) * 100.0, dims=["Y", "X"]),
            },
            execution={
                "dt": "1d",
                "time_start": "2000-01-01",
                "time_end": "2000-01-31",
            },
        )

        model = compile_model(blueprint, config)
        runner = Runner.simulation(chunk_size=15)
        final_state, _ = runner.run(model, output_path=str(tmp_path / "output"))

        # Net growth rate is positive (0.0002 - 0.0001 = 0.0001)
        # So biomass should increase
        assert np.all(np.asarray(final_state["biomass"]) > 100.0)


class TestE2EChunkedVmap:
    """Tests for chunked + vmap combination."""

    def test_chunked_vmap_matches_full_vmap(self):
        """Chunked+vmap produces the same results as full vmap."""
        blueprint = Blueprint.from_dict(
            {
                "id": "toy-growth",
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
                        "func": "biol:chunked_growth",
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

        @functional(name="biol:chunked_growth")
        def chunked_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        n_days, ny, nx = 10, 3, 3
        config = Config(
            parameters={"growth_rate": xr.DataArray(0.0001)},
            forcings={
                "temperature": xr.DataArray(
                    np.ones((n_days, ny, nx)) * 20.0, dims=["T", "Y", "X"]
                ),
                "mask": xr.DataArray(np.ones((ny, nx)), dims=["Y", "X"]),
            },
            initial_state={
                "biomass": xr.DataArray(np.ones((ny, nx)) * 100.0, dims=["Y", "X"]),
            },
            execution={
                "dt": "1d",
                "time_start": "2000-01-01",
                "time_end": "2000-01-11",
            },
        )
        model = compile_model(blueprint, config)

        population = {"growth_rate": jnp.array([0.00005, 0.0001, 0.00015])}

        # Full vmap (no chunking)
        runner_full = Runner(RunnerConfig(vmap_params=True, output_mode="raw"))
        outputs_full = runner_full(model, population, export_variables=["biomass"])

        # Chunked+vmap
        runner_chunked = Runner(RunnerConfig(vmap_params=True, chunk_size=5, output_mode="raw"))
        outputs_chunked = runner_chunked(model, population, export_variables=["biomass"])

        # Results must match
        np.testing.assert_allclose(
            np.asarray(outputs_full["biomass"]),
            np.asarray(outputs_chunked["biomass"]),
            rtol=1e-5,
        )
