"""End-to-end integration tests for the engine.

Tests the complete pipeline: Blueprint → Compile → Run
"""

import numpy as np
import pytest

from seapopym.blueprint import Blueprint, Config, clear_registry, functional
from seapopym.compiler import compile_model
from seapopym.engine import GradientRunner, StreamingRunner


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear registry before each test."""
    clear_registry()
    yield
    clear_registry()


class TestE2EBasicSimulation:
    """End-to-end tests for basic simulation workflow."""

    def test_e2e_numpy_streaming(self, tmp_path):
        """Test complete workflow: Blueprint → Compile → StreamingRunner (NumPy)."""
        # 1. Define blueprint
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
                            "tendency": {
                                "target": "tendencies.biomass",
                                "type": "tendency",
                            }
                        },
                    }
                ],
            }
        )

        # 2. Register function
        @functional(name="biol:simple_growth", backend="numpy")
        def simple_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        # 3. Create config with test data
        n_days = 30
        ny, nx = 10, 10

        config = Config.from_dict(
            {
                "parameters": {"growth_rate": {"value": 0.0001}},  # Very small for stability
                "forcings": {
                    "temperature": np.random.uniform(15, 25, (n_days, ny, nx)),
                    "mask": np.ones((ny, nx)),
                },
                "initial_state": {
                    "biomass": np.ones((ny, nx)) * 100.0,
                },
                "execution": {
                    "dt": "1d",
                    "time_start": "2000-01-01",
                    "time_end": "2000-01-31",
                },
            }
        )

        # 4. Compile
        model = compile_model(blueprint, config, backend="numpy")

        assert model.shapes["Y"] == ny
        assert model.shapes["X"] == nx
        assert model.shapes["T"] == n_days

        # 5. Run
        runner = StreamingRunner(model, chunk_size=10)
        final_state, _ = runner.run(str(tmp_path / "output"))

        # 6. Verify
        assert "biomass" in final_state
        assert final_state["biomass"].shape == (ny, nx)
        # Biomass should have grown (rate > 0, temp > 0)
        assert np.all(final_state["biomass"] >= 100.0)
        assert not np.any(np.isnan(final_state["biomass"]))

    def test_e2e_jax_streaming(self, tmp_path):
        """Test complete workflow: Blueprint → Compile → StreamingRunner (JAX)."""
        pytest.importorskip("jax")
        import jax.numpy as jnp

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
                            "tendency": {
                                "target": "tendencies.biomass",
                                "type": "tendency",
                            }
                        },
                    }
                ],
            }
        )

        @functional(name="biol:simple_growth", backend="jax")
        def simple_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        n_days = 20
        ny, nx = 8, 8

        config = Config.from_dict(
            {
                "parameters": {"growth_rate": {"value": 0.0001}},
                "forcings": {
                    "temperature": np.random.uniform(15, 25, (n_days, ny, nx)),
                    "mask": np.ones((ny, nx)),
                },
                "initial_state": {
                    "biomass": np.ones((ny, nx)) * 100.0,
                },
                "execution": {
                    "dt": "1d",
                    "time_start": "2000-01-01",
                    "time_end": "2000-01-21",
                },
            }
        )

        model = compile_model(blueprint, config, backend="jax")
        runner = StreamingRunner(model, chunk_size=10)
        final_state, _ = runner.run(str(tmp_path / "output"))

        assert "biomass" in final_state
        assert jnp.all(final_state["biomass"] >= 100.0)

    def test_e2e_jax_gradient(self):
        """Test complete workflow: Blueprint → Compile → GradientRunner (JAX)."""
        pytest.importorskip("jax")
        import jax.numpy as jnp

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
                            "tendency": {
                                "target": "tendencies.biomass",
                                "type": "tendency",
                            }
                        },
                    }
                ],
            }
        )

        @functional(name="biol:simple_growth", backend="jax")
        def simple_growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        # Smaller grid for gradient runner (memory)
        n_days = 10
        ny, nx = 5, 5

        config = Config.from_dict(
            {
                "parameters": {"growth_rate": {"value": 0.0001}},
                "forcings": {
                    "temperature": np.ones((n_days, ny, nx)) * 20.0,
                    "mask": np.ones((ny, nx)),
                },
                "initial_state": {
                    "biomass": np.ones((ny, nx)) * 100.0,
                },
                "execution": {
                    "dt": "1d",
                    "time_start": "2000-01-01",
                    "time_end": "2000-01-11",
                },
            }
        )

        model = compile_model(blueprint, config, backend="jax")
        runner = GradientRunner(model)
        final_state, outputs = runner.run()

        assert "biomass" in final_state
        assert jnp.all(final_state["biomass"] >= 100.0)


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
                            "tendency": {
                                "target": "tendencies.biomass",
                                "type": "tendency",
                            }
                        },
                    }
                ],
            }
        )

        @functional(name="biol:growth", backend="numpy")
        def growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        # Create mask with land (zeros)
        mask = np.ones((10, 10))
        mask[:3, :] = 0  # First 3 rows are land

        config = Config.from_dict(
            {
                "parameters": {"growth_rate": {"value": 0.0001}},
                "forcings": {
                    "temperature": np.ones((20, 10, 10)) * 20.0,
                    "mask": mask,
                },
                "initial_state": {
                    "biomass": np.ones((10, 10)) * 100.0,
                },
                "execution": {
                    "dt": "1d",
                    "time_start": "2000-01-01",
                    "time_end": "2000-01-21",
                },
            }
        )

        model = compile_model(blueprint, config, backend="numpy")
        runner = StreamingRunner(model, chunk_size=10)
        final_state, _ = runner.run(str(tmp_path / "output"))

        # Masked regions should be zero
        np.testing.assert_array_equal(final_state["biomass"][:3, :], 0.0)
        # Unmasked regions should have grown
        assert np.all(final_state["biomass"][3:, :] > 0)


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
                            "tendency": {
                                "target": "tendencies.biomass_growth",
                                "type": "tendency",
                            }
                        },
                    },
                    {
                        "func": "proc:mortality",
                        "inputs": {
                            "biomass": "state.biomass",
                            "rate": "parameters.mortality_rate",
                        },
                        "outputs": {
                            "tendency": {
                                "target": "tendencies.biomass_mortality",
                                "type": "tendency",
                            }
                        },
                    },
                ],
            }
        )

        @functional(name="proc:growth", backend="numpy")
        def growth(biomass, rate, temp):
            return biomass * rate * (temp / 20.0)

        @functional(name="proc:mortality", backend="numpy")
        def mortality(biomass, rate):
            return -biomass * rate  # Negative tendency

        config = Config.from_dict(
            {
                "parameters": {
                    "growth_rate": {"value": 0.0002},
                    "mortality_rate": {"value": 0.0001},
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
                    "time_end": "2000-01-31",
                },
            }
        )

        model = compile_model(blueprint, config, backend="numpy")
        runner = StreamingRunner(model, chunk_size=15)
        final_state, _ = runner.run(str(tmp_path / "output"))

        # Net growth rate is positive (0.0002 - 0.0001 = 0.0001)
        # So biomass should increase
        assert np.all(final_state["biomass"] > 100.0)
