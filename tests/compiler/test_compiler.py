"""Integration tests for the Compiler."""

import numpy as np
import pytest
import xarray as xr

from seapopym.blueprint import Blueprint, Config, functional
from seapopym.blueprint.registry import REGISTRY
from seapopym.compiler import CompiledModel, compile_model
from seapopym.dims import CANONICAL_DIMS


@pytest.fixture(autouse=True)
def setup_registry():
    """Save registry, register test functions, restore after."""
    saved = dict(REGISTRY)
    REGISTRY.clear()

    @functional(name="test:simple_growth")
    def simple_growth(biomass, rate, temp):
        """Simple growth function for testing."""
        return biomass * rate * (temp / 20.0)

    yield
    REGISTRY.clear()
    REGISTRY.update(saved)


class TestCompiler:
    """Tests for Compiler class."""

    @pytest.fixture
    def toy_blueprint(self):
        """Create a toy blueprint for testing."""
        return Blueprint.from_dict(
            {
                "id": "toy-test",
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
                        "func": "test:simple_growth",
                        "inputs": {
                            "biomass": "state.biomass",
                            "rate": "parameters.growth_rate",
                            "temp": "forcings.temperature",
                        },
                        "outputs": {
                            "tendency": "derived.growth_flux",
                        },
                    },
                ],
                "tendencies": {
                    "biomass": [{"source": "derived.growth_flux"}],
                },
            }
        )

    @pytest.fixture
    def toy_config(self):
        """Create a toy config for testing."""
        return Config.from_dict(
            {
                "parameters": {"growth_rate": {"value": 0.1}},
                "forcings": {
                    "temperature": xr.DataArray(
                        np.random.uniform(15, 25, (30, 10, 10)),
                        dims=["T", "Y", "X"],
                    ),
                    "mask": xr.DataArray(
                        np.ones((10, 10)),
                        dims=["Y", "X"],
                    ),
                },
                "initial_state": {
                    "biomass": xr.DataArray(
                        np.ones((10, 10)) * 100.0,
                        dims=["Y", "X"],
                    ),
                },
                "execution": {
                    "dt": "1d",
                    "time_start": "2000-01-01",
                    "time_end": "2000-01-31",  # 30 days
                },
            }
        )

    def test_compile_basic(self, toy_blueprint, toy_config):
        """Test basic compilation."""
        compiled = compile_model(toy_blueprint, toy_config)

        assert isinstance(compiled, CompiledModel)

    def test_compile_shapes(self, toy_blueprint, toy_config):
        """Test that shapes are correctly inferred."""
        compiled = compile_model(toy_blueprint, toy_config)

        assert compiled.shapes["T"] == 30
        assert compiled.shapes["Y"] == 10
        assert compiled.shapes["X"] == 10

    def test_compile_state(self, toy_blueprint, toy_config):
        """Test that state is correctly prepared."""
        compiled = compile_model(toy_blueprint, toy_config)

        assert "biomass" in compiled.state
        assert compiled.state["biomass"].shape == (10, 10)
        np.testing.assert_array_equal(np.asarray(compiled.state["biomass"]), 100.0)

    def test_compile_forcings(self, toy_blueprint, toy_config):
        """Test that forcings are correctly prepared."""
        compiled = compile_model(toy_blueprint, toy_config)

        assert "temperature" in compiled.forcings
        assert "mask" in compiled.forcings
        assert compiled.forcings["temperature"].shape == (30, 10, 10)

    def test_compile_parameters(self, toy_blueprint, toy_config):
        """Test that parameters are correctly prepared."""
        compiled = compile_model(toy_blueprint, toy_config)

        assert "growth_rate" in compiled.parameters
        np.testing.assert_allclose(np.asarray(compiled.parameters["growth_rate"]), 0.1, rtol=1e-6)

    def test_compile_dt(self, toy_blueprint, toy_config):
        """Test that dt is correctly parsed."""
        compiled = compile_model(toy_blueprint, toy_config)

        assert compiled.dt == 86400.0  # 1 day in seconds

    def test_compile_mask_property(self, toy_blueprint, toy_config):
        """Test that mask property works."""
        compiled = compile_model(toy_blueprint, toy_config)

        assert compiled.mask is not None
        assert compiled.mask.shape == (10, 10)

    def test_compile_nodes_exist(self, toy_blueprint, toy_config):
        """Test that compute_nodes and data_nodes are included."""
        compiled = compile_model(toy_blueprint, toy_config)

        assert len(compiled.compute_nodes) > 0
        assert len(compiled.data_nodes) > 0

    def test_compile_tendency_map(self, toy_blueprint, toy_config):
        """Test that tendency_map is populated."""
        compiled = compile_model(toy_blueprint, toy_config)

        assert "biomass" in compiled.tendency_map
        assert len(compiled.tendency_map["biomass"]) == 1
        assert compiled.tendency_map["biomass"][0].source == "derived.growth_flux"

    def test_compile_jax_arrays(self, toy_blueprint, toy_config):
        """Test compilation produces JAX arrays."""
        compiled = compile_model(toy_blueprint, toy_config)

        # Check arrays are JAX arrays
        assert hasattr(compiled.state["biomass"], "device")

    def test_compile_parameters(self, toy_blueprint):
        """Test that parameters are compiled as JAX arrays."""
        config = Config.from_dict(
            {
                "parameters": {"growth_rate": {"value": 0.1}},
                "forcings": {
                    "temperature": xr.DataArray(np.random.rand(10, 5, 5), dims=["T", "Y", "X"]),
                    "mask": xr.DataArray(np.ones((5, 5)), dims=["Y", "X"]),
                },
                "initial_state": {
                    "biomass": xr.DataArray(np.ones((5, 5)), dims=["Y", "X"]),
                },
                "execution": {
                    "dt": "1d",
                    "time_start": "2000-01-01",
                    "time_end": "2000-01-11",  # 10 days
                },
            }
        )

        compiled = compile_model(toy_blueprint, config)

        assert "growth_rate" in compiled.parameters


class TestCompileModelFunction:
    """Tests for compile_model convenience function."""

    def test_basic_usage(self):
        """Test basic usage of compile_model function."""
        blueprint = Blueprint.from_dict(
            {
                "id": "test",
                "version": "0.1.0",
                "declarations": {
                    "state": {"x": {"dims": ["Y", "X"]}},
                    "parameters": {"a": {}},
                    "forcings": {"f": {"dims": ["T", "Y", "X"]}},
                },
                "process": [],
            }
        )

        config = Config.from_dict(
            {
                "parameters": {"a": {"value": 1.0}},
                "forcings": {"f": xr.DataArray(np.ones((5, 3, 4)), dims=["T", "Y", "X"])},
                "initial_state": {"x": xr.DataArray(np.ones((3, 4)), dims=["Y", "X"])},
                "execution": {
                    "dt": "1d",
                    "time_start": "2000-01-01",
                    "time_end": "2000-01-06",  # 5 days
                },
            }
        )

        compiled = compile_model(blueprint, config)

        assert isinstance(compiled, CompiledModel)
        assert compiled.shapes == {"T": 5, "Y": 3, "X": 4}


class TestCompiledModel:
    """Tests for CompiledModel dataclass."""

    @pytest.fixture
    def compiled_model(self):
        """Create a compiled model for testing."""
        blueprint = Blueprint.from_dict(
            {
                "id": "test",
                "version": "0.1.0",
                "declarations": {"state": {"x": {"dims": ["Y", "X"]}}},
                "process": [],
            }
        )

        config = Config.from_dict(
            {
                "forcings": {"f": xr.DataArray(np.ones((10, 5, 5)), dims=["T", "Y", "X"])},
                "initial_state": {"x": xr.DataArray(np.ones((5, 5)), dims=["Y", "X"])},
                "execution": {
                    "dt": "1d",
                    "time_start": "2000-01-01",
                    "time_end": "2000-01-11",
                },
            }
        )

        return compile_model(blueprint, config)

    def test_n_timesteps(self, compiled_model):
        """Test n_timesteps property."""
        assert compiled_model.n_timesteps == 10

    def test_get_state_shape(self, compiled_model):
        """Test get_state_shape method."""
        shape = compiled_model.get_state_shape("x")
        assert shape == (5, 5)

    def test_get_state_shape_missing(self, compiled_model):
        """Test get_state_shape with missing variable."""
        with pytest.raises(KeyError):
            compiled_model.get_state_shape("nonexistent")

    def test_to_numpy(self):
        """Test to_numpy conversion."""
        blueprint = Blueprint.from_dict(
            {
                "id": "test",
                "version": "0.1.0",
                "declarations": {"state": {"x": {"dims": ["Y", "X"]}}},
                "process": [],
            }
        )

        config = Config.from_dict(
            {
                "forcings": {"f": xr.DataArray(np.ones((5, 3, 4)), dims=["T", "Y", "X"])},
                "initial_state": {"x": xr.DataArray(np.ones((3, 4)), dims=["Y", "X"])},
                "execution": {
                    "dt": "1d",
                    "time_start": "2000-01-01",
                    "time_end": "2000-01-06",
                },
            }
        )

        compiled = compile_model(blueprint, config)
        numpy_model = compiled.to_numpy()

        assert isinstance(numpy_model.state["x"], np.ndarray)


class TestDimensionMapping:
    """Tests for dimension mapping during compilation."""

    def test_custom_dimension_names(self):
        """Test compilation with custom dimension names."""
        blueprint = Blueprint.from_dict(
            {
                "id": "test",
                "version": "0.1.0",
                "declarations": {
                    "state": {"x": {"dims": ["Y", "X"]}},
                    "forcings": {"f": {"dims": ["T", "Y", "X"]}},
                },
                "process": [],
            }
        )

        # Data has non-canonical dimension names
        config = Config.from_dict(
            {
                "forcings": {
                    "f": xr.DataArray(
                        np.ones((5, 3, 4)),
                        dims=["time", "lat", "lon"],
                    ),
                },
                "initial_state": {
                    "x": xr.DataArray(np.ones((3, 4)), dims=["lat", "lon"]),
                },
                "dimension_mapping": {
                    "time": "T",
                    "lat": "Y",
                    "lon": "X",
                },
                "execution": {
                    "dt": "1d",
                    "time_start": "2000-01-01",
                    "time_end": "2000-01-06",
                },
            }
        )

        compiled = compile_model(blueprint, config)

        # Shapes should use canonical names
        assert compiled.shapes == {"T": 5, "Y": 3, "X": 4}


class TestCanonicalDims:
    """Tests for canonical dimension handling."""

    def test_canonical_dims_constant(self):
        """Test CANONICAL_DIMS constant."""
        assert CANONICAL_DIMS == ("E", "T", "F", "C", "Z", "Y", "X")

    def test_transpose_to_canonical(self):
        """Test that arrays are transposed to canonical order."""
        blueprint = Blueprint.from_dict(
            {
                "id": "test",
                "version": "0.1.0",
                "declarations": {
                    "forcings": {"f": {"dims": ["T", "Y", "X"]}},
                },
                "process": [],
            }
        )

        # Data in wrong order (X, Y, T instead of T, Y, X)
        config = Config.from_dict(
            {
                "forcings": {
                    "f": xr.DataArray(
                        np.arange(60).reshape(4, 3, 5),  # X=4, Y=3, T=5
                        dims=["X", "Y", "T"],
                    ),
                },
                "execution": {
                    "dt": "1d",
                    "time_start": "2000-01-01",
                    "time_end": "2000-01-06",
                },
            }
        )

        compiled = compile_model(blueprint, config)

        # After transposition, should be (T, Y, X) = (5, 3, 4)
        assert compiled.forcings["f"].shape == (5, 3, 4)
