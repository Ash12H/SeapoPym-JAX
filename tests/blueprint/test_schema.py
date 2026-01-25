"""Tests for Blueprint and Config schema classes."""

from pathlib import Path

import pytest

from seapopym.blueprint import (
    Blueprint,
    Config,
    Declarations,
    ExecutionParams,
    ParameterValue,
    ProcessOutput,
    ProcessStep,
    VariableDeclaration,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestVariableDeclaration:
    """Tests for VariableDeclaration."""

    def test_minimal_declaration(self):
        """Test creating a minimal declaration."""
        decl = VariableDeclaration()
        assert decl.units is None
        assert decl.dims is None
        assert decl.description is None

    def test_full_declaration(self):
        """Test creating a full declaration."""
        decl = VariableDeclaration(
            units="g",
            dims=["Y", "X", "C"],
            description="Biomass field",
        )
        assert decl.units == "g"
        assert decl.dims == ["Y", "X", "C"]
        assert decl.description == "Biomass field"


class TestParameterValue:
    """Tests for ParameterValue."""

    def test_simple_value(self):
        """Test creating a simple parameter value."""
        param = ParameterValue(value=0.1)
        assert param.value == 0.1
        assert param.trainable is False
        assert param.bounds is None

    def test_trainable_with_bounds(self):
        """Test creating a trainable parameter with bounds."""
        param = ParameterValue(value=0.1, trainable=True, bounds=[0.01, 0.5])
        assert param.trainable is True
        assert param.bounds == (0.01, 0.5)

    def test_bounds_validation(self):
        """Test that invalid bounds raise error."""
        with pytest.raises(ValueError):
            ParameterValue(value=0.1, bounds=[1, 2, 3])


class TestProcessStep:
    """Tests for ProcessStep."""

    def test_valid_process_step(self):
        """Test creating a valid process step."""
        step = ProcessStep(
            func="biol:growth",
            inputs={"biomass": "state.biomass", "rate": "parameters.growth_rate"},
            outputs={"tendency": ProcessOutput(target="tendencies.growth", type="tendency")},
        )
        assert step.func == "biol:growth"
        assert "biomass" in step.inputs

    def test_invalid_func_format(self):
        """Test that invalid function format raises error."""
        with pytest.raises(ValueError, match="namespace:name"):
            ProcessStep(
                func="invalid_format",
                inputs={},
                outputs={},
            )


class TestDeclarations:
    """Tests for Declarations."""

    def test_empty_declarations(self):
        """Test creating empty declarations."""
        decl = Declarations()
        assert decl.state == {}
        assert decl.parameters == {}

    def test_flat_declarations(self):
        """Test flat variable declarations."""
        decl = Declarations(
            state={"biomass": {"units": "g", "dims": ["Y", "X"]}},
            parameters={"growth_rate": {"units": "1/d"}},
        )

        all_vars = decl.get_all_variables()
        assert "state.biomass" in all_vars
        assert "parameters.growth_rate" in all_vars
        assert all_vars["state.biomass"].units == "g"

    def test_hierarchical_declarations(self):
        """Test hierarchical (functional group) declarations."""
        decl = Declarations(
            state={
                "tuna": {
                    "biomass": {"units": "g", "dims": ["Y", "X", "C"]},
                },
                "zooplankton": {
                    "biomass": {"units": "g", "dims": ["Y", "X"]},
                },
            }
        )

        all_vars = decl.get_all_variables()
        assert "state.tuna.biomass" in all_vars
        assert "state.zooplankton.biomass" in all_vars


class TestBlueprint:
    """Tests for Blueprint class."""

    def test_from_dict(self):
        """Test creating Blueprint from dict."""
        data = {
            "id": "test-model",
            "version": "0.1.0",
            "declarations": {
                "state": {"biomass": {"units": "g", "dims": ["Y", "X"]}},
                "parameters": {"rate": {"units": "1/d"}},
                "forcings": {},
                "derived": {},
            },
            "process": [
                {
                    "func": "biol:growth",
                    "inputs": {"biomass": "state.biomass", "rate": "parameters.rate"},
                    "outputs": {"out": {"target": "tendencies.growth", "type": "tendency"}},
                }
            ],
        }

        bp = Blueprint.from_dict(data)
        assert bp.id == "test-model"
        assert bp.version == "0.1.0"
        assert len(bp.process) == 1

    def test_load_from_yaml(self):
        """Test loading Blueprint from YAML file."""
        yaml_path = FIXTURES_DIR / "toy_model.yaml"
        if not yaml_path.exists():
            pytest.skip("Fixture file not found")

        bp = Blueprint.load(yaml_path)
        assert bp.id == "toy-growth"
        assert bp.version == "0.1.0"

    def test_get_variable(self):
        """Test getting variable by path."""
        data = {
            "id": "test",
            "version": "0.1.0",
            "declarations": {
                "state": {"biomass": {"units": "g"}},
                "parameters": {},
                "forcings": {},
                "derived": {},
            },
            "process": [],
        }

        bp = Blueprint.from_dict(data)
        var = bp.get_variable("state.biomass")
        assert var is not None
        assert var.units == "g"


class TestConfig:
    """Tests for Config class."""

    def test_from_dict(self):
        """Test creating Config from dict."""
        data = {
            "parameters": {"growth_rate": {"value": 0.1}},
            "forcings": {"temperature": "/path/to/temp.nc"},
            "initial_state": {"biomass": "/path/to/init.nc"},
            "execution": {"dt": "1d"},
        }

        cfg = Config.from_dict(data)
        assert cfg.parameters["growth_rate"]["value"] == 0.1
        assert cfg.execution.dt == "1d"

    def test_load_from_yaml(self):
        """Test loading Config from YAML file."""
        yaml_path = FIXTURES_DIR / "toy_config.yaml"
        if not yaml_path.exists():
            pytest.skip("Fixture file not found")

        cfg = Config.load(yaml_path)
        assert cfg.model == "./toy_model.yaml"

    def test_get_parameter_value(self):
        """Test getting parameter value by path."""
        cfg = Config.from_dict(
            {
                "parameters": {
                    "simple": {"value": 0.1},
                    "group": {
                        "nested": {"value": 0.2},
                    },
                },
                "forcings": {},
                "initial_state": {},
            }
        )

        assert cfg.get_parameter_value("simple")["value"] == 0.1
        assert cfg.get_parameter_value("group.nested")["value"] == 0.2
        assert cfg.get_parameter_value("nonexistent") is None


class TestExecutionParams:
    """Tests for ExecutionParams."""

    def test_defaults(self):
        """Test default values."""
        params = ExecutionParams()
        assert params.dt == "1d"
        assert params.time_range is None

    def test_time_range_conversion(self):
        """Test time_range list to tuple conversion."""
        params = ExecutionParams(time_range=["2020-01-01", "2020-12-31"])
        assert params.time_range == ("2020-01-01", "2020-12-31")
