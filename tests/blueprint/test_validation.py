"""Tests for Blueprint validation pipeline."""

import numpy as np
import pytest
import xarray as xr

from seapopym.blueprint import (
    Blueprint,
    BlueprintValidationError,
    Config,
    ConfigValidationError,
    functional,
    validate_blueprint,
    validate_config,
)
from seapopym.blueprint.registry import REGISTRY
from seapopym.blueprint.validation import BlueprintValidator


@pytest.fixture(autouse=True)
def setup_registry():
    """Save registry, register test functions, restore after."""
    saved = dict(REGISTRY)
    REGISTRY.clear()

    @functional(name="test:simple")
    def simple(x):
        return x * 2

    @functional(
        name="test:growth",
        units={"biomass": "g", "rate": "1/d", "return": "g/d"},
    )
    def growth(biomass, rate):
        return biomass * rate

    @functional(
        name="test:multi",
        outputs=["out1", "out2"],
    )
    def multi(x):
        return x, x * 2

    @functional(name="test:core_func", core_dims={"x": ["C"]})
    def core_func(x):
        return x

    @functional(name="test:core_out", core_dims={"field": ["C"]}, out_dims=["Z"])
    def core_out(field):
        return field

    @functional(name="test:forcing_func")
    def forcing_func(temp):
        return temp

    yield
    REGISTRY.clear()
    REGISTRY.update(saved)


class TestValidateBlueprint:
    """Tests for validate_blueprint."""

    def test_function_not_found(self):
        """Test validation fails for unknown function."""
        bp = Blueprint.from_dict(
            {
                "id": "test",
                "version": "0.1.0",
                "declarations": {
                    "state": {},
                    "parameters": {},
                    "forcings": {},
                    "derived": {},
                },
                "process": [
                    {
                        "func": "nonexistent:func",
                        "inputs": {},
                        "outputs": {},
                    }
                ],
            }
        )

        with pytest.raises(BlueprintValidationError) as exc_info:
            validate_blueprint(bp)
        assert any(e.code == "E101" for e in exc_info.value.validation_errors)

    def test_missing_required_input(self):
        """Test validation fails for missing required input."""
        bp = Blueprint.from_dict(
            {
                "id": "test",
                "version": "0.1.0",
                "declarations": {
                    "state": {"biomass": {"units": "g"}},
                    "parameters": {},
                    "forcings": {},
                    "derived": {},
                },
                "process": [
                    {
                        "func": "test:growth",
                        "inputs": {"biomass": "state.biomass"},  # Missing 'rate'
                        "outputs": {"out": "derived.result"},
                    }
                ],
            }
        )

        with pytest.raises(BlueprintValidationError) as exc_info:
            validate_blueprint(bp)
        assert any(e.code == "E102" for e in exc_info.value.validation_errors)

    def test_output_count_mismatch(self):
        """Test validation fails for wrong number of outputs."""
        bp = Blueprint.from_dict(
            {
                "id": "test",
                "version": "0.1.0",
                "declarations": {
                    "state": {"x": {"units": "1"}},
                    "parameters": {},
                    "forcings": {},
                    "derived": {},
                },
                "process": [
                    {
                        "func": "test:multi",  # Returns 2 outputs
                        "inputs": {"x": "state.x"},
                        "outputs": {"out": "derived.result"},  # Only 1 declared
                    }
                ],
            }
        )

        with pytest.raises(BlueprintValidationError) as exc_info:
            validate_blueprint(bp)
        assert any(e.code == "E107" for e in exc_info.value.validation_errors)

    def test_extra_input_argument(self):
        """Extra input arg not in function signature → E102."""
        bp = Blueprint.from_dict(
            {
                "id": "test",
                "version": "0.1.0",
                "declarations": {
                    "state": {"value": {"units": "g"}, "other": {"units": "1"}},
                },
                "process": [
                    {
                        "func": "test:simple",  # signature: (x)
                        "inputs": {"x": "state.value", "typo": "state.other"},
                        "outputs": {"out": "derived.result"},
                    }
                ],
            }
        )
        with pytest.raises(BlueprintValidationError) as exc_info:
            validate_blueprint(bp)
        e102 = [e for e in exc_info.value.validation_errors if e.code == "E102"]
        assert len(e102) == 1
        assert "typo" in e102[0].extra

    def test_nodes_built_on_success(self):
        """Test that compute_nodes and data_nodes are built when validation succeeds."""
        bp = Blueprint.from_dict(
            {
                "id": "test",
                "version": "0.1.0",
                "declarations": {
                    "state": {"value": {"units": "1"}},
                    "parameters": {},
                    "forcings": {},
                    "derived": {},
                },
                "process": [
                    {
                        "func": "test:simple",
                        "inputs": {"x": "state.value"},
                        "outputs": {"out": "derived.result"},
                    }
                ],
            }
        )

        result = validate_blueprint(bp)
        assert result.valid
        assert len(result.compute_nodes) == 1
        assert len(result.data_nodes) > 0
        assert "state.value" in result.data_nodes
        assert "derived.result" in result.data_nodes

    def test_unit_mismatch_error(self):
        """Test validation fails for unit mismatch (strict equality)."""
        bp = Blueprint.from_dict(
            {
                "id": "test",
                "version": "0.1.0",
                "declarations": {
                    "state": {
                        "biomass": {"units": "kg"},  # Mismatch: function expects 'g'
                    },
                    "parameters": {
                        "rate": {"units": "1/d"},
                    },
                    "forcings": {},
                    "derived": {},
                },
                "process": [
                    {
                        "func": "test:growth",
                        "inputs": {"biomass": "state.biomass", "rate": "parameters.rate"},
                        "outputs": {"out": "derived.result"},
                    }
                ],
            }
        )

        with pytest.raises(BlueprintValidationError) as exc_info:
            validate_blueprint(bp)
        assert any(e.code == "E105" for e in exc_info.value.validation_errors)
        error_msgs = [str(e) for e in exc_info.value.validation_errors if e.code == "E105"]
        assert any("Unit mismatch" in msg for msg in error_msgs)

    def test_tendency_unit_error(self):
        """Test validation fails if tendency source lacks time dimension."""
        from seapopym.blueprint import functional

        @functional(name="test:bad_tendency", units={"return": "count"})
        def bad_tendency(x):
            return x

        bp = Blueprint.from_dict(
            {
                "id": "test",
                "version": "0.1.0",
                "declarations": {
                    "state": {"pop": {"units": "count"}},
                    "parameters": {},
                    "forcings": {},
                },
                "process": [
                    {
                        "func": "test:bad_tendency",
                        "inputs": {"x": "state.pop"},
                        "outputs": {"tendency": "derived.bad_flux"},
                    }
                ],
                "tendencies": {
                    "pop": [{"source": "derived.bad_flux"}],
                },
            }
        )

        with pytest.raises(BlueprintValidationError) as exc_info:
            validate_blueprint(bp)
        assert any(e.code == "E105" for e in exc_info.value.validation_errors)
        error_msgs = [str(e) for e in exc_info.value.validation_errors if e.code == "E105"]
        assert any("lacks a time dimension" in msg for msg in error_msgs)

    def test_aggregated_error_count(self):
        """Test that BlueprintValidationError aggregates all errors."""
        bp = Blueprint.from_dict(
            {
                "id": "test",
                "version": "0.1.0",
                "declarations": {
                    "state": {},
                    "parameters": {},
                    "forcings": {},
                    "derived": {},
                },
                "process": [
                    {
                        "func": "nonexistent:func1",
                        "inputs": {},
                        "outputs": {},
                    },
                    {
                        "func": "nonexistent:func2",
                        "inputs": {},
                        "outputs": {},
                    },
                ],
            }
        )

        with pytest.raises(BlueprintValidationError) as exc_info:
            validate_blueprint(bp)

        err = exc_info.value
        assert err.error_count() == 2
        assert len(err.validation_errors) == err.error_count()
        assert all(e.code == "E101" for e in err.validation_errors)

    def test_aggregated_error_has_warnings(self):
        """Test that BlueprintValidationError carries warnings."""
        # Use the low-level validator to inspect warnings separately
        bp = Blueprint.from_dict(
            {
                "id": "test",
                "version": "0.1.0",
                "declarations": {
                    "state": {"value": {"units": "g"}},
                    "parameters": {},
                    "forcings": {},
                    "derived": {},
                },
                "process": [
                    {
                        "func": "test:simple",
                        "inputs": {"x": "state.value"},
                        "outputs": {"out": "derived.result"},
                    }
                ],
            }
        )

        # Valid blueprint — no exception, just verify result has warnings list
        result = validate_blueprint(bp)
        assert isinstance(result.warnings, list)

    def test_low_level_validator_returns_result(self):
        """Test that BlueprintValidator.validate() returns result without raising."""
        bp = Blueprint.from_dict(
            {
                "id": "test",
                "version": "0.1.0",
                "declarations": {
                    "state": {},
                    "parameters": {},
                    "forcings": {},
                    "derived": {},
                },
                "process": [
                    {
                        "func": "nonexistent:func",
                        "inputs": {},
                        "outputs": {},
                    }
                ],
            }
        )

        # Low-level: returns ValidationResult without raising
        validator = BlueprintValidator()
        result = validator.validate(bp)
        assert not result.valid
        assert any(e.code == "E101" for e in result.errors)

    def test_core_dims_mismatch(self):
        """Input has dims [Y,X] but core_dims needs [C] → E103."""
        bp = Blueprint.from_dict(
            {
                "id": "test",
                "version": "0.1.0",
                "declarations": {
                    "state": {"value": {"units": "g", "dims": ["Y", "X"]}},
                },
                "process": [
                    {
                        "func": "test:core_func",
                        "inputs": {"x": "state.value"},
                        "outputs": {"out": "derived.result"},
                    }
                ],
            }
        )
        with pytest.raises(BlueprintValidationError) as exc_info:
            validate_blueprint(bp)
        assert any(e.code == "E103" for e in exc_info.value.validation_errors)

    def test_core_dims_warning_on_scalar(self):
        """Input without dims but core_dims expected → warning (no error)."""
        bp = Blueprint.from_dict(
            {
                "id": "test",
                "version": "0.1.0",
                "declarations": {
                    "state": {"value": {"units": "g"}},  # No dims
                },
                "process": [
                    {
                        "func": "test:core_func",
                        "inputs": {"x": "state.value"},
                        "outputs": {"out": "derived.result"},
                    }
                ],
            }
        )
        validator = BlueprintValidator()
        result = validator.validate(bp)
        assert any("no dimensions" in w for w in result.warnings)

    def test_forcing_T_stripped(self):
        """Forcing with [T,Y,X] → compute node input gets (Y,X) after T-stripping."""
        bp = Blueprint.from_dict(
            {
                "id": "test",
                "version": "0.1.0",
                "declarations": {
                    "forcings": {"temp": {"units": "degC", "dims": ["T", "Y", "X"]}},
                },
                "process": [
                    {
                        "func": "test:forcing_func",
                        "inputs": {"temp": "forcings.temp"},
                        "outputs": {"out": "derived.result"},
                    }
                ],
            }
        )
        result = validate_blueprint(bp)
        assert result.valid
        node = result.compute_nodes[0]
        assert "T" not in node.input_dims["temp"]
        assert "Y" in node.input_dims["temp"]
        assert "X" in node.input_dims["temp"]

    def test_undeclared_input_variable(self):
        """Input refs state.nonexistent → E106."""
        bp = Blueprint.from_dict(
            {
                "id": "test",
                "version": "0.1.0",
                "declarations": {
                    "state": {"value": {"units": "g"}},
                },
                "process": [
                    {
                        "func": "test:simple",
                        "inputs": {"x": "state.nonexistent"},
                        "outputs": {"out": "derived.result"},
                    }
                ],
            }
        )
        with pytest.raises(BlueprintValidationError) as exc_info:
            validate_blueprint(bp)
        assert any(e.code == "E106" for e in exc_info.value.validation_errors)

    def test_tendency_source_unproduced_warning(self):
        """Tendency refs unproduced derived → warning."""
        bp = Blueprint.from_dict(
            {
                "id": "test",
                "version": "0.1.0",
                "declarations": {
                    "state": {"value": {"units": "g"}},
                },
                "process": [
                    {
                        "func": "test:simple",
                        "inputs": {"x": "state.value"},
                        "outputs": {"out": "derived.result"},
                    }
                ],
                "tendencies": {
                    "value": [{"source": "derived.ghost"}],  # not produced
                },
            }
        )
        validator = BlueprintValidator()
        result = validator.validate(bp)
        assert any("derived.ghost" in w for w in result.warnings)

    def test_output_dims_with_out_dims(self):
        """core_dims + out_dims → broadcast + out_dims."""
        bp = Blueprint.from_dict(
            {
                "id": "test",
                "version": "0.1.0",
                "declarations": {
                    "state": {"field": {"units": "g", "dims": ["Y", "X", "C"]}},
                },
                "process": [
                    {
                        "func": "test:core_out",
                        "inputs": {"field": "state.field"},
                        "outputs": {"out": "derived.result"},
                    }
                ],
            }
        )
        result = validate_blueprint(bp)
        assert result.valid
        # broadcast = (Y, X) [C removed as core_dim], out_dims = [Z]
        output_node = result.data_nodes["derived.result"]
        assert output_node.dims is not None
        assert "Z" in output_node.dims
        assert "C" not in output_node.dims

    def test_output_dims_broadcast_only(self):
        """core_dims without out_dims → broadcast only."""
        bp = Blueprint.from_dict(
            {
                "id": "test",
                "version": "0.1.0",
                "declarations": {
                    "state": {"value": {"units": "g", "dims": ["Y", "X", "C"]}},
                },
                "process": [
                    {
                        "func": "test:core_func",
                        "inputs": {"x": "state.value"},
                        "outputs": {"out": "derived.result"},
                    }
                ],
            }
        )
        result = validate_blueprint(bp)
        assert result.valid
        # broadcast = (Y, X) [C removed as core_dim], no out_dims
        output_node = result.data_nodes["derived.result"]
        assert output_node.dims is not None
        assert "C" not in output_node.dims
        assert "Y" in output_node.dims
        assert "X" in output_node.dims


class TestValidateConfig:
    """Tests for validate_config."""

    def test_valid_config(self):
        """Test validation of a valid config."""
        bp = Blueprint.from_dict(
            {
                "id": "test",
                "version": "0.1.0",
                "declarations": {
                    "state": {"biomass": {"units": "g"}},
                    "parameters": {"rate": {"units": "1/d"}},
                    "forcings": {"temp": {"units": "degC"}},
                    "derived": {},
                },
                "process": [],
            }
        )

        cfg = Config(
            parameters={"rate": xr.DataArray(0.1)},
            forcings={"temp": xr.DataArray(np.random.rand(10), dims=["T"])},
            initial_state={"biomass": xr.DataArray(np.ones(5), dims=["Y"])},
            execution={"time_start": "2000-01-01", "time_end": "2000-12-31"},
        )

        result = validate_config(cfg, bp)
        assert result.valid

    def test_missing_parameter(self):
        """Test validation fails for missing parameter."""
        bp = Blueprint.from_dict(
            {
                "id": "test",
                "version": "0.1.0",
                "declarations": {
                    "state": {},
                    "parameters": {"required_param": {"units": "1/d"}},
                    "forcings": {},
                    "derived": {},
                },
                "process": [],
            }
        )

        cfg = Config.from_dict(
            {
                "parameters": {},  # Missing required_param
                "forcings": {},
                "initial_state": {},
                "execution": {"time_start": "2000-01-01", "time_end": "2000-12-31"},
            }
        )

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(cfg, bp)
        assert any(e.code == "E106" for e in exc_info.value.validation_errors)

    def test_missing_forcing(self):
        """Test validation fails for missing forcing."""
        bp = Blueprint.from_dict(
            {
                "id": "test",
                "version": "0.1.0",
                "declarations": {
                    "state": {},
                    "parameters": {},
                    "forcings": {"temperature": {"units": "degC"}},
                    "derived": {},
                },
                "process": [],
            }
        )

        cfg = Config.from_dict(
            {
                "parameters": {},
                "forcings": {},  # Missing temperature
                "initial_state": {},
                "execution": {"time_start": "2000-01-01", "time_end": "2000-12-31"},
            }
        )

        with pytest.raises(ConfigValidationError):
            validate_config(cfg, bp)

    def test_missing_initial_state(self):
        """Test validation fails for missing initial state."""
        bp = Blueprint.from_dict(
            {
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
        )

        cfg = Config.from_dict(
            {
                "parameters": {},
                "forcings": {},
                "initial_state": {},  # Missing biomass
                "execution": {"time_start": "2000-01-01", "time_end": "2000-12-31"},
            }
        )

        with pytest.raises(ConfigValidationError):
            validate_config(cfg, bp)

    def test_config_aggregated_errors(self):
        """Test that ConfigValidationError aggregates all errors."""
        bp = Blueprint.from_dict(
            {
                "id": "test",
                "version": "0.1.0",
                "declarations": {
                    "state": {"biomass": {"units": "g"}},
                    "parameters": {"rate": {"units": "1/d"}},
                    "forcings": {"temp": {"units": "degC"}},
                    "derived": {},
                },
                "process": [],
            }
        )

        cfg = Config.from_dict(
            {
                "parameters": {},  # Missing rate
                "forcings": {},  # Missing temp
                "initial_state": {},  # Missing biomass
                "execution": {"time_start": "2000-01-01", "time_end": "2000-12-31"},
            }
        )

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(cfg, bp)

        err = exc_info.value
        assert err.error_count() == 3
        assert len(err.validation_errors) == 3
        assert isinstance(err.validation_warnings, list)
