"""Tests for Blueprint validation pipeline."""

import pytest

from seapopym.blueprint import (
    Blueprint,
    Config,
    clear_registry,
    functional,
    validate_blueprint,
    validate_config,
)


@pytest.fixture(autouse=True)
def setup_registry():
    """Setup test functions in registry."""
    clear_registry()

    @functional(name="test:simple", backend="jax")
    def simple(x):
        return x * 2

    @functional(
        name="test:growth",
        backend="jax",
        units={"biomass": "g", "rate": "1/d", "return": "g/d"},
    )
    def growth(biomass, rate):
        return biomass * rate

    @functional(
        name="test:multi",
        backend="jax",
        outputs=["out1", "out2"],
    )
    def multi(x):
        return x, x * 2

    yield
    clear_registry()


class TestValidateBlueprint:
    """Tests for validate_blueprint."""

    def test_valid_simple_blueprint(self):
        """Test validation of a simple valid blueprint."""
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
                        "outputs": {"out": {"target": "derived.result", "type": "derived"}},
                    }
                ],
            }
        )

        result = validate_blueprint(bp, backend="jax")
        assert result.valid
        assert len(result.errors) == 0

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

        result = validate_blueprint(bp, backend="jax")
        assert not result.valid
        assert any(e.code == "E101" for e in result.errors)

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
                        "outputs": {"out": {"target": "derived.result", "type": "derived"}},
                    }
                ],
            }
        )

        result = validate_blueprint(bp, backend="jax")
        assert not result.valid
        assert any(e.code == "E102" for e in result.errors)

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
                        "outputs": {"out": {"target": "derived.result", "type": "derived"}},  # Only 1 declared
                    }
                ],
            }
        )

        result = validate_blueprint(bp, backend="jax")
        assert not result.valid
        assert any(e.code == "E107" for e in result.errors)

    def test_graph_built_on_success(self):
        """Test that graph is built when validation succeeds."""
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
                        "outputs": {"out": {"target": "derived.result", "type": "derived"}},
                    }
                ],
            }
        )

        result = validate_blueprint(bp, backend="jax")
        assert result.valid
        assert result.graph is not None
        assert len(result.graph.nodes) > 0


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

        cfg = Config.from_dict(
            {
                "parameters": {"rate": {"value": 0.1}},
                "forcings": {"temp": "/path/to/temp.nc"},
                "initial_state": {"biomass": "/path/to/init.nc"},
            }
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
            }
        )

        result = validate_config(cfg, bp)
        assert not result.valid
        assert any(e.code == "E106" for e in result.errors)

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
            }
        )

        result = validate_config(cfg, bp)
        assert not result.valid

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
            }
        )

        result = validate_config(cfg, bp)
        assert not result.valid
