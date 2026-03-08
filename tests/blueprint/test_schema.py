"""Tests for Blueprint and Config schema classes."""


import pytest
import xarray as xr

from seapopym.blueprint import (
    Blueprint,
    Config,
    Declarations,
    ExecutionParams,
    ProcessStep,
    TendencySource,
    VariableDeclaration,
)




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


class TestProcessStep:
    """Tests for ProcessStep."""

    def test_valid_process_step(self):
        """Test creating a valid process step."""
        step = ProcessStep(
            func="biol:growth",
            inputs={"biomass": "state.biomass", "rate": "parameters.growth_rate"},
            outputs={"tendency": "derived.growth_flux"},
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

    def test_output_must_be_derived(self):
        """Test that output targets must start with 'derived.'."""
        with pytest.raises(ValueError, match="must start with 'derived.'"):
            ProcessStep(
                func="biol:growth",
                inputs={"x": "state.x"},
                outputs={"out": "tendencies.growth"},
            )


class TestTendencySource:
    """Tests for TendencySource."""

    def test_default_sign(self):
        """Test default sign is +1.0."""
        src = TendencySource(source="derived.growth_flux")
        assert src.source == "derived.growth_flux"
        assert src.sign == 1.0

    def test_negative_sign(self):
        """Test negative sign."""
        src = TendencySource(source="derived.predation", sign=-1.0)
        assert src.sign == -1.0


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

    def test_raw_dicts_coerced_to_variable_declaration(self):
        """Test that raw dicts in YAML are correctly coerced to VariableDeclaration."""
        decl = Declarations(
            state={"biomass": {"units": "g", "dims": ["Y", "X"]}},
            forcings={"temperature": {"units": "degC", "dims": ["T", "Y", "X"]}},
        )
        assert isinstance(decl.state["biomass"], VariableDeclaration)
        assert decl.state["biomass"].units == "g"
        assert isinstance(decl.forcings["temperature"], VariableDeclaration)

    def test_invalid_declaration_value_rejected(self):
        """Test that non-dict values are rejected in declarations."""
        with pytest.raises(ValueError, match="Invalid declaration"):
            Declarations(state={"biomass": "not a dict"})


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
                    "outputs": {"out": "derived.growth_flux"},
                }
            ],
            "tendencies": {
                "biomass": [{"source": "derived.growth_flux"}],
            },
        }

        bp = Blueprint.from_dict(data)
        assert bp.id == "test-model"
        assert bp.version == "0.1.0"
        assert len(bp.process) == 1
        assert "biomass" in bp.tendencies


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

    def test_tendency_validates_state_keys(self):
        """Test that tendencies must reference valid state variables."""
        with pytest.raises(ValueError, match="not a declared state variable"):
            Blueprint.from_dict(
                {
                    "id": "test",
                    "version": "0.1.0",
                    "declarations": {
                        "state": {"biomass": {"units": "g"}},
                    },
                    "process": [],
                    "tendencies": {
                        "nonexistent": [{"source": "derived.flux"}],
                    },
                }
            )

    def test_tendency_source_must_be_derived(self):
        """Test that tendency sources must start with 'derived.'."""
        with pytest.raises(ValueError, match="must start with 'derived.'"):
            Blueprint.from_dict(
                {
                    "id": "test",
                    "version": "0.1.0",
                    "declarations": {
                        "state": {"biomass": {"units": "g"}},
                    },
                    "process": [],
                    "tendencies": {
                        "biomass": [{"source": "state.biomass"}],
                    },
                }
            )


class TestConfig:
    """Tests for Config class."""

    def test_creation_with_xr_data(self):
        """Test creating Config with xr.DataArray data."""
        cfg = Config(
            parameters={"growth_rate": xr.DataArray(0.1)},
            forcings={"temperature": xr.DataArray([20.0, 21.0], dims=["T"])},
            initial_state={"biomass": xr.DataArray([[1.0]], dims=["Y", "X"])},
            execution=ExecutionParams(
                time_start="2000-01-01",
                time_end="2001-01-01",
                dt="1d",
            ),
        )
        assert float(cfg.parameters["growth_rate"]) == 0.1
        assert cfg.execution.time_start == "2000-01-01"
        assert cfg.execution.time_end == "2001-01-01"
        assert cfg.execution.dt == "1d"

    def test_parameter_access(self):
        """Test direct parameter access from dict."""
        cfg = Config(
            parameters={
                "simple": xr.DataArray(0.1),
                "vector": xr.DataArray([1.0, 2.0, 3.0], dims=["C"]),
            },
            forcings={},
            initial_state={},
            execution=ExecutionParams(
                time_start="2000-01-01",
                time_end="2001-01-01",
            ),
        )

        assert float(cfg.parameters["simple"]) == 0.1
        assert list(cfg.parameters["vector"].values) == [1.0, 2.0, 3.0]
        assert cfg.parameters.get("nonexistent") is None


class TestExecutionParams:
    """Tests for ExecutionParams."""

    def test_defaults(self):
        """Test default values for optional fields."""
        params = ExecutionParams(time_start="2000-01-01", time_end="2001-01-01")
        assert params.dt == "1d"
        assert params.forcing_interpolation == "constant"

    def test_time_fields(self):
        """Test time_start and time_end fields."""
        params = ExecutionParams(time_start="2020-01-01", time_end="2020-12-31")
        assert params.time_start == "2020-01-01"
        assert params.time_end == "2020-12-31"

    @pytest.mark.parametrize("dt", ["1d", "0.05d", "6h", "30min", "3600s", "1s"])
    def test_valid_dt_accepted(self, dt):
        """Test that valid dt formats are accepted."""
        params = ExecutionParams(time_start="2000-01-01", time_end="2001-01-01", dt=dt)
        assert params.dt == dt

    @pytest.mark.parametrize("dt", ["foo", "2x", "d1", "abc123"])
    def test_invalid_dt_rejected(self, dt):
        """Test that invalid dt formats are rejected."""
        with pytest.raises(ValueError, match="Invalid dt format"):
            ExecutionParams(time_start="2000-01-01", time_end="2001-01-01", dt=dt)

    def test_invalid_datetime_rejected(self):
        """Test that unparseable datetime raises ValueError."""
        with pytest.raises(ValueError, match="Invalid datetime format"):
            ExecutionParams(time_start="not-a-date", time_end="2001-01-01")

    def test_inverted_time_range_rejected(self):
        """Test that time_end < time_start raises ValueError."""
        with pytest.raises(ValueError, match="must be after"):
            ExecutionParams(time_start="2020-01-01", time_end="2019-01-01")
