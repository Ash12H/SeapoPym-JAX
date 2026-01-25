from datetime import timedelta

import pytest
import xarray as xr

from seapopym.blueprint import Blueprint
from seapopym.blueprint.exceptions import ConfigurationError
from seapopym.controller import SimulationConfig, SimulationController
from seapopym.gsm.exceptions import StateValidationError

# --- Fixtures ---


def dummy_func(state_var):
    """A dummy function that computes a tendency for a state variable."""
    return {"tendency": state_var * 0.1}


def dummy_func_no_tendency(state_var):
    """A dummy function that just computes something."""
    return {"output": state_var * 2}


@pytest.fixture
def simple_config():
    return SimulationConfig(
        start_date="2020-01-01",
        end_date="2020-01-02",
        timestep=timedelta(days=1),
    )


# --- Tests ---


def test_register_state_variables():
    """Test that state variables are correctly registered with prefixes."""
    bp = Blueprint()
    bp.register_group(
        "GroupA",
        units=[],
        state_variables={"biomass": {"dims": ("time", "lat", "lon"), "units": "g/m2"}},
    )

    assert "GroupA/biomass" in bp.registered_variables
    assert "GroupA/biomass" in bp.get_state_variables()
    node = bp._data_nodes["GroupA/biomass"]
    assert node.is_state
    assert node.units == "g/m2"


def test_tendency_validation_success():
    """Test that a tendency targeting a declared state variable is valid."""
    bp = Blueprint()
    bp.register_group(
        "GroupA",
        units=[
            {
                "func": dummy_func,
                "input_mapping": {"state_var": "biomass"},
                "output_mapping": {"tendency": "biomass_tendency"},
                "output_tendencies": {"tendency": "biomass"},
            }
        ],
        state_variables={"biomass": {"dims": ("time",), "units": "g"}},
    )
    # Should build without error
    plan = bp.build()
    # State variable should be in initial_variables (as root)
    assert "GroupA/biomass" in plan.initial_variables


def test_tendency_validation_error():
    """Test that a tendency targeting an UNDECLARED variable raises an error."""
    bp = Blueprint()
    # On enregistre un forcing pour que l'input soit résolu
    # Mais ce n'est PAS une variable d'état, donc la validation de tendance doit échouer
    bp.register_forcing("GroupA/biomass")

    with pytest.raises(ConfigurationError, match="tendency targets but not declared"):
        bp.register_group(
            "GroupA",
            units=[
                {
                    "func": dummy_func,
                    "input_mapping": {"state_var": "biomass"},
                    "output_mapping": {"tendency": "biomass_tendency"},
                    "output_tendencies": {"tendency": "biomass"},
                }
            ],
            # No state_variables declared!
        )
        bp.build()


def test_controller_validation_success(simple_config):
    """Test that Controller accepts a valid initial state."""
    bp = Blueprint()
    bp.register_group(
        "GroupA", units=[], state_variables={"biomass": {"dims": ("time",), "units": "g"}}
    )

    initial_state = xr.Dataset(
        {"GroupA/biomass": xr.DataArray([1.0], coords={"time": [0]}, dims="time")}
    )

    controller = SimulationController(simple_config)
    controller.setup(
        lambda b: b.register_group(
            "GroupA", [], state_variables={"biomass": {"dims": ("time",), "units": "g"}}
        ),
        initial_state,
    )
    assert "GroupA/biomass" in controller.state


def test_controller_validation_error(simple_config):
    """Test that Controller rejects an incomplete initial state."""

    def configure(bp):
        bp.register_group(
            "GroupA", units=[], state_variables={"biomass": {"dims": ("time",), "units": "g"}}
        )

    # Missing GroupA/biomass
    initial_state = xr.Dataset({"other": 1})

    controller = SimulationController(simple_config)
    with pytest.raises(StateValidationError, match="Missing required"):
        controller.setup(configure, initial_state)
