from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pint
import pytest
import xarray as xr

from seapopym.blueprint import Blueprint
from seapopym.controller.configuration import SimulationConfig
from seapopym.controller.core import SimulationController

# --- Mock Data & Config ---


@dataclass
class MockParams:
    rate: float  # Will be provided as 1/day, expected 1/s
    length: float  # Will be provided as cm, expected m


def mock_compute(state_var, rate):
    return {"output": state_var * rate}


def configure_model(bp: Blueprint):
    bp.register_forcing("temperature", units="degC")

    # Define a group with parameters
    bp.register_group(
        group_prefix="TestGroup",
        units=[
            {
                "func": mock_compute,
                "output_mapping": {"output": "result"},
                "input_mapping": {"state_var": "temperature"},  # rate will be auto-resolved
            }
        ],
        parameters={
            "rate": {"units": "1/s"},  # The model expects seconds
            "length": {"units": "m"},  # The model expects meters
        },
    )


@pytest.fixture
def controller():
    config = SimulationConfig(
        start_date=datetime(2020, 1, 1), end_date=datetime(2020, 1, 2), timestep=timedelta(days=1)
    )
    return SimulationController(config)


def test_parameter_ingestion_and_unit_conversion(controller):
    """Test that parameters are correctly ingested and converted to model units."""

    # 1. Initial State
    initial_state = xr.Dataset(
        {"temperature": (("time", "lat", "lon"), np.ones((1, 10, 10)))},
        coords={"time": [datetime(2020, 1, 1)], "lat": np.arange(10), "lon": np.arange(10)},
        attrs={"units": "degC"},  # Correct unit for temperature
    )
    initial_state["temperature"].attrs["units"] = "degC"

    # 2. Parameters provided in "Human" units
    # rate: 1 per day (should become ~1.157e-5 per second)
    # length: 100 cm (should become 1 meter)
    ureg = pint.get_application_registry()

    params = {"TestGroup/rate": 1.0 * ureg("1/day"), "TestGroup/length": 100.0 * ureg("cm")}

    # 3. Setup
    controller.setup(
        model_configuration_func=configure_model, initial_state=initial_state, parameters=params
    )

    # 4. Verify State
    ds = controller.state

    # Check rate conversion
    rate_var = ds["TestGroup/rate"]
    assert rate_var.attrs["units"] == "1 / second"  # Pint standard format
    np.testing.assert_allclose(rate_var.values, 1.0 / 86400.0)

    # Check length conversion
    length_var = ds["TestGroup/length"]
    assert length_var.attrs["units"] == "meter"
    np.testing.assert_allclose(length_var.values, 1.0)


def test_incompatible_units_raises_error(controller):
    """Test that providing incompatible units raises a ValueError."""

    initial_state = xr.Dataset(
        {"temperature": (("time", "lat", "lon"), np.ones((1, 10, 10)))},
        coords={"time": [datetime(2020, 1, 1)], "lat": np.arange(10), "lon": np.arange(10)},
    )
    initial_state["temperature"].attrs["units"] = "degC"

    ureg = pint.get_application_registry()

    # Error: rate is length (meter) instead of frequency (1/s)
    params = {"TestGroup/rate": 1.0 * ureg("meter"), "TestGroup/length": 1.0 * ureg("meter")}

    with pytest.raises(ValueError, match="Unit conversion failed"):
        controller.setup(
            model_configuration_func=configure_model, initial_state=initial_state, parameters=params
        )


def test_forcing_unit_conversion():
    """Test that forcings with units are properly converted by the Controller."""
    from seapopym.controller.configuration import SimulationConfig
    from seapopym.controller.core import SimulationController

    config = SimulationConfig(
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2020, 1, 2),
        timestep=timedelta(days=1),
    )

    def configure_npp_model(bp: Blueprint):
        # Blueprint expects NPP in g/m²/s (gram/m²/s)
        bp.register_forcing("npp", units="g/m**2/s")

    controller = SimulationController(config)

    # Provide NPP in g/m²/day (human-friendly units)
    initial_state = xr.Dataset()

    forcings = xr.Dataset(
        {
            "npp": (
                ("time", "lat", "lon"),
                np.ones((2, 5, 5)) * 86400.0,  # 86400 g/m²/day = 1 g/m²/s
            )
        },
        coords={
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "lat": np.arange(5),
            "lon": np.arange(5),
        },
    )
    forcings["npp"].attrs["units"] = "g/m**2/day"

    controller.setup(
        model_configuration_func=configure_npp_model,
        initial_state=initial_state,
        forcings=forcings,
    )

    # After setup, forcings should be standardized to g/m²/s
    # The controller stores the standardized forcings
    # We can check via the forcing_manager
    from datetime import datetime as dt

    current_forcings = controller.forcing_manager.get_forcings(dt(2020, 1, 1))

    # The converted value should be ~1.0 g/m²/s (86400 / 86400)
    assert "npp" in current_forcings
    np.testing.assert_allclose(current_forcings["npp"].values, 1.0, rtol=1e-5)


def test_output_units_in_blueprint():
    """Test that output_units can be specified in Blueprint.register_unit()."""
    bp = Blueprint()

    def dummy_func(input_var):
        return {"output": input_var * 2}

    bp.register_forcing("input_var", units="degC")

    bp.register_unit(
        func=dummy_func,
        output_mapping={"output": "output_var"},
        output_units={"output": "degC"},
    )

    # Build the plan
    # Build the plan
    bp.build()

    # Check that the output node has units defined
    output_node = bp._data_nodes["output_var"]
    assert output_node.units == "degC"
