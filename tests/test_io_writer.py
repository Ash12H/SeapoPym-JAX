from datetime import timedelta

import numpy as np
import pytest
import xarray as xr

from seapopym.blueprint import Blueprint
from seapopym.controller import SimulationConfig, SimulationController
from seapopym.standard.coordinates import Coordinates


# --- Fixtures ---
@pytest.fixture
def simple_blueprint():
    """A minimal blueprint for testing."""
    bp = Blueprint()
    bp.register_forcing(
        "temperature", dims=(Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value)
    )
    bp.register_forcing(
        "biomass", dims=(Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value)
    )
    return bp


@pytest.fixture
def simple_config():
    return SimulationConfig(
        start_date="2020-01-01",
        end_date="2020-01-04",  # 3 days
        timestep=timedelta(days=1),
    )


@pytest.fixture
def initial_state(simple_config):
    # Create a simple state
    times = [simple_config.start_date]
    lats = [0.0]
    lons = [0.0]

    biomass = xr.DataArray(
        np.array([[[10.0]]]),
        coords={Coordinates.T.value: times, Coordinates.Y.value: lats, Coordinates.X.value: lons},
        dims=(Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value),
        name="biomass",
    )
    temperature = xr.DataArray(
        np.array([[[20.0]]]),
        coords={Coordinates.T.value: times, Coordinates.Y.value: lats, Coordinates.X.value: lons},
        dims=(Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value),
        name="temperature",
    )
    return xr.Dataset({"biomass": biomass, "temperature": temperature})


def configure_model(bp):
    from seapopym.standard.coordinates import Coordinates

    bp.register_group(
        group_prefix="Global",
        units=[],
        state_variables={
            "biomass": {
                "dims": (Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value),
                "units": "dimensionless",
            },
            "temperature": {
                "dims": (Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value),
                "units": "degC",
            },
        },
    )


# --- Tests ---


def test_memory_writer_integration(simple_config, simple_blueprint, initial_state):
    """Test that results are correctly stored in memory."""
    controller = SimulationController(simple_config)
    controller.setup(
        configure_model,
        {"Global": initial_state},
        output_path=None,  # Memory mode
    )
    controller.run()

    results = controller.results
    assert isinstance(results, xr.Dataset)
    assert "Global/biomass" in results
    assert "Global/temperature" in results

    # Check time dimension
    # 4 timesteps saved: initial state (Jan 1) + 3 steps (Jan 2, Jan 3, Jan 4)
    # Total: 4 timesteps.
    # 3 timesteps saved: no initial state + 3 steps (Jan 2, Jan 3, Jan 4)
    assert len(results[Coordinates.T.value]) == 3


def test_zarr_writer_integration(simple_config, simple_blueprint, initial_state, tmp_path):
    """Test that results are correctly stored to Zarr."""
    output_path = tmp_path / "test_output.zarr"

    controller = SimulationController(simple_config)
    controller.setup(configure_model, {"Global": initial_state}, output_path=output_path)
    controller.run()

    # Check file exists
    assert output_path.exists()

    # Check results from controller (should have original names with slashes)
    results = controller.results
    assert isinstance(results, xr.Dataset)
    assert len(results[Coordinates.T.value]) == 3
    assert "Global/biomass" in results
    assert "Global/temperature" in results

    # Verify data persistence: Zarr store has sanitized names (underscores replace slashes)
    ds_loaded = xr.open_zarr(output_path)
    assert "Global__biomass" in ds_loaded  # Note: double underscore instead of slash
    assert "Global__temperature" in ds_loaded
    assert len(ds_loaded[Coordinates.T.value]) == 3


def test_variable_selection(simple_config, simple_blueprint, initial_state):
    """Test that only selected variables are saved."""
    controller = SimulationController(simple_config)
    controller.setup(
        configure_model,
        {"Global": initial_state},
        output_path=None,
        output_variables=["Global/biomass"],  # Only biomass
    )
    controller.run()

    results = controller.results
    assert "Global/biomass" in results
    assert "Global/temperature" not in results


def test_metadata_storage(simple_config, simple_blueprint, initial_state):
    """Test that metadata is saved."""
    meta = {"author": "Test", "version": 1.0}
    controller = SimulationController(simple_config)
    controller.setup(
        configure_model, {"Global": initial_state}, output_path=None, output_metadata=meta
    )
    controller.run()

    results = controller.results
    assert results.attrs["author"] == "Test"
    assert results.attrs["version"] == 1.0


def test_zarr_overwrite_error(simple_config, initial_state, tmp_path):
    """Test that ZarrWriter raises an error if the path already exists."""
    output_path = tmp_path / "existing.zarr"
    output_path.mkdir()

    controller = SimulationController(simple_config)
    controller.setup(configure_model, {"Global": initial_state}, output_path=output_path)

    # Should raise FileExistsError when trying to initialize the writer
    with pytest.raises(FileExistsError, match="already exists"):
        controller.run()
