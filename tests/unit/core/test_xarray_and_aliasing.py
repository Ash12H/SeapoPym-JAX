"""Unit tests for xarray support in sensing units and UnitInstance aliasing."""

import numpy as np
import xarray as xr

from seapopym_message.core.group import FunctionalGroup, UnitInstance
from seapopym_message.core.kernel import Kernel
from seapopym_message.kernels.sensing import diel_migration, extract_layer


def test_extract_layer_with_xarray():
    """Test extract_layer with xarray DataArray (robust dimension selection)."""
    # Create 3D data with named dimensions
    temp_3d = xr.DataArray(
        np.array([[[25.0]], [[15.0]], [[5.0]]]),  # (3, 1, 1)
        dims=("depth", "lat", "lon"),
        coords={"depth": [0, 50, 100], "lat": [0], "lon": [0]},
    )

    forcings = {"forcing_nd": temp_3d}
    params = {"layer_index": 0}

    # Extract surface layer using Unit.execute
    state = {}  # No state inputs for this unit
    result_dict = extract_layer.execute(state, dt=1.0, params=params, forcings=forcings)
    result = result_dict["forcing_2d"]

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("lat", "lon")
    assert result.values[0, 0] == 25.0


def test_extract_layer_auto_detect_dimension():
    """Test that extract_layer auto-detects depth dimension name."""
    # Use 'Z' instead of 'depth'
    temp_3d = xr.DataArray(
        np.array([[[25.0]], [[15.0]], [[5.0]]]),
        dims=("Z", "lat", "lon"),
        coords={"Z": [0, 50, 100], "lat": [0], "lon": [0]},
    )

    forcings = {"forcing_nd": temp_3d}
    params = {"layer_index": 1}  # Middle layer

    state = {}
    result_dict = extract_layer.execute(state, dt=1.0, params=params, forcings=forcings)
    result = result_dict["forcing_2d"]

    assert result.values[0, 0] == 15.0


def test_diel_migration_with_xarray():
    """Test diel_migration with xarray."""
    temp_3d = xr.DataArray(
        np.array([[[25.0]], [[15.0]], [[5.0]]]),  # (depth=3, lat=1, lon=1)
        dims=("depth", "lat", "lon"),
    )

    day_length = xr.DataArray(np.array([[0.5]]), dims=("lat", "lon"))

    forcings = {"forcing_nd": temp_3d, "day_length": day_length}
    params = {
        "day_layer_index": 2,  # Deep (5°C)
        "night_layer_index": 0,  # Surface (25°C)
    }

    state = {}
    result_dict = diel_migration.execute(state, dt=1.0, params=params, forcings=forcings)
    result = result_dict["forcing_effective"]

    # Expected: 5 * 0.5 + 25 * 0.5 = 15
    assert np.isclose(result.values[0, 0], 15.0)


def test_unit_instance_aliasing():
    """Test UnitInstance for reusing same unit with different names."""
    from seapopym_message.core.unit import unit

    # Simple test unit
    @unit(name="multiply", inputs=["x"], outputs=["y"], scope="local")
    def multiply_unit(x, dt, params):
        return x * params["factor"]

    # Create group with two instances of the same unit
    group = FunctionalGroup(
        name="test",
        units=[
            UnitInstance(multiply_unit, alias="double"),
            UnitInstance(multiply_unit, alias="triple"),
        ],
        variable_map={
            "double.x": "test/input",
            "double.y": "test/doubled",
            "triple.x": "test/input",
            "triple.y": "test/tripled",
        },
    )

    kernel = Kernel([group])

    # Check that units were aliased correctly
    unit_names = [u.name for u in kernel.local_units]
    assert "test/double" in unit_names
    assert "test/triple" in unit_names

    # Check inputs/outputs
    double_unit = [u for u in kernel.local_units if u.name == "test/double"][0]
    triple_unit = [u for u in kernel.local_units if u.name == "test/triple"][0]

    assert double_unit.inputs == ["test/input"]
    assert double_unit.outputs == ["test/doubled"]
    assert triple_unit.inputs == ["test/input"]
    assert triple_unit.outputs == ["test/tripled"]
