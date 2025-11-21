"""Unit tests for FunctionalGroup and Unit binding."""

import jax.numpy as jnp

from seapopym_message.core.group import FunctionalGroup
from seapopym_message.core.unit import unit


@unit(
    name="growth",
    inputs=["biomass"],
    outputs=["biomass"],
    scope="local",
    forcings=["temperature"],
)
def compute_growth(biomass, dt, params, forcings):
    """Simple growth function: B = B + B * r * T * dt."""
    r = params["rate"]
    T = forcings["temperature"]
    return biomass + biomass * r * T * dt


def test_unit_binding():
    """Test that binding a Unit updates its variable names correctly."""
    # Define mapping
    variable_map = {
        "biomass": "tuna/biomass",
        "temperature": "env/temp_surface",
    }

    # Bind unit manually
    original_unit = compute_growth
    bound_unit = original_unit.bind(variable_map)

    # Check inputs/outputs/forcings lists (External Names)
    assert bound_unit.inputs == ["tuna/biomass"]
    assert bound_unit.outputs == ["tuna/biomass"]
    assert bound_unit.forcings == ["env/temp_surface"]

    # Check mappings (Internal -> External)
    assert bound_unit.input_mapping["biomass"] == "tuna/biomass"
    assert bound_unit.output_mapping["biomass"] == "tuna/biomass"
    assert bound_unit.forcing_mapping["temperature"] == "env/temp_surface"

    # Check internals remain unchanged
    assert bound_unit.internal_inputs == ["biomass"]


def test_unit_execution_bound():
    """Test execution of a bound unit with namespaced state."""
    variable_map = {
        "biomass": "tuna/biomass",
        "temperature": "env/temp_surface",
    }
    bound_unit = compute_growth.bind(variable_map)

    # Prepare state with namespaced variables
    state = {"tuna/biomass": jnp.array([10.0])}
    forcings = {"env/temp_surface": jnp.array([2.0])}
    params = {"rate": 0.1}
    dt = 1.0

    # Execute
    result = bound_unit.execute(state, dt=dt, params=params, forcings=forcings)

    # Check result key is namespaced
    assert "tuna/biomass" in result

    # Check calculation: 10 + 10 * 0.1 * 2 * 1 = 12
    assert result["tuna/biomass"][0] == 12.0


def test_functional_group_mapping():
    """Test FunctionalGroup helper method."""
    group = FunctionalGroup(name="tuna", units=[], variable_map={"biomass": "tuna_pop/biomass"})

    # Explicit mapping
    assert group.get_mapped_name("biomass") == "tuna_pop/biomass"

    # Default namespacing
    assert group.get_mapped_name("mortality") == "tuna/mortality"
