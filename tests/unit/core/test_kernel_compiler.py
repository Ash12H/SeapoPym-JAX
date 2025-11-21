"""Tests for Kernel Compiler features (FunctionalGroup support)."""

import jax.numpy as jnp

from seapopym_message.core.group import FunctionalGroup
from seapopym_message.core.kernel import Kernel
from seapopym_message.core.unit import unit


@unit(name="growth", inputs=["biomass"], outputs=["biomass"], scope="local")
def compute_growth(biomass, dt, params):
    return biomass * (1.0 + params["rate"] * dt)


@unit(name="mortality", inputs=["biomass"], outputs=["biomass"], scope="local")
def compute_mortality(biomass, dt, params):
    return biomass * (1.0 - params["m"] * dt)


def test_kernel_with_functional_group():
    """Test initializing Kernel with a FunctionalGroup."""

    # Define a group with specific mapping
    tuna_group = FunctionalGroup(
        name="tuna", units=[compute_growth], variable_map={"biomass": "tuna/biomass"}
    )

    kernel = Kernel([tuna_group])

    # Check that unit was flattened and renamed
    assert len(kernel.local_units) == 1
    unit = kernel.local_units[0]
    assert unit.name == "tuna/growth"
    assert unit.inputs == ["tuna/biomass"]
    assert unit.outputs == ["tuna/biomass"]


def test_kernel_with_multiple_groups():
    """Test Kernel with multiple groups sharing the same unit definition."""

    tuna_group = FunctionalGroup(
        name="tuna", units=[compute_growth], variable_map={"biomass": "tuna/biomass"}
    )

    shark_group = FunctionalGroup(
        name="shark", units=[compute_growth], variable_map={"biomass": "shark/biomass"}
    )

    kernel = Kernel([tuna_group, shark_group])

    assert len(kernel.local_units) == 2

    # Check names
    names = sorted([u.name for u in kernel.local_units])
    assert names == ["shark/growth", "tuna/growth"]

    # Check execution
    state = {"tuna/biomass": jnp.array([10.0]), "shark/biomass": jnp.array([5.0])}
    params = {"rate": 0.1}
    dt = 1.0

    result = kernel.execute_local_phase(state, dt=dt, params=params)

    # Both should grow by 10%
    assert result["tuna/biomass"][0] == 11.0
    assert result["shark/biomass"][0] == 5.5


def test_kernel_mixed_units_and_groups():
    """Test Kernel with both standalone Units and FunctionalGroups."""

    @unit(name="global_env", inputs=[], outputs=["env"], scope="local")
    def env_unit(dt, params):
        return jnp.array([1.0])

    tuna_group = FunctionalGroup(
        name="tuna", units=[compute_growth], variable_map={"biomass": "tuna/biomass"}
    )

    kernel = Kernel([env_unit, tuna_group])

    assert len(kernel.local_units) == 2
    names = {u.name for u in kernel.local_units}
    assert "global_env" in names
    assert "tuna/growth" in names


def test_visualize_graph():
    """Test DOT graph generation."""
    tuna_group = FunctionalGroup(
        name="tuna",
        units=[compute_growth, compute_mortality],
        variable_map={"biomass": "tuna/biomass"},
    )

    kernel = Kernel([tuna_group])
    dot_output = kernel.visualize_graph()

    assert "digraph Kernel" in dot_output
    assert '"tuna/growth"' in dot_output
    assert '"tuna/mortality"' in dot_output
    # Check edge: growth reads biomass, mortality reads biomass
    # Since both read/write same var, dependency depends on order in list
    # In FunctionalGroup, order is preserved.
    # growth -> mortality (if growth is first)
    # Wait, topological sort might reorder if they are independent?
    # No, if they modify the same variable, there is a dependency.
    # growth: reads biomass, writes biomass
    # mortality: reads biomass, writes biomass
    # mortality depends on growth's output.

    assert '"tuna/growth" -> "tuna/mortality"' in dot_output
