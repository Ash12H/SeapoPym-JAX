"""Test the new Blueprint and NetworkX architecture."""

from typing import Any

import jax.numpy as jnp
import pytest

from seapopym_message.core.blueprint import Blueprint
from seapopym_message.core.group import FunctionalGroup
from seapopym_message.core.kernel import Kernel
from seapopym_message.core.unit import unit


# --- Define Toy Units ---
@unit(name="photosynthesis", inputs=["sunlight"], outputs=["energy"], scope="local")
def photosynthesis(sunlight: jnp.ndarray, efficiency: float) -> jnp.ndarray:
    """Convert sunlight to energy."""
    return sunlight * efficiency


@unit(name="growth", inputs=["energy", "biomass"], outputs=["biomass_out"], scope="local")
def growth(energy: jnp.ndarray, biomass: jnp.ndarray, rate: float) -> jnp.ndarray:
    """Grow biomass using energy."""
    return biomass + (energy * rate)


def test_blueprint_workflow() -> None:
    """Test the full workflow: Blueprint -> Group -> Kernel -> Execution."""

    # 1. Create Blueprint
    bp = Blueprint()

    # 2. Create a Functional Group
    # We define parameters here. 'efficiency' goes to photosynthesis, 'rate' to growth.
    # Note: 'sunlight' is an external forcing/input. 'biomass' is state.
    plant_params = {"efficiency": 0.5, "rate": 0.1}

    # We define the units for this group
    plant_group = FunctionalGroup(
        name="algae",
        units=[photosynthesis, growth],
        params=plant_params,
        variable_map={
            # Map internal 'sunlight' to global 'forcing/sun'
            "sunlight": "forcing/sun",
            # 'biomass' will be automatically namespaced to 'algae/biomass'
            # 'energy' will be automatically namespaced to 'algae/energy'
        },
    )

    # 3. Add Group to Blueprint
    plant_group.add_to_blueprint(bp)

    # 4. Build Execution Plan
    # This validates the graph and sorts units
    ordered_units = bp.build()

    assert len(ordered_units) == 2
    # Check order: photosynthesis produces energy, which growth needs.
    # So photosynthesis MUST come before growth.
    assert "algae/photosynthesis" in ordered_units[0].name
    assert "algae/growth" in ordered_units[1].name

    # 5. Create Kernel
    kernel = Kernel(ordered_units)

    # 6. Execute Kernel (Simulate one step)
    # Initial State
    state = {
        "forcing/sun": jnp.array([100.0]),
        "algae/biomass": jnp.array([10.0]),
    }

    # Parameters are already bound via partial in the blueprint,
    # so we pass an empty dict or global params if needed.
    # dt is passed but not used by these simple units.

    # Get the first stage (which should be local)
    scope, stage_units = kernel.stages[0]
    assert scope == "local"

    new_state = kernel.execute_stage(stage_units, state, dt=1.0, params={})

    # 7. Validate Results
    # Energy = 100 * 0.5 = 50
    # Growth = 50 * 0.1 = 5
    # New Biomass = 10 + 5 = 15

    assert "algae/energy" in new_state
    assert new_state["algae/energy"] == 50.0
    assert "algae/biomass_out" in new_state
    assert new_state["algae/biomass_out"] == 15.0


def test_cycle_detection() -> None:
    """Test that the Blueprint detects cyclic dependencies."""
    bp = Blueprint()

    # A depends on B
    @unit(name="unit_A", inputs=["var_B"], outputs=["var_A"])
    def unit_a(var_B: Any) -> Any:
        return var_B

    # B depends on A
    @unit(name="unit_B", inputs=["var_A"], outputs=["var_B"])
    def unit_b(var_A: Any) -> Any:
        return var_A

    bp.add_unit(unit_a)
    bp.add_unit(unit_b)

    with pytest.raises(ValueError, match="Dependency cycle detected"):
        bp.build()


def test_visualization_output() -> None:
    """Test that visualization generates a string."""
    bp = Blueprint()
    bp.add_unit(photosynthesis)
    dot_code = bp.visualize()

    assert "digraph Blueprint" in dot_code
    assert "photosynthesis" in dot_code
    assert "sunlight" in dot_code
