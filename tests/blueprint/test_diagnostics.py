import pytest

from seapopym.blueprint.core import Blueprint
from seapopym.blueprint.exceptions import MissingInputError


def test_register_diagnostic_nominal():
    """Test nominal registration and execution order of a diagnostic function."""
    bp = Blueprint()
    bp.register_forcing("temperature")

    # Store side effects
    logs = []

    def check_temp(temperature):
        logs.append(f"Temperature is {temperature}")

    bp.register_diagnostic(check_temp, input_mapping={"temperature": "temperature"})

    # Check graph structure
    assert len(bp.graph.nodes) == 2  # 1 DataNode (temp) + 1 ComputeNode (diagnostic)

    plan = bp.build()

    # Check that diagnostic is in the plan
    # Flatten groups to find tasks
    tasks = [task for group in plan.task_groups for task in group[1]]
    assert len(tasks) == 1
    assert tasks[0].name == "check_temp"


def test_diagnostic_hard_check():
    """Test that a diagnostic raising an exception stops execution (simulation)."""
    # Note: We can't fully simulate 'execute' here without the Controller,
    # but we verify that the Blueprint accepts the function.
    # The actual execution logic is in Controller, but Blueprint builds the graph.

    bp = Blueprint()
    bp.register_forcing("biomass")

    def check_positive(biomass):
        if biomass < 0:
            raise ValueError("Negative biomass")

    bp.register_diagnostic(check_positive)

    plan = bp.build()
    assert len(plan.task_groups[0][1]) == 1


def test_diagnostic_missing_input():
    """Test error when input is missing for diagnostic."""
    bp = Blueprint()
    # No forcing registered

    def my_diag(x):
        pass

    with pytest.raises(MissingInputError):
        bp.register_diagnostic(my_diag)


def test_diagnostic_with_group():
    """Test diagnostic registration within a group."""
    bp = Blueprint()
    bp.register_forcing("Tuna/biomass")

    def check_tuna(biomass):
        pass

    bp.register_group("Tuna", units=[], state_variables=None)

    # Manually mimic context manager or use register_group's unit list if we supported it there.
    # Since register_diagnostic is a method, we can call it.
    # But register_group helper uses a list of dicts for units, strictly for register_unit.
    # So we have to set context manually or verify if register_diagnostic allows explicit naming.
    # The current implementation of register_group takes 'units' which map to register_unit.
    # It doesn't support diagnostics in that list yet.
    # Let's test calling register_diagnostic inside a manual context block if possible,
    # or just verifying prefix handling if we pass name.

    # Actually Blueprint._group_context is internal.
    # Let's verify we can register a diagnostic that uses a grouped variable.

    bp.register_diagnostic(check_tuna, input_mapping={"biomass": "Tuna/biomass"})
    # Should work

    plan = bp.build()
    tasks = [task for group in plan.task_groups for task in group[1]]
    assert tasks[0].input_mapping["biomass"] == "Tuna/biomass"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
