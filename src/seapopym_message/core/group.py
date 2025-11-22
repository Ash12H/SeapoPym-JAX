"""Functional Group definition.

This module defines the FunctionalGroup class, which encapsulates the behavior
(units), parameters, and variable mappings for a specific biological entity
(e.g., a species or life stage).
"""

import inspect
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

from seapopym_message.core.unit import Unit

if TYPE_CHECKING:
    from seapopym_message.core.blueprint import Blueprint


@dataclass
class UnitInstance:
    """Wrapper for using a Unit with a specific context (alias + local params).

    Allows reusing the same Unit multiple times within a FunctionalGroup,
    each with a unique alias and optional parameters.

    Attributes:
        unit: The Unit to instantiate.
        alias: Unique name for this instance within the group.
        local_params: Optional parameters specific to this instance.

    Example:
        >>> from seapopym_message.kernels.sensing import extract_layer
        >>> temp_extraction = UnitInstance(extract_layer, alias="extract_temp")
        >>> sal_extraction = UnitInstance(extract_layer, alias="extract_salinity")
    """

    unit: Unit
    alias: str
    local_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionalGroup:
    """A functional group (e.g., species, life stage) in the simulation.

    A FunctionalGroup defines a specific entity that evolves in the simulation.
    It binds generic computational Units to specific state variables and parameters.

    Attributes:
        name: Unique identifier for the group (e.g., "tuna", "zooplankton").
        units: List of Units or UnitInstances that define the group's behavior.
               The order matters: units are executed sequentially (respecting dependencies).
        variable_map: Dictionary mapping variable names to global State names.
                      For UnitInstances, use "alias.var_name" format.
                      Example: {"extract_temp.forcing_nd": "forcing/temperature_3d",
                                "extract_temp.forcing_2d": "tuna/temperature"}
        params: Dictionary of parameters specific to this group.
                Example: {"growth_rate": 0.1, "mortality": 0.05}
    """

    name: str
    units: list[Unit | UnitInstance]
    variable_map: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Ensure name is valid for namespacing
        if "/" in self.name:
            msg = f"Group name '{self.name}' cannot contain '/' character."
            raise ValueError(msg)

    def get_mapped_name(self, internal_name: str, alias: str | None = None) -> str:
        """Resolve the global name for a variable.

        If the variable is in the variable_map, return the mapped name.
        Otherwise, assume it's a group-specific variable and prefix it with the group name.

        Args:
            internal_name: The variable name used in the Unit definition.
            alias: Optional alias prefix (for UnitInstances).

        Returns:
            The global variable name in the State.
        """
        # Try with alias prefix first
        if alias:
            aliased_name = f"{alias}.{internal_name}"
            if aliased_name in self.variable_map:
                return self.variable_map[aliased_name]

        # Try without alias
        if internal_name in self.variable_map:
            return self.variable_map[internal_name]

        # Default behavior: namespace with group name
        return f"{self.name}/{internal_name}"

    def add_to_blueprint(self, blueprint: "Blueprint") -> None:
        """Register all units in this group to the Blueprint.

        This method:
        1. Resolves variable names (namespacing).
        2. Binds group-specific parameters to the units (using partial).
        3. Adds the configured units to the Blueprint.

        Args:
            blueprint: The Blueprint instance to populate.
        """
        for item in self.units:
            if isinstance(item, UnitInstance):
                unit = item.unit
                alias = item.alias
                local_params = item.local_params
            else:
                unit = item
                alias = None
                local_params = {}

            # 1. Resolve Names
            # Create a map for bind()
            var_map = {}
            # We map all internals to their global scoped names
            for name in unit.internal_inputs + unit.internal_outputs + unit.internal_forcings:
                var_map[name] = self.get_mapped_name(name, alias)

            # 2. Bind variables (Creates a new Unit copy)
            new_unit = unit.bind(var_map)

            # 3. Update Name (Unique ID in Graph)
            if alias:
                new_unit.name = f"{self.name}/{alias}"
            else:
                new_unit.name = f"{self.name}/{unit.name}"

            # 4. Parameter Injection
            # Merge group params and local params
            combined_params = {**self.params, **local_params}

            if combined_params:
                # Inspect function to see which params are applicable
                sig = inspect.signature(unit.func)
                relevant_params = {}
                for p_name, p_value in combined_params.items():
                    if p_name in sig.parameters:
                        relevant_params[p_name] = p_value

                if relevant_params:
                    # Bind parameters to the function
                    # This creates a new function where these params are fixed
                    new_func = partial(unit.func, **relevant_params)
                    new_unit.func = new_func

                    # Re-compile if necessary since func changed
                    if new_unit.compiled:
                        import jax

                        new_unit._compiled_func = jax.jit(new_func)
                    else:
                        new_unit._compiled_func = new_func

            # 5. Add to Blueprint
            blueprint.add_unit(new_unit)
