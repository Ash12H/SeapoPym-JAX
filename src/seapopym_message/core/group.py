"""Functional Group definition.

This module defines the FunctionalGroup class, which encapsulates the behavior
(units), parameters, and variable mappings for a specific biological entity
(e.g., a species or life stage).
"""

from dataclasses import dataclass, field
from typing import Any

from seapopym_message.core.unit import Unit


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
