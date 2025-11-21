"""Functional Group definition.

This module defines the FunctionalGroup class, which encapsulates the behavior
(units), parameters, and variable mappings for a specific biological entity
(e.g., a species or life stage).
"""

from dataclasses import dataclass, field
from typing import Any

from seapopym_message.core.unit import Unit


@dataclass
class FunctionalGroup:
    """A functional group (e.g., species, life stage) in the simulation.

    A FunctionalGroup defines a specific entity that evolves in the simulation.
    It binds generic computational Units to specific state variables and parameters.

    Attributes:
        name: Unique identifier for the group (e.g., "tuna", "zooplankton").
        units: List of computational Units that define the group's behavior.
               The order matters: units are executed sequentially (respecting dependencies).
        variable_map: Dictionary mapping internal Unit variable names to global State names.
                      Example: {"biomass": "tuna/biomass", "temperature": "forcing/temp_surface"}
        params: Dictionary of parameters specific to this group.
                Example: {"growth_rate": 0.1, "mortality": 0.05}
    """

    name: str
    units: list[Unit]
    variable_map: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Ensure name is valid for namespacing
        if "/" in self.name:
            msg = f"Group name '{self.name}' cannot contain '/' character."
            raise ValueError(msg)

    def get_mapped_name(self, internal_name: str) -> str:
        """Resolve the global name for a variable.

        If the variable is in the variable_map, return the mapped name.
        Otherwise, assume it's a group-specific variable and prefix it with the group name.

        Args:
            internal_name: The variable name used in the Unit definition.

        Returns:
            The global variable name in the State.
        """
        if internal_name in self.variable_map:
            return self.variable_map[internal_name]

        # Default behavior: namespace with group name
        return f"{self.name}/{internal_name}"
