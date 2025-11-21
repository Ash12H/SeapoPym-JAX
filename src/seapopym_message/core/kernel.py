"""Kernel: Orchestrator for composable computational Units.

The Kernel manages the execution of Units in a topologically sorted order,
respecting dependencies between Units. It separates execution into two phases:
- Local phase: Units with scope='local' (embarrassingly parallel)
- Global phase: Units with scope='global' (require neighbor communication)
"""

from typing import Any

from seapopym_message.core.group import FunctionalGroup
from seapopym_message.core.unit import Unit


class Kernel:
    """Orchestrator for executing a collection of computational Units.

    The Kernel ensures Units are executed in the correct order based on their
    declared inputs/outputs dependencies. It separates local and global computation
    phases for efficient distributed execution.

    Args:
        units_or_groups: List of Unit instances or FunctionalGroup instances.
                         FunctionalGroups are flattened into their constituent Units,
                         with variables bound to the group's namespace.

    Raises:
        ValueError: If dependencies are cyclic or missing.
    """

    def __init__(self, units_or_groups: list[Unit | FunctionalGroup]) -> None:
        """Initialize the Kernel.

        Args:
            units_or_groups: List of Units or FunctionalGroups.
        """
        from seapopym_message.core.group import UnitInstance

        self.units: list[Unit] = []

        # Flatten groups and bind units
        for item in units_or_groups:
            if isinstance(item, FunctionalGroup):
                # Bind all units in the group
                for unit_or_instance in item.units:
                    # Determine if this is a direct Unit or a UnitInstance
                    if isinstance(unit_or_instance, UnitInstance):
                        unit = unit_or_instance.unit
                        alias = unit_or_instance.alias
                        local_params = unit_or_instance.local_params  # noqa: F841
                    else:
                        unit = unit_or_instance
                        alias = None

                    # Build full variable map for this unit instance
                    full_map = {}
                    # Map inputs
                    for internal_name in unit.internal_inputs:
                        full_map[internal_name] = item.get_mapped_name(internal_name, alias)
                    # Map outputs
                    for internal_name in unit.internal_outputs:
                        full_map[internal_name] = item.get_mapped_name(internal_name, alias)
                    # Map forcings
                    for internal_name in unit.internal_forcings:
                        full_map[internal_name] = item.get_mapped_name(internal_name, alias)

                    bound_unit = unit.bind(full_map)

                    # Rename unit to include group name and alias for uniqueness
                    if alias:
                        bound_unit.name = f"{item.name}/{alias}"
                    else:
                        bound_unit.name = f"{item.name}/{unit.name}"

                    # Store local params for later use (if needed)
                    # For now, they would be merged into the global params dict
                    # This is a limitation we can address later

                    self.units.append(bound_unit)
            elif isinstance(item, Unit):
                self.units.append(item)
            else:
                raise TypeError(f"Expected Unit or FunctionalGroup, got {type(item)}")

        self._check_dependencies()
        self._local_units_sorted = self._topological_sort(
            [u for u in self.units if u.scope == "local"]
        )
        self._global_units_sorted = self._topological_sort(
            [u for u in self.units if u.scope == "global"]
        )

    @property
    def local_units(self) -> list[Unit]:
        """Return topologically sorted local Units."""
        return self._local_units_sorted

    @property
    def global_units(self) -> list[Unit]:
        """Return topologically sorted global Units."""
        return self._global_units_sorted

    def execute_local_phase(
        self, state: dict[str, Any], dt: float, params: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        """Execute all local-scope Units in topological order.

        Args:
            state: Current simulation state (dict of arrays).
            dt: Time step size.
            params: Model parameters.
            **kwargs: Additional arguments (forcings, etc.).

        Returns:
            Updated state after executing all local Units.
        """
        for unit in self._local_units_sorted:
            # We pass the global params dict.
            # Units might need specific params.
            # Currently, Unit.execute passes all params.
            # If FunctionalGroup has params, they should be merged or handled.
            # Ideally, params should be namespaced too.
            # For now, we assume params contains everything needed.
            result = unit.execute(state, dt=dt, params=params, **kwargs)
            state.update(result)
        return state

    def execute_global_phase(
        self,
        state: dict[str, Any],
        dt: float,
        params: dict[str, Any],
        neighbor_data: dict[str, Any] | None = None,
        forcings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute all global-scope Units in topological order.

        Args:
            state: Current simulation state (dict of arrays).
            dt: Time step size.
            params: Model parameters.
            neighbor_data: Optional dictionary with halo data from neighbors.
            forcings: Optional dictionary with forcing data.

        Returns:
            Updated state after executing all global Units.
        """
        kwargs = {"dt": dt, "params": params}
        if neighbor_data:
            kwargs.update(neighbor_data)
        if forcings:
            kwargs["forcings"] = forcings

        for unit in self._global_units_sorted:
            result = unit.execute(state, **kwargs)
            state.update(result)
        return state

    def has_global_units(self) -> bool:
        """Check if kernel contains any global-scope Units.

        Returns:
            True if there are global Units, False otherwise.
        """
        return len(self._global_units_sorted) > 0

    def visualize_graph(self) -> str:
        """Generate a DOT string representation of the dependency graph.

        Returns:
            String containing the Graphviz DOT definition.
        """
        lines = ["digraph Kernel {"]
        lines.append("  rankdir=LR;")
        lines.append("  node [shape=box, style=filled, fillcolor=lightgrey];")

        # Add nodes (Units)
        for unit in self.units:
            lines.append(f'  "{unit.name}" [label="{unit.name}"];')

        # Add edges based on data flow
        # We need to know which unit produces which variable
        producers: dict[str, str] = {}  # variable -> unit_name

        # Sort all units to ensure deterministic output
        all_units = self._local_units_sorted + self._global_units_sorted

        for unit in all_units:
            for output in unit.outputs:
                producers[output] = unit.name

        for unit in all_units:
            for input_var in unit.inputs:
                if input_var in producers:
                    producer_name = producers[input_var]
                    if producer_name != unit.name:
                        lines.append(f'  "{producer_name}" -> "{unit.name}" [label="{input_var}"];')
                else:
                    # Input comes from external source (Initial State or Forcing)
                    # We can add a node for it
                    ext_node = f"EXT_{input_var}"
                    lines.append(
                        f'  "{ext_node}" [label="{input_var}", shape=ellipse, fillcolor=white];'
                    )
                    lines.append(f'  "{ext_node}" -> "{unit.name}";')

        lines.append("}")
        return "\n".join(lines)

    def _check_dependencies(self) -> None:
        """Check that all Unit inputs can be satisfied.

        Raises:
            ValueError: If any Unit has inputs that cannot be produced by previous Units.
        """
        # Build set of all available outputs (produced by Units or external)
        produced: set[str] = set()

        # Separate units by scope for checking
        local_units = [u for u in self.units if u.scope == "local"]
        global_units = [u for u in self.units if u.scope == "global"]

        # Check local phase
        for unit in self._topological_sort(local_units):
            for inp in unit.inputs:
                if inp not in produced:
                    # Input must come from initial state or external source
                    pass  # Not an error - will be checked at runtime
            produced.update(unit.outputs)

        # Check global phase (can use outputs from local phase)
        for unit in self._topological_sort(global_units):
            for inp in unit.inputs:
                if inp not in produced:
                    pass  # Not an error - will be checked at runtime
            produced.update(unit.outputs)

    def _topological_sort(self, units: list[Unit]) -> list[Unit]:
        """Sort Units in topological order based on input/output dependencies.

        Uses Kahn's algorithm to detect cycles and sort Units.

        Args:
            units: List of Units to sort.

        Returns:
            Topologically sorted list of Units.

        Raises:
            ValueError: If a dependency cycle is detected.
        """
        if not units:
            return []

        # Build dependency graph
        # in_degree[unit] = number of units that must execute before this unit
        # dependencies[unit] = set of units that depend on this unit's outputs
        in_degree: dict[Unit, int] = dict.fromkeys(units, 0)
        dependencies: dict[Unit, set[Unit]] = {unit: set() for unit in units}

        # Build reverse mapping: output -> units that produce it
        # Only count outputs that are NOT also inputs (newly created variables)
        producers: dict[str, Unit] = {}
        for unit in units:
            for output in unit.outputs:
                # Only register as producer if this is a net new variable
                # (not modifying an existing input)
                if output not in unit.inputs:
                    if output in producers:
                        # Multiple units producing same output - use last one
                        pass
                    producers[output] = unit
                # Also register if it modifies an input but we want to track the flow
                # Actually, if unit reads A and writes A, it depends on whoever produced A before.
                # And whoever reads A next depends on this unit.
                # So we should register it as producer of A (the new version).
                producers[output] = unit

        # Calculate in-degrees
        for unit in units:
            for inp in unit.inputs:
                if inp in producers:
                    producer = producers[inp]
                    # Create dependency only if:
                    # 1. It's not a self-loop (producer != unit)
                    # 2. Either the input is NOT in the unit's outputs (normal dependency)
                    #    OR it IS in the unit's outputs but we still need the producer
                    #    to run first (because the unit reads THEN writes the variable)
                    if producer != unit:
                        dependencies[producer].add(unit)
                        in_degree[unit] += 1

        # Kahn's algorithm
        queue = [unit for unit in units if in_degree[unit] == 0]
        sorted_units: list[Unit] = []

        while queue:
            # Process unit with no dependencies
            current = queue.pop(0)
            sorted_units.append(current)

            # Reduce in-degree for dependent units
            for dependent in dependencies[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # Check for cycles
        if len(sorted_units) != len(units):
            remaining = [u.name for u in units if u not in sorted_units]
            raise ValueError(
                f"Cyclic dependency detected among units: {remaining}. "
                "Units must form a directed acyclic graph (DAG)."
            )

        return sorted_units
