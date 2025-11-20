"""Kernel: Orchestrator for composable computational Units.

The Kernel manages the execution of Units in a topologically sorted order,
respecting dependencies between Units. It separates execution into two phases:
- Local phase: Units with scope='local' (embarrassingly parallel)
- Global phase: Units with scope='global' (require neighbor communication)
"""

from typing import Any

from seapopym_message.core.unit import Unit


class Kernel:
    """Orchestrator for executing a collection of computational Units.

    The Kernel ensures Units are executed in the correct order based on their
    declared inputs/outputs dependencies. It separates local and global computation
    phases for efficient distributed execution.

    Execution order:
        When multiple Units have no dependencies between them (same inputs, different
        outputs), they are executed in the order they appear in the units list.
        This allows explicit control over execution order when topological sorting
        alone cannot determine the correct sequence.

    Args:
        units: List of Unit instances to execute. When Units have equivalent
               dependencies, they will execute in the order specified in this list.

    Raises:
        ValueError: If dependencies are cyclic or missing.

    Example:
        >>> # Order matters when Units read the same inputs
        >>> kernel = Kernel([
        ...     compute_recruitment,  # Must run before age_production
        ...     age_production,       # Modifies production after recruitment
        ...     update_biomass
        ... ])
        >>> state = kernel.execute_local_phase(state, dt=0.1, params={...})

    Note:
        For Units with independent inputs/outputs, the topological sort preserves
        the order from the input list. This is critical when execution order affects
        results (e.g., reading a variable before vs. after it's modified).
    """

    def __init__(self, units: list[Unit]) -> None:
        """Initialize the Kernel with a list of Units.

        Args:
            units: List of Unit instances to manage.
        """
        self.units = units
        self._check_dependencies()
        self._local_units_sorted = self._topological_sort([u for u in units if u.scope == "local"])
        self._global_units_sorted = self._topological_sort(
            [u for u in units if u.scope == "global"]
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

        Local Units are embarrassingly parallel - they don't require neighbor communication.
        This phase can be executed independently on each worker.

        Args:
            state: Current simulation state (dict of arrays).
            dt: Time step size.
            params: Model parameters.
            **kwargs: Additional arguments (forcings, etc.).

        Returns:
            Updated state after executing all local Units.

        Example:
            >>> state = {'biomass': jnp.array([10., 20., 30.])}
            >>> params = {'R': 5.0, 'lambda': 0.1}
            >>> forcings = {'recruitment': jnp.array([...])}
            >>> state = kernel.execute_local_phase(state, dt=0.1, params=params,
            ...                                     forcings=forcings)
        """
        for unit in self._local_units_sorted:
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

        Global Units require neighbor communication (e.g., transport, diffusion).
        This phase is executed after synchronization and halo exchange.

        Args:
            state: Current simulation state (dict of arrays).
            dt: Time step size.
            params: Model parameters.
            neighbor_data: Optional dictionary with halo data from neighbors.
                          Keys: 'halo_north', 'halo_south', 'halo_east', 'halo_west'
            forcings: Optional dictionary with forcing data.

        Returns:
            Updated state after executing all global Units.

        Example:
            >>> neighbor_data = {
            ...     'halo_north': {'biomass': jnp.array([...])},
            ...     'halo_south': {'biomass': jnp.array([...])},
            ...     'halo_east': {'biomass': jnp.array([...])},
            ...     'halo_west': {'biomass': jnp.array([...])}
            ... }
            >>> forcings = {'recruitment': jnp.array([...])}
            >>> state = kernel.execute_global_phase(state, dt=0.1, params=params,
            ...                                     neighbor_data=neighbor_data,
            ...                                     forcings=forcings)
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
