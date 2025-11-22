"""Kernel: Orchestrator for executing computational Units.

The Kernel is responsible for executing the Units in the order determined by the
Blueprint. It groups Units into "stages" (Local vs Global) to optimize distributed
execution.
"""

from typing import Any

from seapopym_message.core.unit import Unit


class Kernel:
    """Executor for a sequence of computational Units.

    The Kernel takes a topologically sorted list of Units (from Blueprint) and
    executes them. It groups consecutive Units of the same scope into "stages"
    to minimize synchronization overhead in a distributed setting.

    Args:
        ordered_units: List of Units in execution order.
    """

    def __init__(self, ordered_units: list[Unit]) -> None:
        """Initialize the Kernel with a fixed execution plan."""
        self.units = ordered_units

    @property
    def stages(self) -> list[tuple[str, list[Unit]]]:
        """Group units into execution stages based on scope.

        Returns:
            List of (scope, units) tuples.
            Example: [('local', [u1, u2]), ('global', [u3]), ('local', [u4])]
        """
        stages: list[tuple[str, list[Unit]]] = []
        if not self.units:
            return stages

        current_scope = self.units[0].scope
        current_batch = []

        for unit in self.units:
            if unit.scope == current_scope:
                current_batch.append(unit)
            else:
                stages.append((current_scope, current_batch))
                current_scope = unit.scope
                current_batch = [unit]

        if current_batch:
            stages.append((current_scope, current_batch))

        return stages

    def execute_stage(
        self,
        stage_units: list[Unit],
        state: dict[str, Any],
        dt: float,
        params: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a list of units (a stage) sequentially.

        Args:
            stage_units: List of units to execute.
            state: Current simulation state.
            dt: Time step size.
            params: Global parameters (fallback if not bound).
            **kwargs: Additional arguments (forcings, neighbor_data).

        Returns:
            Updated state dictionary.
        """
        # Prepare common kwargs
        exec_kwargs = {"dt": dt, "params": params, **kwargs}

        for unit in stage_units:
            # Execute unit
            # Note: unit.func might already have params bound via partial
            result = unit.execute(state, **exec_kwargs)
            state.update(result)

        return state
