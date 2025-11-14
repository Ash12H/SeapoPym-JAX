"""Unit: Composable computational unit for the Kernel.

A Unit represents a single computational step in the simulation.
Each Unit has:
- inputs: variables it needs from the state
- outputs: variables it produces
- scope: 'local' (embarrassingly parallel) or 'global' (requires neighbor communication)
- func: the actual computation function
- compiled: whether to JIT compile with JAX
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import jax


@dataclass
class Unit:
    """Composable computational unit.

    Args:
        name: Unique identifier for this unit.
        func: Function to execute. Should accept state variables as kwargs.
        inputs: List of state variable names required by this unit.
        outputs: List of state variable names produced by this unit.
        scope: Execution scope - 'local' for independent computation,
               'global' for operations requiring neighbor communication.
        compiled: If True, wraps func with jax.jit for compilation.
        forcings: List of forcing variable names required by this unit.

    Example:
        >>> @unit(name='growth', inputs=['biomass'], outputs=['biomass'],
        ...       scope='local', compiled=True, forcings=['recruitment'])
        ... def compute_growth(biomass: jnp.ndarray, dt: float, params: dict,
        ...                    forcings: dict) -> jnp.ndarray:
        ...     return biomass + forcings['recruitment'] * dt
    """

    name: str
    func: Callable
    inputs: list[str]
    outputs: list[str]
    scope: Literal["local", "global"] = "local"
    compiled: bool = False
    forcings: list[str] = field(default_factory=list)
    _compiled_func: Callable | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Compile function if requested."""
        if self.compiled:
            self._compiled_func = jax.jit(self.func)
        else:
            self._compiled_func = self.func

    def __hash__(self) -> int:
        """Make Unit hashable based on name."""
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        """Compare Units based on name."""
        if not isinstance(other, Unit):
            return NotImplemented
        return self.name == other.name

    def can_execute(self, available_vars: set[str]) -> bool:
        """Check if all required inputs are available.

        Args:
            available_vars: Set of variable names currently in state.

        Returns:
            True if all inputs are available, False otherwise.
        """
        return set(self.inputs).issubset(available_vars)

    def execute(self, state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Execute the unit's function.

        Args:
            state: Current simulation state (dict of arrays).
            **kwargs: Additional arguments (dt, params, halo data, forcings, etc.).

        Returns:
            Dictionary with output variable names and their computed values.

        Raises:
            ValueError: If required inputs are missing from state.
            ValueError: If required forcings are missing.
        """
        # Check inputs availability
        available = set(state.keys())
        missing = set(self.inputs) - available
        if missing:
            raise ValueError(
                f"Unit '{self.name}' missing required inputs: {missing}. " f"Available: {available}"
            )

        # Extract inputs from state
        inputs_dict = {key: state[key] for key in self.inputs}

        # Handle forcings if requested
        if self.forcings:
            # Check if forcings dict is provided in kwargs
            if "forcings" not in kwargs:
                raise ValueError(
                    f"Unit '{self.name}' requires forcings {self.forcings} "
                    "but no 'forcings' dict was provided"
                )

            all_forcings = kwargs["forcings"]
            missing_forcings = set(self.forcings) - set(all_forcings.keys())
            if missing_forcings:
                raise ValueError(
                    f"Unit '{self.name}' missing required forcings: {missing_forcings}. "
                    f"Available: {set(all_forcings.keys())}"
                )

            # Filter only the forcings this unit needs
            filtered_forcings = {key: all_forcings[key] for key in self.forcings}

            # Replace full forcings dict with filtered version
            kwargs = {**kwargs, "forcings": filtered_forcings}

        # Merge with additional kwargs
        all_args = {**inputs_dict, **kwargs}

        # Filter args to match function signature
        assert self._compiled_func is not None
        sig = inspect.signature(self.func)
        filtered_args = {}
        for param_name in sig.parameters:
            if param_name in all_args:
                filtered_args[param_name] = all_args[param_name]

        # Execute function
        result = self._compiled_func(**filtered_args)

        # Wrap single output in dict
        if len(self.outputs) == 1:
            if not isinstance(result, dict):
                result = {self.outputs[0]: result}
        else:
            # Multiple outputs: expect dict or tuple
            if isinstance(result, tuple):
                if len(result) != len(self.outputs):
                    raise ValueError(
                        f"Unit '{self.name}' returned {len(result)} values "
                        f"but declared {len(self.outputs)} outputs"
                    )
                result = dict(zip(self.outputs, result, strict=False))
            elif not isinstance(result, dict):
                raise ValueError(
                    f"Unit '{self.name}' with multiple outputs must return dict or tuple"
                )

        # Validate outputs
        for out_var in self.outputs:
            if out_var not in result:
                raise ValueError(f"Unit '{self.name}' did not produce declared output '{out_var}'")

        # Type assertion for mypy
        assert isinstance(result, dict)
        return result


def unit(
    name: str,
    inputs: list[str],
    outputs: list[str],
    scope: Literal["local", "global"] = "local",
    compiled: bool = False,
    forcings: list[str] | None = None,
) -> Callable[[Callable], Unit]:
    """Decorator to create a Unit from a function.

    Args:
        name: Unit name.
        inputs: List of input variable names.
        outputs: List of output variable names.
        scope: 'local' or 'global'.
        compiled: Whether to JIT compile with JAX.
        forcings: List of forcing variable names required by this unit.

    Returns:
        Decorator that wraps a function into a Unit.

    Example:
        >>> @unit(name='mortality', inputs=['biomass'], outputs=['mortality_rate'],
        ...       scope='local', compiled=True)
        ... def compute_mortality(biomass: jnp.ndarray, params: dict) -> jnp.ndarray:
        ...     return params['lambda'] * biomass
        >>>
        >>> @unit(name='growth', inputs=['biomass'], outputs=['biomass'],
        ...       scope='local', compiled=True, forcings=['recruitment'])
        ... def compute_growth(biomass: jnp.ndarray, dt: float, params: dict,
        ...                    forcings: dict) -> jnp.ndarray:
        ...     return biomass + forcings['recruitment'] * dt
    """
    if forcings is None:
        forcings = []

    def decorator(func: Callable) -> Unit:
        return Unit(
            name=name,
            func=func,
            inputs=inputs,
            outputs=outputs,
            scope=scope,
            compiled=compiled,
            forcings=forcings,
        )

    return decorator
