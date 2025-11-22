"""Unit: Composable computational unit for the Kernel.

A Unit represents a single computational step in the simulation.
Each Unit has:
- inputs: variables it needs from the state
- outputs: variables it produces
- scope: 'local' (embarrassingly parallel) or 'global' (requires neighbor communication)
- func: the actual computation function
- compiled: whether to JIT compile with JAX
"""

import copy
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

        internal_inputs: List of internal function argument names.
        internal_outputs: List of internal function return names.
        internal_forcings: List of internal forcing argument names.

        input_mapping: Mapping from function argument names to state variable names.
        output_mapping: Mapping from function output names to state variable names.
        forcing_mapping: Mapping from function forcing argument names to global forcing names.

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

    # Internal names (Function Signature) - These never change
    internal_inputs: list[str] = field(default_factory=list)
    internal_outputs: list[str] = field(default_factory=list)
    internal_forcings: list[str] = field(default_factory=list)

    # Mappings: Internal Name (Function) -> External Name (State/Forcing)
    input_mapping: dict[str, str] = field(default_factory=dict)
    output_mapping: dict[str, str] = field(default_factory=dict)
    forcing_mapping: dict[str, str] = field(default_factory=dict)

    _compiled_func: Callable | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Compile function if requested and initialize mappings if empty."""
        if self.compiled:
            self._compiled_func = jax.jit(self.func)
        else:
            self._compiled_func = self.func

        # Initialize internals if empty (first creation)
        if not self.internal_inputs:
            self.internal_inputs = list(self.inputs)
        if not self.internal_outputs:
            self.internal_outputs = list(self.outputs)
        if not self.internal_forcings:
            self.internal_forcings = list(self.forcings)

        # Initialize mappings with identity if not provided
        if not self.input_mapping:
            self.input_mapping = {name: name for name in self.internal_inputs}
        if not self.output_mapping:
            self.output_mapping = {name: name for name in self.internal_outputs}
        if not self.forcing_mapping:
            self.forcing_mapping = {name: name for name in self.internal_forcings}

    def __hash__(self) -> int:
        """Make Unit hashable based on name."""
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        """Compare Units based on name."""
        if not isinstance(other, Unit):
            return NotImplemented
        return self.name == other.name

    def bind(self, variable_map: dict[str, str]) -> "Unit":
        """Create a new Unit with bound variable names.

        Args:
            variable_map: Dictionary mapping internal names to global names.
                          Any name not in the map is assumed to be unchanged.

        Returns:
            A new Unit instance with updated mappings and input/output lists.
        """
        new_unit = copy.copy(self)

        # Update Input Mapping and List
        new_input_mapping = {}
        new_inputs = []
        for internal_name in self.internal_inputs:
            global_name = variable_map.get(internal_name, internal_name)
            new_input_mapping[internal_name] = global_name
            new_inputs.append(global_name)

        # Update Output Mapping and List
        new_output_mapping = {}
        new_outputs = []
        for internal_name in self.internal_outputs:
            global_name = variable_map.get(internal_name, internal_name)
            new_output_mapping[internal_name] = global_name
            new_outputs.append(global_name)

        # Update Forcing Mapping and List
        new_forcing_mapping = {}
        new_forcings = []
        for internal_name in self.internal_forcings:
            global_name = variable_map.get(internal_name, internal_name)
            new_forcing_mapping[internal_name] = global_name
            new_forcings.append(global_name)

        new_unit.input_mapping = new_input_mapping
        new_unit.inputs = new_inputs
        new_unit.output_mapping = new_output_mapping
        new_unit.outputs = new_outputs
        new_unit.forcing_mapping = new_forcing_mapping
        new_unit.forcings = new_forcings

        return new_unit

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
                f"Unit '{self.name}' missing required inputs: {missing}. Available: {available}"
            )

        # Extract inputs from state using Mapping (Internal -> Global)
        func_args = {}
        for internal_name, global_name in self.input_mapping.items():
            func_args[internal_name] = state[global_name]

        # Handle forcings if requested
        if self.forcings:
            if "forcings" not in kwargs:
                raise ValueError(
                    f"Unit '{self.name}' requires forcings {self.forcings} "
                    "but no 'forcings' dict was provided"
                )

            all_forcings = kwargs["forcings"]
            # Check availability using Global Names
            missing_forcings = set(self.forcings) - set(all_forcings.keys())
            if missing_forcings:
                raise ValueError(
                    f"Unit '{self.name}' missing required forcings: {missing_forcings}. "
                    f"Available: {set(all_forcings.keys())}"
                )

            # Create a forcings dict for the function using Internal Names
            func_forcings = {}
            for internal_name, global_name in self.forcing_mapping.items():
                func_forcings[internal_name] = all_forcings[global_name]

            # Replace full forcings dict with filtered/mapped version
            kwargs = {**kwargs, "forcings": func_forcings}

        # Merge with additional kwargs
        all_args = {**func_args, **kwargs}

        # Filter args to match function signature
        assert self._compiled_func is not None
        sig = inspect.signature(self.func)
        filtered_args = {}
        for param_name in sig.parameters:
            if param_name in all_args:
                filtered_args[param_name] = all_args[param_name]

        # Execute function
        result = self._compiled_func(**filtered_args)

        # Map results back to Global Names
        mapped_result = {}

        if len(self.internal_outputs) == 1:
            # Single output case
            internal_key = self.internal_outputs[0]
            global_key = self.output_mapping[internal_key]

            if isinstance(result, dict):
                if internal_key in result:
                    mapped_result[global_key] = result[internal_key]
                else:
                    raise ValueError(
                        f"Unit '{self.name}' did not produce internal output '{internal_key}'"
                    )
            else:
                mapped_result[global_key] = result
        else:
            # Multiple outputs
            if isinstance(result, tuple):
                if len(result) != len(self.internal_outputs):
                    raise ValueError(
                        f"Unit '{self.name}' returned {len(result)} values "
                        f"but declared {len(self.internal_outputs)} outputs"
                    )
                # Map tuple elements to outputs in order
                for i, value in enumerate(result):
                    internal_key = self.internal_outputs[i]
                    global_key = self.output_mapping[internal_key]
                    mapped_result[global_key] = value

            elif isinstance(result, dict):
                for internal_key, value in result.items():
                    if internal_key in self.output_mapping:
                        global_key = self.output_mapping[internal_key]
                        mapped_result[global_key] = value
            else:
                raise ValueError(
                    f"Unit '{self.name}' with multiple outputs must return dict or tuple"
                )

        # Validate outputs (Global Names)
        for out_var in self.outputs:
            if out_var not in mapped_result:
                raise ValueError(f"Unit '{self.name}' did not produce declared output '{out_var}'")

        return mapped_result


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
            # Internal lists are copies of the initial lists
            internal_inputs=list(inputs),
            internal_outputs=list(outputs),
            internal_forcings=list(forcings),
        )

    return decorator
