"""Step function builder for time-stepping logic.

The step function encapsulates the logic for a single timestep:
1. Execute process graph to compute tendencies
2. Apply Euler integration
3. Apply mask
4. Extract diagnostics
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from seapopym.blueprint.nodes import ComputeNode
from seapopym.engine.vectorize import (
    compute_broadcast_dims,
    compute_output_transpose_axes,
    wrap_with_vmap,
)

if TYPE_CHECKING:
    from seapopym.compiler import CompiledModel

from seapopym.types import Array, Forcings, Outputs, Params, State


def build_step_fn(
    model: CompiledModel,
) -> Callable[..., tuple[Any, Outputs]]:
    """Build a step function from a compiled model.

    The returned function executes one timestep of the model:
    1. Computes tendencies via the process graph
    2. Integrates state using Euler explicit
    3. Applies mask to state
    4. Returns new state and diagnostic outputs

    Args:
        model: Compiled model containing graph, parameters, and metadata.

    Returns:
        Step function with signature ((state, params), forcings_t) -> ((new_state, params), outputs).
        This signature is compatible with jax.lax.scan and jax.grad.
    """
    # Extract what we need from model (closure captures these)
    graph = model.graph
    default_parameters = model.parameters
    dt = model.dt
    backend = model.backend

    # Get ordered list of ComputeNode from graph
    # Nodes are already in topological order from validation
    import networkx as nx

    compute_nodes: list[ComputeNode] = [node for node in nx.topological_sort(graph) if isinstance(node, ComputeNode)]

    # Pre-compute vmapped functions for nodes with core_dims
    # Store (vmapped_func, arg_order, transpose_axes) - arg_order for positional calling,
    # transpose_axes to convert vmap output order back to canonical order
    vmapped_funcs: dict[str, tuple[Callable[..., Any], list[str] | None, tuple[int, ...] | None]] = {}
    for compute_node in compute_nodes:
        if compute_node.core_dims:
            # Get argument order from input_mapping keys
            arg_order = list(compute_node.input_mapping.keys())
            broadcast_dims = compute_broadcast_dims(compute_node.input_dims, compute_node.core_dims)
            transpose_axes = compute_output_transpose_axes(broadcast_dims, compute_node.out_dims)
            vmapped_funcs[compute_node.name] = (
                wrap_with_vmap(
                    compute_node.func,
                    compute_node.input_dims,
                    compute_node.core_dims,
                    arg_order,
                ),
                arg_order,  # Store arg_order for positional calling
                transpose_axes,  # Store transpose axes for output reordering
            )
        else:
            # No core_dims, use original function (element-wise broadcasting)
            vmapped_funcs[compute_node.name] = (compute_node.func, None, None)

    def _execute_step(state: State, forcings_t: Forcings, parameters: Params) -> tuple[State, Outputs]:
        """Core step execution logic.

        Args:
            state: Current state variables.
            forcings_t: Forcings at current timestep (includes mask).
            parameters: Model parameters.

        Returns:
            Tuple of (new_state, outputs).
        """
        # Get mask (default to 1.0 if not present)
        mask = forcings_t.get("mask", 1.0)

        # Accumulate tendencies per state variable
        tendencies: dict[str, list[Array]] = {}

        # Store intermediate results for diagnostics
        intermediates: dict[str, Array] = {}

        # Execute each ComputeNode in topological order
        for compute_node in compute_nodes:
            # Build function inputs from state, forcings, parameters, intermediates
            func_inputs = _resolve_inputs(compute_node.input_mapping, state, forcings_t, parameters, intermediates)

            # Call the vmapped function (or original if no core_dims)
            func, arg_order, transpose_axes = vmapped_funcs[compute_node.name]
            if arg_order is not None:
                # vmapped function requires positional arguments in specific order
                result = func(*[func_inputs[arg] for arg in arg_order])
                # Transpose output from vmap order to canonical order
                if transpose_axes is not None:
                    result = _transpose_vmap_output(result, transpose_axes)
            else:
                # Original function can use keyword arguments
                result = func(**func_inputs)

            # Handle outputs using the node's output_mapping
            _handle_compute_outputs(result, compute_node.output_mapping, tendencies, intermediates)

        # Euler explicit integration
        new_state = _integrate_euler(state, tendencies, dt, backend)

        # Apply mask to state
        new_state = _apply_mask(new_state, mask, backend)

        # Build outputs (include state for I/O)
        outputs: Outputs = {**intermediates}
        # Also include state variables in outputs for saving
        for var_name, value in new_state.items():
            outputs[var_name] = value

        return new_state, outputs

    def step_fn(
        carry: tuple[State, Params], forcings_t: Forcings
    ) -> tuple[tuple[State, Params], Outputs]:
        """Execute one timestep with parameters as part of the carry.

        This signature is compatible with jax.lax.scan and jax.grad.

        Args:
            carry: Tuple of (state, params).
            forcings_t: Forcings at current timestep.

        Returns:
            Tuple of ((new_state, params), outputs).
        """
        state, params = carry
        new_state, outputs = _execute_step(state, forcings_t, params)
        return (new_state, params), outputs

    return step_fn


def _transpose_vmap_output(result: Any, axes: tuple[int, ...]) -> Any:
    """Transpose vmap output from vmap order to canonical order.

    Handles both single outputs and tuples of outputs.
    Only transposes arrays with the expected number of dimensions.

    Args:
        result: Function result (array or tuple of arrays).
        axes: Transpose axes to apply.

    Returns:
        Transposed result.
    """
    import jax.numpy as jnp

    expected_ndim = len(axes)

    def transpose_if_matching(arr: Any) -> Any:
        """Transpose array only if it has the expected number of dimensions."""
        if hasattr(arr, "ndim") and arr.ndim == expected_ndim:
            return jnp.transpose(arr, axes)
        return arr

    if isinstance(result, tuple):
        # Multi-output: transpose each if dimensions match
        return tuple(transpose_if_matching(r) for r in result)
    else:
        return transpose_if_matching(result)


def _resolve_inputs(
    inputs_mapping: dict[str, str],
    state: State,
    forcings_t: Forcings,
    parameters: dict[str, Array],
    intermediates: dict[str, Array],
) -> dict[str, Array]:
    """Resolve input references to actual arrays.

    Args:
        inputs_mapping: Mapping from arg name to variable path (e.g., "state.biomass").
        state: Current state dict.
        forcings_t: Current forcings dict.
        parameters: Parameters dict.
        intermediates: Intermediate results from earlier processes.

    Returns:
        Dict mapping argument names to arrays.
    """
    result = {}

    for arg_name, var_path in inputs_mapping.items():
        parts = var_path.split(".")

        if len(parts) < 2:
            # Direct reference (e.g., just "biomass")
            # Try to find in state, then forcings, then parameters
            if var_path in state:
                result[arg_name] = state[var_path]
            elif var_path in forcings_t:
                result[arg_name] = forcings_t[var_path]
            elif var_path in parameters:
                result[arg_name] = parameters[var_path]
            elif var_path in intermediates:
                result[arg_name] = intermediates[var_path]
            else:
                raise KeyError(f"Cannot resolve input '{var_path}' for argument '{arg_name}'")
        else:
            category = parts[0]
            var_name = ".".join(parts[1:])

            if category == "state":
                result[arg_name] = state[var_name]
            elif category == "forcings":
                result[arg_name] = forcings_t[var_name]
            elif category == "parameters":
                result[arg_name] = parameters[var_name]
            elif category == "intermediates" or category == "derived":
                result[arg_name] = intermediates[var_name]
            else:
                raise KeyError(f"Unknown category '{category}' in path '{var_path}'")

    return result


def _handle_compute_outputs(
    result: Any,
    output_mapping: dict[str, str],
    tendencies: dict[str, list[Array]],
    intermediates: dict[str, Array],
) -> None:
    """Handle ComputeNode outputs based on target path.

    Args:
        result: Function return value (single array or tuple of arrays).
        output_mapping: Mapping from output key to target path.
        tendencies: Dict to accumulate tendencies.
        intermediates: Dict to store intermediate results.
    """
    output_items = list(output_mapping.items())

    # If multiple outputs, result MUST be a tuple/list matching the mapping order
    if len(output_items) > 1:
        if not isinstance(result, tuple | list):
            raise TypeError(f"Function returned {type(result)} but expected tuple for {len(output_items)} outputs.")
        if len(result) != len(output_items):
            raise ValueError(f"Function returned {len(result)} items but expected {len(output_items)}.")

        for idx, (_out_key, target) in enumerate(output_items):
            val = result[idx]
            _dispatch_single_output(val, target, tendencies, intermediates)
    else:
        # Single output
        target = output_items[0][1]
        _dispatch_single_output(result, target, tendencies, intermediates)


def _dispatch_single_output(
    val: Array, target: str, tendencies: dict[str, list[Array]], intermediates: dict[str, Array]
) -> None:
    """Helper to dispatch a single value to tendencies or intermediates."""
    parts = target.split(".")

    if len(parts) >= 2 and parts[0] == "tendencies":
        # It's a tendency - parse state variable from target
        # "tendencies.biomass" -> state_var = "biomass"
        # "tendencies.biomass_growth" -> state_var = "biomass"
        var_name = parts[1]
        state_var = var_name.split("_")[0] if "_" in var_name else var_name

        if state_var not in tendencies:
            tendencies[state_var] = []
        tendencies[state_var].append(val)
    else:
        # It's a derived/diagnostic value
        var_name = parts[-1] if len(parts) > 1 else target
        intermediates[var_name] = val


def _integrate_euler(
    state: State,
    tendencies: dict[str, list[Array]],
    dt: float,
    backend: str,
) -> State:
    """Integrate state using Euler explicit method.

    Args:
        state: Current state.
        tendencies: Accumulated tendencies per state variable.
        dt: Timestep in seconds.
        backend: Backend name for array operations.

    Returns:
        New state after integration.
    """
    if backend == "jax":
        import jax.numpy as jnp

        new_state = {}
        for var_name, value in state.items():
            if var_name in tendencies:
                total_tendency = sum(tendencies[var_name])
                new_state[var_name] = jnp.maximum(value + total_tendency * dt, 0.0)
            else:
                new_state[var_name] = value
        return new_state
    else:
        import numpy as np

        new_state = {}
        for var_name, value in state.items():
            if var_name in tendencies:
                total_tendency = sum(tendencies[var_name])
                new_state[var_name] = np.maximum(value + total_tendency * dt, 0.0)
            else:
                new_state[var_name] = np.copy(value)
        return new_state


def _apply_mask(state: State, mask: Array, _backend: str) -> State:
    """Apply mask to state variables.

    Args:
        state: Current state.
        mask: Binary mask (1 = valid, 0 = masked).
        backend: Backend name.

    Returns:
        Masked state.
    """
    # If mask is scalar 1.0, no masking needed
    if isinstance(mask, int | float) and mask == 1.0:
        return state

    new_state = {}
    for var_name, value in state.items():
        # Broadcast mask to match value shape if needed
        new_state[var_name] = value * mask

    return new_state
