"""Step function builder for time-stepping logic.

The step function encapsulates the logic for a single timestep:
1. Execute process chain to compute derived values
2. Integrate tendencies into state (Euler explicit)
3. Apply mask
4. Extract diagnostics
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from seapopym.blueprint.nodes import ComputeNode
from seapopym.engine.vectorize import (
    compute_broadcast_dims,
    compute_output_transpose_axes,
    wrap_with_vmap,
)
from seapopym.types import Array, Forcings, Outputs, Params, State

if TYPE_CHECKING:
    from seapopym.blueprint.schema import TendencySource
    from seapopym.compiler import CompiledModel


def build_step_fn(
    model: CompiledModel,
    export_variables: list[str] | None = None,
) -> Callable[..., tuple[Any, Outputs]]:
    """Build a step function from a compiled model.

    The returned function executes one timestep of the model:
    1. Computes derived values via the process chain
    2. Integrates state using tendency_map + Euler explicit
    3. Applies mask to state
    4. Returns new state and diagnostic outputs

    Args:
        model: Compiled model containing compute_nodes, tendency_map, parameters, and metadata.
        export_variables: If provided, only these variables are included in
            the scan outputs. Filtering happens *inside* the step function,
            so ``lax.scan`` never accumulates the excluded variables.

    Returns:
        Step function with signature ((state, params), forcings_t) -> ((new_state, params), outputs).
        This signature is compatible with jax.lax.scan and jax.grad.
    """
    # Extract what we need from model (closure captures these)
    compute_nodes: list[ComputeNode] = model.compute_nodes
    tendency_map = model.tendency_map
    dt = model.dt
    statics = model.forcings.get_statics()

    # Pre-compute vmapped functions for nodes with broadcast dimensions
    vmapped_funcs: dict[str, tuple[Callable[..., Any], list[str] | None, tuple[int, ...] | None]] = {}
    for compute_node in compute_nodes:
        arg_order = list(compute_node.input_mapping.keys())
        broadcast_dims = compute_broadcast_dims(compute_node.input_dims, compute_node.core_dims)
        if broadcast_dims:
            transpose_axes = compute_output_transpose_axes(broadcast_dims, compute_node.out_dims)
            vmapped_funcs[compute_node.name] = (
                wrap_with_vmap(
                    compute_node.func,
                    compute_node.input_dims,
                    compute_node.core_dims,
                    arg_order,
                ),
                arg_order,
                transpose_axes,
            )
        else:
            vmapped_funcs[compute_node.name] = (compute_node.func, None, None)

    def _execute_step(state: State, forcings_t: Forcings, parameters: Params) -> tuple[State, Outputs]:
        """Core step execution logic.

        Args:
            state: Current state variables.
            forcings_t: Dynamic forcings at current timestep.
            parameters: Model parameters.

        Returns:
            Tuple of (new_state, outputs).
        """
        all_forcings = {**statics, **forcings_t}
        mask = all_forcings.get("mask", 1.0)

        # Store all derived values
        intermediates: dict[str, Array] = {}

        # Execute each ComputeNode in process order
        for compute_node in compute_nodes:
            func_inputs = _resolve_inputs(compute_node.input_mapping, state, all_forcings, parameters, intermediates)

            func, arg_order, transpose_axes = vmapped_funcs[compute_node.name]
            if arg_order is not None:
                result = func(*[func_inputs[arg] for arg in arg_order])
                if transpose_axes is not None:
                    result = _transpose_vmap_output(result, transpose_axes)
            else:
                result = func(**func_inputs)

            # All outputs go to intermediates (keyed by derived short name)
            _handle_compute_outputs(result, compute_node.output_mapping, intermediates)

        # Euler explicit integration using tendency_map
        new_state = _integrate_euler(state, intermediates, tendency_map, dt)

        # Apply mask to state
        new_state = _apply_mask(new_state, mask)

        # Build outputs (include state variables for saving)
        outputs: Outputs = {**intermediates, **new_state}
        if export_variables is not None:
            outputs = {k: v for k, v in outputs.items() if k in export_variables}

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
    expected_ndim = len(axes)

    def transpose_if_matching(arr: Any) -> Any:
        """Transpose array only if it has the expected number of dimensions."""
        if hasattr(arr, "ndim") and arr.ndim == expected_ndim:
            return jnp.transpose(arr, axes)
        return arr

    if isinstance(result, tuple):
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
            elif category == "derived":
                result[arg_name] = intermediates[var_name]
            else:
                raise KeyError(f"Unknown category '{category}' in path '{var_path}'")

    return result


def _handle_compute_outputs(
    result: Any,
    output_mapping: dict[str, str],
    intermediates: dict[str, Array],
) -> None:
    """Handle ComputeNode outputs — all go to intermediates.

    Args:
        result: Function return value (single array or tuple of arrays).
        output_mapping: Mapping from output key to derived target path.
        intermediates: Dict to store intermediate results (keyed by short name after 'derived.').
    """
    output_items = list(output_mapping.items())

    if len(output_items) > 1:
        if not isinstance(result, tuple | list):
            raise TypeError(f"Function returned {type(result)} but expected tuple for {len(output_items)} outputs.")
        if len(result) != len(output_items):
            raise ValueError(f"Function returned {len(result)} items but expected {len(output_items)}.")

        for idx, (_out_key, target) in enumerate(output_items):
            # Strip "derived." prefix for intermediates key
            var_name = target.removeprefix("derived.")
            intermediates[var_name] = result[idx]
    else:
        target = output_items[0][1]
        var_name = target.removeprefix("derived.")
        intermediates[var_name] = result


def _integrate_euler(
    state: State,
    intermediates: dict[str, Array],
    tendency_map: dict[str, list[TendencySource]],
    dt: float,
) -> State:
    """Integrate state using Euler explicit method with declarative tendency_map.

    Args:
        state: Current state.
        intermediates: All computed derived values.
        tendency_map: Mapping from state var name to list of TendencySource.
        dt: Timestep in seconds.

    Returns:
        New state after integration.
    """
    new_state = {}
    for var_name, value in state.items():
        if var_name in tendency_map:
            sources = tendency_map[var_name]
            total = sum(
                src.sign * intermediates[src.source.removeprefix("derived.")]
                for src in sources
            )
            # Clamp to zero: biomass and other state variables are physically
            # non-negative quantities.  Euler explicit can overshoot below zero
            # when the loss tendency is large relative to dt, so we enforce the
            # constraint here.
            new_state[var_name] = jnp.maximum(value + total * dt, 0.0)
        else:
            new_state[var_name] = value
    return new_state


def _apply_mask(state: State, mask: Array) -> State:
    """Apply mask to state variables.

    Args:
        state: Current state.
        mask: Binary mask (1 = valid, 0 = masked).

    Returns:
        Masked state.
    """
    if isinstance(mask, int | float) and mask == 1.0:
        return state

    new_state = {}
    for var_name, value in state.items():
        new_state[var_name] = value * mask

    return new_state
