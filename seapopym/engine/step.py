"""Step function builder for time-stepping logic.

The step function encapsulates the logic for a single timestep:
1. Execute process chain to compute derived values
2. Integrate tendencies into state (configurable: Euler, RK2, RK4)
3. Extract diagnostics
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


def _apply_clamp(
    state: State,
    clamp_map: dict[str, tuple[float | None, float | None]] | None,
) -> State:
    """Apply per-variable clamping bounds to state.

    Args:
        state: State to clamp.
        clamp_map: Per-variable clamping bounds.

    Returns:
        Clamped state.
    """
    if not clamp_map:
        return state
    result = {}
    for var_name, value in state.items():
        if var_name in clamp_map:
            lo, hi = clamp_map[var_name]
            if lo is not None:
                value = jnp.maximum(value, lo)
            if hi is not None:
                value = jnp.minimum(value, hi)
        result[var_name] = value
    return result


def build_step_fn(
    model: CompiledModel,
    export_variables: list[str] | None = None,
) -> Callable[..., tuple[Any, Outputs]]:
    """Build a step function from a compiled model.

    The returned function executes one timestep of the model:
    1. Computes derived values via the process chain
    2. Integrates state using the configured integrator (Euler, RK2, or RK4)
    3. Returns new state and diagnostic outputs

    Args:
        model: Compiled model containing compute_nodes, tendency_map, parameters, and metadata.
        export_variables: If provided, only these variables are included in
            the scan outputs. Filtering happens *inside* the step function,
            so ``lax.scan`` never accumulates the excluded variables.

    Returns:
        Step function with signature ``((state, params), (forcings_t, time_params_t)) ->
        ((new_state, params), outputs)``.  This is compatible with ``jax.lax.scan``
        and ``jax.grad``.  When there are no time-indexed parameters the second
        element of *xs* is an empty dict (no JAX leaves, zero overhead).
    """
    # Extract what we need from model (closure captures these)
    compute_nodes: list[ComputeNode] = model.compute_nodes
    tendency_map = model.tendency_map
    dt = model.dt
    statics = model.forcings.get_statics()
    clamp_map = model.clamp_map
    integrator = model.integrator

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

    # --- Method of lines: decoupled process chain and integration ---

    def _run_process_chain(
        state: State, all_forcings: Forcings, parameters: Params
    ) -> dict[str, Array]:
        """Execute the full process chain and return intermediates.

        Args:
            state: Current state variables.
            all_forcings: All forcings (static + dynamic).
            parameters: Model parameters.

        Returns:
            Dict of all intermediate (derived) values.
        """
        intermediates: dict[str, Array] = {}

        for compute_node in compute_nodes:
            func_inputs = _resolve_inputs(compute_node.input_mapping, state, all_forcings, parameters, intermediates)

            func, arg_order, transpose_axes = vmapped_funcs[compute_node.name]
            if arg_order is not None:
                result = func(*[func_inputs[arg] for arg in arg_order])
                if transpose_axes is not None:
                    result = _transpose_vmap_output(result, transpose_axes)
            else:
                result = func(**func_inputs)

            _handle_compute_outputs(result, compute_node.output_mapping, intermediates)

        return intermediates

    def _sum_tendencies(intermediates: dict[str, Array]) -> dict[str, Array]:
        """Sum tendency sources for each state variable.

        Args:
            intermediates: All computed derived values.

        Returns:
            Dict mapping state variable names to their total tendency.
        """
        tendencies: dict[str, Array] = {}
        for var_name, sources in tendency_map.items():
            tendencies[var_name] = sum(
                src.sign * intermediates[src.source.removeprefix("derived.")] for src in sources
            )
        return tendencies

    def _eval_tendencies(state: State, all_forcings: Forcings, parameters: Params) -> dict[str, Array]:
        """Evaluate the RHS of the ODE: run process chain then sum tendencies.

        Args:
            state: State at which to evaluate.
            all_forcings: All forcings for this timestep.
            parameters: Model parameters.

        Returns:
            Dict mapping state variable names to their total tendency.
        """
        return _sum_tendencies(_run_process_chain(state, all_forcings, parameters))

    def _advance_state(state: State, tendencies: dict[str, Array], scale: float) -> State:
        """Compute state + scale * tendencies for variables with tendencies.

        Variables without tendencies are passed through unchanged.

        Args:
            state: Base state.
            tendencies: Tendency per state variable (only vars with tendencies).
            scale: Scalar multiplier (typically dt or fraction of dt).

        Returns:
            Advanced state.
        """
        return {
            var: (value + scale * tendencies[var] if var in tendencies else value)
            for var, value in state.items()
        }

    # --- Integration functions ---
    # Each returns (new_state, intermediates) where intermediates are from the
    # initial state evaluation (diagnostics at the beginning of the timestep).

    def _step_euler(
        state: State, all_forcings: Forcings, parameters: Params
    ) -> tuple[State, dict[str, Array]]:
        """Euler explicit: 1 evaluation."""
        intermediates = _run_process_chain(state, all_forcings, parameters)
        tendencies = _sum_tendencies(intermediates)
        new_state = _advance_state(state, tendencies, dt)
        new_state = _apply_clamp(new_state, clamp_map)
        return new_state, intermediates

    def _step_rk2(
        state: State, all_forcings: Forcings, parameters: Params
    ) -> tuple[State, dict[str, Array]]:
        """Heun's RK2: 2 evaluations."""
        intermediates = _run_process_chain(state, all_forcings, parameters)
        k1 = _sum_tendencies(intermediates)
        k2 = _eval_tendencies(_advance_state(state, k1, dt), all_forcings, parameters)
        combined = {var: 0.5 * (k1[var] + k2[var]) for var in k1}
        new_state = _advance_state(state, combined, dt)
        new_state = _apply_clamp(new_state, clamp_map)
        return new_state, intermediates

    def _step_rk4(
        state: State, all_forcings: Forcings, parameters: Params
    ) -> tuple[State, dict[str, Array]]:
        """Classical RK4: 4 evaluations."""
        intermediates = _run_process_chain(state, all_forcings, parameters)
        k1 = _sum_tendencies(intermediates)
        k2 = _eval_tendencies(_advance_state(state, k1, 0.5 * dt), all_forcings, parameters)
        k3 = _eval_tendencies(_advance_state(state, k2, 0.5 * dt), all_forcings, parameters)
        k4 = _eval_tendencies(_advance_state(state, k3, dt), all_forcings, parameters)
        combined = {var: (k1[var] + 2 * k2[var] + 2 * k3[var] + k4[var]) / 6 for var in k1}
        new_state = _advance_state(state, combined, dt)
        new_state = _apply_clamp(new_state, clamp_map)
        return new_state, intermediates

    # Select integrator at build-time (not per-step)
    _integrators = {"euler": _step_euler, "rk2": _step_rk2, "rk4": _step_rk4}
    integrate_fn = _integrators[integrator]

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

        new_state, intermediates = integrate_fn(state, all_forcings, parameters)

        # Build outputs (include state variables for saving)
        outputs: Outputs = {**intermediates, **new_state}
        if export_variables is not None:
            outputs = {k: v for k, v in outputs.items() if k in export_variables}

        return new_state, outputs

    def step_fn(
        carry: tuple[State, Params],
        xs: tuple[Forcings, Params],
    ) -> tuple[tuple[State, Params], Outputs]:
        """Execute one timestep with parameters as part of the carry.

        This signature is compatible with jax.lax.scan and jax.grad.
        Time-indexed parameters are passed via *xs* (second element) so that
        ``jax.grad`` can differentiate through them per-timestep.

        Args:
            carry: Tuple of (state, static_params).
            xs: Tuple of (forcings_t, time_params_t).  time_params_t is an
                empty dict when there are no time-indexed parameters.

        Returns:
            Tuple of ((new_state, static_params), outputs).
        """
        state, static_params = carry
        forcings_t, time_params_t = xs
        all_params = {**static_params, **time_params_t}
        new_state, outputs = _execute_step(state, forcings_t, all_params)
        return (new_state, static_params), outputs

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
    clamp_map: dict[str, tuple[float | None, float | None]] | None = None,
) -> State:
    """Integrate state using Euler explicit method with declarative tendency_map.

    Args:
        state: Current state.
        intermediates: All computed derived values.
        tendency_map: Mapping from state var name to list of TendencySource.
        dt: Timestep in seconds.
        clamp_map: Per-variable clamping bounds. ``None`` or missing key means
            no clamping for that variable.  ``(0.0, None)`` clamps to non-negative.

    Returns:
        New state after integration.
    """
    new_state = {}
    for var_name, value in state.items():
        if var_name in tendency_map:
            sources = tendency_map[var_name]
            total = sum(src.sign * intermediates[src.source.removeprefix("derived.")] for src in sources)
            updated = value + total * dt
            # Apply per-variable clamping if specified
            if clamp_map and var_name in clamp_map:
                lo, hi = clamp_map[var_name]
                if lo is not None:
                    updated = jnp.maximum(updated, lo)
                if hi is not None:
                    updated = jnp.minimum(updated, hi)
            new_state[var_name] = updated
        else:
            new_state[var_name] = value
    return new_state
