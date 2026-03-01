"""Automatic vectorization of functions using jax.vmap.

This module provides utilities to wrap functions with vmap for automatic
broadcasting over non-core dimensions. Functions are written for their
core dimensions only, and vmap handles the rest.

Example:
    A function with core_dims={"production": ["C"]} expects production
    with shape (C,). If the actual data has shape (C, Y, X), this module
    wraps the function with vmap over Y and X dimensions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax

from seapopym.dims import CANONICAL_DIMS


def compute_broadcast_dims(
    input_dims: dict[str, tuple[str, ...]],
    core_dims: dict[str, list[str]],
) -> list[str]:
    """Compute which dimensions need to be vmapped (broadcast dimensions).

    Broadcast dimensions are those that appear in input_dims but are not
    in core_dims for any input. These are the dimensions we vmap over.

    Args:
        input_dims: Actual dimensions of each input.
                    Format: {"production": ("C", "Y", "X"), "temp": ("Y", "X")}.
        core_dims: Core dimensions per input (from FunctionMetadata).
                   Format: {"production": ["C"]}.

    Returns:
        List of dimension names to vmap over, in canonical order.

    Example:
        >>> compute_broadcast_dims(
        ...     input_dims={"production": ("C", "Y", "X"), "cohort_ages": ("C",)},
        ...     core_dims={"production": ["C"], "cohort_ages": ["C"]},
        ... )
        ["Y", "X"]
    """
    # Collect all dimensions from all inputs
    all_dims: set[str] = set()
    for dims in input_dims.values():
        all_dims.update(dims)

    # Collect all core dimensions
    all_core_dims: set[str] = set()
    for dims in core_dims.values():
        all_core_dims.update(dims)

    # Broadcast dims = all dims - core dims
    broadcast_dims = all_dims - all_core_dims

    # Return in canonical order for consistent vmap nesting
    return [d for d in CANONICAL_DIMS if d in broadcast_dims]


def compute_in_axes(
    input_dims: dict[str, tuple[str, ...]],
    broadcast_dim: str,
    arg_order: list[str],
) -> tuple[int | None, ...]:
    """Compute in_axes tuple for a single vmap level.

    For each argument, find the axis index of the broadcast dimension,
    or None if the argument doesn't have that dimension.

    Args:
        input_dims: Current dimensions of each input (updated after each vmap).
        broadcast_dim: The dimension we're vmapping over.
        arg_order: Order of arguments as they appear in the function signature.

    Returns:
        Tuple of axis indices (or None) for jax.vmap in_axes parameter.

    Example:
        >>> compute_in_axes(
        ...     input_dims={"production": ("C", "Y", "X"), "rate": ()},
        ...     broadcast_dim="X",
        ...     arg_order=["production", "rate"],
        ... )
        (2, None)
    """
    axes: list[int | None] = []

    for arg_name in arg_order:
        if arg_name not in input_dims:
            # Argument not in input_dims (e.g., scalar constant)
            axes.append(None)
            continue

        dims = input_dims[arg_name]
        if broadcast_dim in dims:
            axes.append(dims.index(broadcast_dim))
        else:
            axes.append(None)

    return tuple(axes)


def remove_dim_from_inputs(
    input_dims: dict[str, tuple[str, ...]],
    dim_to_remove: str,
) -> dict[str, tuple[str, ...]]:
    """Remove a dimension from input_dims after vmapping over it.

    After vmap over a dimension, that dimension is "consumed" and the
    inner function sees arrays without that dimension.

    Args:
        input_dims: Current dimensions of each input.
        dim_to_remove: The dimension that was vmapped over.

    Returns:
        New input_dims with the dimension removed.
    """
    new_dims: dict[str, tuple[str, ...]] = {}
    for name, dims in input_dims.items():
        new_dims[name] = tuple(d for d in dims if d != dim_to_remove)
    return new_dims


def compute_output_transpose_axes(
    broadcast_dims: list[str],
    out_dims: list[str] | None,
) -> tuple[int, ...] | None:
    """Compute transpose axes to convert vmap output order to canonical order.

    After vmap, output has shape (broadcast_dims..., out_dims...).
    Canonical order places dimensions like C before Y, X.

    Args:
        broadcast_dims: Dimensions that were vmapped over (in canonical order).
        out_dims: Output core dimensions (from function metadata).

    Returns:
        Tuple of axes for transposition, or None if no transpose needed.
    """
    if not out_dims:
        # No core dims in output, just broadcast dims (already canonical)
        return None

    # vmap output order: (broadcast_dims..., out_dims...)
    # Example: broadcast_dims = [Y, X], out_dims = [C]
    # vmap output order: (Y, X, C)
    # canonical order: (C, Y, X)
    vmap_order = list(broadcast_dims) + list(out_dims)

    # Get canonical order for these dims
    canonical_order = [d for d in CANONICAL_DIMS if d in vmap_order]

    if vmap_order == canonical_order:
        return None

    # Compute permutation: for each dim in canonical_order, find its position in vmap_order
    axes = tuple(vmap_order.index(d) for d in canonical_order)
    return axes


def wrap_with_vmap(
    func: Callable[..., Any],
    input_dims: dict[str, tuple[str, ...]],
    core_dims: dict[str, list[str]],
    arg_order: list[str],
) -> Callable[..., Any]:
    """Wrap a function with vmap for automatic broadcasting.

    The function is wrapped with nested vmap calls, one for each broadcast
    dimension. The order of vmap nesting follows the canonical dimension
    order (reversed, so innermost vmap is for the last canonical dimension).

    Args:
        func: The original function to wrap.
        input_dims: Actual dimensions of each input after canonical transposition.
        core_dims: Core dimensions per input (from FunctionMetadata).
        arg_order: Order of arguments as they appear in the function signature.

    Returns:
        Wrapped function that accepts full-dimensional arrays and
        applies vmap over non-core dimensions.

    Example:
        >>> # Function expects production with shape (C,)
        >>> # Actual data has shape (C, Y, X)
        >>> wrapped = wrap_with_vmap(
        ...     func=aging_flow_core,
        ...     input_dims={"production": ("C", "Y", "X"), "cohort_ages": ("C",)},
        ...     core_dims={"production": ["C"], "cohort_ages": ["C"]},
        ...     arg_order=["production", "cohort_ages", "rec_age"],
        ... )
        >>> # wrapped now accepts (C, Y, X) arrays and vmaps over Y, X
    """
    # Compute which dimensions to vmap over
    broadcast_dims = compute_broadcast_dims(input_dims, core_dims)

    if not broadcast_dims:
        # No broadcast dimensions, return original function
        return func

    # Compute in_axes for each vmap level
    # We iterate in canonical order (outer to inner) to compute axes correctly
    # as each dimension is "peeled" from the perspective of inner vmaps
    axes_list: list[tuple[int | None, ...]] = []
    current_dims = dict(input_dims)

    for dim in broadcast_dims:
        in_axes = compute_in_axes(current_dims, dim, arg_order)
        axes_list.append(in_axes)
        current_dims = remove_dim_from_inputs(current_dims, dim)

    # Apply vmaps in reverse order (innermost first, then wrap with outer)
    # This way the first element of axes_list corresponds to the outermost vmap
    # Note: vmap places the mapped dimension at position 0 by default.
    # The output shape will be (broadcast_dims..., core_dims...).
    vmapped = func
    for in_axes in reversed(axes_list):
        vmapped = jax.vmap(vmapped, in_axes=in_axes)

    return vmapped
