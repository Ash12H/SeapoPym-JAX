"""Functional execution engine for compiled models.

Provides two entry points:

``run(step_fn, model, state, params, ...)``
    Pure execution engine — one loop of chunks, each processed by ``lax.scan``,
    outputs delegated to a pluggable writer.

``simulate(model, ...)``
    Convenience wrapper — builds step_fn and writer, manages writer lifecycle,
    calls ``run()``.

Example::

    # Low-level (optimization, custom transforms)
    step_fn = build_step_fn(model, export_variables=["biomass"])
    state, outputs = run(step_fn, model, dict(model.state), params)

    # High-level (simulation)
    state, dataset = simulate(model, chunk_size=365, output_path="/results/")
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax.lax as lax

from seapopym.types import Outputs, Params, State

from .exceptions import ChunkingError
from .io import WriterRaw, build_writer
from .step import build_step_fn

if TYPE_CHECKING:
    from seapopym.compiler import CompiledModel
    from seapopym.engine.io import OutputWriter

logger = logging.getLogger(__name__)


def chunk_ranges(n_timesteps: int, chunk_size: int) -> Iterator[tuple[int, int]]:
    """Yield (start, end) pairs for temporal chunks.

    Args:
        n_timesteps: Total number of timesteps.
        chunk_size: Number of timesteps per chunk.

    Yields:
        Tuples of (start_index, end_index).
    """
    for start in range(0, n_timesteps, chunk_size):
        yield start, min(start + chunk_size, n_timesteps)


def run(
    step_fn: Callable,
    model: CompiledModel,
    state: State,
    params: Params,
    chunk_size: int | None = None,
    writer: OutputWriter | None = None,
) -> tuple[State, Any]:
    """Execute the model with a pluggable writer.

    Pure execution engine: loops over temporal chunks, runs ``lax.scan``
    for each chunk, and delegates output accumulation to the writer.

    This function never applies ``jax.vmap`` — that is the caller's
    responsibility (e.g. the optimizer wraps ``run`` in ``jax.vmap``).

    Args:
        step_fn: Step function from ``build_step_fn``.
        model: Compiled model (provides forcings and n_timesteps).
        state: Initial state dict.
        params: Parameters dict.
        chunk_size: Timesteps per chunk. ``None`` = all timesteps in one chunk.
        writer: Output writer. ``None`` = ``WriterRaw()`` (JAX-traceable).

    Returns:
        Tuple of (final_state, writer.finalize() result).
    """
    if writer is None:
        writer = WriterRaw()

    n_timesteps = model.n_timesteps
    effective_chunk = chunk_size if chunk_size is not None else n_timesteps

    if effective_chunk <= 0:
        raise ChunkingError(effective_chunk, n_timesteps, "chunk_size must be positive")

    n_chunks = math.ceil(n_timesteps / effective_chunk)
    logger.info(
        "Starting run: %d steps in %d chunk(s) (chunk_size=%d)",
        n_timesteps, n_chunks, effective_chunk,
    )

    for start, end in chunk_ranges(n_timesteps, effective_chunk):
        chunk_len = end - start
        forcings = model.forcings.get_chunk(start, end)
        (state, params), outputs = lax.scan(step_fn, (state, params), forcings, length=chunk_len)
        writer.append(outputs)

    return state, writer.finalize()


def simulate(
    model: CompiledModel,
    chunk_size: int | None = None,
    output_path: str | Path | None = None,
    export_variables: list[str] | None = None,
) -> tuple[State, Any]:
    """Run a full simulation with automatic writer management.

    Convenience wrapper that builds the step function and writer, manages
    the writer lifecycle (initialize/close), and calls :func:`run`.

    Args:
        model: Compiled model to execute.
        chunk_size: Timesteps per chunk. ``None`` = all timesteps at once.
        output_path: Path for Zarr output (DiskWriter). ``None`` = in-memory
            (MemoryWriter returning xarray.Dataset).
        export_variables: Variables to export. Defaults to all state variables.

    Returns:
        Tuple of (final_state, outputs). Outputs type depends on the writer:
        ``xr.Dataset`` for MemoryWriter, ``None`` for DiskWriter.
    """
    output_vars = export_variables if export_variables is not None else list(model.state.keys())
    step_fn = build_step_fn(model, export_variables=output_vars)
    writer = build_writer(model, output_path, output_vars)

    try:
        result = run(step_fn, model, dict(model.state), dict(model.parameters),
                     chunk_size=chunk_size, writer=writer)
    finally:
        writer.close()

    return result
