"""Runner implementations for model execution.

Provides:
- StreamingRunner: Production mode with chunking and disk I/O

Note on forcings:
    Forcings can be dynamic (with time dimension) or static (spatial-only).
    Static forcings (e.g., mask, bathymetry) are automatically broadcast
    over time during execution. This supports both spatial models and
    box models without additional configuration.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.lax as lax

from seapopym.types import State

from .exceptions import ChunkingError
from .io import DiskWriter, MemoryWriter, resolve_var_dims
from .step import build_step_fn

if TYPE_CHECKING:
    from seapopym.compiler import CompiledModel
    from seapopym.engine.io import OutputWriter

logger = logging.getLogger(__name__)


def _scan(
    step_fn: Callable,
    init: tuple[State, dict[str, Any]],
    xs: Any,
    length: int,
) -> tuple[tuple[State, dict[str, Any]], dict[str, Any]]:
    """JIT-compiled lax.scan wrapper."""

    @jax.jit
    def _jitted(init_state: tuple[State, dict[str, Any]], inputs: Any) -> tuple[tuple[State, dict[str, Any]], dict[str, Any]]:
        return lax.scan(step_fn, init_state, inputs, length=length)

    return _jitted(init, xs)


class StreamingRunner:
    """Runner for production simulations with chunking and disk I/O.

    Processes the simulation in temporal chunks, writing each chunk
    to disk.

    Example:
        >>> model = compile_model(blueprint, config)
        >>> runner = StreamingRunner(model, chunk_size=365)
        >>> runner.run("/results/sim_001/")
    """

    def __init__(
        self,
        model: CompiledModel,
        chunk_size: int | None = None,
    ) -> None:
        """Initialize streaming runner.

        Args:
            model: Compiled model to execute.
            chunk_size: Number of timesteps per chunk. If None, uses model.chunk_size.
                If both None, processes entire simulation in one chunk.
        """
        self.model = model
        self.chunk_size: int
        # Priority: chunk_size (parameter) > model.chunk_size (config) > model.n_timesteps (all)
        if chunk_size is not None:
            self.chunk_size = chunk_size
        elif getattr(model, "chunk_size", None) is not None:
            self.chunk_size = cast(int, model.chunk_size)
        else:
            self.chunk_size = model.n_timesteps

    def _build_writer(
        self,
        output_path: str | Path | None,
        output_vars: list[str],
    ) -> OutputWriter:
        """Build and initialize the appropriate output writer.

        Args:
            output_path: Path for disk output, or None for in-memory.
            output_vars: Variables to export.

        Returns:
            Initialized OutputWriter.
        """
        n_timesteps = self.model.n_timesteps

        # Prepare coordinates for writer (with real timestamps from time_grid)
        writer_coords = dict(self.model.coords)
        if self.model.time_grid is not None:
            writer_coords["T"] = self.model.time_grid.coords[:n_timesteps]

        # Resolve per-variable dims from data_nodes
        var_dims = resolve_var_dims(self.model.data_nodes, output_vars)

        # Initialize Writer Strategy
        writer: OutputWriter
        writer = (
            DiskWriter(output_path)
            if output_path is not None
            else MemoryWriter(self.model)
        )
        writer.initialize(self.model.shapes, output_vars, coords=writer_coords, var_dims=var_dims)
        return writer

    def run(
        self,
        output_path: str | Path | None = None,
        export_variables: list[str] | None = None,
    ) -> tuple[State, Any | None]:
        """Execute the full simulation with streaming output.

        Args:
            output_path: Path to write outputs. If None, outputs are returned in memory as xarray.Dataset.
            export_variables: List of variables to export/keep. If None, defaults to state variables.

        Returns:
            Tuple of (final_state, outputs).
            - If output_path is set, outputs is None (data written to disk).
            - If output_path is None, outputs is an xarray.Dataset containing requested variables.
        """
        n_timesteps = self.model.n_timesteps

        # Validate chunking
        if self.chunk_size <= 0:
            raise ChunkingError(self.chunk_size, n_timesteps, "chunk_size must be positive")

        n_chunks = math.ceil(n_timesteps / self.chunk_size)

        logger.info(f"Starting simulation: {n_timesteps} steps in {n_chunks} chunks (chunk_size={self.chunk_size})")

        # Build step function
        step_fn = build_step_fn(self.model)

        # Initialize carry: (state, params)
        state = dict(self.model.state)  # Copy to avoid modifying original
        params = dict(self.model.parameters)

        # Determine variables to export
        output_vars = export_variables if export_variables is not None else list(state.keys())

        writer = self._build_writer(output_path, output_vars)

        try:
            # Process chunks
            for chunk_idx in range(n_chunks):
                start_t = chunk_idx * self.chunk_size
                end_t = min(start_t + self.chunk_size, n_timesteps)
                chunk_len = end_t - start_t

                logger.debug(f"Processing chunk {chunk_idx + 1}/{n_chunks} (t={start_t}-{end_t})")

                # Extract forcings for this chunk
                forcings_chunk = self.model.forcings.get_chunk(start_t, end_t)

                # Run scan on chunk
                (state, params), outputs = _scan(
                    step_fn, (state, params), forcings_chunk, chunk_len
                )

                # Append outputs to writer (it will filter what it needs)
                writer.append(outputs)

            # Finalize and get results
            final_results = writer.finalize()

        finally:
            writer.close()

        logger.info("Simulation complete")
        return state, final_results
