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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .exceptions import ChunkingError
from .io import DiskWriter, MemoryWriter, OutputWriter
from .step import build_step_fn

if TYPE_CHECKING:
    from seapopym.compiler import CompiledModel

logger = logging.getLogger(__name__)

from seapopym.types import State


class StreamingRunner:
    """Runner for production simulations with chunking and disk I/O.

    Processes the simulation in temporal chunks, writing each chunk
    to disk while computing the next one.

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
        # Priority: chunk_size (parameter) > model.chunk_size (config) > model.n_timesteps (all)
        self.chunk_size = chunk_size or getattr(model, "chunk_size", None) or model.n_timesteps

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

        # Calculate number of chunks
        n_full_chunks = n_timesteps // self.chunk_size
        remainder = n_timesteps % self.chunk_size
        n_chunks = n_full_chunks + (1 if remainder > 0 else 0)

        logger.info(f"Starting simulation: {n_timesteps} steps in {n_chunks} chunks (chunk_size={self.chunk_size})")

        # Build step function
        step_fn = build_step_fn(self.model)

        # Initialize carry: (state, params)
        state = dict(self.model.state)  # Copy to avoid modifying original
        params = dict(self.model.parameters)

        # Determine variables to export
        output_vars = export_variables if export_variables is not None else list(state.keys())

        # Prepare coordinates for writer (with real timestamps from time_grid)
        writer_coords = dict(self.model.coords)
        if self.model.time_grid is not None:
            # Use the full time coordinates from time_grid
            # (sliced to n_timesteps in case of any discrepancy)
            writer_coords["T"] = self.model.time_grid.coords[:n_timesteps]

        # Resolve per-variable dims from data_nodes
        var_dims: dict[str, tuple[str, ...]] = {}
        for node in self.model.data_nodes.values():
            if node.dims is None:
                continue
            short = node.name.split(".")[-1] if "." in node.name else node.name
            if node.name in output_vars:
                var_dims[node.name] = tuple(node.dims)
            elif short in output_vars:
                var_dims[short] = tuple(node.dims)

        # Initialize Writer Strategy
        writer: OutputWriter
        writer = (
            DiskWriter(output_path)
            if output_path is not None
            else MemoryWriter(self.model)
        )

        try:
            writer.initialize(self.model.shapes, output_vars, coords=writer_coords, var_dims=var_dims)

            # Process chunks
            for chunk_idx in range(n_chunks):
                start_t = chunk_idx * self.chunk_size
                end_t = min(start_t + self.chunk_size, n_timesteps)
                chunk_len = end_t - start_t

                logger.debug(f"Processing chunk {chunk_idx + 1}/{n_chunks} (t={start_t}-{end_t})")

                # Extract forcings for this chunk
                forcings_chunk = self.model.forcings.get_chunk(start_t, end_t)

                # Run scan on chunk
                (state, params), outputs = self._scan(
                    step_fn, (state, params), forcings_chunk, chunk_len
                )

                # Append outputs to writer (it will filter what it needs)
                # Note: 'outputs' from JAX contains state + derived variables
                writer.append(outputs, chunk_idx)

            # Finalize and get results
            final_results = writer.finalize()

        finally:
            # Ensure resources are released
            if hasattr(writer, "close"):
                writer.close()

        logger.info("Simulation complete")
        return state, final_results

    @staticmethod
    def _scan(step_fn, init, xs, length):
        """JIT-compiled lax.scan wrapper."""
        import jax
        import jax.lax as lax

        @jax.jit
        def _jitted(init_state, inputs):
            return lax.scan(step_fn, init_state, inputs, length=length)

        return _jitted(init, xs)
