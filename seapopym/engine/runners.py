"""Runner implementations for model execution.

Provides two runners:
- StreamingRunner: Production mode with chunking and async I/O
- GradientRunner: Optimization mode with full scan (JAX only)

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

from .backends import get_backend
from .exceptions import BackendError, ChunkingError
from .io import AsyncWriter
from .step import build_step_fn

if TYPE_CHECKING:
    from seapopym.compiler import CompiledModel

logger = logging.getLogger(__name__)

# Type aliases
Array = Any  # np.ndarray | jax.Array
State = dict[str, Array]
Outputs = dict[str, Array]


class StreamingRunner:
    """Runner for production simulations with chunking and async I/O.

    Processes the simulation in temporal chunks, writing each chunk
    to disk asynchronously while computing the next one.

    Example:
        >>> model = compile_model(blueprint, config)
        >>> runner = StreamingRunner(model, chunk_size=365)
        >>> runner.run("/results/sim_001/")
    """

    def __init__(
        self,
        model: CompiledModel,
        chunk_size: int | None = None,
        io_workers: int = 2,
    ) -> None:
        """Initialize streaming runner.

        Args:
            model: Compiled model to execute.
            chunk_size: Number of timesteps per chunk. If None, uses model.batch_size.
                If both None, processes entire simulation in one batch.
            io_workers: Number of async I/O workers.
        """
        self.model = model
        # Priority: chunk_size (parameter) > model.batch_size (config) > model.n_timesteps (all)
        self.chunk_size = chunk_size or getattr(model, "batch_size", None) or model.n_timesteps
        self.io_workers = io_workers
        self.backend = get_backend(model.backend)

    def run(self, output_path: str | Path | None = None) -> tuple[State, Outputs | None]:
        """Execute the full simulation with streaming output.

        Args:
            output_path: Path to write outputs. If None, outputs are returned in memory.

        Returns:
            Tuple of (final_state, outputs). Outputs is None if output_path is provided.
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

        # Initialize state
        state = dict(self.model.state)  # Copy to avoid modifying original

        # Get output variable names
        output_vars = list(state.keys())

        # Prepare for execution
        accumulated_outputs: dict[str, list[Array]] | None = {} if output_path is None else None

        # Setup context manager for disk usage (dummy for in-memory)
        from contextlib import nullcontext

        writer_ctx = (
            AsyncWriter(output_path, max_workers=self.io_workers)  # type: ignore
            if output_path is not None
            else nullcontext()
        )

        with writer_ctx as writer:
            if writer is not None:
                writer.initialize(self.model.shapes, output_vars)

            # Process chunks
            for chunk_idx in range(n_chunks):
                start_t = chunk_idx * self.chunk_size
                end_t = min(start_t + self.chunk_size, n_timesteps)
                chunk_len = end_t - start_t

                logger.debug(f"Processing chunk {chunk_idx + 1}/{n_chunks} (t={start_t}-{end_t})")

                # Extract forcings for this chunk
                forcings_chunk = self._slice_forcings(start_t, end_t)

                # Run scan on chunk
                state, outputs = self.backend.scan(
                    step_fn=step_fn,
                    init=state,
                    xs=forcings_chunk,
                    length=chunk_len,
                )

                if writer is not None:
                    # Write outputs asynchronously
                    writer.write_async(outputs, chunk_idx)
                else:
                    # Accumulate in memory
                    if accumulated_outputs is None:
                        # Should not happen if writer is None (see logic above)
                        accumulated_outputs = {}

                    for k, v in outputs.items():
                        if k not in accumulated_outputs:
                            accumulated_outputs[k] = []
                        accumulated_outputs[k].append(v)

            if writer is not None:
                # Ensure all writes complete
                writer.flush()

        # Finalize in-memory outputs
        final_outputs: Outputs | None = None
        if accumulated_outputs is not None and accumulated_outputs:
            import numpy as np

            if self.model.backend == "jax":
                import jax.numpy as jnp

                concat = jnp.concatenate
            else:
                concat = np.concatenate

            final_outputs = {k: concat(v, axis=0) for k, v in accumulated_outputs.items()}

        logger.info("Simulation complete")
        return state, final_outputs

    def _slice_forcings(self, start: int, end: int) -> dict[str, Array]:
        """Slice forcings for a temporal chunk.

        Args:
            start: Start timestep (inclusive).
            end: End timestep (exclusive).

        Returns:
            Dict of sliced forcing arrays.
        """
        import numpy as np

        forcings = self.model.forcings
        n_timesteps = self.model.n_timesteps
        chunk_len = end - start
        sliced = {}

        for name, arr in forcings.items():
            arr_np = np.asarray(arr)
            # Check if array has time dimension (shape[0] matches total timesteps)
            if arr_np.ndim > 0 and arr_np.shape[0] == n_timesteps:
                sliced[name] = arr_np[start:end]
            else:
                # Static forcing (e.g., mask) - broadcast to chunk length
                # Stack it chunk_len times along a new first axis
                sliced[name] = np.broadcast_to(arr_np, (chunk_len,) + arr_np.shape)

        return sliced


class GradientRunner:
    """Runner for optimization with full gradient support.

    Executes the entire simulation in a single scan, preserving
    the gradient chain for automatic differentiation.

    Note: JAX backend only. Memory usage scales with total timesteps.

    Example:
        >>> model = compile_model(blueprint, config, backend="jax")
        >>> runner = GradientRunner(model)
        >>> final_state, outputs = runner.run()
        >>>
        >>> # For optimization
        >>> loss, grads = jax.value_and_grad(runner.loss_fn)(params)
    """

    def __init__(self, model: CompiledModel) -> None:
        """Initialize gradient runner.

        Args:
            model: Compiled model to execute (must use JAX backend).
        """
        if model.backend != "jax":
            raise BackendError(
                model.backend,
                "GradientRunner requires JAX backend for autodiff support",
            )

        self.model = model
        self.backend = get_backend("jax")
        self._step_fn = build_step_fn(model)

    def run(self) -> tuple[State, Outputs]:
        """Execute the full simulation in a single scan.

        Returns:
            Tuple of (final_state, all_outputs).
        """
        state = dict(self.model.state)
        forcings = self._prepare_forcings()

        final_state, outputs = self.backend.scan(
            step_fn=self._step_fn,
            init=state,
            xs=forcings,
        )

        return final_state, outputs

    def _prepare_forcings(self) -> dict[str, Array]:
        """Prepare forcings for scan (broadcast static forcings).

        Returns:
            Dict of forcings with consistent time dimension.
        """
        import jax.numpy as jnp

        forcings = self.model.forcings
        n_timesteps = self.model.n_timesteps
        prepared = {}

        for name, arr in forcings.items():
            # Check if array has time dimension
            if arr.ndim > 0 and arr.shape[0] == n_timesteps:
                prepared[name] = arr
            else:
                # Static forcing - broadcast to all timesteps
                prepared[name] = jnp.broadcast_to(arr, (n_timesteps,) + arr.shape)

        return prepared

    def run_with_params(
        self,
        params: dict[str, Array],
    ) -> tuple[State, Outputs]:
        """Execute simulation with updated parameters.

        This creates a new step function with the given parameters,
        useful for optimization loops.

        Args:
            params: Dict of parameter values to use.

        Returns:
            Tuple of (final_state, all_outputs).
        """
        # Update model parameters (creates a modified view)
        model_params = dict(self.model.parameters)
        model_params.update(params)

        # We need to rebuild the step function with new params
        # For now, just update in-place (not ideal for JIT)
        # TODO: Better parameter injection mechanism
        original_params = self.model.parameters
        self.model.parameters.update(params)

        try:
            result = self.run()
        finally:
            # Restore original parameters
            self.model.parameters.clear()
            self.model.parameters.update(original_params)

        return result

    def loss_fn(
        self,
        params: dict[str, Array],
        observations: dict[str, Array],
        loss_weights: dict[str, float] | None = None,
    ) -> Array:
        """Compute loss for optimization.

        Args:
            params: Current parameter values.
            observations: Dict of observation arrays keyed by variable name.
            loss_weights: Optional weights per variable.

        Returns:
            Scalar loss value.
        """
        import jax.numpy as jnp

        final_state, outputs = self.run_with_params(params)

        # Compute MSE loss for each observed variable
        total_loss = jnp.array(0.0)
        weights = loss_weights or {}

        for var_name, obs in observations.items():
            # Try to find corresponding output
            if var_name in outputs:
                pred = outputs[var_name]
            elif var_name in final_state:
                pred = final_state[var_name]
            else:
                continue

            # Get mask if available
            mask = self.model.forcings.get("mask", 1.0)

            # Compute masked MSE
            diff = (pred - obs) ** 2
            if not isinstance(mask, float):
                diff = diff * mask

            weight = weights.get(var_name, 1.0)
            total_loss = total_loss + weight * jnp.mean(diff)

        return total_loss
