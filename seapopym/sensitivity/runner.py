"""Batched model runner for Sobol sensitivity analysis.

Evaluates the model for a batch of parameter sets simultaneously
using jax.vmap over the parameter axis, with temporal chunking
for memory management and point extraction for efficiency.

Memory layout per time chunk:
    GPU: (batch_size, T_chunk, Y, X) — full spatial grid
    Accumulated: (batch_size, T_chunk, n_points) — extracted points only
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import jax
import jax.lax as lax
import jax.numpy as jnp

from seapopym.engine.step import build_step_fn

if TYPE_CHECKING:
    from seapopym.compiler import CompiledModel

logger = logging.getLogger(__name__)

Array = Any  # jax.Array | np.ndarray


class SobolRunner:
    """Runs batched simulations with temporal chunking and point extraction.

    Uses jax.vmap to parallelize model evaluations across parameter sets,
    and temporal chunking to control GPU memory usage. At each chunk,
    only the values at extraction points are kept.

    Args:
        model: Compiled SeapoPym model.
        extraction_points: List of (y, x) index tuples for point extraction.
        output_variable: Name of the output variable to extract (e.g., "biomass").
        chunk_size: Number of timesteps per temporal chunk.
    """

    def __init__(
        self,
        model: CompiledModel,
        extraction_points: list[tuple[int, int]],
        output_variable: str,
        chunk_size: int,
    ) -> None:
        self.model = model
        self.extraction_points = extraction_points
        self.output_variable = output_variable
        self.chunk_size = chunk_size

        # Pre-compute extraction indices as JAX arrays
        self._y_indices = jnp.array([p[0] for p in extraction_points])
        self._x_indices = jnp.array([p[1] for p in extraction_points])

        # Build step function with params externalized
        self._step_fn = build_step_fn(model, params_as_argument=True)

        # Cache of JIT-compiled vmapped scans, keyed by chunk length.
        # Typically 1 entry (chunk_size), or 2 if remainder != 0.
        self._scan_cache: dict[int, Any] = {}

    def run_batch(self, params_batch: dict[str, Array], batch_size: int) -> Array:
        """Run simulation for a batch of parameter sets.

        Processes the full time series in temporal chunks, extracting
        point values at each chunk to minimize memory usage.

        Each distinct chunk length triggers one JIT compilation (cached).
        Typically at most 2 compilations: one for chunk_size and one for
        the remainder.

        Args:
            params_batch: Dict of parameter arrays, each with shape (batch_size,).
            batch_size: Expected batch size (for padding if needed).

        Returns:
            Time series at extraction points, shape (batch_size, T_total, n_points).
        """
        n_timesteps = self.model.n_timesteps

        # Compute chunks
        n_full_chunks = n_timesteps // self.chunk_size
        remainder = n_timesteps % self.chunk_size
        chunks = []
        for i in range(n_full_chunks):
            chunks.append((i * self.chunk_size, (i + 1) * self.chunk_size))
        if remainder > 0:
            chunks.append((n_full_chunks * self.chunk_size, n_timesteps))

        # Initialize batched state: replicate initial state for each param set
        state_batch = {k: jnp.broadcast_to(v, (batch_size,) + v.shape) for k, v in self.model.state.items()}

        # Accumulate extracted time series
        extracted_chunks: list[Array] = []

        for chunk_idx, (start, end) in enumerate(chunks):
            chunk_len = end - start
            logger.debug(f"Processing chunk {chunk_idx + 1}/{len(chunks)} (t={start}-{end})")

            # Slice forcings for this chunk (exact length, no padding)
            forcings_chunk = self.model.forcings.get_chunk(start, end)

            # Run vmapped scan with the exact chunk length
            state_batch, outputs_batch = self._run_vmapped_scan(
                state_batch, params_batch, forcings_chunk, chunk_len
            )

            # Extract points from output variable
            var_output = outputs_batch[self.output_variable]
            extracted = self._extract_points(var_output)
            extracted_chunks.append(extracted)

        # Concatenate along time axis: (batch_size, T_total, n_points)
        return jnp.concatenate(extracted_chunks, axis=1)

    def _run_vmapped_scan(
        self,
        state_batch: dict[str, Array],
        params_batch: dict[str, Array],
        forcings_chunk: dict[str, Array],
        chunk_len: int,
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        """Run vmapped lax.scan over a temporal chunk.

        Each distinct chunk_len triggers one JIT compilation, cached in
        self._scan_cache. Typically at most 2 entries: chunk_size and remainder.

        Args:
            state_batch: Batched state, each array has leading batch dim.
            params_batch: Batched parameters, each array has shape (batch_size,).
            forcings_chunk: Forcings for this chunk (exact length).
            chunk_len: Number of timesteps in this chunk.

        Returns:
            Tuple of (new_state_batch, outputs_batch).
        """
        if chunk_len not in self._scan_cache:
            step_fn = self._step_fn
            scan_length = chunk_len

            def single_run(state, params, forcings):
                """Run scan for a single parameter set."""
                carry = (state, params)
                (final_state, _), outputs = lax.scan(step_fn, carry, forcings, length=scan_length)
                return final_state, outputs

            def vmapped_scan(state_batch, params_batch, forcings):
                return jax.vmap(single_run, in_axes=(0, 0, None))(state_batch, params_batch, forcings)

            self._scan_cache[chunk_len] = jax.jit(vmapped_scan)

        return self._scan_cache[chunk_len](state_batch, params_batch, forcings_chunk)

    def _extract_points(self, output: Array) -> Array:
        """Extract values at specified grid points.

        Args:
            output: Array with spatial dims, shape (batch_size, T_chunk, ..., Y, X)
                    or (batch_size, T_chunk) for 0D models.

        Returns:
            Extracted values, shape (batch_size, T_chunk, n_points).
        """
        if output.ndim == 2:
            # 0D model: no spatial dims, treat single value as one point
            return output[:, :, None]

        if output.ndim == 3:
            # (batch, T, X) — 1D model
            return output[:, :, self._x_indices]

        # (batch, T, Y, X) — 2D model
        return output[:, :, self._y_indices, self._x_indices]

