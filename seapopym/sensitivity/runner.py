"""Batched model runner for Sobol sensitivity analysis.

Evaluates the model for a batch of parameter sets simultaneously
using jax.vmap over the parameter axis, with temporal chunking
for memory management and point extraction for efficiency.

Memory optimization: extraction happens INSIDE the lax.scan body,
so only (batch_size, T_chunk, n_points) is accumulated instead of
the full spatial grids (batch_size, T_chunk, C, Y, X).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import jax
import jax.lax as lax
import jax.numpy as jnp

from seapopym.engine.step import build_step_fn
from seapopym.types import Array

if TYPE_CHECKING:
    from seapopym.compiler import CompiledModel

logger = logging.getLogger(__name__)


class SobolRunner:
    """Runs batched simulations with temporal chunking and point extraction.

    Uses jax.vmap to parallelize model evaluations across parameter sets,
    and temporal chunking to control device memory usage. Point extraction
    is performed inside the scan body so that lax.scan only accumulates
    a small (n_points,) vector per timestep instead of full spatial grids.

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

        # Pre-compute extraction indices as JAX integer arrays
        self._y_indices = jnp.array([p[0] for p in extraction_points], dtype=jnp.int32)
        self._x_indices = jnp.array([p[1] for p in extraction_points], dtype=jnp.int32)

        # Build step function with params externalized
        self._step_fn = build_step_fn(model)

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

            # Run vmapped scan — returns extracted points directly
            state_batch, extracted = self._run_vmapped_scan(
                state_batch, params_batch, forcings_chunk, chunk_len
            )

            # extracted is already (batch_size, chunk_len, n_points)
            extracted_chunks.append(extracted)

        # Concatenate along time axis: (batch_size, T_total, n_points)
        return jnp.concatenate(extracted_chunks, axis=1)

    def _run_vmapped_scan(
        self,
        state_batch: dict[str, Array],
        params_batch: dict[str, Array],
        forcings_chunk: dict[str, Array],
        chunk_len: int,
    ) -> tuple[dict[str, Array], Array]:
        """Run vmapped lax.scan over a temporal chunk with in-scan extraction.

        Point extraction happens inside the scan body: at each timestep,
        only the output variable values at extraction points are returned.
        This prevents lax.scan from accumulating full spatial grids
        (especially the large production tensor).

        Each distinct chunk_len triggers one JIT compilation, cached in
        self._scan_cache. Typically at most 2 entries: chunk_size and remainder.

        Args:
            state_batch: Batched state, each array has leading batch dim.
            params_batch: Batched parameters, each array has shape (batch_size,).
            forcings_chunk: Forcings for this chunk (exact length).
            chunk_len: Number of timesteps in this chunk.

        Returns:
            Tuple of (new_state_batch, extracted_points).
            extracted_points has shape (batch_size, chunk_len, n_points).
        """
        if chunk_len not in self._scan_cache:
            step_fn = self._step_fn
            scan_length = chunk_len
            output_var = self.output_variable
            y_idx = self._y_indices
            x_idx = self._x_indices
            n_spatial_dims = len(self.model.state[output_var].shape)

            def _extract_from_state(var):
                """Extract points from a state variable (handles 0D/1D/2D)."""
                if n_spatial_dims == 0:
                    return var[None]  # scalar → (1,)
                if n_spatial_dims == 1:
                    return var[x_idx]  # (X,) → (n_points,)
                return var[y_idx, x_idx]  # (Y, X) → (n_points,)

            def single_run(state, params, forcings):
                """Run scan for a single parameter set with in-scan extraction."""
                carry = (state, params)

                def step_extract(carry, forcings_t):
                    new_carry, _outputs = step_fn(carry, forcings_t)
                    extracted = _extract_from_state(new_carry[0][output_var])
                    return new_carry, extracted

                (final_state, _), extracted = lax.scan(
                    step_extract, carry, forcings, length=scan_length
                )
                return final_state, extracted  # extracted: (chunk, n_points)

            def vmapped_scan(state_batch, params_batch, forcings):
                return jax.vmap(single_run, in_axes=(0, 0, None))(state_batch, params_batch, forcings)

            self._scan_cache[chunk_len] = jax.jit(vmapped_scan)

        return self._scan_cache[chunk_len](state_batch, params_batch, forcings_chunk)
