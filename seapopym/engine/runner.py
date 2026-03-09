"""Composable Runner — unified entry point for simulation and optimization.

Use factory methods to create instances for different use-cases::

    # Simulation (chunked, disk or memory output)
    runner = Runner.simulation(chunk_size=365)
    final_state, outputs = runner.run(model, output_path="/results/sim_001/")

    # Optimization (single-shot, callable interface)
    runner = Runner.optimization()
    outputs = runner(model, free_params)

    # Optimization with vmap + chunking (reduced memory)
    runner = Runner.optimization(vmap=True)
    outputs = runner(model, population_params)

Execution paths
===============

Two internal paths handle all use-cases via functional composition:

``_run_chunked``
    Python loop over temporal chunks, each processed by ``lax.scan``.
    Compatible with simulation and vmap (vmap wraps each chunk's scan,
    forcings shared in closure). NOT compatible with grad (Python loop
    is opaque to the JAX tracer).

``_run_full``
    Single ``lax.scan`` over all timesteps. All dynamic forcings loaded
    once via ``get_all_dynamic()``. Compatible with all transforms
    (simulation, vmap, grad).

JAX transforms (vmap, grad, checkpoint) are composed functionally::

    step_fn = build_step_fn(model)       # statics captured in closure
    # checkpoint: step_fn = jax.checkpoint(step_fn)
    # vmap:       jax.vmap(eval_one)(population_params)
    # grad:       jax.value_and_grad(eval_one)(params)

Compatibility matrix::

                        _run_chunked        _run_full
                        (Python loop)       (single scan)
    ──────────────────────────────────────────────────────
    simulation            ✅                  ✅
    vmap                  ✅                  ✅
    grad                  ❌                  ✅
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.lax as lax
import jax.numpy as jnp

from seapopym.types import Outputs, Params, State

from .exceptions import ChunkingError, EngineError
from .io import DiskWriter, MemoryWriter, resolve_var_dims
from .step import build_step_fn

if TYPE_CHECKING:
    from seapopym.compiler import CompiledModel
    from seapopym.engine.io import OutputWriter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunnerConfig:
    """Configuration for Runner execution strategy.

    Attributes:
        vmap_params: Whether to vmap over a population of parameter sets.
        chunk_size: Number of timesteps per chunk. None = no chunking.
            Chunking is compatible with vmap (vmap inside each chunk's scan).
        output_mode: Where to write outputs.
            ``"memory"`` — return xarray.Dataset (default for simulation).
            ``"disk"`` — write to Zarr.
            ``"raw"`` — return raw JAX arrays (default for optimization).
    """

    vmap_params: bool = False
    chunk_size: int | None = None
    output_mode: Literal["disk", "memory", "raw"] = "raw"

    def __post_init__(self) -> None:
        if self.vmap_params and self.output_mode != "raw":
            msg = "vmap_params=True requires output_mode='raw'"
            raise EngineError(msg)


def _scan(
    step_fn: Callable,
    init: tuple[State, dict[str, Any]],
    xs: Any,
    length: int,
) -> tuple[tuple[State, dict[str, Any]], dict[str, Any]]:
    """JIT-compiled lax.scan wrapper."""

    @jax.jit
    def _jitted(
        init_state: tuple[State, dict[str, Any]], inputs: Any
    ) -> tuple[tuple[State, dict[str, Any]], dict[str, Any]]:
        return lax.scan(step_fn, init_state, inputs, length=length)

    return _jitted(init, xs)


class Runner:
    """Unified runner for simulation and optimization.

    Use the factory methods to create pre-configured instances:

    - :meth:`simulation` — chunked production runs with disk/memory output.
    - :meth:`optimization` — single-shot runs returning raw JAX arrays.

    Example::

        # Simulation
        runner = Runner.simulation(chunk_size=365)
        final_state, dataset = runner.run(model)

        # Optimization
        runner = Runner.optimization()
        outputs = runner(model, {"growth_rate": jnp.array(0.5)})
    """

    def __init__(self, config: RunnerConfig) -> None:
        self.config = config

    # --- Presets ---

    @classmethod
    def simulation(
        cls,
        chunk_size: int | None = None,
        output: Literal["memory", "disk"] = "memory",
    ) -> Runner:
        """Create a runner configured for production simulation.

        Args:
            chunk_size: Timesteps per chunk. None = use model's chunk_size
                or process all at once.
            output: ``"memory"`` returns xarray.Dataset,
                ``"disk"`` writes Zarr (requires output_path in run()).
        """
        return cls(
            RunnerConfig(
                vmap_params=False,
                chunk_size=chunk_size,
                output_mode=output,
            )
        )

    @classmethod
    def optimization(cls, vmap: bool = False, chunk_size: int | None = None) -> Runner:
        """Create a runner configured for optimization/calibration.

        Args:
            vmap: Whether to vmap over a population of parameter sets.
            chunk_size: Number of timesteps per chunk. None = no chunking
                (all timesteps in a single scan). Chunking reduces memory
                for long time series and is compatible with vmap.
        """
        return cls(
            RunnerConfig(
                vmap_params=vmap,
                chunk_size=chunk_size,
                output_mode="raw",
            )
        )

    # --- Simulation interface ---

    def run(
        self,
        model: CompiledModel,
        output_path: str | Path | None = None,
        export_variables: list[str] | None = None,
    ) -> tuple[State, Any | None]:
        """Execute the full simulation with streaming output.

        Args:
            model: Compiled model to execute.
            output_path: Path to write outputs (disk mode). If None and
                output_mode is "memory", returns xarray.Dataset.
            export_variables: Variables to export. Defaults to state variables.

        Returns:
            Tuple of (final_state, outputs).
        """
        state = dict(model.state)
        params = dict(model.parameters)
        output_vars = export_variables if export_variables is not None else list(state.keys())
        step_fn = build_step_fn(model, export_variables=output_vars)
        writer = self._build_writer(model, output_path, output_vars)

        try:
            if self.config.chunk_size is not None:
                state, _ = self._run_chunked(model, step_fn, state, params, writer)
            else:
                state, outputs = self._run_full(model, step_fn, state, params)
                writer.append(outputs)
            final_results = writer.finalize()
        finally:
            writer.close()

        logger.info("Simulation complete")
        return state, final_results

    # --- Optimization interface ---

    def __call__(
        self,
        model: CompiledModel,
        free_params: Params,
        export_variables: list[str] | None = None,
    ) -> Outputs:
        """Run the model with merged parameters (optimization interface).

        Args:
            model: Compiled model.
            free_params: Free parameters to override in model.parameters.
            export_variables: If provided, only these variables are accumulated
                by ``lax.scan``. This reduces memory when only a subset (e.g.
                ``["biomass"]``) is needed for the objective function.

        Returns:
            Model outputs (dict of arrays).
        """
        step_fn = build_step_fn(model, export_variables=export_variables)

        def eval_one(single_free: Params) -> Outputs:
            merged = {**model.parameters, **single_free}
            state = dict(model.state)
            if self.config.chunk_size is not None:
                _, outputs = self._run_chunked(model, step_fn, state, merged, writer=None)
            else:
                _, outputs = self._run_full(model, step_fn, state, merged)
            return outputs

        if self.config.vmap_params:
            return jax.vmap(eval_one)(free_params)
        return eval_one(free_params)

    # --- Internal: simulation ---

    def _resolve_chunk_size(self, model: CompiledModel) -> int:
        """Resolve chunk_size from config or default to all timesteps."""
        if self.config.chunk_size is not None:
            return self.config.chunk_size
        return model.n_timesteps

    def _build_writer(
        self,
        model: CompiledModel,
        output_path: str | Path | None,
        output_vars: list[str],
    ) -> OutputWriter:
        """Build and initialize the appropriate output writer."""
        n_timesteps = model.n_timesteps
        writer_coords = dict(model.coords)
        if model.time_grid is not None:
            writer_coords["T"] = model.time_grid.coords[:n_timesteps]
        var_dims = resolve_var_dims(model.data_nodes, output_vars)

        writer: OutputWriter
        writer = (
            DiskWriter(output_path)
            if output_path is not None
            else MemoryWriter()
        )
        writer.initialize(model.shapes, output_vars, coords=writer_coords, var_dims=var_dims)
        return writer

    def _run_chunked(
        self,
        model: CompiledModel,
        step_fn: Callable,
        state: State,
        params: Params,
        writer: OutputWriter | None,
    ) -> tuple[State, Outputs | None]:
        """Run with temporal chunking (Python loop over chunks).

        Args:
            model: Compiled model.
            step_fn: Step function from build_step_fn.
            state: Initial state.
            params: Parameters.
            writer: Output writer (None for raw mode).

        Returns:
            Tuple of (final_state, last_chunk_outputs or None).
        """
        n_timesteps = model.n_timesteps
        chunk_size = self._resolve_chunk_size(model)

        if chunk_size <= 0:
            raise ChunkingError(chunk_size, n_timesteps, "chunk_size must be positive")

        n_chunks = math.ceil(n_timesteps / chunk_size)
        logger.info(
            f"Starting chunked run: {n_timesteps} steps in {n_chunks} chunks "
            f"(chunk_size={chunk_size})"
        )

        collected: list[Outputs] = []
        for chunk_idx in range(n_chunks):
            start_t = chunk_idx * chunk_size
            end_t = min(start_t + chunk_size, n_timesteps)
            chunk_len = end_t - start_t

            logger.debug(f"Processing chunk {chunk_idx + 1}/{n_chunks} (t={start_t}-{end_t})")

            forcings_chunk = model.forcings.get_chunk(start_t, end_t)
            (state, params), chunk_outputs = _scan(step_fn, (state, params), forcings_chunk, chunk_len)

            if writer is not None:
                writer.append(chunk_outputs)
            else:
                collected.append(chunk_outputs)

        if collected:
            all_outputs = jax.tree.map(lambda *arrays: jnp.concatenate(arrays, axis=0), *collected)
            return state, all_outputs
        return state, None

    # --- Internal: full (no chunking) ---

    def _run_full(
        self,
        model: CompiledModel,
        step_fn: Callable,
        state: State,
        params: Params,
    ) -> tuple[State, Outputs]:
        """Run full scan without chunking.

        All dynamic forcings are loaded once via get_all_dynamic().

        Args:
            model: Compiled model.
            step_fn: Step function from build_step_fn.
            state: Initial state.
            params: Parameters.

        Returns:
            Tuple of (final_state, outputs).
        """
        forcings = model.forcings.get_all_dynamic()
        (state, params), outputs = lax.scan(step_fn, (state, params), forcings)
        return state, outputs
