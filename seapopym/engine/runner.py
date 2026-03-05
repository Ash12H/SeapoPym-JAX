"""Composable Runner — unified entry point for simulation and optimization.

Use factory methods to create instances for different use-cases::

    # Simulation (chunked, disk or memory output)
    runner = Runner.simulation(chunk_size=365)
    final_state, outputs = runner.run(model, output_path="/results/sim_001/")

    # Optimization (single-shot, no chunking, callable interface)
    runner = Runner.optimization()
    outputs = runner(model, free_params)

Design Notes — Why optimization cannot use time chunking
========================================================

Simulation mode splits the time axis into chunks processed by successive
``lax.scan`` calls inside a **plain Python loop**.  This works because
nothing needs to trace through the loop boundary.

Optimization mode must keep the **entire time axis inside a single
``lax.scan``** call.  Two independent JAX transforms require this:

1. **``jax.value_and_grad``** (gradient-based optimizers like Adam) —
   Automatic differentiation traces the computation graph through
   ``lax.scan``.  A Python ``for`` loop is opaque to the JAX tracer, so
   splitting the scan into chunks would break gradient computation.

2. **``jax.vmap``** (evolutionary optimizers like CMA-ES / GA) —
   Population evaluation vectorises ``eval_one`` over all individuals
   with ``jax.vmap``.  Inside ``eval_one``, the model runs as a single
   ``lax.scan``.  ``vmap`` can vectorise JAX primitives (including
   ``lax.scan``) but **cannot** vectorise a Python ``for`` loop.  This
   ``vmap`` is what enables parallel evaluation of the whole population
   on GPU/TPU in one kernel launch — removing it would serialise
   individuals and drastically hurt performance.

In summary::

    Simulation:   Python for-loop  →  lax.scan per chunk   ✅ works
    Grad optim:   jax.value_and_grad  →  lax.scan           ✅ works
                  jax.value_and_grad  →  Python for-loop    ❌ breaks grad
    Evol optim:   jax.vmap  →  lax.scan                     ✅ works
                  jax.vmap  →  Python for-loop              ❌ breaks vmap

Memory considerations for long time series
-------------------------------------------

Two things consume GPU memory during optimization:

* **Intermediate states (carry)** — stored at every timestep by
  ``lax.scan`` for the backward pass.  ``jax.checkpoint`` can trade
  compute for memory by recomputing them instead of storing them.
  However, this only helps gradient-based optimizers (which have a
  backward pass); evolutionary optimizers do not benefit.

* **Forcing data** — loaded in full via ``model.forcings.get_all()``
  before ``lax.scan`` starts.  ``jax.checkpoint`` does **not** reduce
  this cost.  Only chunking (``get_chunk(start, end)``) avoids loading
  all forcings at once, but chunking is incompatible with ``vmap`` and
  ``grad`` as explained above.

For very long time series, the forcing data is likely the dominant memory
bottleneck, not the intermediate states.
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
        param_mode: How parameters flow through the scan.
            ``"carry"`` — parameters are part of the carry (default).
            ``"closure"`` — parameters captured in step_fn closure.
        loop_mode: JAX loop primitive to use.
            ``"scan"`` — ``lax.scan`` (default, supports chunking).
            ``"fori_loop"`` — ``lax.fori_loop`` (lower memory).
        vmap_params: Whether to vmap over a population of parameter sets.
        chunk_size: Number of timesteps per chunk (simulation only).
        output_mode: Where to write outputs.
            ``"memory"`` — return xarray.Dataset (default for simulation).
            ``"disk"`` — write to Zarr.
            ``"raw"`` — return raw JAX arrays (default for optimization).
    """

    param_mode: Literal["closure", "carry"] = "carry"
    loop_mode: Literal["scan", "fori_loop"] = "scan"
    vmap_params: bool = False
    chunk_size: int | None = None
    output_mode: Literal["disk", "memory", "raw"] = "raw"

    def __post_init__(self) -> None:
        if self.output_mode == "disk" and self.loop_mode != "scan":
            msg = "output_mode='disk' requires loop_mode='scan' (chunking needs scan)"
            raise EngineError(msg)
        if self.vmap_params and self.output_mode != "raw":
            msg = "vmap_params=True requires output_mode='raw'"
            raise EngineError(msg)
        if self.vmap_params and self.chunk_size is not None:
            msg = "vmap_params=True is incompatible with chunking (chunk_size must be None)"
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
                param_mode="carry",
                loop_mode="scan",
                vmap_params=False,
                chunk_size=chunk_size,
                output_mode=output,
            )
        )

    @classmethod
    def optimization(cls, vmap: bool = False) -> Runner:
        """Create a runner configured for optimization/calibration.

        Args:
            vmap: Whether to vmap over a population of parameter sets.
        """
        return cls(
            RunnerConfig(
                param_mode="carry",
                loop_mode="scan",
                vmap_params=vmap,
                chunk_size=None,
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
        return self._run_simulation(model, output_path, export_variables)

    # --- Optimization interface ---

    def __call__(self, model: CompiledModel, free_params: Params) -> Outputs:
        """Run the model with merged parameters (optimization interface).

        Args:
            model: Compiled model.
            free_params: Free parameters to override in model.parameters.

        Returns:
            Model outputs (dict of arrays).
        """
        if self.config.vmap_params:
            return self._run_optimization_vmap(model, free_params)
        return self._run_optimization(model, free_params)

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
            else MemoryWriter(model)
        )
        writer.initialize(model.shapes, output_vars, coords=writer_coords, var_dims=var_dims)
        return writer

    def _run_simulation(
        self,
        model: CompiledModel,
        output_path: str | Path | None,
        export_variables: list[str] | None,
    ) -> tuple[State, Any | None]:
        """Run simulation with chunking and writer output."""
        n_timesteps = model.n_timesteps
        chunk_size = self._resolve_chunk_size(model)

        if chunk_size <= 0:
            raise ChunkingError(chunk_size, n_timesteps, "chunk_size must be positive")

        n_chunks = math.ceil(n_timesteps / chunk_size)
        logger.info(
            f"Starting simulation: {n_timesteps} steps in {n_chunks} chunks "
            f"(chunk_size={chunk_size})"
        )

        step_fn = build_step_fn(model)
        state = dict(model.state)
        params = dict(model.parameters)
        output_vars = export_variables if export_variables is not None else list(state.keys())
        writer = self._build_writer(model, output_path, output_vars)

        try:
            for chunk_idx in range(n_chunks):
                start_t = chunk_idx * chunk_size
                end_t = min(start_t + chunk_size, n_timesteps)
                chunk_len = end_t - start_t

                logger.debug(
                    f"Processing chunk {chunk_idx + 1}/{n_chunks} (t={start_t}-{end_t})"
                )

                forcings_chunk = model.forcings.get_chunk(start_t, end_t)
                (state, params), outputs = _scan(
                    step_fn, (state, params), forcings_chunk, chunk_len
                )
                writer.append(outputs)

            final_results = writer.finalize()
        finally:
            writer.close()

        logger.info("Simulation complete")
        return state, final_results

    # --- Internal: optimization ---

    def _run_optimization(self, model: CompiledModel, free_params: Params) -> Outputs:
        """Single-shot run with merged parameters."""
        merged = {**model.parameters, **free_params}
        step_fn = build_step_fn(model)
        forcings = model.forcings.get_all()
        (final_state, _), outputs = lax.scan(
            step_fn, (dict(model.state), merged), forcings
        )
        return outputs

    def _run_optimization_vmap(self, model: CompiledModel, free_params: Params) -> Outputs:
        """Vmap over a population of free parameter sets."""

        def run_one(single_free: Params) -> Outputs:
            merged = {**model.parameters, **single_free}
            step_fn = build_step_fn(model)
            forcings = model.forcings.get_all()
            (_, _), outputs = lax.scan(
                step_fn, (dict(model.state), merged), forcings
            )
            return outputs

        return jax.vmap(run_one)(free_params)
