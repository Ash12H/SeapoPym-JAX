"""Engine package for execution of compiled models.

This package provides:
- run: Pure execution engine (chunked lax.scan with pluggable writer)
- simulate: Convenience wrapper (builds step_fn + writer, manages lifecycle)
- build_step_fn: Step function builder for time-stepping logic
- Writers: WriterRaw (JAX-traceable), MemoryWriter (xarray), DiskWriter (Zarr)

Example::

    from seapopym.compiler import compile_model
    from seapopym.engine import simulate

    model = compile_model(blueprint, config)
    final_state, dataset = simulate(model, chunk_size=365)

    # Low-level
    from seapopym.engine import run, build_step_fn

    step_fn = build_step_fn(model, export_variables=["biomass"])
    state, outputs = run(step_fn, model, dict(model.state), params)
"""

from .exceptions import (
    ChunkingError,
    EngineError,
    EngineIOError,
)
from .io import DiskWriter, MemoryWriter, WriterRaw, build_writer
from .run import run, simulate
from .runner import Runner, RunnerConfig  # Deprecated shim
from .step import build_step_fn

__all__ = [
    # Functions
    "run",
    "simulate",
    "build_step_fn",
    "build_writer",
    # Writers
    "WriterRaw",
    "MemoryWriter",
    "DiskWriter",
    # Deprecated
    "Runner",
    "RunnerConfig",
    # Exceptions
    "EngineError",
    "ChunkingError",
    "EngineIOError",
]
