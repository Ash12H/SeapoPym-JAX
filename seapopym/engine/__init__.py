"""Engine package for execution of compiled models.

This package provides:
- Backends: JAXBackend (lax.scan) and NumpyBackend (for loop)
- Runners: StreamingRunner (production)
- Step function builder for time-stepping logic
- Async I/O for streaming outputs

Example:
    >>> from seapopym.compiler import compile_model
    >>> from seapopym.engine import StreamingRunner
    >>>
    >>> # Production mode (with disk I/O)
    >>> model = compile_model(blueprint, config)
    >>> runner = StreamingRunner(model, chunk_size=365)
    >>> final_state = runner.run(output_path="/results/sim_001/")
"""

from .backends import Backend, JAXBackend, NumpyBackend, get_backend
from .exceptions import (
    BackendError,
    ChunkingError,
    EngineError,
    EngineIOError,
    StepError,
)
from .io import DiskWriter
from .runners import StreamingRunner
from .step import build_step_fn

__all__ = [
    # Runners
    "StreamingRunner",
    # Backends
    "Backend",
    "JAXBackend",
    "NumpyBackend",
    "get_backend",
    # Step
    "build_step_fn",
    # I/O
    "DiskWriter",
    # Exceptions
    "EngineError",
    "StepError",
    "BackendError",
    "ChunkingError",
    "EngineIOError",
]
