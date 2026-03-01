"""Engine package for execution of compiled models.

This package provides:
- Runners: StreamingRunner (production, with chunking and disk I/O)
- Step function builder for time-stepping logic

Example:
    >>> from seapopym.compiler import compile_model
    >>> from seapopym.engine import StreamingRunner
    >>>
    >>> model = compile_model(blueprint, config)
    >>> runner = StreamingRunner(model, chunk_size=365)
    >>> final_state = runner.run(output_path="/results/sim_001/")
"""

from .exceptions import (
    ChunkingError,
    EngineError,
    EngineIOError,
)
from .io import DiskWriter
from .runners import StreamingRunner
from .step import build_step_fn

__all__ = [
    # Runners
    "StreamingRunner",
    # Step
    "build_step_fn",
    # I/O
    "DiskWriter",
    # Exceptions
    "EngineError",
    "ChunkingError",
    "EngineIOError",
]
