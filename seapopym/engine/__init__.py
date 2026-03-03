"""Engine package for execution of compiled models.

This package provides:
- Runner: Composable runner with presets for simulation and optimization
- Step function builder for time-stepping logic

Example:
    >>> from seapopym.compiler import compile_model
    >>> from seapopym.engine import Runner
    >>>
    >>> model = compile_model(blueprint, config)
    >>> runner = Runner.simulation(chunk_size=365)
    >>> final_state, outputs = runner.run(model, output_path="/results/sim_001/")
"""

from .exceptions import (
    ChunkingError,
    EngineError,
    EngineIOError,
)
from .io import DiskWriter
from .runner import Runner, RunnerConfig
from .step import build_step_fn

__all__ = [
    # Runners
    "Runner",
    "RunnerConfig",
    # Step
    "build_step_fn",
    # I/O
    "DiskWriter",
    # Exceptions
    "EngineError",
    "ChunkingError",
    "EngineIOError",
]
