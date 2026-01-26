"""Engine-specific exceptions.

Error codes as per architecture:
- E300: EngineError (base)
- E301: StepError
- E302: BackendError
- E303: ChunkingError
- E304: IOError
"""

from __future__ import annotations


class EngineError(Exception):
    """Base class for engine errors."""

    code: str = "E300"

    def __init__(self, message: str) -> None:
        """Initialize with error message."""
        self.message = message
        super().__init__(f"[{self.code}] {message}")


class StepError(EngineError):
    """E301: Error during step function execution."""

    code = "E301"

    def __init__(self, timestep: int, reason: str) -> None:
        """Initialize with timestep and failure reason."""
        self.timestep = timestep
        self.reason = reason
        super().__init__(f"Step failed at timestep {timestep}: {reason}")


class BackendError(EngineError):
    """E302: Backend not available or misconfigured."""

    code = "E302"

    def __init__(self, backend: str, reason: str) -> None:
        """Initialize with backend name and failure reason."""
        self.backend = backend
        self.reason = reason
        super().__init__(f"Backend '{backend}' error: {reason}")


class ChunkingError(EngineError):
    """E303: Error in temporal chunking."""

    code = "E303"

    def __init__(self, chunk_size: int, total_steps: int, reason: str) -> None:
        """Initialize with chunking parameters and failure reason."""
        self.chunk_size = chunk_size
        self.total_steps = total_steps
        self.reason = reason
        super().__init__(f"Chunking error (chunk_size={chunk_size}, total={total_steps}): {reason}")


class EngineIOError(EngineError):
    """E304: Error during I/O operations."""

    code = "E304"

    def __init__(self, path: str, reason: str) -> None:
        """Initialize with path and failure reason."""
        self.path = path
        self.reason = reason
        super().__init__(f"I/O error at '{path}': {reason}")
