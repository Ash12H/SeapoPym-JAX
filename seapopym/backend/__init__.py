"""Execution backends for Seapopym."""

from .base import ComputeBackend
from .distributed import DistributedBackend
from .exceptions import BackendConfigurationError, ExecutionError
from .sequential import SequentialBackend

__all__ = [
    "ComputeBackend",
    "SequentialBackend",
    "DistributedBackend",
    "BackendConfigurationError",
    "ExecutionError",
]
