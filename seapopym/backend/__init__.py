"""Execution backends for Seapopym."""

from .base import ComputeBackend
from .data_parallel import DataParallelBackend
from .distributed import DistributedBackend
from .exceptions import BackendConfigurationError, ExecutionError
from .sequential import SequentialBackend
from .task_parallel import TaskParallelBackend

__all__ = [
    "ComputeBackend",
    "SequentialBackend",
    "TaskParallelBackend",
    "DataParallelBackend",
    "DistributedBackend",
    "BackendConfigurationError",
    "ExecutionError",
]
