"""Execution backends for Seapopym."""

from .base import ComputeBackend
from .dask import DaskBackend
from .data_parallel import DataParallelBackend
from .exceptions import BackendConfigurationError, ExecutionError
from .sequential import SequentialBackend
from .task_parallel import TaskParallelBackend

__all__ = [
    "ComputeBackend",
    "SequentialBackend",
    "TaskParallelBackend",
    "DataParallelBackend",
    "DaskBackend",  # Deprecated, kept for backwards compatibility
    "BackendConfigurationError",
    "ExecutionError",
]
