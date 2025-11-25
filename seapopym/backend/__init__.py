"""Execution backends for Seapopym."""

from .base import ComputeBackend
from .dask import DaskBackend
from .sequential import SequentialBackend

__all__ = ["ComputeBackend", "DaskBackend", "SequentialBackend"]
