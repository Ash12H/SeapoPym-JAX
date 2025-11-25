"""Execution backends for Seapopym."""

from .base import ComputeBackend
from .sequential import SequentialBackend

__all__ = ["ComputeBackend", "SequentialBackend"]
