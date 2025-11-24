"""Functional Group module for executing simulation logic."""

from .core import FunctionalGroup
from .exceptions import ExecutionError, FunctionalGroupError

__all__ = ["FunctionalGroup", "ExecutionError", "FunctionalGroupError"]
