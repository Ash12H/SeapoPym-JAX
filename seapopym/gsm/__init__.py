"""Global State Manager package for simulation state management."""

from .core import StateManager
from .exceptions import StateValidationError

__all__ = ["StateManager", "StateValidationError"]
