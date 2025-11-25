"""Standard coordinate names used across the application."""

from enum import Enum


class Coordinates(str, Enum):
    """Standard coordinate names."""

    T = "time"
    X = "x"
    Y = "y"
    Z = "z"
