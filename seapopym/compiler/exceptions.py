"""Compiler-specific exceptions.

Error codes as per SPEC_02 §9:
- E201: ShapeInferenceError
- E202: GridAlignmentError
- E203: MissingDimensionError
- E204: TransposeError
- E205: UnitError
"""

from __future__ import annotations


class CompilerError(Exception):
    """Base class for compiler errors."""

    code: str = "E200"

    def __init__(self, message: str) -> None:
        """Initialize with error message."""
        self.message = message
        super().__init__(f"[{self.code}] {message}")


class ShapeInferenceError(CompilerError):
    """E201: Failed to read metadata or infer shapes from data files."""

    code = "E201"

    def __init__(self, path: str, reason: str) -> None:
        """Initialize with file path and failure reason."""
        self.path = path
        self.reason = reason
        super().__init__(f"Cannot infer shape from '{path}': {reason}")


class GridAlignmentError(CompilerError):
    """E202: Dimension sizes are inconsistent between files."""

    code = "E202"

    def __init__(self, dimension: str, sizes: dict[str, int]) -> None:
        """Initialize with dimension name and conflicting sizes."""
        self.dimension = dimension
        self.sizes = sizes
        size_details = ", ".join(f"{k}={v}" for k, v in sizes.items())
        super().__init__(f"Dimension '{dimension}' has inconsistent sizes: {size_details}")


class MissingDimensionError(CompilerError):
    """E203: A required dimension is missing from the data."""

    code = "E203"

    def __init__(self, dimension: str, variable: str, available: list[str]) -> None:
        """Initialize with missing dimension, variable name, and available dims."""
        self.dimension = dimension
        self.variable = variable
        self.available = available
        super().__init__(
            f"Variable '{variable}' is missing required dimension '{dimension}'. " f"Available: {available}"
        )


class TransposeError(CompilerError):
    """E204: Failed to transpose data to canonical order."""

    code = "E204"

    def __init__(self, variable: str, reason: str) -> None:
        """Initialize with variable name and failure reason."""
        self.variable = variable
        self.reason = reason
        super().__init__(f"Cannot transpose '{variable}': {reason}")


class UnitError(CompilerError):
    """E205: Unit validation error (incompatible or incorrect units).

    Raised when:
    - Units declared in Blueprint don't match function signature expectations
    - Output units from one function don't match input units of the next
    - Units are dimensionally incompatible or have different canonical forms
    - Tendencies lack required time dimension (/s)
    """

    code = "E205"

    def __init__(self, message: str) -> None:
        """Initialize with descriptive error message.

        Args:
            message: Detailed explanation of the unit mismatch,
                     including expected vs actual units and context.
        """
        super().__init__(message)
