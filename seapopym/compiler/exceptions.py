"""Compiler-specific exceptions.

Error codes:
- E201: ShapeInferenceError
- E202: GridAlignmentError
- E204: TransposeError
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


class TransposeError(CompilerError):
    """E204: Failed to transpose data to canonical order."""

    code = "E204"

    def __init__(self, variable: str, reason: str) -> None:
        """Initialize with variable name and failure reason."""
        self.variable = variable
        self.reason = reason
        super().__init__(f"Cannot transpose '{variable}': {reason}")


