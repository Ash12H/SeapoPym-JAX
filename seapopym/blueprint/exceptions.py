"""Custom exceptions for the Blueprint module.

Error codes:
- E1xx: Validation errors
- E100: Blueprint validation (aggregated)
- E101: Function not found in registry
- E102: Signature mismatch
- E103: Dimension mismatch
- E105: Unit error
- E106: Missing data
- E107: Output count mismatch
- E110: Config validation (aggregated)
"""

from __future__ import annotations


class BlueprintError(Exception):
    """Base class for exceptions in this module."""

    code: str = "E000"
    message: str = "Blueprint error"

    def __init__(self, message: str | None = None) -> None:
        """Initialize the exception with an optional custom message."""
        self.message = message or self.message
        super().__init__(f"[{self.code}] {self.message}")


class ValidationError(BlueprintError):
    """Base class for validation errors."""

    code = "E1XX"
    message = "Validation failed"


class FunctionNotFoundError(ValidationError):
    """E101: Function not found in the registry."""

    code = "E101"
    message = "Function not found in registry"

    def __init__(self, func_name: str, backend: str | None = None) -> None:
        """Initialize with function name and optional backend."""
        self.func_name = func_name
        self.backend = backend
        if backend:
            msg = f"Function '{func_name}' not found in registry for backend '{backend}'"
        else:
            msg = f"Function '{func_name}' not found in registry"
        super().__init__(msg)


class SignatureMismatchError(ValidationError):
    """E102: Function signature does not match expected inputs."""

    code = "E102"
    message = "Function signature mismatch"

    def __init__(self, func_name: str, missing: list[str] | None = None, extra: list[str] | None = None) -> None:
        """Initialize with function name and mismatched arguments."""
        self.func_name = func_name
        self.missing = missing or []
        self.extra = extra or []
        parts = [f"Signature mismatch for function '{func_name}'"]
        if self.missing:
            parts.append(f"missing arguments: {self.missing}")
        if self.extra:
            parts.append(f"unexpected arguments: {self.extra}")
        super().__init__("; ".join(parts))


class DimensionMismatchError(ValidationError):
    """E103: Variable dimensions do not match expected dimensions."""

    code = "E103"
    message = "Dimension mismatch"

    def __init__(self, var_name: str, expected: list[str], actual: list[str] | None = None) -> None:
        """Initialize with variable name and dimension info."""
        self.var_name = var_name
        self.expected = expected
        self.actual = actual
        if actual is not None:
            msg = f"Dimension mismatch for '{var_name}': expected {expected}, got {actual}"
        else:
            msg = f"Dimension mismatch for '{var_name}': expected {expected}"
        super().__init__(msg)


class UnitError(ValidationError):
    """E105: Unit validation error (incompatible or incorrect units).

    Raised when:
    - Units declared in Blueprint don't match function signature expectations
    - Output units from one function don't match input units of the next
    - Units are dimensionally incompatible or have different canonical forms
    - Tendencies lack required time dimension (/s)
    """

    code = "E105"
    message = "Unit error"

    def __init__(self, message: str) -> None:
        """Initialize with descriptive error message."""
        super().__init__(message)


class MissingDataError(ValidationError):
    """E106: Required data is missing."""

    code = "E106"
    message = "Missing required data"

    def __init__(self, var_name: str, data_type: str = "variable") -> None:
        """Initialize with missing variable name."""
        self.var_name = var_name
        self.data_type = data_type
        super().__init__(f"Missing required {data_type}: '{var_name}'")


class OutputCountMismatchError(ValidationError):
    """E107: Number of declared outputs does not match function return."""

    code = "E107"
    message = "Output count mismatch"

    def __init__(self, func_name: str, expected: int, actual: int) -> None:
        """Initialize with function name and output counts."""
        self.func_name = func_name
        self.expected = expected
        self.actual = actual
        super().__init__(f"Output count mismatch for '{func_name}': expected {expected}, got {actual}")


class BlueprintValidationError(ValidationError):
    """E100: Aggregation of all Blueprint validation errors.

    Similar to pydantic.ValidationError: a single exception
    containing all individual errors.
    """

    code = "E100"

    def __init__(
        self,
        errors: list[ValidationError],
        warnings: list[str] | None = None,
    ) -> None:
        self.validation_errors = errors
        self.validation_warnings = warnings or []
        summary = f"{len(errors)} blueprint validation error(s)"
        details = "\n".join(f"  {e}" for e in errors)
        super().__init__(f"{summary}\n{details}")

    def error_count(self) -> int:
        return len(self.validation_errors)


class ConfigValidationError(ValidationError):
    """E110: Aggregation of all Config validation errors."""

    code = "E110"

    def __init__(
        self,
        errors: list[ValidationError],
        warnings: list[str] | None = None,
    ) -> None:
        self.validation_errors = errors
        self.validation_warnings = warnings or []
        summary = f"{len(errors)} config validation error(s)"
        details = "\n".join(f"  {e}" for e in errors)
        super().__init__(f"{summary}\n{details}")

    def error_count(self) -> int:
        return len(self.validation_errors)
