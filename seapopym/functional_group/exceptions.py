"""Custom exceptions for the Functional Group module."""


class FunctionalGroupError(Exception):
    """Base exception for Functional Group module."""

    pass


class ExecutionError(FunctionalGroupError):
    """Raised when an error occurs during unit execution."""

    pass
