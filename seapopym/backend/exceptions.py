"""Custom exceptions for the Backend module."""


class BackendError(Exception):
    """Base exception for Backend module."""

    pass


class ExecutionError(BackendError):
    """Raised when an error occurs during unit execution."""

    pass
