"""Custom exceptions for the Blueprint module."""


class BlueprintError(Exception):
    """Base class for exceptions in this module."""

    pass


class MissingInputError(BlueprintError):
    """Raised when an input variable cannot be resolved for a unit."""

    pass


class CycleError(BlueprintError):
    """Raised when a cycle is detected in the dependency graph."""

    pass


class ConfigurationError(BlueprintError):
    """Raised when there is a configuration issue (e.g. duplicate names)."""

    pass
