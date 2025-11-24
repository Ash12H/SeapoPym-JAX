"""Custom exceptions for the GSM module."""


class GSMError(Exception):
    """Base exception for GSM module."""

    pass


class StateValidationError(GSMError):
    """Exception levée lorsque l'état est invalide (variables manquantes, dimensions incorrectes)."""

    pass
