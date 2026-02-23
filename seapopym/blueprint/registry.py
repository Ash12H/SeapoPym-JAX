"""Function registry for the Blueprint module.

This module provides the @functional decorator and the global REGISTRY
for registering computation functions with metadata.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .exceptions import FunctionNotFoundError


@dataclass(frozen=True)
class FunctionMetadata:
    """Metadata for a registered function.

    Attributes:
        name: Unique identifier in format "namespace:function_name".
        core_dims: Dimensions that are not broadcast (per input).
        out_dims: Output dimensions (for single output).
        outputs: Names of outputs (for multiple outputs).
        units: Expected units per argument and return value.
        func: The wrapped function.
    """

    name: str
    func: Callable[..., Any]
    core_dims: dict[str, list[str]] = field(default_factory=dict)
    out_dims: list[str] | None = None
    outputs: list[str] | None = None
    units: dict[str, str] = field(default_factory=dict)

    @property
    def is_multi_output(self) -> bool:
        """Return True if the function returns multiple outputs."""
        return self.outputs is not None and len(self.outputs) > 1

    @property
    def output_names(self) -> list[str]:
        """Return the list of output names."""
        if self.outputs:
            return self.outputs
        return ["return"]

    def get_signature(self) -> inspect.Signature:
        """Return the function signature."""
        return inspect.signature(self.func)

    def get_required_inputs(self) -> list[str]:
        """Return list of required input argument names (no default value)."""
        sig = self.get_signature()
        return [name for name, param in sig.parameters.items() if param.default is inspect.Parameter.empty]


# Global registry: {name: FunctionMetadata}
REGISTRY: dict[str, FunctionMetadata] = {}


def functional(
    name: str,
    core_dims: dict[str, list[str]] | None = None,
    out_dims: list[str] | None = None,
    outputs: list[str] | None = None,
    units: dict[str, str] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a function in the global registry.

    Args:
        name: Unique identifier in format "namespace:function_name".
              Example: "biol:growth", "phys:advection".
        core_dims: Dimensions that are explicitly operated on (not broadcast).
                   Format: {"input_name": ["dim1", "dim2"]}.
        out_dims: Output dimensions for single-output functions.
        outputs: Names of outputs for multi-output functions.
                 The function must return a tuple in this order.
        units: Expected units per argument. Use "return" key for output unit.
               Example: {"biomass": "g", "rate": "1/d", "return": "g/d"}.

    Returns:
        Decorator function.

    Example:
        >>> @functional(
        ...     name="biol:growth",
        ...     core_dims={"biomass": ["C"]},
        ...     out_dims=["C"],
        ...     units={"biomass": "g", "rate": "1/d", "temp": "degC", "return": "g/d"}
        ... )
        ... def growth(biomass, rate, temp):
        ...     return biomass * rate * jnp.exp(temp / 10)

    Example (multi-output):
        >>> @functional(
        ...     name="biol:predation",
        ...     outputs=["prey_loss", "predator_gain"],
        ...     units={"prey_loss": "g/d", "predator_gain": "g/d"}
        ... )
        ... def predation(prey_biomass, predator_biomass, rate):
        ...     flux = rate * prey_biomass * predator_biomass
        ...     return -flux, +flux
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Validate name format
        if ":" not in name:
            raise ValueError(f"Function name must be in format 'namespace:function_name', got '{name}'")

        # Create metadata
        metadata = FunctionMetadata(
            name=name,
            func=func,
            core_dims=core_dims or {},
            out_dims=out_dims,
            outputs=outputs,
            units=units or {},
        )

        # Register in global registry
        if name in REGISTRY:
            import warnings

            warnings.warn(
                f"Function '{name}' is already registered. Overwriting.",
                stacklevel=2,
            )
        REGISTRY[name] = metadata

        # Preserve function metadata
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        # Attach metadata to wrapper for introspection
        wrapper._functional_metadata = metadata  # type: ignore[attr-defined]

        return wrapper

    return decorator


def get_function(name: str) -> FunctionMetadata:
    """Retrieve a function from the registry.

    Args:
        name: Function identifier (e.g., "biol:growth").

    Returns:
        FunctionMetadata for the requested function.

    Raises:
        FunctionNotFoundError: If function is not registered.
    """
    if name not in REGISTRY:
        raise FunctionNotFoundError(name)

    return REGISTRY[name]


def list_functions() -> list[str]:
    """List all registered function names.

    Returns:
        Sorted list of function names.
    """
    return sorted(REGISTRY.keys())


def clear_registry() -> None:
    """Clear the registry (useful for testing)."""
    REGISTRY.clear()
