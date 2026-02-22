"""Blueprint package for declarative model definition and validation.

This package provides:
- Blueprint: Declarative model definition (topology, contracts)
- Config: Experiment configuration (values, data paths)
- @functional: Decorator to register computation functions
- validate_blueprint: Validation pipeline for blueprints

Example:
    >>> from seapopym.blueprint import Blueprint, Config, functional
    >>>
    >>> # Load model and config
    >>> blueprint = Blueprint.load("model.yaml")
    >>> config = Config.load("run.yaml")
    >>>
    >>> # Register a custom function
    >>> @functional(name="biol:growth", backend="jax")
    ... def growth(biomass, rate, temp):
    ...     return biomass * rate * jnp.exp(temp / 10)
"""

from .exceptions import (
    BlueprintError,
    DimensionMismatchError,
    FunctionNotFoundError,
    MissingDataError,
    OutputCountMismatchError,
    SignatureMismatchError,
    UnitMismatchError,
    ValidationError,
)
from .execution import ExecutionPlan
from .nodes import ComputeNode, DataNode
from .registry import (
    FunctionMetadata,
    clear_registry,
    functional,
    get_function,
    list_functions,
)
from .schema import (
    Blueprint,
    Config,
    Declarations,
    ExecutionParams,
    ParameterValue,
    ProcessStep,
    TendencySource,
    VariableDeclaration,
)
from .validation import (
    BlueprintValidator,
    ValidationResult,
    validate_blueprint,
    validate_config,
)

__all__ = [
    # Core classes
    "Blueprint",
    "Config",
    "ExecutionPlan",
    # Schema classes
    "VariableDeclaration",
    "ParameterValue",
    "ProcessStep",
    "TendencySource",
    "Declarations",
    "ExecutionParams",
    # Registry
    "functional",
    "get_function",
    "list_functions",
    "clear_registry",
    "FunctionMetadata",
    # Validation
    "validate_blueprint",
    "validate_config",
    "BlueprintValidator",
    "ValidationResult",
    # Nodes
    "DataNode",
    "ComputeNode",
    # Exceptions
    "BlueprintError",
    "ValidationError",
    "FunctionNotFoundError",
    "SignatureMismatchError",
    "DimensionMismatchError",
    "UnitMismatchError",
    "MissingDataError",
    "OutputCountMismatchError",
]
