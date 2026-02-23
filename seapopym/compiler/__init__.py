"""Compiler package for transforming Blueprint + Config into executable JAX structures.

This package provides:
- compile_model: Compile a Blueprint + Config into a CompiledModel
- CompiledModel: Dataclass containing pytrees ready for execution
- TimeGrid: Temporal grid configuration
- Shape inference, dimension mapping, and preprocessing utilities

Example:
    >>> from seapopym.blueprint import Blueprint, Config
    >>> from seapopym.compiler import compile_model
    >>>
    >>> blueprint = Blueprint.load("model.yaml")
    >>> config = Config.load("run.yaml")
    >>> compiled = compile_model(blueprint, config)
    >>>
    >>> # Access compiled data
    >>> compiled.state["biomass"]  # jax.Array
    >>> compiled.shapes  # {"Y": 180, "X": 360, ...}
"""

from .compiler import compile_model
from .exceptions import CompilerError, GridAlignmentError, ShapeInferenceError, TransposeError
from .forcing import ForcingStore
from .inference import infer_shapes
from .model import CompiledModel
from .time_grid import TimeGrid
from seapopym.blueprint.units import UnitValidator, validate_units

__all__ = [
    "compile_model",
    "CompiledModel",
    "ForcingStore",
    "TimeGrid",
    "infer_shapes",
    "UnitValidator",
    "validate_units",
    "CompilerError",
    "ShapeInferenceError",
    "GridAlignmentError",
    "TransposeError",
]
