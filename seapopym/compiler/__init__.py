"""Compiler package for transforming Blueprint + Config into executable JAX structures.

This package provides:
- Compiler: Main class for compiling blueprints
- CompiledModel: Dataclass containing pytrees ready for execution
- compile_model: Convenience function for compilation
- Shape inference, dimension mapping, and preprocessing utilities

Example:
    >>> from seapopym.blueprint import Blueprint, Config
    >>> from seapopym.compiler import Compiler, compile_model
    >>>
    >>> blueprint = Blueprint.load("model.yaml")
    >>> config = Config.load("run.yaml")
    >>>
    >>> # Using the class
    >>> compiler = Compiler(backend="jax")
    >>> compiled = compiler.compile(blueprint, config)
    >>>
    >>> # Or using the convenience function
    >>> compiled = compile_model(blueprint, config)
    >>>
    >>> # Access compiled data
    >>> compiled.state["biomass"]  # jax.Array
    >>> compiled.shapes  # {"Y": 180, "X": 360, ...}
"""

from .compiler import Compiler, compile_model
from .exceptions import (
    CompilerError,
    GridAlignmentError,
    MissingDimensionError,
    ShapeInferenceError,
    TransposeError,
)
from .inference import infer_shapes
from .model import CANONICAL_DIMS, CompiledModel
from .preprocessing import prepare_array, preprocess_nan, strip_xarray
from .transpose import (
    apply_dimension_mapping,
    get_canonical_order,
    transpose_array,
    transpose_canonical,
)

__all__ = [
    # Main API
    "Compiler",
    "compile_model",
    "CompiledModel",
    # Constants
    "CANONICAL_DIMS",
    # Inference
    "infer_shapes",
    # Transpose
    "apply_dimension_mapping",
    "get_canonical_order",
    "transpose_canonical",
    "transpose_array",
    # Preprocessing
    "prepare_array",
    "preprocess_nan",
    "strip_xarray",
    # Exceptions
    "CompilerError",
    "ShapeInferenceError",
    "GridAlignmentError",
    "MissingDimensionError",
    "TransposeError",
]
