"""Main Compiler class for transforming Blueprint + Config into CompiledModel.

The Compiler orchestrates the full compilation pipeline:
1. Infer shapes from data metadata
2. Rename dimensions to canonical names
3. Transpose to canonical order
4. Strip xarray and preprocess NaN
5. Package into CompiledModel
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from seapopym.blueprint import Blueprint, Config, ParameterValue, validate_blueprint, validate_config

from .inference import infer_shapes
from .model import CANONICAL_DIMS, Array, CompiledModel
from .preprocessing import extract_coords, prepare_array


def _parse_dt(dt_str: str) -> float:
    """Parse timestep string to seconds.

    Args:
        dt_str: Timestep string like "1d", "6h", "30m".

    Returns:
        Timestep in seconds.
    """
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}

    # Try to parse as number + unit
    for unit, multiplier in units.items():
        if dt_str.endswith(unit):
            try:
                value = float(dt_str[:-1])
                return value * multiplier
            except ValueError:
                pass

    # Try as plain number (assume seconds)
    try:
        return float(dt_str)
    except ValueError:
        return 86400.0  # Default: 1 day


Backend = Literal["jax", "numpy"]


class Compiler:
    """Compiler for transforming Blueprint + Config into executable structures.

    The Compiler reads data from files or in-memory arrays, applies dimension
    mappings, transposes to canonical order, and packages everything into
    a CompiledModel ready for JAX execution.

    Attributes:
        backend: Target backend ("jax" or "numpy").
        canonical_order: Dimension order for transposition.
        fill_nan: Value to replace NaN with.
    """

    backend: Backend

    def __init__(
        self,
        backend: Backend = "jax",
        canonical_order: tuple[str, ...] = CANONICAL_DIMS,
        fill_nan: float = 0.0,
    ) -> None:
        """Initialize the Compiler.

        Args:
            backend: Target backend for arrays.
            canonical_order: Canonical dimension order.
            fill_nan: Value to replace NaN values.
        """
        self.backend = backend
        self.canonical_order = canonical_order
        self.fill_nan = fill_nan

    def compile(
        self,
        blueprint: Blueprint,
        config: Config,
        validate: bool = True,
    ) -> CompiledModel:
        """Compile a Blueprint + Config into a CompiledModel.

        Args:
            blueprint: Model definition.
            config: Experiment configuration with data.
            validate: Whether to run validation first.

        Returns:
            CompiledModel ready for execution.

        Raises:
            ValidationError: If validation fails.
            ShapeInferenceError: If shapes cannot be inferred.
            GridAlignmentError: If dimensions are inconsistent.
        """
        # Step 1: Validate (optional but recommended)
        if validate:
            bp_result = validate_blueprint(blueprint, self.backend)
            if not bp_result.valid:
                raise bp_result.errors[0]

            cfg_result = validate_config(config, blueprint, self.backend)
            if not cfg_result.valid:
                raise cfg_result.errors[0]

            graph = bp_result.graph
        else:
            # Still need the graph
            bp_result = validate_blueprint(blueprint, self.backend)
            graph = bp_result.graph

        if graph is None:
            raise ValueError("Failed to build dependency graph")

        # Step 2: Build dims mapping from blueprint
        blueprint_dims = self._extract_blueprint_dims(blueprint)

        # Step 3: Infer shapes from data
        shapes = infer_shapes(config, blueprint_dims)

        # Step 4: Get dimension mapping from config and apply to shapes
        dim_mapping = config.dimension_mapping
        if dim_mapping:
            shapes = {dim_mapping.get(k, k): v for k, v in shapes.items()}

        # Step 5: Prepare forcings
        forcings, coords = self._prepare_forcings(config, dim_mapping, shapes)

        # Step 6: Prepare initial state
        state = self._prepare_state(config, blueprint, dim_mapping)

        # Step 7: Prepare parameters
        parameters, trainable = self._prepare_parameters(config, blueprint)

        # Step 8: Parse dt
        dt = _parse_dt(config.execution.dt)

        # Step 9: Build CompiledModel
        return CompiledModel(
            blueprint=blueprint,
            graph=graph,
            state=state,
            forcings=forcings,
            parameters=parameters,
            shapes=shapes,
            coords=coords,
            dt=dt,
            backend=self.backend,
            trainable_params=trainable,
        )

    def _extract_blueprint_dims(self, blueprint: Blueprint) -> dict[str, list[str] | None]:
        """Extract dimension declarations from blueprint."""
        dims_map: dict[str, list[str] | None] = {}
        all_vars = blueprint.declarations.get_all_variables()

        for var_path, var_decl in all_vars.items():
            dims_map[var_path] = var_decl.dims

        return dims_map

    def _prepare_forcings(
        self,
        config: Config,
        dim_mapping: dict[str, str] | None,
        shapes: dict[str, int],
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        """Prepare forcing arrays."""
        forcings: dict[str, Array] = {}
        coords: dict[str, Array] = {}
        coords_extracted = False

        for name, source in config.forcings.items():
            arr, dims, mask = prepare_array(
                source,
                dimension_mapping=dim_mapping,
                backend=self.backend,
                fill_nan=self.fill_nan,
            )
            forcings[name] = arr

            # If this is the mask, keep track of it separately
            # (it's also in forcings for uniformity)

            # Extract coords from first file source
            if not coords_extracted and isinstance(source, str):
                try:
                    coords = extract_coords(source, dim_mapping, self.backend)
                    coords_extracted = True
                except Exception:
                    pass  # Coords extraction is optional

        # Generate mask if not provided
        if "mask" not in forcings:
            # Create default mask from shapes
            mask_shape = tuple(shapes.get(d, 1) for d in ["Y", "X"] if d in shapes)
            if mask_shape:
                if self.backend == "jax":
                    import jax.numpy as jnp

                    forcings["mask"] = jnp.ones(mask_shape, dtype=jnp.bool_)
                else:
                    forcings["mask"] = np.ones(mask_shape, dtype=np.bool_)

        return forcings, coords

    def _prepare_state(
        self,
        config: Config,
        _blueprint: Blueprint,
        dim_mapping: dict[str, str] | None,
    ) -> dict[str, Array]:
        """Prepare initial state arrays."""
        state: dict[str, Array] = {}

        def process_state(data: dict[str, Any], prefix: str = "") -> None:
            for name, value in data.items():
                full_name = f"{prefix}{name}" if prefix else name

                if isinstance(value, dict) and not hasattr(value, "shape"):
                    # Nested group
                    process_state(value, f"{full_name}.")
                else:
                    arr, dims, mask = prepare_array(
                        value,
                        dimension_mapping=dim_mapping,
                        backend=self.backend,
                        fill_nan=self.fill_nan,
                    )
                    state[full_name] = arr

        process_state(config.initial_state)
        return state

    def _prepare_parameters(
        self,
        config: Config,
        _blueprint: Blueprint,
    ) -> tuple[dict[str, Array], list[str]]:
        """Prepare parameter arrays and identify trainable params."""
        parameters: dict[str, Array] = {}
        trainable: list[str] = []

        def process_params(data: dict[str, Any], prefix: str = "") -> None:
            for name, value in data.items():
                full_name = f"{prefix}{name}" if prefix else name

                if isinstance(value, dict):
                    if "value" in value:
                        # It's a ParameterValue-like dict
                        pv = ParameterValue.model_validate(value)
                        param_value = pv.value
                        if pv.trainable:
                            trainable.append(full_name)
                    else:
                        # Nested group
                        process_params(value, f"{full_name}.")
                        continue
                elif isinstance(value, ParameterValue):
                    param_value = value.value
                    if value.trainable:
                        trainable.append(full_name)
                else:
                    param_value = value

                # Convert to array
                if self.backend == "jax":
                    import jax.numpy as jnp

                    parameters[full_name] = jnp.asarray(param_value)
                else:
                    parameters[full_name] = np.asarray(param_value)

        process_params(config.parameters)
        return parameters, trainable


def compile_model(
    blueprint: Blueprint,
    config: Config,
    backend: Literal["jax", "numpy"] = "jax",
    validate: bool = True,
) -> CompiledModel:
    """Convenience function to compile a model.

    Args:
        blueprint: Model definition.
        config: Experiment configuration.
        backend: Target backend.
        validate: Whether to validate first.

    Returns:
        CompiledModel ready for execution.
    """
    compiler = Compiler(backend=backend)
    return compiler.compile(blueprint, config, validate=validate)
