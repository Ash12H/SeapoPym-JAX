"""Compiler module for transforming Blueprint + Config into CompiledModel.

The compilation pipeline:
1. Validate blueprint and config
2. Compute temporal grid
3. Build dims mapping from blueprint
4. Infer shapes from data (with time_grid priority for T dimension)
5. Get dimension mapping from config and apply to shapes
6. Prepare forcings (with temporal validation)
7. Prepare initial state
8. Prepare parameters
9. Build CompiledModel
"""

from __future__ import annotations

import logging
from typing import Any

import jax.numpy as jnp
import numpy as np
import xarray as xr

from seapopym.blueprint import Blueprint, Config, validate_blueprint, validate_config

from .forcing import ForcingStore
from .inference import infer_shapes
from .model import Array, CompiledModel
from .preprocessing import extract_coords, prepare_array
from .time_grid import TimeGrid

logger = logging.getLogger(__name__)


def _extract_blueprint_dims(blueprint: Blueprint) -> dict[str, list[str] | None]:
    """Extract dimension declarations from blueprint."""
    dims_map: dict[str, list[str] | None] = {}
    all_vars = blueprint.declarations.get_all_variables()

    for var_path, var_decl in all_vars.items():
        dims_map[var_path] = var_decl.dims

    return dims_map


def _prepare_forcings(
    config: Config,
    dim_mapping: dict[str, str] | None,
    shapes: dict[str, int],
    time_grid: TimeGrid,
    blueprint_dims: dict[str, list[str] | None],
    fill_nan: float,
) -> tuple[ForcingStore, dict[str, Array]]:
    """Prepare forcings into a ForcingStore with deferred materialization.

    xr.DataArray forcings are kept lazy — only metadata is validated at
    compile time. Materialization happens at runtime in ForcingStore.get_chunk().

    Raw arrays (numpy/scalar) are materialized eagerly since they are
    already in memory.

    Args:
        config: Configuration with forcings data.
        dim_mapping: Optional dimension renaming map.
        shapes: Inferred dimension sizes (including T from time_grid).
        time_grid: Temporal grid for validation and coord generation.
        blueprint_dims: Dimension declarations from blueprint.
        fill_nan: Value to replace NaN with.

    Returns:
        Tuple of (ForcingStore, coords dict) with temporal coords added.

    Raises:
        ValueError: If a forcing's temporal range does not cover the
            simulation range defined by time_grid.
    """
    from .transpose import apply_dimension_mapping, transpose_canonical

    raw_forcings: dict[str, Any] = {}
    dynamic_forcings: set[str] = set()
    coords: dict[str, Array] = {}
    coords_extracted = False
    n_timesteps = shapes.get("T", 1)
    interp_method = config.execution.forcing_interpolation

    for name, source in config.forcings.items():
        # Determine dynamic/static from blueprint declarations
        bp_dims = blueprint_dims.get(f"forcings.{name}")
        is_dynamic = bp_dims is not None and "T" in bp_dims

        if is_dynamic:
            dynamic_forcings.add(name)

        if isinstance(source, xr.DataArray):
            # --- Lazy xr.DataArray path ---
            da = apply_dimension_mapping(source, dim_mapping)
            da = transpose_canonical(da)

            # Validate temporal coverage on metadata (no materialization)
            if is_dynamic and "T" in da.coords:
                forcing_coords = da.coords["T"]
                forcing_start = forcing_coords.values[0]
                forcing_end = forcing_coords.values[-1]

                if time_grid.start < forcing_start or time_grid.end > forcing_end:
                    raise ValueError(
                        f"Forcing '{name}' temporal range [{forcing_start}, {forcing_end}] "
                        f"does not cover simulation range [{time_grid.start}, {time_grid.end}]. "
                        f"Ensure forcing data spans the entire simulation period."
                    )

                # Slice at xarray level (still lazy)
                da = da.sel(T=slice(time_grid.start, time_grid.end))

            # Keep as lazy DataArray — ForcingStore will materialize & NaN-fill at runtime
            raw_forcings[name] = da

            # Extract coords from first xr.DataArray source
            if not coords_extracted:
                try:
                    coords = extract_coords(da, dim_mapping)
                    coords_extracted = True
                except (KeyError, ValueError, TypeError) as e:
                    logger.debug("Failed to extract coords from DataArray '%s': %s", name, e)

        else:
            # --- Raw array / scalar / file path: eager materialization ---
            arr, _dims, _mask = prepare_array(
                source,
                dimension_mapping=dim_mapping,
                fill_nan=fill_nan,
            )
            raw_forcings[name] = arr

            # Extract coords from first file source
            if not coords_extracted and isinstance(source, str):
                try:
                    coords = extract_coords(source, dim_mapping)
                    coords_extracted = True
                except (KeyError, ValueError, TypeError) as e:
                    logger.debug("Failed to extract coords from file '%s': %s", source, e)

    # Generate mask if not provided
    if "mask" not in raw_forcings:
        mask_shape = tuple(shapes.get(d, 1) for d in ["Y", "X"] if d in shapes)
        if mask_shape:
            raw_forcings["mask"] = jnp.ones(mask_shape, dtype=jnp.bool_)

    # Add temporal coordinates from time_grid
    coords["T"] = time_grid.coords

    forcing_store = ForcingStore(
        _forcings=raw_forcings,
        n_timesteps=n_timesteps,
        interp_method=interp_method,
        fill_nan=fill_nan,
        _dynamic_forcings=dynamic_forcings,
        _time_coords=time_grid.coords,
    )

    return forcing_store, coords


def _prepare_state(
    config: Config,
    dim_mapping: dict[str, str] | None,
    fill_nan: float,
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
                arr, _dims, _mask = prepare_array(
                    value,
                    dimension_mapping=dim_mapping,
                    fill_nan=fill_nan,
                )
                state[full_name] = arr

    process_state(config.initial_state)
    return state


def _prepare_parameters(
    config: Config,
) -> tuple[dict[str, Array], list[str]]:
    """Prepare parameter arrays and identify trainable params."""
    parameters: dict[str, Array] = {}
    trainable: list[str] = []

    for name, pv in config.parameters.items():
        if pv.trainable:
            trainable.append(name)

        parameters[name] = jnp.asarray(pv.value)
    return parameters, trainable


def compile_model(
    blueprint: Blueprint,
    config: Config,
    fill_nan: float = 0.0,
) -> CompiledModel:
    """Compile a Blueprint + Config into a CompiledModel.

    This is the main entry point for the compiler module.

    Args:
        blueprint: Model definition.
        config: Experiment configuration.
        fill_nan: Value to replace NaN values with.

    Returns:
        CompiledModel ready for execution.

    Raises:
        BlueprintValidationError: If blueprint validation fails.
        ConfigValidationError: If config validation fails.
        ShapeInferenceError: If shapes cannot be inferred.
        GridAlignmentError: If dimensions are inconsistent.
    """
    # Step 1: Validate (raises aggregated errors if invalid)
    bp_result = validate_blueprint(blueprint)
    validate_config(config, blueprint)

    compute_nodes = bp_result.compute_nodes
    data_nodes = bp_result.data_nodes
    tendency_map = dict(blueprint.tendencies)

    # Step 2: Compute temporal grid from execution params
    time_grid = TimeGrid.from_config(
        config.execution.time_start,
        config.execution.time_end,
        config.execution.dt,
    )

    # Step 3: Build dims mapping from blueprint
    blueprint_dims = _extract_blueprint_dims(blueprint)

    # Step 4: Infer shapes from data (with time_grid priority for T dimension)
    shapes = infer_shapes(config, blueprint_dims, time_grid=time_grid)

    # Step 5: Get dimension mapping from config and apply to shapes
    dim_mapping = config.dimension_mapping
    if dim_mapping:
        shapes = {dim_mapping.get(k, k): v for k, v in shapes.items()}

    # Step 6: Prepare forcings (with temporal validation)
    forcings, coords = _prepare_forcings(config, dim_mapping, shapes, time_grid, blueprint_dims, fill_nan)

    # Step 7: Prepare initial state
    state = _prepare_state(config, dim_mapping, fill_nan)

    # Step 8: Prepare parameters
    parameters, trainable = _prepare_parameters(config)

    # Step 9: Build CompiledModel
    return CompiledModel(
        blueprint=blueprint,
        compute_nodes=compute_nodes,
        data_nodes=data_nodes,
        tendency_map=tendency_map,
        state=state,
        forcings=forcings,
        parameters=parameters,
        shapes=shapes,
        coords=coords,
        dt=time_grid.dt_seconds,
        trainable_params=trainable,
        time_grid=time_grid,
        chunk_size=config.execution.chunk_size,
    )
