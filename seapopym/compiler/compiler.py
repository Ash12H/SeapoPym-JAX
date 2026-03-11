"""Compiler module for transforming Blueprint + Config into CompiledModel.

The compilation pipeline:
1. Validate blueprint and config (validate_model)
2. Compute temporal grid
3. Build dims mapping from blueprint
4. Infer shapes from data
5. Prepare forcings (transpose, temporal slice, ForcingStore)
6. Prepare initial state (flat dict, DataArray → JAX)
7. Prepare parameters (DataArray → JAX)
8. Build CompiledModel
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np
import xarray as xr

from seapopym.blueprint import Blueprint, Config, validate_model

from .forcing import ForcingStore
from .inference import infer_shapes
from .model import Array, CompiledModel
from .preprocessing import extract_coords
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
) -> tuple[ForcingStore, dict[str, Array]]:
    """Prepare forcings into a ForcingStore with deferred materialization.

    All forcings are xr.DataArray (enforced by Config typing). They are
    kept lazy — materialization happens at runtime in ForcingStore.

    Args:
        config: Configuration with forcings data.
        dim_mapping: Optional dimension renaming map.
        shapes: Inferred dimension sizes (including T from time_grid).
        time_grid: Temporal grid for validation and coord generation.
        blueprint_dims: Dimension declarations from blueprint.

    Returns:
        Tuple of (ForcingStore, coords dict) with temporal coords added.
    """
    from .transpose import apply_dimension_mapping, transpose_canonical

    processed_forcings: dict[str, xr.DataArray] = {}
    coords: dict[str, Array] = {}
    coords_extracted = False
    n_timesteps = shapes.get("T", 1)
    interp_method = config.execution.forcing_interpolation

    for name, source in config.forcings.items():
        bp_dims = blueprint_dims.get(f"forcings.{name}")
        is_dynamic = bp_dims is not None and "T" in bp_dims

        da = apply_dimension_mapping(source, dim_mapping)
        da = transpose_canonical(da)  # type: ignore[reportArgumentType]

        # Slice dynamic forcings at xarray level (still lazy)
        if is_dynamic and "T" in da.coords:
            da = da.sel(T=slice(time_grid.start, time_grid.end))

        processed_forcings[name] = da

        # Extract coords from first source
        if not coords_extracted:
            try:
                coords = extract_coords(da, dim_mapping)
                coords_extracted = True
            except (KeyError, ValueError, TypeError) as e:
                logger.debug("Failed to extract coords from DataArray '%s': %s", name, e)

    # Generate mask if not provided
    if "mask" not in processed_forcings:
        mask_shape = tuple(shapes.get(d, 1) for d in ["Y", "X"] if d in shapes)
        if mask_shape:
            processed_forcings["mask"] = xr.DataArray(
                data=np.ones(mask_shape, dtype=bool),
                dims=[d for d in ["Y", "X"] if d in shapes],
            )

    coords["T"] = time_grid.coords

    forcing_store = ForcingStore.from_config(
        forcings=processed_forcings,
        blueprint_dims=blueprint_dims,
        n_timesteps=n_timesteps,
        interp_method=interp_method,
        time_coords=time_grid.coords,
    )

    return forcing_store, coords


def _check_nan(name: str, arr: np.ndarray, category: str) -> None:
    """Raise ValueError if array contains NaN values."""
    if np.issubdtype(arr.dtype, np.floating) and np.isnan(arr).any():
        raise ValueError(f"{category} '{name}' contains NaN values. Handle NaN upstream.")


def _prepare_state(config: Config, dim_mapping: dict[str, str] | None) -> dict[str, Array]:
    """Prepare initial state arrays (transpose to canonical order, DataArray → JAX)."""
    from .transpose import apply_dimension_mapping, transpose_canonical

    result = {}
    for name, da in config.initial_state.items():
        da = apply_dimension_mapping(da, dim_mapping)
        da = transpose_canonical(da)  # type: ignore[reportArgumentType]
        values = da.values
        _check_nan(name, values, "Initial state")
        result[name] = jnp.asarray(values)
    return result


def _prepare_parameters(config: Config, dim_mapping: dict[str, str] | None) -> dict[str, Array]:
    """Prepare parameter arrays (transpose to canonical order, DataArray → JAX)."""
    from .transpose import apply_dimension_mapping, transpose_canonical

    result = {}
    for name, da in config.parameters.items():
        da = apply_dimension_mapping(da, dim_mapping)
        da = transpose_canonical(da)  # type: ignore[reportArgumentType]
        values = da.values
        _check_nan(name, values, "Parameter")
        result[name] = jnp.asarray(values)
    return result


def compile_model(
    blueprint: Blueprint,
    config: Config,
) -> CompiledModel:
    """Compile a Blueprint + Config into a CompiledModel.

    This is the main entry point for the compiler module.
    Validation is performed first via validate_model(), then mechanical
    transformation produces the CompiledModel.

    Args:
        blueprint: Model definition.
        config: Experiment configuration.

    Returns:
        CompiledModel ready for execution.

    Raises:
        BlueprintValidationError: If blueprint validation fails.
        ConfigValidationError: If config validation fails.
        ShapeInferenceError: If shapes cannot be inferred.
        GridAlignmentError: If dimensions are inconsistent.
    """
    # Step 1: Validate (raises aggregated errors if invalid)
    bp_result = validate_model(blueprint, config)

    compute_nodes = bp_result.compute_nodes
    data_nodes = bp_result.data_nodes
    tendency_map = dict(blueprint.tendencies)

    # Step 2: Compute temporal grid
    time_grid = TimeGrid.from_config(
        config.execution.time_start,
        config.execution.time_end,
        config.execution.dt,
    )

    # Step 3: Build dims mapping from blueprint
    blueprint_dims = _extract_blueprint_dims(blueprint)

    # Step 4: Infer shapes from data
    shapes = infer_shapes(config, time_grid=time_grid)

    dim_mapping = config.dimension_mapping
    if dim_mapping:
        shapes = {dim_mapping.get(k, k): v for k, v in shapes.items()}

    # Step 5: Prepare forcings
    forcings, coords = _prepare_forcings(config, dim_mapping, shapes, time_grid, blueprint_dims)

    # Step 6: Prepare initial state
    state = _prepare_state(config, dim_mapping)

    # Step 7: Prepare parameters
    parameters = _prepare_parameters(config, dim_mapping)

    # Step 8: Build CompiledModel
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
        time_grid=time_grid,
    )
