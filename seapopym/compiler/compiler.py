"""Main Compiler class for transforming Blueprint + Config into CompiledModel.

The Compiler orchestrates the full compilation pipeline:
1. Infer shapes from data metadata
2. Rename dimensions to canonical names
3. Transpose to canonical order
4. Strip xarray and preprocess NaN
5. Package into CompiledModel
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

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


@dataclass
class TimeGrid:
    """Temporal grid configuration computed from user parameters.

    This class represents the temporal discretization of a simulation,
    computed from explicit start/end times and timestep duration.

    Attributes:
        start: Simulation start time (datetime64).
        end: Simulation end time (datetime64).
        dt_seconds: Timestep duration in seconds.
        n_timesteps: Number of timesteps in the simulation.
        coords: Temporal coordinates array (datetime64, shape: (n_timesteps,)).

    Example:
        >>> grid = TimeGrid.from_config("2000-01-01", "2000-01-10", "1d")
        >>> grid.n_timesteps
        9
        >>> grid.coords[0]
        numpy.datetime64('2000-01-01')
    """

    start: np.datetime64
    end: np.datetime64
    dt_seconds: float
    n_timesteps: int
    coords: np.ndarray  # dtype=datetime64[ns]

    @classmethod
    def from_config(cls, time_start: str, time_end: str, dt_str: str) -> TimeGrid:
        """Compute temporal grid from configuration strings.

        Args:
            time_start: Start time (ISO format, e.g., "2000-01-01").
            time_end: End time (ISO format, e.g., "2020-12-31").
            dt_str: Timestep duration (e.g., "1d", "0.05d", "6h").

        Returns:
            TimeGrid instance with computed n_timesteps and coordinates.

        Raises:
            ValueError: If time_end <= time_start, or if time range is not
                evenly divisible by dt (remainder > 1 second).

        Example:
            >>> grid = TimeGrid.from_config("2000-01-01", "2000-01-10", "1d")
            >>> grid.n_timesteps
            9
        """
        # 1. Parse dates
        start_pd = pd.to_datetime(time_start)
        end_pd = pd.to_datetime(time_end)

        if end_pd <= start_pd:
            raise ValueError(f"time_end ({time_end}) must be after time_start ({time_start})")

        start_dt64 = start_pd.to_datetime64()
        end_dt64 = end_pd.to_datetime64()

        # 2. Parse timestep
        dt_seconds = _parse_dt(dt_str)
        dt_td = pd.Timedelta(seconds=dt_seconds)

        # 3. Compute number of timesteps
        duration = end_pd - start_pd
        n_timesteps_float = duration / dt_td

        # Round to nearest integer
        n_timesteps = int(np.round(n_timesteps_float))

        # 4. Validate: remainder should be negligible (< 1 second)
        expected_duration = n_timesteps * dt_td
        remainder = abs(duration - expected_duration)

        if remainder > pd.Timedelta(seconds=1):
            raise ValueError(
                f"Time range [{time_start}, {time_end}] is not evenly divisible by dt={dt_str}. "
                f"Duration: {duration}, n_timesteps*dt: {expected_duration}, remainder: {remainder}. "
                f"Adjust time_end or dt to ensure exact alignment."
            )

        # 5. Generate temporal coordinates
        # Use freq instead of periods to avoid inclusive end (want [start, end))
        # Generate exactly n_timesteps starting from start with spacing dt
        coords = pd.date_range(start=start_pd, periods=n_timesteps, freq=dt_td).to_numpy()

        return cls(
            start=start_dt64,
            end=end_dt64,
            dt_seconds=dt_seconds,
            n_timesteps=n_timesteps,
            coords=coords,
        )


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

        # Step 2: Compute temporal grid from execution params
        time_grid = TimeGrid.from_config(
            config.execution.time_start,
            config.execution.time_end,
            config.execution.dt,
        )

        # Step 3: Build dims mapping from blueprint
        blueprint_dims = self._extract_blueprint_dims(blueprint)

        # Step 4: Infer shapes from data (with time_grid priority for T dimension)
        shapes = infer_shapes(config, blueprint_dims, time_grid=time_grid)

        # Step 5: Get dimension mapping from config and apply to shapes
        dim_mapping = config.dimension_mapping
        if dim_mapping:
            shapes = {dim_mapping.get(k, k): v for k, v in shapes.items()}

        # Step 6: Prepare forcings (with temporal validation)
        forcings, coords = self._prepare_forcings(config, dim_mapping, shapes, time_grid)

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
            time_grid=time_grid,
            batch_size=config.execution.batch_size,
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
        time_grid: TimeGrid,
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        """Prepare forcing arrays with temporal validation and interpolation.

        Forcings are processed as follows:
        1. Static forcings (no T dimension): kept as-is (broadcasted at runtime)
        2. Temporally aligned forcings (shape[0] == n_timesteps): kept as-is
        3. Under-sampled forcings (shape[0] < n_timesteps): interpolated

        The interpolation method is specified in config.execution.forcing_interpolation.

        Args:
            config: Configuration with forcings data.
            dim_mapping: Optional dimension renaming map.
            shapes: Inferred dimension sizes (including T from time_grid).
            time_grid: Temporal grid for validation and coord generation.

        Returns:
            Tuple of (forcings dict, coords dict) with temporal coords added.

        Raises:
            ValueError: If a forcing's temporal range does not cover the
                simulation range defined by time_grid.
        """
        forcings: dict[str, Array] = {}
        coords: dict[str, Array] = {}
        coords_extracted = False
        n_timesteps = shapes.get("T", 1)
        interp_method = config.execution.forcing_interpolation

        for name, source in config.forcings.items():
            arr, _dims, _mask = prepare_array(
                source,
                dimension_mapping=dim_mapping,
                backend=self.backend,
                fill_nan=self.fill_nan,
            )

            # Validate temporal range if forcing has time coordinates
            if hasattr(source, "coords") and "T" in source.coords:
                forcing_coords = source.coords["T"]
                forcing_start = forcing_coords.values[0]
                forcing_end = forcing_coords.values[-1]

                # Check if forcing covers simulation range
                if time_grid.start < forcing_start or time_grid.end > forcing_end:
                    raise ValueError(
                        f"Forcing '{name}' temporal range [{forcing_start}, {forcing_end}] "
                        f"does not cover simulation range [{time_grid.start}, {time_grid.end}]. "
                        f"Ensure forcing data spans the entire simulation period."
                    )

            # Check if temporal interpolation is needed
            # Only interpolate if:
            # 1. Array is valid
            # 2. Time dimension is present (first canonical dimension)
            # 3. Timesteps don't match
            # 4. Interpolation is enabled
            time_dim = "T"
            if (
                arr.ndim > 0 and time_dim in _dims and arr.shape[0] != n_timesteps and interp_method != "constant"
            ):  # Forcing has a time dimension but doesn't match expected timesteps
                # Apply interpolation
                arr = self._interpolate_forcing(arr, arr.shape[0], n_timesteps, interp_method)

            forcings[name] = arr

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

        # Add temporal coordinates from time_grid
        coords["T"] = time_grid.coords

        return forcings, coords

    def _interpolate_forcing(
        self,
        forcing: Array,
        source_len: int,
        target_len: int,
        method: str,
    ) -> Array:
        """Interpolate forcing data to target number of timesteps.

        Args:
            forcing: Original forcing array (shape: (T_source, ...)).
            source_len: Source number of timesteps.
            target_len: Target number of timesteps.
            method: Interpolation method ("nearest", "linear", "ffill").

        Returns:
            Interpolated forcing (shape: (T_target, ...)).
        """
        import numpy as np

        # Convert to numpy for interpolation (if JAX)
        forcing_np = np.asarray(forcing)

        # Source and target indices
        source_indices = np.arange(source_len)
        target_indices = np.linspace(0, source_len - 1, target_len)

        if method == "nearest":
            # Nearest neighbor interpolation
            nearest_idx = np.round(target_indices).astype(int)
            nearest_idx = np.clip(nearest_idx, 0, source_len - 1)
            result = forcing_np[nearest_idx]

        elif method == "linear":
            # Linear interpolation
            from scipy.interpolate import interp1d

            # Interpolate along first axis (time)
            # Note: fill_value="extrapolate" allows extrapolation beyond bounds
            f = interp1d(source_indices, forcing_np, axis=0, kind="linear", fill_value="extrapolate")  # type: ignore[arg-type]
            result = f(target_indices)

        elif method == "ffill":
            # Forward fill: repeat each value until next sample
            # Map target indices to source indices (floor)
            ffill_idx = np.floor(target_indices).astype(int)
            ffill_idx = np.clip(ffill_idx, 0, source_len - 1)
            result = forcing_np[ffill_idx]

        else:
            # Should not happen (validated by Pydantic)
            raise ValueError(f"Unknown interpolation method: {method}")

        # Convert back to target backend
        if self.backend == "jax":
            import jax.numpy as jnp

            return jnp.asarray(result)
        else:
            return result

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
                    arr, _dims, _mask = prepare_array(
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
