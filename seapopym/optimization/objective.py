"""Observation objective for calibration.

An Objective groups observations with how to extract corresponding
predictions from model outputs.  Two modes (mutually exclusive):

- ``target``: auto-extraction by variable name via coords_to_indices.
- ``transform``: custom JAX callable for derived quantities.

No metric or likelihood — the consumer (Optimizer or Sampler) decides
**how** to compare predictions with observations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from seapopym.compiler.coords import coords_to_indices
from seapopym.types import Array

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr


@dataclass
class PreparedObjective:
    """Objective ready for JAX evaluation.

    Produced by :meth:`Objective.setup`.  Contains a pure function and
    a JAX array — both safe to pass through JIT boundaries.

    Attributes:
        extract_fn: ``outputs → predictions`` at observation locations.
        obs_array: Observation values as a JAX array.
    """

    extract_fn: Callable[[dict[str, Array]], Array]
    obs_array: Array


class Objective:
    """Observation data with extraction method.

    Args:
        observations: Observation data.
            *target mode*: ``xarray.DataArray`` (gridded) or
            ``pandas.DataFrame`` (sparse).
            *transform mode*: JAX / NumPy array.
        target: Name of the model output variable to extract.
        transform: Custom JAX function ``(outputs) → predictions``.

    Raises:
        ValueError: If neither or both of *target* / *transform* are given.

    Example::

        # Target mode — auto-extraction
        Objective(observations=obs_xr, target="biomass")

        # Transform mode — custom derived quantity
        Objective(
            observations=obs_jax,
            transform=lambda o: (o["biomass"][idx_t, idx_y, idx_x] > 0).astype(float),
        )
    """

    def __init__(
        self,
        observations: Any,
        *,
        target: str | None = None,
        transform: Callable[[dict[str, Array]], Array] | None = None,
    ) -> None:
        if target is None and transform is None:
            msg = "Objective requires either 'target' or 'transform'"
            raise ValueError(msg)
        if target is not None and transform is not None:
            msg = "'target' and 'transform' are mutually exclusive"
            raise ValueError(msg)

        self.observations = observations
        self.target = target
        self.transform = transform

    def __repr__(self) -> str:
        mode = f"target={self.target!r}" if self.target else "transform=<fn>"
        return f"Objective({mode})"

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(
        self,
        model_coords: dict[str, Array],
        dummy_outputs: dict[str, Array] | None = None,
    ) -> PreparedObjective:
        """Prepare the objective for JAX evaluation.

        Call once at calibration setup time (before JIT compilation).

        Args:
            model_coords: Coordinate arrays keyed by dimension name
                (typically ``CompiledModel.coords``).
            dummy_outputs: Abstract output pytree for shape validation
                (transform mode only).

        Returns:
            A :class:`PreparedObjective` with ``extract_fn`` and ``obs_array``.
        """
        if self.target is not None:
            return self._setup_target(model_coords)
        return self._setup_transform(dummy_outputs)

    # ------------------------------------------------------------------
    # Target mode
    # ------------------------------------------------------------------

    def _setup_target(self, model_coords: dict[str, Array]) -> PreparedObjective:
        import pandas as pd
        import xarray as xr

        if isinstance(self.observations, xr.DataArray):
            return self._setup_xarray(model_coords)
        if isinstance(self.observations, pd.DataFrame):
            return self._setup_dataframe(model_coords)

        msg = f"target mode requires xarray.DataArray or pandas.DataFrame, got {type(self.observations).__name__}"
        raise TypeError(msg)

    def _setup_xarray(self, model_coords: dict[str, Array]) -> PreparedObjective:
        """Gridded observations → subgrid extraction via ``jnp.ix_``."""
        da: xr.DataArray = self.observations

        # One index array per observation dimension
        dim_coords = {str(dim): da.coords[dim].values for dim in da.dims}
        indices = coords_to_indices(model_coords, sel_kwargs=None, **dim_coords)

        target_name: str = self.target  # type: ignore[assignment]
        idx = tuple(jnp.asarray(i, dtype=jnp.int32) for i in indices)

        def extract_fn(outputs: dict[str, Array]) -> Array:
            return outputs[target_name][jnp.ix_(*idx)]

        return PreparedObjective(
            extract_fn=extract_fn,
            obs_array=jnp.asarray(da.values),
        )

    def _setup_dataframe(self, model_coords: dict[str, Array]) -> PreparedObjective:
        """Sparse point observations → direct fancy indexing."""
        df: pd.DataFrame = self.observations
        target_name: str = self.target  # type: ignore[assignment]

        if target_name not in df.columns:
            msg = f"target column '{target_name}' not in DataFrame: {list(df.columns)}"
            raise ValueError(msg)

        obs_values = df[target_name].values

        # All columns except target are coordinate dimensions.
        # .to_numpy() ensures plain ndarray (not ExtensionArray).
        dim_coords = {col: df[col].to_numpy() for col in df.columns if col != target_name}
        indices = coords_to_indices(model_coords, sel_kwargs=None, **dim_coords)

        idx = tuple(jnp.asarray(i, dtype=jnp.int32) for i in indices)

        def extract_fn(outputs: dict[str, Array]) -> Array:
            return outputs[target_name][idx]

        return PreparedObjective(
            extract_fn=extract_fn,
            obs_array=jnp.asarray(obs_values),
        )

    # ------------------------------------------------------------------
    # Transform mode
    # ------------------------------------------------------------------

    def _setup_transform(self, dummy_outputs: dict[str, Array] | None) -> PreparedObjective:
        """Validate shape and wrap the user-provided transform."""
        obs_array = jnp.asarray(self.observations)
        transform = self.transform  # guaranteed not None by __init__
        if transform is None:  # unreachable, but satisfies type checker
            msg = "transform is None"
            raise RuntimeError(msg)

        if dummy_outputs is not None:
            pred_shape = jax.eval_shape(transform, dummy_outputs).shape
            if pred_shape != obs_array.shape:
                msg = f"transform output shape {pred_shape} != observations shape {obs_array.shape}"
                raise ValueError(msg)

        return PreparedObjective(
            extract_fn=transform,
            obs_array=obs_array,
        )
