"""Shape inference from data metadata.

Infers dimension sizes from xr.DataArray dims/sizes without
loading data into memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from seapopym.blueprint import Config

if TYPE_CHECKING:
    from seapopym.compiler.time_grid import TimeGrid

from .exceptions import GridAlignmentError


def infer_shapes(
    config: Config,
    time_grid: TimeGrid | None = None,
) -> dict[str, int]:
    """Infer all dimension sizes from config data sources.

    All data (forcings, initial_state, parameters) are xr.DataArray,
    so dimensions are read directly from .dims and .sizes.

    If time_grid is provided, the time dimension is computed from the
    temporal grid configuration rather than inferred from data sources.

    Args:
        config: Configuration with forcings, initial_state, and parameters.
        time_grid: Optional TimeGrid with computed n_timesteps.

    Returns:
        Dict mapping dimension names to sizes.

    Raises:
        GridAlignmentError: If same dimension has different sizes in different sources.
    """
    time_dim = config.execution.time_dim
    shapes: dict[str, int] = {}
    size_sources: dict[str, dict[str, int]] = {}

    # If time_grid provided, set time dimension from it (priority over data inference)
    if time_grid is not None:
        shapes[time_dim] = time_grid.n_timesteps
        size_sources[time_dim] = {"time_grid": time_grid.n_timesteps}

    def record_shape(source_name: str, dim: str, size: int) -> None:
        """Record a dimension size and check for conflicts."""
        if dim == time_dim and time_grid is not None:
            return

        if dim not in size_sources:
            size_sources[dim] = {}
        size_sources[dim][source_name] = size

        if dim in shapes and shapes[dim] != size:
            raise GridAlignmentError(dim, size_sources[dim])
        shapes[dim] = size

    # Infer from forcings
    for name, da in config.forcings.items():
        for dim, size in zip(da.dims, da.shape, strict=True):
            record_shape(f"forcings.{name}", str(dim), size)

    # Infer from initial_state (flat dict)
    for name, da in config.initial_state.items():
        for dim, size in zip(da.dims, da.shape, strict=True):
            record_shape(f"initial_state.{name}", str(dim), size)

    # Infer from parameters
    for name, da in config.parameters.items():
        for dim, size in zip(da.dims, da.shape, strict=True):
            record_shape(f"parameters.{name}", str(dim), size)

    return shapes
