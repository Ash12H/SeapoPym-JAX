"""Sensing Units: Perception of the environment.

This module defines Units that transform raw environmental forcings (potentially N-dimensional)
into effective fields perceived by the biological entities (2D).

These units use xarray DataArrays to preserve dimension metadata and allow robust
dimension selection by name (not index).
"""

import xarray as xr

from seapopym_message.core.unit import unit


@unit(
    name="extract_layer",
    inputs=[],
    outputs=["forcing_2d"],
    scope="local",
    compiled=False,  # xarray not compatible with JIT
    forcings=["forcing_nd"],
)
def extract_layer(
    dt: float,  # noqa: ARG001
    params: dict,
    forcings: dict[str, xr.DataArray],
) -> xr.DataArray:
    """Extract a specific layer from an N-D forcing field using dimension metadata.

    Uses xarray dimension selection by name for robustness.
    Supports common depth dimension names: 'depth', 'Z', 'z', 'lev', 'level'.

    Args:
        dt: Time step (unused).
        params: Dictionary containing:
            - layer_index: Integer index of the layer to extract.
            - dimension: Optional dimension name (default: auto-detect from known depth dims).
        forcings: Dictionary containing:
            - forcing_nd: N-D DataArray with dimension metadata.

    Returns:
        2D DataArray with the depth dimension removed.

    Example:
        >>> import xarray as xr
        >>> import numpy as np
        >>> temp_3d = xr.DataArray(
        ...     np.random.rand(3, 5, 5),
        ...     dims=("depth", "lat", "lon")
        ... )
        >>> forcings = {"forcing_nd": temp_3d}
        >>> params = {"layer_index": 0}
        >>> temp_2d = extract_layer(0, params, forcings)
        >>> temp_2d.dims  # ('lat', 'lon')
    """
    forcing_nd = forcings["forcing_nd"]
    layer_index = params["layer_index"]

    # Auto-detect depth dimension if not specified
    depth_dim_names = ["depth", "Z", "z", "lev", "level"]
    dim_to_extract = params.get("dimension", None)

    if dim_to_extract is None:
        # Auto-detect
        for dim_name in depth_dim_names:
            if dim_name in forcing_nd.dims:
                dim_to_extract = dim_name
                break
        if dim_to_extract is None:
            raise ValueError(
                f"No depth dimension found in {forcing_nd.dims}. "
                f"Expected one of: {depth_dim_names}"
            )

    # Select layer using xarray (robust!)
    return forcing_nd.isel({dim_to_extract: layer_index})


@unit(
    name="diel_migration",
    inputs=[],
    outputs=["forcing_effective"],
    scope="local",
    compiled=False,  # xarray not compatible with JIT
    forcings=["forcing_nd", "day_length"],
)
def diel_migration(
    dt: float,  # noqa: ARG001
    params: dict,
    forcings: dict[str, xr.DataArray],
) -> xr.DataArray:
    """Compute effective forcing based on diel vertical migration.

    Calculates a weighted average between day and night layers based on day length.
    Uses xarray for robust dimension selection.

    Equation:
        F_eff = F[z_day] * day_length + F[z_night] * (1 - day_length)

    Args:
        dt: Time step (unused).
        params: Dictionary containing:
            - day_layer_index: Integer index of the day depth layer.
            - night_layer_index: Integer index of the night depth layer.
            - dimension: Optional dimension name (default: auto-detect).
        forcings: Dictionary containing:
            - forcing_nd: N-D DataArray (depth, lat, lon) or similar.
            - day_length: Fraction of day (0-1), DataArray or array (lat, lon).

    Returns:
        Effective 2D DataArray (lat, lon).
    """
    forcing_nd = forcings["forcing_nd"]
    day_length = forcings["day_length"]

    day_idx = params["day_layer_index"]
    night_idx = params["night_layer_index"]

    # Auto-detect depth dimension if not specified
    depth_dim_names = ["depth", "Z", "z", "lev", "level"]
    dim_to_extract = params.get("dimension", None)

    if dim_to_extract is None:
        # Auto-detect
        for dim_name in depth_dim_names:
            if dim_name in forcing_nd.dims:
                dim_to_extract = dim_name
                break
        if dim_to_extract is None:
            raise ValueError(
                f"No depth dimension found in {forcing_nd.dims}. "
                f"Expected one of: {depth_dim_names}"
            )

    # Extract layers using xarray
    val_day = forcing_nd.isel({dim_to_extract: day_idx})
    val_night = forcing_nd.isel({dim_to_extract: night_idx})

    # Get day_length values (handle both DataArray and ndarray)
    day_length_vals = day_length.values if isinstance(day_length, xr.DataArray) else day_length

    return val_day * day_length_vals + val_night * (1.0 - day_length_vals)
