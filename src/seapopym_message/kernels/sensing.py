"""Sensing Units: Perception of the environment.

This module defines Units that transform raw environmental forcings (potentially N-dimensional)
into effective fields perceived by the biological entities (2D).
"""

import jax.numpy as jnp

from seapopym_message.core.unit import unit


@unit(
    name="extract_layer",
    inputs=[],  # No state inputs needed
    outputs=["forcing_2d"],
    scope="local",
    forcings=["forcing_3d"],
)
def extract_layer(
    dt: float,  # noqa: ARG001
    params: dict,
    forcings: dict,
) -> jnp.ndarray:
    """Extract a specific layer from a 3D forcing field.

    Args:
        dt: Time step (unused).
        params: Dictionary containing:
            - layer_index: Integer index of the depth layer to extract.
        forcings: Dictionary containing:
            - forcing_3d: 3D field (depth, lat, lon).

    Returns:
        2D field (lat, lon) at the specified layer.
    """
    forcing_3d = forcings["forcing_3d"]
    layer_index = params["layer_index"]

    # Extract layer
    return forcing_3d[layer_index]


@unit(
    name="diel_migration",
    inputs=[],
    outputs=["forcing_effective"],
    scope="local",
    forcings=["forcing_3d", "day_length"],
)
def diel_migration(
    dt: float,  # noqa: ARG001
    params: dict,
    forcings: dict,
) -> jnp.ndarray:
    """Compute effective forcing based on diel vertical migration.

    Calculates a weighted average between day and night layers based on day length.

    Equation:
        F_eff = F[z_day] * day_length + F[z_night] * (1 - day_length)

    Args:
        dt: Time step (unused).
        params: Dictionary containing:
            - day_layer_index: Integer index of the day depth layer.
            - night_layer_index: Integer index of the night depth layer.
        forcings: Dictionary containing:
            - forcing_3d: 3D field (depth, lat, lon).
            - day_length: Fraction of day (0-1), shape (lat, lon).

    Returns:
        Effective 2D field (lat, lon).
    """
    forcing_3d = forcings["forcing_3d"]
    day_length = forcings["day_length"]

    day_idx = params["day_layer_index"]
    night_idx = params["night_layer_index"]

    val_day = forcing_3d[day_idx]
    val_night = forcing_3d[night_idx]

    return val_day * day_length + val_night * (1.0 - day_length)
