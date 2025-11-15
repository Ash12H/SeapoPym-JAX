"""Boundary conditions for transport computations.

This module defines boundary condition types and utilities for applying them
to scalar fields. Boundary conditions control behavior at domain edges:
- CLOSED: No-flux walls (Neumann BC)
- PERIODIC: Wrap-around (cyclic)
- OPEN: Zero-gradient outflow (similar to CLOSED for scalars)

References:
    - Neumann BC for diffusion: IA/Diffusion-euler-explicite-description.md
      (lines 47-58: CLOSED, lines 65-81: PERIODIC)
    - Implementation approach: IA/TRANSPORT_ANALYSIS.md
"""

from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp


class BoundaryType(Enum):
    """Types of boundary conditions for domain edges.

    Attributes:
        CLOSED: No-flux boundary (Neumann BC). Ghost cell values equal edge cell values.
                Used for physical walls or closed domain edges.
                Implementation: C_ghost = C_edge → ∂C/∂n = 0

        PERIODIC: Cyclic wrap-around. Used for global domains (e.g., longitude).
                  Implementation: C_west_ghost = C_east_edge, C_east_ghost = C_west_edge

        OPEN: Zero-gradient outflow boundary. For scalar transport, this is equivalent
              to CLOSED (∂C/∂n = 0).
              Implementation: C_ghost = C_edge
    """

    CLOSED = "closed"
    PERIODIC = "periodic"
    OPEN = "open"


@dataclass(frozen=True)
class BoundaryConditions:
    """Boundary conditions for all four edges of a 2D domain.

    Attributes:
        north: Boundary type for northern edge (top, j=nlat)
        south: Boundary type for southern edge (bottom, j=0)
        east: Boundary type for eastern edge (right, i=nlon)
        west: Boundary type for western edge (left, i=0)

    Example:
        >>> # Global ocean: periodic in longitude, closed at poles
        >>> bc = BoundaryConditions(
        ...     north=BoundaryType.CLOSED,
        ...     south=BoundaryType.CLOSED,
        ...     east=BoundaryType.PERIODIC,
        ...     west=BoundaryType.PERIODIC,
        ... )
    """

    north: BoundaryType
    south: BoundaryType
    east: BoundaryType
    west: BoundaryType

    def __post_init__(self) -> None:
        """Validate boundary conditions."""
        # Check that east/west are consistent for periodicity
        if (
            self.east == BoundaryType.PERIODIC or self.west == BoundaryType.PERIODIC
        ) and self.east != self.west:
            raise ValueError(
                "East and West boundaries must both be PERIODIC or neither. "
                f"Got east={self.east}, west={self.west}"
            )


def apply_boundary_conditions(
    field: jnp.ndarray,
    bc: BoundaryConditions,
) -> jnp.ndarray:
    """Apply boundary conditions to a 2D field by creating ghost cells.

    This function adds a 1-cell halo around the input field and fills ghost cells
    according to the specified boundary conditions. The resulting array has shape
    (nlat+2, nlon+2) where the interior [1:-1, 1:-1] contains the original field.

    Args:
        field: Input 2D array of shape (nlat, nlon)
        bc: Boundary conditions to apply

    Returns:
        Array of shape (nlat+2, nlon+2) with ghost cells filled

    Mathematical implementation:
        - CLOSED/OPEN: C[0, :] = C[1, :] (south), C[-1, :] = C[-2, :] (north)
                       C[:, 0] = C[:, 1] (west),  C[:, -1] = C[:, -2] (east)
        - PERIODIC (E/W): C[:, 0] = C[:, -2] (west), C[:, -1] = C[:, 1] (east)

    Reference:
        - IA/Diffusion-euler-explicite-description.md:
          Lines 50-58 (CLOSED), Lines 76-80 (PERIODIC)

    Example:
        >>> field = jnp.ones((10, 20))
        >>> bc = BoundaryConditions(
        ...     BoundaryType.CLOSED, BoundaryType.CLOSED,
        ...     BoundaryType.PERIODIC, BoundaryType.PERIODIC
        ... )
        >>> field_with_ghosts = apply_boundary_conditions(field, bc)
        >>> field_with_ghosts.shape
        (12, 22)
        >>> jnp.array_equal(field_with_ghosts[1:-1, 1:-1], field)
        True
    """
    nlat, nlon = field.shape

    # Create array with halo (1 ghost cell on each side)
    field_with_ghosts = jnp.zeros((nlat + 2, nlon + 2), dtype=field.dtype)

    # Copy interior
    field_with_ghosts = field_with_ghosts.at[1:-1, 1:-1].set(field)

    # Apply South boundary (j=0)
    if bc.south == BoundaryType.CLOSED or bc.south == BoundaryType.OPEN:
        # Ghost cells equal edge cells: C[0, :] = C[1, :]
        field_with_ghosts = field_with_ghosts.at[0, 1:-1].set(field[0, :])
    # PERIODIC in N/S not typically used, but could be implemented here

    # Apply North boundary (j=nlat+1)
    if bc.north == BoundaryType.CLOSED or bc.north == BoundaryType.OPEN:
        # Ghost cells equal edge cells: C[-1, :] = C[-2, :]
        field_with_ghosts = field_with_ghosts.at[-1, 1:-1].set(field[-1, :])

    # Apply West boundary (i=0)
    if bc.west == BoundaryType.CLOSED or bc.west == BoundaryType.OPEN:
        # Ghost cells equal edge cells: C[:, 0] = C[:, 1]
        field_with_ghosts = field_with_ghosts.at[1:-1, 0].set(field[:, 0])
    elif bc.west == BoundaryType.PERIODIC:
        # Wrap from east: C[:, 0] = C[:, -1] (last interior cell)
        field_with_ghosts = field_with_ghosts.at[1:-1, 0].set(field[:, -1])

    # Apply East boundary (i=nlon+1)
    if bc.east == BoundaryType.CLOSED or bc.east == BoundaryType.OPEN:
        # Ghost cells equal edge cells: C[:, -1] = C[:, -2]
        field_with_ghosts = field_with_ghosts.at[1:-1, -1].set(field[:, -1])
    elif bc.east == BoundaryType.PERIODIC:
        # Wrap from west: C[:, -1] = C[:, 0] (first interior cell)
        field_with_ghosts = field_with_ghosts.at[1:-1, -1].set(field[:, 0])

    # Apply corners (simple average or copy from adjacent edge)
    # For corners, we use the edge values (simpler and sufficient for our schemes)

    # Southwest corner
    if bc.south in (BoundaryType.CLOSED, BoundaryType.OPEN):
        if bc.west in (BoundaryType.CLOSED, BoundaryType.OPEN):
            field_with_ghosts = field_with_ghosts.at[0, 0].set(field[0, 0])
        elif bc.west == BoundaryType.PERIODIC:
            field_with_ghosts = field_with_ghosts.at[0, 0].set(field[0, -1])

    # Southeast corner
    if bc.south in (BoundaryType.CLOSED, BoundaryType.OPEN):
        if bc.east in (BoundaryType.CLOSED, BoundaryType.OPEN):
            field_with_ghosts = field_with_ghosts.at[0, -1].set(field[0, -1])
        elif bc.east == BoundaryType.PERIODIC:
            field_with_ghosts = field_with_ghosts.at[0, -1].set(field[0, 0])

    # Northwest corner
    if bc.north in (BoundaryType.CLOSED, BoundaryType.OPEN):
        if bc.west in (BoundaryType.CLOSED, BoundaryType.OPEN):
            field_with_ghosts = field_with_ghosts.at[-1, 0].set(field[-1, 0])
        elif bc.west == BoundaryType.PERIODIC:
            field_with_ghosts = field_with_ghosts.at[-1, 0].set(field[-1, -1])

    # Northeast corner
    if bc.north in (BoundaryType.CLOSED, BoundaryType.OPEN):
        if bc.east in (BoundaryType.CLOSED, BoundaryType.OPEN):
            field_with_ghosts = field_with_ghosts.at[-1, -1].set(field[-1, -1])
        elif bc.east == BoundaryType.PERIODIC:
            field_with_ghosts = field_with_ghosts.at[-1, -1].set(field[-1, 0])

    return field_with_ghosts


def get_neighbors_with_bc(
    field: jnp.ndarray,
    bc: BoundaryConditions,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Get neighbor values for all cells with boundary conditions applied.

    This is a convenience function that applies boundary conditions and extracts
    neighbor arrays for use in stencil operations (advection, diffusion).

    Args:
        field: Input 2D array of shape (nlat, nlon)
        bc: Boundary conditions to apply

    Returns:
        Tuple of (west, east, south, north) neighbor arrays, each shape (nlat, nlon)
        - west[i, j] = field[i, j-1] with BC
        - east[i, j] = field[i, j+1] with BC
        - south[i, j] = field[i-1, j] with BC
        - north[i, j] = field[i+1, j] with BC

    Example:
        >>> field = jnp.arange(12).reshape(3, 4)
        >>> bc = BoundaryConditions(
        ...     BoundaryType.CLOSED, BoundaryType.CLOSED,
        ...     BoundaryType.PERIODIC, BoundaryType.PERIODIC
        ... )
        >>> west, east, south, north = get_neighbors_with_bc(field, bc)
        >>> west.shape == east.shape == south.shape == north.shape == (3, 4)
        True
    """
    # Apply boundary conditions to get field with ghost cells
    field_with_ghosts = apply_boundary_conditions(field, bc)

    # Extract neighbors (interior cells are at [1:-1, 1:-1])
    # Shape of each output: (nlat, nlon)
    west = field_with_ghosts[1:-1, 0:-2]  # Shift left
    east = field_with_ghosts[1:-1, 2:]  # Shift right
    south = field_with_ghosts[0:-2, 1:-1]  # Shift down
    north = field_with_ghosts[2:, 1:-1]  # Shift up

    return west, east, south, north
