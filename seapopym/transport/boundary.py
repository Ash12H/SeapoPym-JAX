"""Boundary conditions for transport computations.

This module provides boundary condition types and utilities for handling
domain edges in advection and diffusion schemes.

References:
    - Original implementation: IA/transport/boundary.py
    - Boundary handling in advection: IA/transport/advection.py
    - Boundary handling in diffusion: IA/transport/diffusion.py
"""

from dataclasses import dataclass
from enum import Enum

import xarray as xr

from seapopym.standard.coordinates import Coordinates


class BoundaryType(Enum):
    """Types of boundary conditions for transport schemes.

    CLOSED: No-flux boundary (Neumann BC: ∂C/∂n = 0)
        - Used at solid boundaries (coastlines, domain edges)
        - Neighbor value = current cell value
        - Prevents mass loss/gain at boundary

    PERIODIC: Wrap-around boundary
        - Used for longitude in global models (0° = 360°)
        - West neighbor of first cell = last cell
        - East neighbor of last cell = first cell

    OPEN: Zero-gradient boundary
        - Used at open ocean boundaries
        - Similar to CLOSED but conceptually different
        - Allows advective flux but not diffusive flux
    """

    CLOSED = "closed"
    PERIODIC = "periodic"
    OPEN = "open"


@dataclass(frozen=True)
class BoundaryConditions:
    """Boundary conditions for all four domain edges.

    Attributes:
        north: Boundary type at northern edge (max latitude)
        south: Boundary type at southern edge (min latitude)
        east: Boundary type at eastern edge (max longitude)
        west: Boundary type at western edge (min longitude)

    Example:
        >>> # Global ocean model: closed poles, periodic longitude
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


def get_neighbors_with_bc(
    data: xr.DataArray,
    boundary: BoundaryConditions,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Get neighbor values with boundary condition handling.

    This function returns the four neighbor arrays (west, east, south, north)
    with proper handling of domain boundaries according to the specified
    boundary conditions.

    Uses xarray.shift() for efficient neighbor access:
    - shift(lon=1): shift data to the right → get west neighbor
    - shift(lon=-1): shift data to the left → get east neighbor
    - shift(lat=1): shift data up → get south neighbor
    - shift(lat=-1): shift data down → get north neighbor

    Boundary conditions are applied by filling NaN values introduced by shift:
    - CLOSED/OPEN: Fill with current cell value (∂C/∂n = 0)
    - PERIODIC: Use xarray's roll instead of shift

    Args:
        data: DataArray with dimensions (lat, lon) or (..., lat, lon)
        boundary: BoundaryConditions object specifying edge behavior

    Returns:
        Tuple of (west, east, south, north) neighbor arrays with same shape as data

    Example:
        >>> data = xr.DataArray(np.ones((10, 20)), dims=["y", "x"])
        >>> bc = BoundaryConditions(CLOSED, CLOSED, PERIODIC, PERIODIC)
        >>> west, east, south, north = get_neighbors_with_bc(data, bc)
    """
    # Use standardized coordinate names from Coordinates enum
    dim_y = Coordinates.Y.value  # "y"
    dim_x = Coordinates.X.value  # "x"

    # East-West neighbors
    if boundary.east == BoundaryType.PERIODIC and boundary.west == BoundaryType.PERIODIC:
        # Use roll for periodic boundaries (no NaN introduced)
        data_west = data.roll({dim_x: 1}, roll_coords=False)
        data_east = data.roll({dim_x: -1}, roll_coords=False)
    else:
        # Use shift for non-periodic (introduces NaN at boundaries)
        data_west = data.shift({dim_x: 1})
        data_east = data.shift({dim_x: -1})

        # Fill NaN at boundaries
        if boundary.west in (BoundaryType.CLOSED, BoundaryType.OPEN):
            # West boundary: fill NaN with current value
            data_west = data_west.fillna(data)
        if boundary.east in (BoundaryType.CLOSED, BoundaryType.OPEN):
            # East boundary: fill NaN with current value
            data_east = data_east.fillna(data)

    # North-South neighbors
    if boundary.north == BoundaryType.PERIODIC and boundary.south == BoundaryType.PERIODIC:
        # Use roll for periodic boundaries
        data_south = data.roll({dim_y: 1}, roll_coords=False)
        data_north = data.roll({dim_y: -1}, roll_coords=False)
    else:
        # Use shift for non-periodic
        data_south = data.shift({dim_y: 1})
        data_north = data.shift({dim_y: -1})

        # Fill NaN at boundaries
        if boundary.south in (BoundaryType.CLOSED, BoundaryType.OPEN):
            # South boundary: fill NaN with current value
            data_south = data_south.fillna(data)
        if boundary.north in (BoundaryType.CLOSED, BoundaryType.OPEN):
            # North boundary: fill NaN with current value
            data_north = data_north.fillna(data)

    return data_west, data_east, data_south, data_north
