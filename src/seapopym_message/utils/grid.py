"""Grid utilities for 2D spatial domains.

This module provides a hierarchy of GridInfo dataclasses for managing
different types of grids (spherical, plane, etc.) with their specific
coordinate systems and metric calculations.
"""

from abc import ABC
from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class GridInfo(ABC):
    """Base class for grid information.

    All grid types share common properties (nlat, nlon) but have
    different coordinate systems and metric calculations.

    Args:
        nlat: Number of cells in the first spatial dimension (latitude/y).
        nlon: Number of cells in the second spatial dimension (longitude/x).
    """

    nlat: int
    nlon: int


@dataclass
class SphericalGridInfo(GridInfo):
    """Information about a 2D latitude/longitude spherical grid.

    This class encapsulates geographic grid geometry and provides convenient
    properties for coordinate arrays and grid spacing on a spherical domain.

    Args:
        lat_min: Minimum latitude [degrees].
        lat_max: Maximum latitude [degrees].
        lon_min: Minimum longitude [degrees].
        lon_max: Maximum longitude [degrees].
        nlat: Number of latitude cells.
        nlon: Number of longitude cells.

    Example:
        >>> grid = SphericalGridInfo(
        ...     lat_min=-10.0, lat_max=10.0,
        ...     lon_min=140.0, lon_max=180.0,
        ...     nlat=20, nlon=40
        ... )
        >>> grid.dlat  # Latitude spacing in degrees
        1.0
        >>> grid.dx  # Approximate zonal spacing in meters (at equator)
        111320.0
    """

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    @property
    def lat_coords(self) -> jnp.ndarray:
        """Cell-centered latitude coordinates [degrees].

        Returns:
            Array of shape (nlat,) with latitude values.
        """
        return jnp.linspace(self.lat_min, self.lat_max, self.nlat)

    @property
    def lon_coords(self) -> jnp.ndarray:
        """Cell-centered longitude coordinates [degrees].

        Returns:
            Array of shape (nlon,) with longitude values.
        """
        return jnp.linspace(self.lon_min, self.lon_max, self.nlon)

    @property
    def dlat(self) -> float:
        """Latitude spacing [degrees].

        Returns:
            Spacing between latitude cells in degrees.
        """
        return (self.lat_max - self.lat_min) / (self.nlat - 1) if self.nlat > 1 else 0.0

    @property
    def dlon(self) -> float:
        """Longitude spacing [degrees].

        Returns:
            Spacing between longitude cells in degrees.
        """
        return (self.lon_max - self.lon_min) / (self.nlon - 1) if self.nlon > 1 else 0.0

    @property
    def dy(self) -> float:
        """Meridional grid spacing [meters].

        Converts latitude spacing from degrees to meters.
        Uses: 1 degree latitude ≈ 111,320 meters (constant).

        Returns:
            Meridional spacing in meters.
        """
        return self.dlat * 111320.0  # meters per degree latitude

    @property
    def dx(self) -> float:
        """Zonal grid spacing [meters] at mean latitude.

        Converts longitude spacing from degrees to meters,
        accounting for spherical geometry using mean latitude.
        Uses: 1 degree longitude = 111,320 * cos(lat) meters.

        Returns:
            Zonal spacing in meters at the grid's mean latitude.
        """
        mean_lat = (self.lat_min + self.lat_max) / 2.0
        # Convert to radians for cos
        mean_lat_rad = jnp.deg2rad(mean_lat)
        return float(self.dlon * 111320.0 * jnp.cos(mean_lat_rad))

    def get_meshgrid(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Generate 2D coordinate meshgrids.

        Returns:
            Tuple of (LAT, LON) arrays, each of shape (nlat, nlon).
            LAT[i,j] contains the latitude of cell (i,j).
            LON[i,j] contains the longitude of cell (i,j).

        Example:
            >>> grid = GridInfo(0, 10, 0, 20, 5, 10)
            >>> LAT, LON = grid.get_meshgrid()
            >>> LAT.shape, LON.shape
            ((5, 10), (5, 10))
        """
        lon_1d = self.lon_coords
        lat_1d = self.lat_coords
        LON, LAT = jnp.meshgrid(lon_1d, lat_1d)
        return LAT, LON

    def __repr__(self) -> str:
        """String representation of SphericalGridInfo."""
        return (
            f"SphericalGridInfo(lat=[{self.lat_min:.2f}, {self.lat_max:.2f}], "
            f"lon=[{self.lon_min:.2f}, {self.lon_max:.2f}], "
            f"shape=({self.nlat}, {self.nlon}), "
            f"dx={self.dx/1000:.1f}km, dy={self.dy/1000:.1f}km)"
        )


@dataclass
class PlaneGridInfo(GridInfo):
    """Information about a 2D Cartesian (plane) grid.

    This class encapsulates plane grid geometry with uniform spacing
    in both directions. Suitable for idealized simulations or local
    domains where spherical geometry can be approximated as flat.

    Args:
        dx: Grid spacing in x/longitude direction [meters].
        dy: Grid spacing in y/latitude direction [meters].
        nlat: Number of cells in y direction.
        nlon: Number of cells in x direction.

    Example:
        >>> grid = PlaneGridInfo(dx=10e3, dy=10e3, nlat=100, nlon=100)
        >>> grid.dx
        10000.0
        >>> grid.total_area_km2
        10000.0
    """

    dx: float
    dy: float

    @property
    def total_area(self) -> float:
        """Total domain area [m²].

        Returns:
            Total area of the domain in square meters.
        """
        return self.nlat * self.nlon * self.dx * self.dy

    @property
    def total_area_km2(self) -> float:
        """Total domain area [km²].

        Returns:
            Total area of the domain in square kilometers.
        """
        return self.total_area / 1e6

    @property
    def domain_length_x(self) -> float:
        """Domain length in x direction [meters].

        Returns:
            Total length of domain in x direction.
        """
        return self.nlon * self.dx

    @property
    def domain_length_y(self) -> float:
        """Domain length in y direction [meters].

        Returns:
            Total length of domain in y direction.
        """
        return self.nlat * self.dy

    def __repr__(self) -> str:
        """String representation of PlaneGridInfo."""
        return (
            f"PlaneGridInfo(shape=({self.nlat}, {self.nlon}), "
            f"dx={self.dx/1000:.1f}km, dy={self.dy/1000:.1f}km, "
            f"domain={self.domain_length_x/1000:.1f}×{self.domain_length_y/1000:.1f}km)"
        )
