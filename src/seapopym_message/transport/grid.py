"""Grid infrastructure for transport computations.

This module provides abstract grid representations for both spherical (lat/lon)
and plane (Cartesian) geometries. Grids pre-compute cell areas and face areas
needed for flux-based transport schemes.

References:
    - Spherical geometry: IA/Diffusion-euler-explicite-description.md (line 26)
    - Grid importance for conservation: IA/TRANSPORT_ANALYSIS.md (Section 8)
"""

from abc import ABC, abstractmethod

import jax.numpy as jnp


class Grid(ABC):
    """Abstract base class for computational grids.

    A Grid provides geometric information needed for flux-based transport:
    - Cell areas (for volume calculations)
    - Face areas (for flux calculations)
    - Grid spacing (for gradient/laplacian approximations)

    All implementations must pre-compute these quantities for efficiency.
    """

    @abstractmethod
    def cell_areas(self) -> jnp.ndarray:
        """Return areas of grid cells.

        Returns:
            Array of shape (nlat, nlon) with cell areas [m²]
        """
        pass

    @abstractmethod
    def face_areas_ew(self) -> jnp.ndarray:
        """Return areas of East/West faces (vertical faces).

        Returns:
            Array of shape (nlat, nlon+1) with E/W face areas [m²]
            Face i separates cells i-1 and i in longitude direction
        """
        pass

    @abstractmethod
    def face_areas_ns(self) -> jnp.ndarray:
        """Return areas of North/South faces (horizontal faces).

        Returns:
            Array of shape (nlat+1, nlon) with N/S face areas [m²]
            Face j separates cells j-1 and j in latitude direction
        """
        pass

    @abstractmethod
    def dx(self) -> jnp.ndarray | float:
        """Return grid spacing in x/longitude direction.

        Returns:
            For SphericalGrid: array of shape (nlat,) varying with latitude [m]
            For PlaneGrid: scalar constant spacing [m]
        """
        pass

    @abstractmethod
    def dy(self) -> float:
        """Return grid spacing in y/latitude direction.

        Returns:
            Constant spacing [m]
        """
        pass


class SphericalGrid(Grid):
    """Spherical lat/lon grid for oceanic transport.

    This grid accounts for the spherical geometry of the Earth:
    - Cell width dx(lat) decreases toward poles: dx(lat) = R·cos(lat)·dλ
    - Cell height dy is constant: dy = R·dφ
    - Cell areas vary with latitude: A(lat) = R²·cos(lat)·dλ·dφ

    Attributes:
        lat_min, lat_max: Latitude bounds [degrees]
        lon_min, lon_max: Longitude bounds [degrees]
        nlat, nlon: Number of cells in each direction
        R: Earth radius [m], default 6371e3

    Mathematical formulas (IA/Diffusion-euler-explicite-description.md, line 26):
        dx(j) = R × cos(lat[j]) × dλ    [m]  (varies with latitude)
        dy    = R × dφ                   [m]  (constant)

    Example:
        >>> grid = SphericalGrid(-60, 60, 0, 360, nlat=120, nlon=360)
        >>> areas = grid.cell_areas()  # Shape (120, 360)
        >>> areas.min() / areas.max()  # Ratio ~ cos(60°) ≈ 0.5 (poles vs equator)
    """

    def __init__(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        nlat: int,
        nlon: int,
        R: float = 6371e3,
    ):
        """Initialize spherical grid.

        Args:
            lat_min: Minimum latitude [degrees], typically -90 or -60
            lat_max: Maximum latitude [degrees], typically 90 or 60
            lon_min: Minimum longitude [degrees], typically 0
            lon_max: Maximum longitude [degrees], typically 360
            nlat: Number of latitude cells
            nlon: Number of longitude cells
            R: Earth radius [m], default 6371e3
        """
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.nlat = nlat
        self.nlon = nlon
        self.R = R

        # Compute grid spacing in degrees
        self.dlat = (lat_max - lat_min) / nlat
        self.dlon = (lon_max - lon_min) / nlon

        # Cell centers
        self.lat = jnp.linspace(lat_min + self.dlat / 2, lat_max - self.dlat / 2, nlat)
        self.lon = jnp.linspace(lon_min + self.dlon / 2, lon_max - self.dlon / 2, nlon)

        # Convert to radians for geometry
        lat_rad = jnp.radians(self.lat)
        dlat_rad = jnp.radians(self.dlat)
        dlon_rad = jnp.radians(self.dlon)

        # Pre-compute geometric quantities
        # Cell areas: A = R² × cos(lat) × dλ × dφ
        # Need to broadcast from (nlat, 1) to (nlat, nlon)
        cell_area_by_lat = (
            self.R**2
            * jnp.cos(lat_rad)  # (nlat,)
            * dlon_rad
            * dlat_rad
        )  # (nlat,)
        self._cell_areas = jnp.broadcast_to(
            cell_area_by_lat[:, None],  # (nlat, 1)
            (nlat, nlon),
        )  # (nlat, nlon)

        # Face areas North/South: A_ns = R × cos(lat_face) × dλ
        # Faces are at cell boundaries (nlat+1 faces)
        lat_faces = jnp.linspace(lat_min, lat_max, nlat + 1)
        lat_faces_rad = jnp.radians(lat_faces)
        face_area_ns_by_lat = (
            self.R
            * jnp.cos(lat_faces_rad)  # (nlat+1,)
            * dlon_rad
        )  # (nlat+1,)
        self._face_areas_ns = jnp.broadcast_to(
            face_area_ns_by_lat[:, None],  # (nlat+1, 1)
            (nlat + 1, nlon),
        )  # (nlat+1, nlon)

        # Face areas East/West: A_ew = R × dφ (constant with lat)
        # These faces span full latitude height
        self._face_areas_ew = self.R * dlat_rad  # scalar, but broadcasted to (nlat, nlon+1)

        # Grid spacing for diffusion
        # dx(lat) varies with latitude
        self._dx = self.R * jnp.cos(lat_rad) * dlon_rad  # (nlat,)
        # dy is constant
        self._dy = self.R * dlat_rad  # scalar

    def cell_areas(self) -> jnp.ndarray:
        """Return cell areas [m²].

        Returns:
            Array (nlat, nlon) with areas varying as cos(lat)
        """
        return self._cell_areas

    def face_areas_ew(self) -> jnp.ndarray:
        """Return E/W face areas [m²].

        For spherical grid, these are vertical faces with constant area.

        Returns:
            Scalar (broadcast to shape (nlat, nlon+1))
        """
        # Return as broadcasted array for consistency
        return jnp.full((self.nlat, self.nlon + 1), self._face_areas_ew)

    def face_areas_ns(self) -> jnp.ndarray:
        """Return N/S face areas [m²].

        For spherical grid, these vary with latitude.

        Returns:
            Array (nlat+1, nlon) with areas varying as cos(lat_face)
        """
        return self._face_areas_ns

    def dx(self) -> jnp.ndarray:
        """Return longitude spacing [m].

        For spherical grid, dx varies with latitude: dx(lat) = R·cos(lat)·dλ

        Returns:
            Array (nlat,) with spacing decreasing toward poles
        """
        return self._dx

    def dy(self) -> float:
        """Return latitude spacing [m].

        For spherical grid, dy is constant: dy = R·dφ

        Returns:
            Scalar constant spacing
        """
        return float(self._dy)


class PlaneGrid(Grid):
    """Uniform Cartesian grid for testing or idealized cases.

    This grid has constant spacing in both directions:
    - dx = constant [m]
    - dy = constant [m]
    - Cell areas = dx × dy (constant)

    Useful for:
    - Testing transport schemes in simple geometry
    - Idealized simulations (e.g., channel flows)
    - Benchmarking against analytical solutions

    Attributes:
        dx, dy: Grid spacing [m]
        nlat, nlon: Number of cells (naming kept for consistency)

    Example:
        >>> grid = PlaneGrid(dx=10e3, dy=10e3, nlat=100, nlon=100)
        >>> areas = grid.cell_areas()
        >>> jnp.all(areas == 100e6)  # All cells have area 100 km²
        True
    """

    def __init__(
        self,
        dx: float,
        dy: float,
        nlat: int,
        nlon: int,
    ):
        """Initialize plane grid.

        Args:
            dx: Grid spacing in x/longitude direction [m]
            dy: Grid spacing in y/latitude direction [m]
            nlat: Number of cells in y direction
            nlon: Number of cells in x direction
        """
        self._dx = dx
        self._dy = dy
        self.nlat = nlat
        self.nlon = nlon

        # Pre-compute constant areas
        self._cell_area = dx * dy
        self._face_area_ew = dy  # East/West faces have height dy
        self._face_area_ns = dx  # North/South faces have width dx

    def cell_areas(self) -> jnp.ndarray:
        """Return cell areas [m²].

        Returns:
            Array (nlat, nlon) with constant area dx×dy
        """
        return jnp.full((self.nlat, self.nlon), self._cell_area)

    def face_areas_ew(self) -> jnp.ndarray:
        """Return E/W face areas [m²].

        Returns:
            Array (nlat, nlon+1) with constant area dy
        """
        return jnp.full((self.nlat, self.nlon + 1), self._face_area_ew)

    def face_areas_ns(self) -> jnp.ndarray:
        """Return N/S face areas [m²].

        Returns:
            Array (nlat+1, nlon) with constant area dx
        """
        return jnp.full((self.nlat + 1, self.nlon), self._face_area_ns)

    def dx(self) -> float:
        """Return x spacing [m].

        Returns:
            Scalar constant spacing
        """
        return self._dx

    def dy(self) -> float:
        """Return y spacing [m].

        Returns:
            Scalar constant spacing
        """
        return self._dy
