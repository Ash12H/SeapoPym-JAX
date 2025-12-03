"""Grid geometry calculations for transport computations.

This module provides functions to compute geometric quantities for spherical
(lat/lon) grids used in ocean transport models. These quantities are essential
for flux-based transport schemes that conserve mass.

Key geometric quantities:
- Cell areas: A(lat) = R² × cos(lat) × dλ × dφ [m²]
- Face areas E/W: A_ew = R × dφ [m]
- Face areas N/S: A_ns(lat) = R × cos(lat_face) × dλ [m]
- Grid spacing dx(lat): R × cos(lat) × dλ [m]
- Grid spacing dy: R × dφ [m]

References:
    - Original implementation: IA/transport/grid.py
    - Spherical geometry: IA/Diffusion-euler-explicite-description.md (line 26)
    - Conservation requirements: IA/TRANSPORT_ANALYSIS.md (Section 8)
"""

import numpy as np
import xarray as xr

from seapopym.standard.coordinates import Coordinates

# Earth radius in meters
EARTH_RADIUS = 6371e3


def compute_spherical_cell_areas(
    lats: xr.DataArray | np.ndarray,
    lons: xr.DataArray | np.ndarray,
    R: float = EARTH_RADIUS,
) -> xr.DataArray:
    """Compute cell areas for a spherical lat/lon grid.

    Cell area formula for spherical grids:
        A(lat) = R² × cos(lat) × dλ × dφ

    where:
    - R = Earth radius [m]
    - lat = latitude [degrees]
    - dλ = longitude spacing [radians]
    - dφ = latitude spacing [radians]

    Cell areas vary with latitude due to the cos(lat) term:
    - Maximum at equator (lat=0°): A = R² × dλ × dφ
    - Minimum at poles (lat=±90°): A → 0

    Args:
        lats: Latitude coordinates [degrees], shape (nlat,)
        lons: Longitude coordinates [degrees], shape (nlon,)
        R: Earth radius [m], default 6371e3

    Returns:
        DataArray of cell areas [m²], shape (lat, lon)
        Coordinates: {lat: lats, lon: lons}

    Example:
        >>> lats = xr.DataArray(np.linspace(-60, 60, 120), dims=["lat"])
        >>> lons = xr.DataArray(np.linspace(0, 360, 360), dims=["lon"])
        >>> areas = compute_spherical_cell_areas(lats, lons)
        >>> areas.shape
        (120, 360)
    """
    # Ensure we have numpy arrays
    lats_np = lats.values if isinstance(lats, xr.DataArray) else np.asarray(lats)

    lons_np = lons.values if isinstance(lons, xr.DataArray) else np.asarray(lons)

    # Compute grid spacing
    dlat = np.abs(lats_np[1] - lats_np[0])  # Assume uniform spacing
    dlon = np.abs(lons_np[1] - lons_np[0])

    # Convert to radians
    lat_rad = np.radians(lats_np)
    dlat_rad = np.radians(dlat)
    dlon_rad = np.radians(dlon)

    # Cell area by latitude: A(lat) = R² × cos(lat) × dλ × dφ
    cell_area_by_lat = R**2 * np.cos(lat_rad) * dlon_rad * dlat_rad  # (nlat,)

    # Broadcast to (nlat, nlon)
    nlat = len(lats_np)
    nlon = len(lons_np)
    cell_areas = np.broadcast_to(cell_area_by_lat[:, None], (nlat, nlon))

    # Return as DataArray with coordinates
    return xr.DataArray(
        cell_areas,
        dims=[Coordinates.Y.value, Coordinates.X.value],
        coords={
            Coordinates.Y.value: lats,
            Coordinates.X.value: lons,
        },
        attrs={"units": "m**2", "long_name": "Cell areas"},
    )


def compute_spherical_face_areas_ew(
    lats: xr.DataArray | np.ndarray,
    lons: xr.DataArray | np.ndarray,
    R: float = EARTH_RADIUS,
) -> xr.DataArray:
    """Compute East/West face areas for a spherical lat/lon grid.

    East/West faces are vertical faces perpendicular to longitude direction.
    Their area is constant (does not vary with latitude):
        A_ew = R × dφ

    where:
    - R = Earth radius [m]
    - dφ = latitude spacing [radians]

    Args:
        lats: Latitude coordinates [degrees], shape (nlat,)
        lons: Longitude coordinates [degrees], shape (nlon,)
        R: Earth radius [m], default 6371e3

    Returns:
        DataArray of E/W face areas [m], shape (lat, lon+1)
        Note: There are nlon+1 faces for nlon cells
        Coordinates: {lat: lats}

    Example:
        >>> lats = xr.DataArray(np.linspace(-60, 60, 120), dims=["lat"])
        >>> lons = xr.DataArray(np.linspace(0, 360, 360), dims=["lon"])
        >>> face_areas_ew = compute_spherical_face_areas_ew(lats, lons)
        >>> face_areas_ew.shape
        (120, 361)
    """
    # Ensure we have numpy arrays
    lats_np = lats.values if isinstance(lats, xr.DataArray) else np.asarray(lats)

    lons_np = lons.values if isinstance(lons, xr.DataArray) else np.asarray(lons)

    # Compute latitude spacing
    dlat = np.abs(lats_np[1] - lats_np[0])
    dlat_rad = np.radians(dlat)

    # E/W face area (constant)
    face_area_ew = R * dlat_rad

    # Shape: (nlat, nlon+1)
    nlat = len(lats_np)
    nlon = len(lons_np)
    face_areas = np.full((nlat, nlon + 1), face_area_ew)

    # Return as DataArray
    # Note: We don't add a longitude coordinate for faces since they're between cells
    return xr.DataArray(
        face_areas,
        dims=[Coordinates.Y.value, "lon_face"],
        coords={Coordinates.Y.value: lats},
        attrs={"units": "m", "long_name": "East/West face areas"},
    )


def compute_spherical_face_areas_ns(
    lats: xr.DataArray | np.ndarray,
    lons: xr.DataArray | np.ndarray,
    R: float = EARTH_RADIUS,
) -> xr.DataArray:
    """Compute North/South face areas for a spherical lat/lon grid.

    North/South faces are horizontal faces perpendicular to latitude direction.
    Their area varies with latitude:
        A_ns(lat_face) = R × cos(lat_face) × dλ

    where:
    - R = Earth radius [m]
    - lat_face = latitude at face [radians]
    - dλ = longitude spacing [radians]

    Faces are located at cell boundaries (nlat+1 faces for nlat cells).

    Args:
        lats: Latitude coordinates [degrees], shape (nlat,)
        lons: Longitude coordinates [degrees], shape (nlon,)
        R: Earth radius [m], default 6371e3

    Returns:
        DataArray of N/S face areas [m], shape (lat+1, lon)
        Note: There are nlat+1 faces for nlat cells
        Coordinates: {lon: lons}

    Example:
        >>> lats = xr.DataArray(np.linspace(-60, 60, 120), dims=["lat"])
        >>> lons = xr.DataArray(np.linspace(0, 360, 360), dims=["lon"])
        >>> face_areas_ns = compute_spherical_face_areas_ns(lats, lons)
        >>> face_areas_ns.shape
        (121, 360)
    """
    # Ensure we have numpy arrays
    lats_np = lats.values if isinstance(lats, xr.DataArray) else np.asarray(lats)

    lons_np = lons.values if isinstance(lons, xr.DataArray) else np.asarray(lons)

    # Compute grid spacing
    dlat = np.abs(lats_np[1] - lats_np[0])
    dlon = np.abs(lons_np[1] - lons_np[0])
    dlon_rad = np.radians(dlon)

    # Compute face latitudes (at boundaries)
    # If cell centers are at lats[i], faces are at lats[i] ± dlat/2
    lat_min = lats_np[0] - dlat / 2
    lat_max = lats_np[-1] + dlat / 2
    lat_faces = np.linspace(lat_min, lat_max, len(lats_np) + 1)
    lat_faces_rad = np.radians(lat_faces)

    # N/S face area by latitude: A_ns(lat_face) = R × cos(lat_face) × dλ
    face_area_by_lat = R * np.cos(lat_faces_rad) * dlon_rad  # (nlat+1,)

    # Broadcast to (nlat+1, nlon)
    nlat = len(lats_np)
    nlon = len(lons_np)
    face_areas = np.broadcast_to(face_area_by_lat[:, None], (nlat + 1, nlon))

    # Return as DataArray
    return xr.DataArray(
        face_areas,
        dims=["lat_face", Coordinates.X.value],
        coords={Coordinates.X.value: lons},
        attrs={"units": "m", "long_name": "North/South face areas"},
    )


def compute_spherical_dx(
    lats: xr.DataArray | np.ndarray,
    lons: xr.DataArray | np.ndarray,
    R: float = EARTH_RADIUS,
) -> xr.DataArray:
    """Compute grid spacing in longitude direction (varying with latitude).

    For spherical grids, longitude spacing decreases toward the poles:
        dx(lat) = R × cos(lat) × dλ

    where:
    - R = Earth radius [m]
    - lat = latitude [radians]
    - dλ = longitude spacing [radians]

    Args:
        lats: Latitude coordinates [degrees], shape (nlat,)
        lons: Longitude coordinates [degrees], shape (nlon,)
        R: Earth radius [m], default 6371e3

    Returns:
        DataArray of dx values [m], shape (lat, lon)
        Varies with latitude (broadcasts across longitude)
        Coordinates: {lat: lats, lon: lons}

    Example:
        >>> lats = xr.DataArray(np.linspace(-60, 60, 120), dims=["lat"])
        >>> lons = xr.DataArray(np.linspace(0, 360, 360), dims=["lon"])
        >>> dx = compute_spherical_dx(lats, lons)
        >>> dx.shape
        (120, 360)
    """
    # Ensure we have numpy arrays
    lats_np = lats.values if isinstance(lats, xr.DataArray) else np.asarray(lats)

    lons_np = lons.values if isinstance(lons, xr.DataArray) else np.asarray(lons)

    # Compute longitude spacing
    dlon = np.abs(lons_np[1] - lons_np[0])
    dlon_rad = np.radians(dlon)

    # Convert latitude to radians
    lat_rad = np.radians(lats_np)

    # dx(lat) = R × cos(lat) × dλ
    dx_by_lat = R * np.cos(lat_rad) * dlon_rad  # (nlat,)

    # Broadcast to (nlat, nlon)
    nlat = len(lats_np)
    nlon = len(lons_np)
    dx = np.broadcast_to(dx_by_lat[:, None], (nlat, nlon))

    # Return as DataArray with coordinates
    return xr.DataArray(
        dx,
        dims=[Coordinates.Y.value, Coordinates.X.value],
        coords={
            Coordinates.Y.value: lats,
            Coordinates.X.value: lons,
        },
        attrs={"units": "m", "long_name": "Grid spacing in longitude direction"},
    )


def compute_spherical_dy(
    lats: xr.DataArray | np.ndarray,
    lons: xr.DataArray | np.ndarray,
    R: float = EARTH_RADIUS,
) -> xr.DataArray:
    """Compute grid spacing in latitude direction (constant).

    For spherical grids, latitude spacing is constant:
        dy = R × dφ

    where:
    - R = Earth radius [m]
    - dφ = latitude spacing [radians]

    Args:
        lats: Latitude coordinates [degrees], shape (nlat,)
        lons: Longitude coordinates [degrees], shape (nlon,)
        R: Earth radius [m], default 6371e3

    Returns:
        DataArray of dy values [m], shape (lat, lon)
        Constant value broadcasted to grid shape
        Coordinates: {lat: lats, lon: lons}

    Example:
        >>> lats = xr.DataArray(np.linspace(-60, 60, 120), dims=["lat"])
        >>> lons = xr.DataArray(np.linspace(0, 360, 360), dims=["lon"])
        >>> dy = compute_spherical_dy(lats, lons)
        >>> dy.shape
        (120, 360)
        >>> np.all(dy == dy[0, 0])
        True
    """
    # Ensure we have numpy arrays
    lats_np = lats.values if isinstance(lats, xr.DataArray) else np.asarray(lats)

    lons_np = lons.values if isinstance(lons, xr.DataArray) else np.asarray(lons)

    # Compute latitude spacing
    dlat = np.abs(lats_np[1] - lats_np[0])
    dlat_rad = np.radians(dlat)

    # dy = R × dφ (constant)
    dy_val = R * dlat_rad

    # Broadcast to (nlat, nlon)
    nlat = len(lats_np)
    nlon = len(lons_np)
    dy = np.full((nlat, nlon), dy_val)

    # Return as DataArray with coordinates
    return xr.DataArray(
        dy,
        dims=[Coordinates.Y.value, Coordinates.X.value],
        coords={
            Coordinates.Y.value: lats,
            Coordinates.X.value: lons,
        },
        attrs={"units": "m", "long_name": "Grid spacing in latitude direction"},
    )
