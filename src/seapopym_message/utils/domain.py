"""Domain decomposition utilities for distributed computing.

This module provides functions to split 2D spatial domains into patches
for parallel processing by multiple workers.
"""


def split_domain_2d(
    nlat_global: int,
    nlon_global: int,
    num_workers_lat: int,
    num_workers_lon: int,
) -> list[dict]:
    """Split 2D domain into rectangular patches for workers.

    Divides a global (nlat_global, nlon_global) grid into
    (num_workers_lat × num_workers_lon) patches.

    Each patch is assigned to a worker and includes information about:
    - Worker ID
    - Latitude and longitude index ranges
    - Neighboring workers (N, S, E, W)

    Args:
        nlat_global: Total number of latitude cells in global domain.
        nlon_global: Total number of longitude cells in global domain.
        num_workers_lat: Number of workers in latitude direction.
        num_workers_lon: Number of workers in longitude direction.

    Returns:
        List of patch dictionaries, one per worker. Each dict contains:
        - 'worker_id': Unique integer ID (0 to num_workers-1)
        - 'lat_start': Starting latitude index (inclusive)
        - 'lat_end': Ending latitude index (exclusive)
        - 'lon_start': Starting longitude index (inclusive)
        - 'lon_end': Ending longitude index (exclusive)
        - 'neighbors': Dict with keys 'north', 'south', 'east', 'west'
                      Values are worker IDs or None if at boundary

    Raises:
        ValueError: If grid cannot be evenly divided.

    Example:
        >>> patches = split_domain_2d(20, 40, 2, 2)
        >>> len(patches)  # 4 workers
        4
        >>> patches[0]  # Top-left worker
        {
            'worker_id': 0,
            'lat_start': 0, 'lat_end': 10,
            'lon_start': 0, 'lon_end': 20,
            'neighbors': {'north': None, 'south': 2, 'east': 1, 'west': None}
        }
    """
    # Validate divisibility
    if nlat_global % num_workers_lat != 0:
        raise ValueError(
            f"nlat_global ({nlat_global}) must be divisible by "
            f"num_workers_lat ({num_workers_lat})"
        )
    if nlon_global % num_workers_lon != 0:
        raise ValueError(
            f"nlon_global ({nlon_global}) must be divisible by "
            f"num_workers_lon ({num_workers_lon})"
        )

    # Patch dimensions
    nlat_per_worker = nlat_global // num_workers_lat
    nlon_per_worker = nlon_global // num_workers_lon

    patches = []

    for i_lat in range(num_workers_lat):
        for i_lon in range(num_workers_lon):
            # Worker ID: row-major ordering
            worker_id = i_lat * num_workers_lon + i_lon

            # Latitude range
            lat_start = i_lat * nlat_per_worker
            lat_end = lat_start + nlat_per_worker

            # Longitude range
            lon_start = i_lon * nlon_per_worker
            lon_end = lon_start + nlon_per_worker

            # Determine neighbors
            neighbors = {
                "north": None if i_lat == 0 else (i_lat - 1) * num_workers_lon + i_lon,
                "south": (
                    None if i_lat == num_workers_lat - 1 else (i_lat + 1) * num_workers_lon + i_lon
                ),
                "west": None if i_lon == 0 else i_lat * num_workers_lon + (i_lon - 1),
                "east": (
                    None if i_lon == num_workers_lon - 1 else i_lat * num_workers_lon + (i_lon + 1)
                ),
            }

            patches.append(
                {
                    "worker_id": worker_id,
                    "lat_start": lat_start,
                    "lat_end": lat_end,
                    "lon_start": lon_start,
                    "lon_end": lon_end,
                    "nlat": nlat_per_worker,
                    "nlon": nlon_per_worker,
                    "neighbors": neighbors,
                }
            )

    return patches


def split_domain_2d_periodic_lon(
    nlat_global: int,
    nlon_global: int,
    num_workers_lat: int,
    num_workers_lon: int,
) -> list[dict]:
    """Split 2D domain with periodic boundary in longitude.

    Similar to split_domain_2d, but wraps around in the longitude direction
    (periodic boundary condition). Useful for global domains (e.g., 0-360°).

    Args:
        nlat_global: Total number of latitude cells.
        nlon_global: Total number of longitude cells.
        num_workers_lat: Number of workers in latitude direction.
        num_workers_lon: Number of workers in longitude direction.

    Returns:
        List of patch dictionaries with periodic east/west neighbors.

    Example:
        >>> patches = split_domain_2d_periodic_lon(20, 40, 2, 4)
        >>> # Worker 0 (i_lat=0, i_lon=0) has west neighbor = worker 3
        >>> # Worker 3 (i_lat=0, i_lon=3) has east neighbor = worker 0
        >>> patches[0]['neighbors']['west']  # Wraps to rightmost
        3
        >>> patches[3]['neighbors']['east']  # Wraps to leftmost
        0
    """
    # Validate divisibility
    if nlat_global % num_workers_lat != 0:
        raise ValueError(
            f"nlat_global ({nlat_global}) must be divisible by "
            f"num_workers_lat ({num_workers_lat})"
        )
    if nlon_global % num_workers_lon != 0:
        raise ValueError(
            f"nlon_global ({nlon_global}) must be divisible by "
            f"num_workers_lon ({num_workers_lon})"
        )

    # Patch dimensions
    nlat_per_worker = nlat_global // num_workers_lat
    nlon_per_worker = nlon_global // num_workers_lon

    patches = []

    for i_lat in range(num_workers_lat):
        for i_lon in range(num_workers_lon):
            worker_id = i_lat * num_workers_lon + i_lon

            lat_start = i_lat * nlat_per_worker
            lat_end = lat_start + nlat_per_worker

            lon_start = i_lon * nlon_per_worker
            lon_end = lon_start + nlon_per_worker

            # Periodic boundary in longitude
            west_i_lon = (i_lon - 1) % num_workers_lon
            east_i_lon = (i_lon + 1) % num_workers_lon

            neighbors = {
                "north": None if i_lat == 0 else (i_lat - 1) * num_workers_lon + i_lon,
                "south": (
                    None if i_lat == num_workers_lat - 1 else (i_lat + 1) * num_workers_lon + i_lon
                ),
                "west": i_lat * num_workers_lon + west_i_lon,
                "east": i_lat * num_workers_lon + east_i_lon,
            }

            patches.append(
                {
                    "worker_id": worker_id,
                    "lat_start": lat_start,
                    "lat_end": lat_end,
                    "lon_start": lon_start,
                    "lon_end": lon_end,
                    "nlat": nlat_per_worker,
                    "nlon": nlon_per_worker,
                    "neighbors": neighbors,
                }
            )

    return patches
