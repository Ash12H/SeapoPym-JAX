"""Example: How to use the new Grid-based mask architecture.

This example shows how to properly create and use ocean/land masks
with the new architecture where masks are stored in the Grid.
"""

import jax.numpy as jnp
import numpy as np
import ray
import xarray as xr

from seapopym_message.transport.worker import TransportWorker

# =============================================================================
# Step 1: Create your land_mask (boolean array)
# =============================================================================

nlat, nlon = 40, 40
lats = np.linspace(-2.0, 2.0, nlat)
lons = np.linspace(-2.0, 2.0, nlon)
LON, LAT = np.meshgrid(lons, lats)

# Island in upper right
island_lat = 1.0
island_lon = 1.0
island_radius = 0.5
dist_to_island = np.sqrt((LAT - island_lat) ** 2 + (LON - island_lon) ** 2)
land_mask = dist_to_island <= island_radius  # True = terre, False = mer

# =============================================================================
# Step 2: Convert to ocean mask (invert) and create xr.DataArray
# =============================================================================

# Convert from land_mask (bool) to ocean_mask (float32)
# IMPORTANT: 1.0 = ocean, 0.0 = land
ocean_mask_array = np.where(land_mask, 0.0, 1.0).astype(np.float32)

# Create xarray.DataArray with coordinates
ocean_mask_da = xr.DataArray(
    ocean_mask_array,
    coords={"lat": lats, "lon": lons},
    dims=["lat", "lon"],
    attrs={"units": "1", "description": "Ocean mask (1=ocean, 0=land)", "dtype": "float32"},
)

# =============================================================================
# Step 3: Create TransportWorker with mask
# =============================================================================

# NEW: Pass mask directly to TransportWorker
transport_worker = TransportWorker.remote(
    grid_type="plane",
    nlat=nlat,
    nlon=nlon,
    dx=11100,  # meters
    dy=11100,  # meters
    lat_bc="closed",
    lon_bc="closed",
    mask=ocean_mask_da,  # ← NEW: Mask is now part of the grid
)

# =============================================================================
# Step 4: Use TransportWorker (no mask parameter needed!)
# =============================================================================

biomass = jnp.ones((nlat, nlon), dtype=jnp.float32)
u = jnp.zeros((nlat, nlon), dtype=jnp.float32)
v = jnp.zeros((nlat, nlon), dtype=jnp.float32)

# OLD way (deprecated):
# result = ray.get(transport_worker.transport_step.remote(
#     biomass=biomass, u=u, v=v, D=1000.0, dt=3600.0,
#     mask=ocean_mask_jax  # ← Had to pass mask every time
# ))

# NEW way:
result = ray.get(
    transport_worker.transport_step.remote(
        biomass=biomass,
        u=u,
        v=v,
        D=1000.0,
        dt=3600.0,
        # No mask parameter! It's already in the grid.
    )
)

# =============================================================================
# Advanced: 3D Bathymetric Mask
# =============================================================================

# For 3D simulations with depth-varying masks
depths = np.array([0, 10, 50, 100, 500])  # meters
n_depths = len(depths)

# Create 3D mask: some islands only appear near surface
ocean_mask_3d = np.ones((n_depths, nlat, nlon), dtype=np.float32)

for k, depth in enumerate(depths):
    # Shallow island disappears at depth > 50m
    if depth <= 50:
        dist = np.sqrt((LAT - island_lat) ** 2 + (LON - island_lon) ** 2)
        ocean_mask_3d[k, dist <= island_radius] = 0.0

# Create 3D DataArray
ocean_mask_3d_da = xr.DataArray(
    ocean_mask_3d,
    coords={"depth": depths, "lat": lats, "lon": lons},
    dims=["depth", "lat", "lon"],
    attrs={
        "units": "1",
        "description": "Bathymetric ocean mask by depth (1=ocean, 0=land)",
        "dtype": "float32",
    },
)

# Use with spherical grid
transport_worker_3d = TransportWorker.remote(
    grid_type="spherical",
    lat_min=-2.0,
    lat_max=2.0,
    lon_min=-2.0,
    lon_max=2.0,
    nlat=nlat,
    nlon=nlon,
    mask=ocean_mask_3d_da,  # 3D bathymetric mask
)

print("✅ Mask architecture examples complete!")
print(f"2D mask shape: {ocean_mask_da.shape}")
print(f"3D mask shape: {ocean_mask_3d_da.shape}")
print(f"Ocean cells (2D): {ocean_mask_da.sum().values:.0f}/{nlat * nlon}")
