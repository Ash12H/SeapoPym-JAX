"""Generate synthetic forcings for North Pacific Sobol experiment.

Creates a Zarr store with realistic-dimension synthetic data:
- Domain: 0°N–66°N, 100°E–100°W (66 Y × 260 X at 1° resolution)
- Time: 2000-01-01 to 2005-01-01 (1826 daily timesteps)
- Chunking: {"T": 1, "Y": -1, "X": -1}

Dynamic forcings (T, Y, X):
  - temperature: seasonal 10–25°C with latitudinal gradient
  - primary_production: seasonal with noise, proportional to temperature

Static forcings (Y, X):
  - u, v: ocean currents (u=0.1 m/s east, v=0)
  - D: diffusion coefficient (500 m²/s)
  - dx, dy: inter-cell distances in meters (latitude-dependent via cos(lat))
  - face_height, face_width: = dy, dx
  - cell_area: dx × dy
  - mask: simplified land/ocean mask (rectangle with terrestrial corners)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# =============================================================================
# Domain parameters
# =============================================================================
NY, NX = 66, 260
LAT_START, LAT_END = 0.5, 65.5  # cell centers
LON_START, LON_END = 100.5, 359.5  # 100°E to 100°W in 0–360° convention
TIME_START, TIME_END = "2000-01-01", "2005-01-01"

# Physical constants
EARTH_RADIUS = 6_371_000.0  # meters
DEG_TO_RAD = np.pi / 180.0
DEG_TO_M_MERIDIONAL = EARTH_RADIUS * DEG_TO_RAD  # ~111 km per degree latitude

OUTPUT_DIR = Path(__file__).parent
OUTPUT_PATH = OUTPUT_DIR / "forcings.zarr"


def make_coords():
    """Create coordinate arrays."""
    latitudes = np.linspace(LAT_START, LAT_END, NY, dtype=np.float64)
    longitudes = np.linspace(LON_START, LON_END, NX, dtype=np.float64)
    times = pd.date_range(start=TIME_START, end=TIME_END, freq="D", inclusive="left")
    return latitudes, longitudes, times


def make_mask(latitudes, longitudes):
    """Create a simplified land/ocean mask.

    Ocean everywhere except for terrestrial corners representing
    approximate land masses (SE Asia, North America).
    """
    mask = np.ones((NY, NX), dtype=np.float32)

    lat2d = latitudes[:, None] * np.ones((1, NX))
    lon2d = np.ones((NY, 1)) * longitudes[None, :]

    # SE Asia / Indonesia corner (low lat, west side)
    mask[(lat2d < 15) & (lon2d < 120)] = 0.0

    # North America (high lat, east side, lon > 230°E = 130°W)
    mask[(lat2d > 40) & (lon2d > 235)] = 0.0

    # Japan / Korea (mid lat, far west)
    mask[(lat2d > 30) & (lat2d < 50) & (lon2d < 135)] = 0.0

    return mask


def make_grid_metrics(latitudes):
    """Compute dx, dy, face_height, face_width, cell_area."""
    dy = np.full((NY, NX), DEG_TO_M_MERIDIONAL, dtype=np.float32)

    # dx varies with latitude: dx = R * cos(lat) * dlon_rad
    cos_lat = np.cos(latitudes * DEG_TO_RAD)
    dx_1d = EARTH_RADIUS * cos_lat * (1.0 * DEG_TO_RAD)  # 1° spacing
    dx = np.broadcast_to(dx_1d[:, None], (NY, NX)).copy().astype(np.float32)

    face_height = dy.copy()
    face_width = dx.copy()
    cell_area = (dx * dy).astype(np.float32)

    return dx, dy, face_height, face_width, cell_area


def make_dynamic_forcings(latitudes, times):
    """Generate temperature and primary production fields."""
    nt = len(times)
    doy = times.dayofyear.values

    # Temperature: seasonal signal with latitudinal gradient
    # Warmer at low latitudes (~25°C), cooler at high latitudes (~5°C)
    lat_factor = 1.0 - latitudes / 90.0  # 1.0 at equator, ~0.27 at 66°N
    base_temp = 5.0 + 20.0 * lat_factor  # 5–25°C range

    # Seasonal amplitude increases with latitude
    seasonal_amp = 2.0 + 6.0 * (latitudes / 66.0)

    # Build (T, Y, X) temperature field
    seasonal = np.sin(2 * np.pi * (doy - 80) / 365.0)  # peak around day 170 (June)
    temperature = (
        base_temp[None, :, None]
        + seasonal_amp[None, :, None] * seasonal[:, None, None]
    )
    temperature = np.broadcast_to(temperature, (nt, NY, NX)).copy().astype(np.float32)

    # Primary production: seasonal, proportional to temperature, with noise
    rng = np.random.default_rng(42)
    noise = rng.normal(1.0, 0.2, size=(nt, NY, NX)).clip(0.5, 2.0)
    # Base NPP: ~300 mgC/m²/day = 300e-3 / 86400 gC/m²/s
    npp_base = 300e-3 / 86400.0
    # Scale by normalized temperature (warmer → more production)
    temp_norm = (temperature - temperature.min()) / (temperature.max() - temperature.min())
    primary_production = (npp_base * (0.5 + temp_norm) * noise).astype(np.float32)

    return temperature, primary_production


def main():
    print("Generating synthetic North Pacific forcings...")
    print(f"  Grid: {NY} Y × {NX} X, domain: {LAT_START}°–{LAT_END}°N, {LON_START}°–{LON_END}°E")

    latitudes, longitudes, times = make_coords()
    nt = len(times)
    print(f"  Time: {times[0].date()} to {times[-1].date()} ({nt} timesteps)")

    # Static fields
    mask = make_mask(latitudes, longitudes)
    dx, dy, face_height, face_width, cell_area = make_grid_metrics(latitudes)
    u = np.full((NY, NX), 0.1, dtype=np.float32)  # 0.1 m/s eastward
    v = np.zeros((NY, NX), dtype=np.float32)
    D = np.full((NY, NX), 500.0, dtype=np.float32)  # 500 m²/s

    ocean_frac = mask.sum() / mask.size * 100
    print(f"  Ocean cells: {mask.sum():.0f}/{mask.size} ({ocean_frac:.1f}%)")

    # Dynamic fields
    temperature, primary_production = make_dynamic_forcings(latitudes, times)
    print(f"  Temperature range: {temperature.min():.1f}–{temperature.max():.1f} °C")
    print(f"  NPP range: {primary_production.min():.2e}–{primary_production.max():.2e} g/m²/s")

    # Build xarray Dataset
    coords = {"T": times, "Y": latitudes, "X": longitudes}
    coords_yx = {"Y": latitudes, "X": longitudes}

    ds = xr.Dataset(
        {
            # Dynamic (T, Y, X)
            "temperature": xr.DataArray(temperature, dims=["T", "Y", "X"], coords=coords),
            "primary_production": xr.DataArray(primary_production, dims=["T", "Y", "X"], coords=coords),
            # Static (Y, X)
            "u": xr.DataArray(u, dims=["Y", "X"], coords=coords_yx),
            "v": xr.DataArray(v, dims=["Y", "X"], coords=coords_yx),
            "D": xr.DataArray(D, dims=["Y", "X"], coords=coords_yx),
            "dx": xr.DataArray(dx, dims=["Y", "X"], coords=coords_yx),
            "dy": xr.DataArray(dy, dims=["Y", "X"], coords=coords_yx),
            "face_height": xr.DataArray(face_height, dims=["Y", "X"], coords=coords_yx),
            "face_width": xr.DataArray(face_width, dims=["Y", "X"], coords=coords_yx),
            "cell_area": xr.DataArray(cell_area, dims=["Y", "X"], coords=coords_yx),
            "mask": xr.DataArray(mask, dims=["Y", "X"], coords=coords_yx),
        },
        attrs={
            "title": "Synthetic North Pacific forcings for Sobol experiment",
            "domain": "0°N–66°N, 100°E–100°W",
            "resolution": "1°",
        },
    )

    # Write to Zarr with appropriate chunking
    dynamic_vars = ["temperature", "primary_production"]
    static_vars = ["u", "v", "D", "dx", "dy", "face_height", "face_width", "cell_area", "mask"]

    encoding = {var: {"chunks": (1, NY, NX)} for var in dynamic_vars}
    encoding.update({var: {"chunks": (NY, NX)} for var in static_vars})

    if OUTPUT_PATH.exists():
        import shutil

        shutil.rmtree(OUTPUT_PATH)

    ds.to_zarr(OUTPUT_PATH, encoding=encoding)

    # Verify
    size_mb = sum(f.stat().st_size for f in OUTPUT_PATH.rglob("*") if f.is_file()) / 1e6
    print(f"\nZarr written: {OUTPUT_PATH}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Dims: T={nt}, Y={NY}, X={NX}")

    # Quick verification by reopening
    ds_check = xr.open_zarr(OUTPUT_PATH)
    print(f"\nVerification (xr.open_zarr):")
    print(f"  temperature chunks: {ds_check['temperature'].encoding.get('chunks', 'N/A')}")
    print(f"  mask shape: {ds_check['mask'].shape}")
    print("Done!")


if __name__ == "__main__":
    main()
