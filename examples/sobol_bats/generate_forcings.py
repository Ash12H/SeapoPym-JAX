"""Extract forcings for Station BATS Sobol experiment from real SEAPODYM data.

Reads the aggregated 2D PHY Zarr and extracts a 20x20 degree domain around
Station BATS (32N, 64W = 296E), time subset 2000-2004 (60 monthly steps).

Source: examples/data/NP-ERA5-NFIX_SEAPODYM-2D_PHY.zarr
  Variables used: sst -> temperature, pp -> primary_production, uo -> u, vo -> v
  Variables ignored: zeu, mldr10_1

Grid metrics (dx, dy, face_height, face_width, cell_area) and mask are
computed from the coordinate grid. Diffusion D is CFL-limited per cell.
Currents (u, v) are time-averaged from monthly fields.

Usage:
    uv run python examples/sobol_bats/generate_forcings.py
"""

from pathlib import Path

import numpy as np
import xarray as xr

# =============================================================================
# Domain — 20x20 degrees around Station BATS (32N, 64W = 296E)
# =============================================================================
LAT_RANGE = (22.5, 41.5)
LON_RANGE = (-73.5, -54.5)
TIME_RANGE = ("2000-01", "2004-12")

# Physical constants
EARTH_RADIUS = 6_371_000.0  # meters
DEG_TO_RAD = np.pi / 180.0
DEG_TO_M_MERIDIONAL = EARTH_RADIUS * DEG_TO_RAD  # ~111 km per degree

DIFFUSION_COEFF = 500.0  # m2/s
DT = 86400.0  # timestep in seconds (1 day)
CFL_LIMIT = 0.4  # safety margin for explicit diffusion (max 0.5)

SOURCE_PATH = Path(__file__).parents[1] / "data" / "NP-ERA5-NFIX_SEAPODYM-2D_PHY.zarr"
OUTPUT_PATH = Path(__file__).parent / "forcings.zarr"

DTYPE = np.float32


def compute_grid_metrics(latitudes, ny, nx):
    """Compute dx, dy, face_height, face_width, cell_area, D from latitudes.

    D is CFL-limited: at high latitudes where dx shrinks, D is reduced to
    ensure the explicit diffusion scheme remains stable (CFL = D*dt/dx² ≤ 0.5).
    """
    dy = np.full((ny, nx), DEG_TO_M_MERIDIONAL, dtype=DTYPE)
    cos_lat = np.cos(latitudes * DEG_TO_RAD)
    dx_1d = EARTH_RADIUS * cos_lat * (1.0 * DEG_TO_RAD)
    dx = np.broadcast_to(dx_1d[:, None], (ny, nx)).copy().astype(DTYPE)
    face_height = dy.copy()
    face_width = dx.copy()
    cell_area = (dx * dy).astype(DTYPE)

    # CFL-limited diffusion: D_max = CFL_LIMIT * dx² / dt
    D_max = CFL_LIMIT * dx**2 / DT
    D = np.minimum(DIFFUSION_COEFF, D_max).astype(DTYPE)

    return dx, dy, face_height, face_width, cell_area, D


def main():
    print("Extracting Station BATS forcings from real SEAPODYM data...")

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Source data not found at {SOURCE_PATH}. "
            "Run the Zarr aggregation first (see examples/data/)."
        )

    # Load and subset
    ds = xr.open_zarr(SOURCE_PATH)
    ds = ds.sel(
        time=slice(*TIME_RANGE),
        latitude=slice(*LAT_RANGE),
        longitude=slice(*LON_RANGE),
    ).load()

    lat = ds.latitude.values.astype(DTYPE)
    lon_orig = ds.longitude.values.astype(DTYPE)
    NY, NX = len(lat), len(lon_orig)
    NT = len(ds.time)

    # Convert longitude to 0-360 convention
    lon = lon_orig.copy()
    lon[lon < 0] += 360

    print(f"  Domain: {lat[0]:.1f} - {lat[-1]:.1f} N, {lon[0]:.1f} - {lon[-1]:.1f} E")
    print(f"  Time: {str(ds.time.values[0])[:10]} to {str(ds.time.values[-1])[:10]} ({NT} monthly steps)")
    print(f"  Grid: {NY} Y x {NX} X")

    # Create mask from SST (NaN at all times = land)
    sst_all = ds["sst"].values
    mask = np.where(np.all(np.isnan(sst_all), axis=0), 0.0, 1.0).astype(DTYPE)
    ocean_frac = mask.sum() / mask.size * 100
    print(f"  Ocean cells: {mask.sum():.0f}/{mask.size} ({ocean_frac:.1f}%)")

    # Dynamic forcings
    temperature = np.nan_to_num(sst_all, nan=0.0).astype(DTYPE)
    temperature = np.maximum(temperature, 0.0)
    primary_production = np.nan_to_num(ds["pp"].values, nan=0.0).astype(DTYPE) * (1e-3 / 86400.0)

    ocean_mask_3d = np.broadcast_to(mask[None, :, :] > 0, temperature.shape)
    print(f"  SST range (ocean): {temperature[ocean_mask_3d].min():.1f} - {temperature[ocean_mask_3d].max():.1f} C")
    print(f"  PP range (ocean): {primary_production[ocean_mask_3d].min():.2e} - {primary_production[ocean_mask_3d].max():.2e}")

    # Static forcings: time-mean of uo, vo
    u = np.nanmean(ds["uo"].values, axis=0).astype(DTYPE)
    v = np.nanmean(ds["vo"].values, axis=0).astype(DTYPE)
    u = np.nan_to_num(u, nan=0.0)
    v = np.nan_to_num(v, nan=0.0)
    ocean_2d = mask > 0
    print(f"  Mean |u|: {np.abs(u[ocean_2d]).mean():.4f} m/s, mean |v|: {np.abs(v[ocean_2d]).mean():.4f} m/s")

    # Grid metrics (D is CFL-limited per cell)
    dx, dy, face_height, face_width, cell_area, D = compute_grid_metrics(lat, NY, NX)
    D_min, D_max = D[ocean_2d].min(), D[ocean_2d].max()
    if D_min < DIFFUSION_COEFF:
        print(f"  D CFL-limited: {D_min:.0f} - {D_max:.0f} m2/s (nominal {DIFFUSION_COEFF:.0f})")
    else:
        print(f"  D constant: {DIFFUSION_COEFF:.0f} m2/s (CFL OK everywhere)")

    # Build output dataset with model dimension names (T, Y, X)
    times = ds.time.values
    coords_tyx = {"T": times, "Y": lat, "X": lon}
    coords_yx = {"Y": lat, "X": lon}

    out = xr.Dataset(
        {
            "temperature": xr.DataArray(temperature, dims=["T", "Y", "X"], coords=coords_tyx),
            "primary_production": xr.DataArray(primary_production, dims=["T", "Y", "X"], coords=coords_tyx),
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
            "title": "Station BATS forcings from NP-ERA5-NFIX SEAPODYM 2D PHY",
            "domain": f"{lat[0]:.1f} - {lat[-1]:.1f} N, {lon[0]:.1f} - {lon[-1]:.1f} E",
            "source": SOURCE_PATH.name,
            "time_range": f"{TIME_RANGE[0]} to {TIME_RANGE[1]}",
        },
    )

    dynamic_vars = ["temperature", "primary_production"]
    static_vars = ["u", "v", "D", "dx", "dy", "face_height", "face_width", "cell_area", "mask"]

    encoding = {var: {"chunks": (1, NY, NX)} for var in dynamic_vars}
    encoding.update({var: {"chunks": (NY, NX)} for var in static_vars})

    if OUTPUT_PATH.exists():
        import shutil
        shutil.rmtree(OUTPUT_PATH)

    out.to_zarr(OUTPUT_PATH, encoding=encoding)

    size_mb = sum(f.stat().st_size for f in OUTPUT_PATH.rglob("*") if f.is_file()) / 1e6
    print(f"\nZarr written: {OUTPUT_PATH}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Dims: T={NT}, Y={NY}, X={NX}")

    ds_check = xr.open_zarr(OUTPUT_PATH)
    print(f"  temperature chunks: {ds_check['temperature'].encoding.get('chunks', 'N/A')}")
    print(f"  mask shape: {ds_check['mask'].shape}")
    print("Done!")


if __name__ == "__main__":
    main()
