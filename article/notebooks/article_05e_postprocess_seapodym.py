#!/usr/bin/env python
"""Post-traitement Seapodym-LMTL (Aggregation Output).

Ce script agrège les fichiers NetCDF quotidiens produits par le modèle
de référence Seapodym-LMTL (C++) en un fichier Zarr unique pour l'analyse.

Objectifs :
1. Lire la série ZPK_D1N1_biomass_*.nc.
2. Concaténer temporellement.
3. Exporter en Zarr (seapodym_lmtl_output_pacific_ref.zarr).
"""

# %%
import contextlib
from pathlib import Path

import xarray as xr

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA_DIR = BASE_DIR.parent / "data"

SOURCE_DIR = DATA_DIR / "LMTL_Pacific_Run" / "output"
PATTERN = "ZPK_D1N1_biomass_*.nc"
OUTPUT_ZARR = DATA_DIR / "seapodym_lmtl_output_pacific_ref.zarr"

print(f"Source: {SOURCE_DIR}/{PATTERN}")
print(f"Output: {OUTPUT_ZARR}")

# =============================================================================
# 1. LOAD (Lazy Loading)
# =============================================================================

# %%
ds = xr.open_mfdataset(f"{SOURCE_DIR}/{PATTERN}")

print("\nDataset agrégé :")
print(ds)

# =============================================================================
# 2. STANDARDIZATION
# =============================================================================

# %%
# Rename biomass to zooplankton for consistency
ds = ds.rename({"biomass": "zooplankton"})

# Add metadata
ds["zooplankton"].attrs["source"] = "Seapodym-LMTL Reference Run (C++)"
ds.attrs["title"] = "Reference Simulation Output (Pacific)"

# Clean coordinate attributes
ds.latitude.attrs = {}
ds.longitude.attrs = {}

# Convert units to g/m² if pint is available
with contextlib.suppress(Exception):
    ds["zooplankton"] = ds["zooplankton"].pint.quantify().pint.to("g/m^2").pint.dequantify()

print("\nVariable standardisée:")
print(ds["zooplankton"])

# =============================================================================
# 3. EXPORT ZARR
# =============================================================================

# %%
# Optimized chunking for analysis (full spatial access or time series)
ds_chunked = ds.chunk({"time": 100, "latitude": -1, "longitude": -1})

print(f"\nExport vers {OUTPUT_ZARR}...")
ds_chunked.to_zarr(OUTPUT_ZARR, mode="w", consolidated=True)
print("Export terminé.")

# =============================================================================
# 4. VERIFICATION
# =============================================================================

# %%
ds_verif = xr.open_zarr(OUTPUT_ZARR)
print("\nStructure finale :")
print(ds_verif)
