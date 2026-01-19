#!/usr/bin/env python

# # Préparation des données Seapodym-LMTL (Pacifique)
#
# Ce notebook prépare les forcings pour la comparaison SeapoPym vs Seapodym-LMTL dans le Pacifique.
#
# **Traitements effectués :**
#
# 1.  Chargement des données globales.
# 2.  Conversion des longitudes : `[-180, 180]` -> `[0, 360]`.
# 3.  Sélection spatiale : Pacifique (`110°E - 290°E`, `±60°N`).
# 4.  Sélection temporelle : 12 ans (`2000-2011`).
# 5.  Correction profondeur : `Z=0` -> `Z=1`.
# 6.  Sauvegarde Zarr optimisée.
#

# %%


import os
from pathlib import Path

import xarray as xr

ZARR_SOURCE = "/Users/adm-lehodey/Documents/Workspace/Projects/phd_optimization/notebooks/Article_1/data/1_global/post_processed_light_global_multiyear_bgc_001_033.zarr"

BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
OUTPUT_DIR = BASE_DIR.parent / "data"
OUTPUT_ZARR = OUTPUT_DIR / "seapodym_lmtl_forcings_pacific.zarr"

print(f"Source : {ZARR_SOURCE}")
print(f"Cible  : {OUTPUT_ZARR}")


# ## 1. Chargement et Conversion des Longitudes
#

# %%


ds = xr.open_zarr(ZARR_SOURCE)

# Renommage standard
ds = ds.rename({"T": "time", "Z": "depth", "Y": "latitude", "X": "longitude"})

# Conversion Longitude -180/180 -> 0/360
# On utilise assign_coords puis sortby pour réorganiser la grille
ds = ds.assign_coords(longitude=(ds.longitude % 360)).sortby(["longitude", "latitude"])

print("Longitudes converties (0-360) et triées.")
print(f"Lon range: {ds.longitude.min().values:.1f} -> {ds.longitude.max().values:.1f}")


# ## 2. Sélection Spatio-Temporelle
#

# %%


# Période : 2000 - 2011 (12 ans)
START_TIME = "1998-01-01"
END_TIME = "2020-12-31"

# Domaine Pacifique large
# On couvre de l'Indonésie (~100°E) à l'Amérique (~290°E soit -70°W)
LON_MIN, LON_MAX = 100, 290
LAT_MIN, LAT_MAX = -60, 60

ds_subset = ds.sel(
    time=slice(START_TIME, END_TIME),
    longitude=slice(LON_MIN, LON_MAX),
    latitude=slice(LAT_MIN, LAT_MAX),
)

print(f"Subset shape: {ds_subset.dims}")
print(f"Time steps: {len(ds_subset.time)}")


# %%

# ## 3. Préparation Finale
#

# %%


# Conservation des variables d'intérêt
vars_keep = ["current_u", "current_v", "primary_production", "temperature", "zooplankton"]
ds_final = ds_subset[vars_keep]

if "depth" in ds_final.dims:
    if ds_final.sizes["depth"] > 1:
        ds_final = ds_final.isel(depth=slice(0, 1))  # Garde la dimension (taille 1)

    # Remplace la valeur de la coordonnée
    ds_final = ds_final.assign_coords(depth=[1.0])

# Nettoyage des encodages pour éviter les conflits de chunks
for var in ds_final.data_vars:
    if "chunks" in ds_final[var].encoding:
        del ds_final[var].encoding["chunks"]

# Définition de nouveaux chunks optimisés pour le traitement temporel
# (Temps global ou par année, espace par blocs)
chunks = {"time": 100, "latitude": len(ds_final.latitude), "longitude": len(ds_final.longitude)}
if "depth" in ds_final.dims:
    chunks["depth"] = 1

ds_final = ds_final.chunk(chunks)

ds_final.attrs["title"] = "Forçages Pacifique Seapodym-LMTL (2000-2011)"

ds_final.latitude.attrs = {}
ds_final.longitude.attrs = {}
ds_final.depth.attrs = {}
ds_final.time.attrs = {}

ds_final = ds_final.assign_coords({"time": ds_final.time.dt.floor("D")})


# %%

# Répertoire de sortie
os.makedirs(os.path.dirname(OUTPUT_ZARR), exist_ok=True)

print(f"Écriture Zarr vers {OUTPUT_ZARR}...")
# mode='w' écrase l'existant
ds_final.to_zarr(OUTPUT_ZARR, mode="w", consolidated=True)
print("Terminé.")
