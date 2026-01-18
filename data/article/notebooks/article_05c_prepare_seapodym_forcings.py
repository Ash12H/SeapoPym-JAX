#!/usr/bin/env python

# # Préparation des Forçages pour Seapodym-LMTL (Format Natif)
#
# Ce notebook convertit les données Zarr Pacifique en fichiers NetCDF journaliers compatibles avec le modèle de référence Seapodym-LMTL (C++).
#
# **Sorties générées :**
#
# 1.  `mask.nc` : Masque terre/mer.
# 2.  `data/data_YYYYMMDD.nc` : Forçages journaliers (U, V, T, PP).
#

# %%


import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Chemins
INPUT_ZARR = "/Users/adm-lehodey/Documents/Workspace/Projects/seapopym-message/data/article/data/seapodym_lmtl_forcings_pacific.zarr"
EXPORT_DIR = "/Users/adm-lehodey/Documents/Workspace/Projects/seapopym-message/data/article/data/LMTL_Pacific_Run/"

# Création structure dossiers
(Path(EXPORT_DIR) / "data").mkdir(parents=True, exist_ok=True)

print(f"Source : {INPUT_ZARR}")
print(f"Cible  : {EXPORT_DIR}")


# ## 1. Chargement et Vérification
#

# %%


ds = xr.open_zarr(INPUT_ZARR).load()

# Vérification des variables
req_vars = ["current_u", "current_v", "temperature", "primary_production"]
for v in req_vars:
    if v not in ds:
        raise ValueError(f"Variable manquante : {v}")

print("Variables présentes.")
print(f"Période : {ds.time.values[0]} -> {ds.time.values[-1]}")
print(f"Grille : {len(ds.latitude)} lat x {len(ds.longitude)} lon")
ds


# ## 2. Génération du Masque (`mask.nc`)
#
# Le masque est défini par la présence de données de température (Océan = 1, Terre = 0).
#

# %%


# On prend le premier pas de temps pour définir le masque statique
sample_temp = ds["temperature"].isel(time=0)
if "depth" in sample_temp.dims:
    sample_temp = sample_temp.isel(depth=0)

# Masque : 1 si valide (pas NaN), 0 sinon
mask = xr.where(np.isnan(sample_temp), 0, 1).astype("int32")
mask.name = "mask"

# Attributs standards
mask.attrs = {"standard_name": "mask", "long_name": "Land Mask (1=Ocean, 0=Land)"}
mask.latitude.attrs = {"standard_name": "latitude", "units": "degrees_north"}
mask.longitude.attrs = {"standard_name": "longitude", "units": "degrees_east"}

output_mask = os.path.join(EXPORT_DIR, "mask.nc")
mask.to_netcdf(output_mask)
print(f"Masque généré : {output_mask}")

# Stats
ocean_pct = (mask.sum() / mask.size) * 100
print(f"Pourcentage Océan : {float(ocean_pct):.2f}%")


# ## 3. Export des Forçages Quotidiens (`data/data_YYYYMMDD.nc`)
#

# %%


import sys

# Préparation du Dataset à exporter
# On s'assure des attributs standard_name pour Seapodym
ds_export = ds[req_vars].copy()
ds_export["current_u"].attrs.update({"standard_name": "longitudinal current"})
ds_export["current_v"].attrs.update({"standard_name": "latitudinal current"})
ds_export["temperature"].attrs.update({"standard_name": "temperature"})
ds_export["primary_production"].attrs.update({"standard_name": "net primary production"})

# Boucle d'export
total_days = len(ds.time)
print(f"Début de l'export de {total_days} fichiers...")

for i, t in enumerate(ds.time.values):
    date_str = pd.to_datetime(t).strftime("%Y%m%d")
    file_name = f"data_{date_str}.nc"
    file_path = os.path.join(EXPORT_DIR, "data", file_name)

    # Extraction du jour courant
    daily_slice = ds_export.sel(time=[t])

    # Export Netcdf (float32 pour compatibilité standard)
    # On force l'encodage pour éviter les problèmes de compression/filters si nécessaire
    encoding = {v: {"dtype": "float32", "zlib": True} for v in req_vars}
    daily_slice.to_netcdf(file_path, encoding=encoding)

    # Barre de progression simple
    if (i + 1) % 100 == 0:
        sys.stdout.write(f"\rExporté {i + 1}/{total_days}")
        sys.stdout.flush()

print(f"\nExport terminé dans {EXPORT_DIR}/data/")


# %%


# Vérification d'un fichier généré
check_file = os.path.join(
    EXPORT_DIR, "data", f"data_{pd.to_datetime(ds.time.values[0]).strftime('%Y%m%d')}.nc"
)
print(f"Vérification de {check_file} :")
ds_check = xr.open_dataset(check_file)
print(ds_check)
