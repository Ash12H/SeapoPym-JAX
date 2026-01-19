#!/usr/bin/env python

# # Post-traitement Seapodym-LMTL (Aggregation Output)
#
# Ce notebook agrège les fichiers NetCDF quotidiens produits par le modèle de référence Seapodym-LMTL en un fichier Zarr unique pour l'analyse.
#
# **Objectifs :**
#
# 1.  Lire la série `ZPK_D1N1_biomass_*.nc`.
# 2.  Concaténer temporellement.
# 3.  Exporter en Zarr (`seapodym_lmtl_output_pacific_ref.zarr`).
#

# %%


from pathlib import Path

import xarray as xr

# Chemins
BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA_DIR = BASE_DIR.parent / "data"

SOURCE_DIR = DATA_DIR / "LMTL_Pacific_Run" / "output"
PATTERN = "ZPK_D1N1_biomass_*.nc"
OUTPUT_ZARR = DATA_DIR / "seapodym_lmtl_output_pacific_ref.zarr"


# ## 1. Chargement (Lazy Loading)
#

# %%


# Chargement multi-fichiers avec Dask
ds = xr.open_mfdataset(f"{SOURCE_DIR}/{PATTERN}")

print("Dataset agrégé :")
print(ds)


# ## 2. Standardisation
#
# On renomme `biomass` en `zooplankton` pour correspondre à notre convention de nommage interne si besoin, ou on garde `biomass` avec un namespace.
# Ici, pour la comparaison, nous utiliserons `zooplankton` comme nom standard de variable pour ce groupe fonctionnel unique.
#

# %%


# Renommage
ds = ds.rename({"biomass": "zooplankton"})

# Ajout métadonnées
ds["zooplankton"].attrs["source"] = "Seapodym-LMTL Reference Run (C++)"
ds.attrs["title"] = "Reference Simulation Output (Pacific)"

ds.latitude.attrs = {}
ds.longitude.attrs = {}

ds["zooplankton"] = ds["zooplankton"].pint.quantify().pint.to("g/m^2").pint.dequantify()

ds["zooplankton"]


# ## 3. Export Zarr
#

# %%


# Chunking optimisé pour la lecture (Séries temporelles ou cartes)
# Ici on favorise l'accès complet spatial ou temporel
ds_chunked = ds.chunk({"time": 100, "latitude": -1, "longitude": -1})

print(f"Export vers {OUTPUT_ZARR}...")
ds_chunked.to_zarr(OUTPUT_ZARR, mode="w", consolidated=True)
print("Export terminé.")


# %%


# Vérification rapide
ds_verif = xr.open_zarr(OUTPUT_ZARR)
print("Structure finale :")
print(ds_verif)
