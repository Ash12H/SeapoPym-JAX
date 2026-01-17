"""Notebook 05E: Comparaison SeapoPym DAG vs Seapodym-LMTL (Pacifique).

Ce script analyse les résultats des simulations Pacifique pour valider
le module de transport couplé.

**Objectifs** :
1. Validation : Comparer la simulation DAG (avec transport) avec la référence Seapodym-LMTL.
2. Impact du Transport : Comparer la simulation DAG avec transport vs sans transport.

**Fichiers attendus** :
- Référence : seapodym_lmtl_output_pacific_ref.zarr (1998-2020, journalier)
- Transport : seapopym_pacific_transport_optimized.zarr (1998-2020, 3-horaire)
- No-Transport : seapopym_pacific_no_transport_optimized.zarr (1998-2020, 3-horaire)
"""

# %% Imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# %% Configuration des chemins
BASE_DIR = Path("/Users/adm-lehodey/Documents/Workspace/Projects/seapopym-message/data/article")
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
SUMMARY_DIR = BASE_DIR / "summary"
FIGURES_DIR.mkdir(exist_ok=True)
SUMMARY_DIR.mkdir(exist_ok=True)

# Fichiers de données (CORRIGÉS pour utiliser les versions optimisées)
FILE_REF = DATA_DIR / "seapodym_lmtl_output_pacific_ref.zarr"
FILE_TRANS = DATA_DIR / "seapopym_pacific_transport_optimized.zarr"
FILE_NO_TRANS = DATA_DIR / "seapopym_pacific_no_transport_optimized.zarr"

print("=" * 70)
print("VÉRIFICATION DES FICHIERS")
print("=" * 70)
print(f"Référence    : {FILE_REF}")
print(f"  Existe     : {FILE_REF.exists()}")
print(f"Transport    : {FILE_TRANS}")
print(f"  Existe     : {FILE_TRANS.exists()}")
print(f"No-Transport : {FILE_NO_TRANS}")
print(f"  Existe     : {FILE_NO_TRANS.exists()}")
print("=" * 70)

# Vérifier que tous les fichiers existent
missing = []
if not FILE_REF.exists():
    missing.append("Référence")
if not FILE_TRANS.exists():
    missing.append("Transport")
if not FILE_NO_TRANS.exists():
    missing.append("No-Transport")

if missing:
    print(f"\n❌ Fichiers manquants: {missing}")
    print("Veuillez d'abord exécuter article_05b_bis_simulation_pacific_optimized.py")
    raise FileNotFoundError(f"Missing files: {missing}")

# %% Chargement des données
print("\n📊 CHARGEMENT DES DONNÉES")
print("-" * 70)

# Référence (Seapodym original)
ds_ref = xr.open_zarr(FILE_REF)
ref = ds_ref["zooplankton"].load()
print(f"Référence   : {ref.shape}, dims={list(ref.dims)}")
print(
    f"  Period    : {pd.to_datetime(ref.time.values).min()} → {pd.to_datetime(ref.time.values).max()}"
)

# Transport (SeapoPym DAG)
ds_trans = xr.open_zarr(FILE_TRANS)
dag_trans = ds_trans["biomass"].load()
print(f"Transport   : {dag_trans.shape}, dims={list(dag_trans.dims)}")
print(
    f"  Period    : {pd.to_datetime(dag_trans.time.values).min()} → {pd.to_datetime(dag_trans.time.values).max()}"
)

# No-Transport (SeapoPym DAG)
ds_no_trans = xr.open_zarr(FILE_NO_TRANS)
dag_no_trans = ds_no_trans["biomass"].load()
print(f"No-Transport: {dag_no_trans.shape}, dims={list(dag_no_trans.dims)}")
print(
    f"  Period    : {pd.to_datetime(dag_no_trans.time.values).min()} → {pd.to_datetime(dag_no_trans.time.values).max()}"
)

# %% Standardisation des noms de coordonnées
print("\n📊 STANDARDISATION DES COORDONNÉES")
print("-" * 70)

# SeapoPym utilise y/x, Seapodym utilise latitude/longitude
rename_dict = {"y": "latitude", "x": "longitude"}

ref = ref.rename({k: v for k, v in rename_dict.items() if k in ref.dims})
dag_trans = dag_trans.rename({k: v for k, v in rename_dict.items() if k in dag_trans.dims})
dag_no_trans = dag_no_trans.rename({k: v for k, v in rename_dict.items() if k in dag_no_trans.dims})

print(f"Ref coords       : {list(ref.dims)}")
print(f"Transport coords : {list(dag_trans.dims)}")
print(f"No-Trans coords  : {list(dag_no_trans.dims)}")

# Vérification de l'alignement spatial
lat_ref = ref.latitude.values
lat_trans = dag_trans.latitude.values
lon_ref = ref.longitude.values
lon_trans = dag_trans.longitude.values

print(
    f"\nGrille Référence   : lat=[{lat_ref.min():.1f}, {lat_ref.max():.1f}], lon=[{lon_ref.min():.1f}, {lon_ref.max():.1f}]"
)
print(
    f"Grille Transport   : lat=[{lat_trans.min():.1f}, {lat_trans.max():.1f}], lon=[{lon_trans.min():.1f}, {lon_trans.max():.1f}]"
)

if len(lat_ref) == len(lat_trans) and len(lon_ref) == len(lon_trans):
    print("✅ Grilles spatiales de même taille")
else:
    print("⚠️ Grilles spatiales de tailles différentes!")

# %% Alignement temporel
print("\n📊 ALIGNEMENT TEMPOREL")
print("-" * 70)

# Période de comparaison : exclure spin-up (2 premières années)
TIME_START = "2000-01-01"
TIME_END = "2019-12-31"

print(f"Période d'analyse : {TIME_START} → {TIME_END}")

# Sélection temporelle
ref_cut = ref.sel(time=slice(TIME_START, TIME_END))
dag_trans_cut = dag_trans.sel(time=slice(TIME_START, TIME_END))
dag_no_trans_cut = dag_no_trans.sel(time=slice(TIME_START, TIME_END))

print(f"\nAprès sélection temporelle:")
print(f"  Référence    : {len(ref_cut.time)} pas de temps")
print(f"  Transport    : {len(dag_trans_cut.time)} pas de temps")
print(f"  No-Transport : {len(dag_no_trans_cut.time)} pas de temps")

# Check if data is available
if len(dag_trans_cut.time) == 0:
    print("\n❌ ERREUR: Transport n'a pas de données dans la période!")
    print("Vérifiez que la simulation a bien été exécutée pour 1998-2020")
    raise ValueError("No transport data in the selected period")

# Resampling des données SeapoPym à la journée (moyenne journalière)
# car la référence est journalière et SeapoPym est 3-horaire
print("\n📊 RESAMPLING À LA JOURNÉE")
print("-" * 70)

dag_trans_daily = dag_trans_cut.resample(time="1D").mean()
dag_no_trans_daily = dag_no_trans_cut.resample(time="1D").mean()

print(f"Transport (daily)    : {len(dag_trans_daily.time)} jours")
print(f"No-Transport (daily) : {len(dag_no_trans_daily.time)} jours")

# Normalisation des timestamps à 00:00 (la référence a 12:00)
ref_cut["time"] = ref_cut.time.dt.floor("D")
dag_trans_daily["time"] = dag_trans_daily.time.dt.floor("D")
dag_no_trans_daily["time"] = dag_no_trans_daily.time.dt.floor("D")

# Alignement des grilles (intersection)
ref_aligned, dag_trans_aligned = xr.align(ref_cut, dag_trans_daily, join="inner")
_, dag_no_trans_aligned = xr.align(ref_aligned, dag_no_trans_daily, join="inner")

print(f"\nAprès alignement:")
print(f"  Période alignée : {len(ref_aligned.time)} jours")
print(f"  Grille alignée  : {len(ref_aligned.latitude)} x {len(ref_aligned.longitude)}")

# %% Métriques de validation
print("\n" + "=" * 70)
print("MÉTRIQUES DE VALIDATION")
print("=" * 70)


def compute_metrics(da_ref, da_model):
    """Calcule RMSE, NRMSE (normalisé par std) et MAPE."""
    diff = da_model - da_ref

    # RMSE
    rmse = np.sqrt((diff**2).mean()).item()

    # NRMSE (normalisé par l'écart-type de la référence)
    std_ref = da_ref.std().item()
    nrmse = rmse / std_ref if std_ref > 0 else np.nan

    # MAPE
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.abs(diff / da_ref).where(da_ref > 1e-6).mean().item() * 100

    return {"RMSE": rmse, "NRMSE": nrmse, "MAPE": mape}


# DAG Transport vs Ref
metrics_trans = compute_metrics(ref_aligned, dag_trans_aligned)
print("\nDAG Transport vs Référence:")
print(f"  RMSE  : {metrics_trans['RMSE']:.2f}")
print(f"  NRMSE : {metrics_trans['NRMSE']:.2f}")
print(f"  MAPE  : {metrics_trans['MAPE']:.2f} %")

# DAG No-Transport vs Ref
metrics_no_trans = compute_metrics(ref_aligned, dag_no_trans_aligned)
print("\nDAG No-Transport vs Référence:")
print(f"  RMSE  : {metrics_no_trans['RMSE']:.2f}")
print(f"  NRMSE : {metrics_no_trans['NRMSE']:.2f}")
print(f"  MAPE  : {metrics_no_trans['MAPE']:.2f} %")

# %% Cartes spatiales de RMSE
print("\n📊 GÉNÉRATION DES FIGURES")
print("-" * 70)

# Calcul des RMSE spatiales (moyennées sur le temps)
diff_trans = dag_trans_aligned - ref_aligned
rmse_trans = np.sqrt((diff_trans**2).mean(dim="time"))

diff_no_trans = dag_no_trans_aligned - ref_aligned
rmse_no_trans = np.sqrt((diff_no_trans**2).mean(dim="time"))

vmax_rmse = float(ref_aligned.quantile(0.75))

# Figure RMSE
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax1 = axes[0]
im1 = rmse_trans.plot(ax=ax1, cmap="viridis_r", vmin=0, vmax=vmax_rmse, add_colorbar=False)
ax1.set_title("RMSE - DAG Transport vs Référence", fontsize=12, fontweight="bold")
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
plt.colorbar(im1, ax=ax1, label="RMSE (g/m²)")

ax2 = axes[1]
im2 = rmse_no_trans.plot(ax=ax2, cmap="viridis_r", vmin=0, vmax=vmax_rmse, add_colorbar=False)
ax2.set_title("RMSE - DAG No-Transport vs Référence", fontsize=12, fontweight="bold")
ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")
plt.colorbar(im2, ax=ax2, label="RMSE (g/m²)")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig_05e_spatial_rmse.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: fig_05e_spatial_rmse.png")

# %% Séries temporelles par zone latitudinale
# Ordre: Nord en haut, Tropique au milieu, Sud en bas
zones = {
    "Nord (> 20°)": (20, 60),
    "Tropicale (-20° à 20°)": (-20, 20),
    "Sud (< -20°)": (-60, -20),
}

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

for ax, (zone_name, (lat_min, lat_max)) in zip(axes, zones.items()):
    # Sélection de la zone latitudinale avec slice explicite
    ref_zone = ref_aligned.sel(latitude=slice(lat_min, lat_max))
    trans_zone = dag_trans_aligned.sel(latitude=slice(lat_min, lat_max))
    no_trans_zone = dag_no_trans_aligned.sel(latitude=slice(lat_min, lat_max))

    # Calcul des séries temporelles (moyenne spatiale)
    ts_ref = ref_zone.mean(["latitude", "longitude"])
    ts_trans = trans_zone.mean(["latitude", "longitude"])
    ts_no_trans = no_trans_zone.mean(["latitude", "longitude"])

    # Tracé
    ts_ref.plot(ax=ax, label="Seapodym (Ref)", linewidth=2, color="black")
    ts_trans.plot(ax=ax, label="DAG (Transport)", linewidth=1.5, linestyle="--", color="blue")
    ts_no_trans.plot(
        ax=ax, label="DAG (No Transport)", linewidth=1, linestyle=":", color="grey", alpha=0.7
    )

    ax.set_title(f"Biomasse Moyenne - Zone {zone_name}")
    ax.set_xlabel("Temps")
    ax.set_ylabel("Biomasse (g/m²)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig_05e_pacific_timeseries_zones.png", dpi=150)
plt.show()
print("✅ Saved: fig_05e_pacific_timeseries_zones.png")

# %% Résumé
print("\n" + "=" * 70)
print("RÉSUMÉ")
print("=" * 70)
print(f"Période d'analyse : {TIME_START} à {TIME_END}")
print(f"Grille : {len(ref_aligned.latitude)} x {len(ref_aligned.longitude)}")
print(f"\nDAG Transport vs Référence:")
print(f"  RMSE  : {metrics_trans['RMSE']:.2f}")
print(f"  NRMSE : {metrics_trans['NRMSE']:.2f}")
print(f"  MAPE  : {metrics_trans['MAPE']:.2f} %")
print(f"\nDAG No-Transport vs Référence:")
print(f"  RMSE  : {metrics_no_trans['RMSE']:.2f}")
print(f"  NRMSE : {metrics_no_trans['NRMSE']:.2f}")
print(f"  MAPE  : {metrics_no_trans['MAPE']:.2f} %")
print("=" * 70)
