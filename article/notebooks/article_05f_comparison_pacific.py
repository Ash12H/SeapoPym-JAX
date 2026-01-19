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
# %% Configuration des chemins
BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA_DIR = BASE_DIR.parent / "data"
FIGURES_DIR = BASE_DIR.parent / "figures"
SUMMARY_DIR = BASE_DIR.parent / "summary"
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

print("\nAprès sélection temporelle:")
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

print("\nAprès alignement:")
print(f"  Période alignée : {len(ref_aligned.time)} jours")
print(f"  Grille alignée  : {len(ref_aligned.latitude)} x {len(ref_aligned.longitude)}")

# %% Application du masque terre/mer
print("\n📊 APPLICATION DU MASQUE TERRE/MER")
print("-" * 70)

# La référence utilise NaN pour la terre, les modèles DAG utilisent 0
# On applique le masque de la référence aux modèles DAG pour une comparaison équitable
# Le masque est 2D (latitude, longitude) - on prend le premier pas de temps
mask_ocean = ~np.isnan(ref_aligned.isel(time=0))

# Appliquer le masque: mettre NaN là où la référence a des NaN
dag_trans_aligned = dag_trans_aligned.where(mask_ocean)
dag_no_trans_aligned = dag_no_trans_aligned.where(mask_ocean)

# Statistiques du masque
n_total = mask_ocean.sizes["latitude"] * mask_ocean.sizes["longitude"]
n_ocean = mask_ocean.sum().item()
pct_ocean = (n_ocean / n_total) * 100

print(f"Points océan : {n_ocean} / {n_total} ({pct_ocean:.1f}%)")
print("✅ Masque appliqué aux modèles DAG")

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

for ax, (zone_name, (lat_min, lat_max)) in zip(axes, zones.items(), strict=False):
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

# %% Génération du Summary
FIGURE_PREFIX = "fig_05f_comparison_pacific"
summary_filename = f"{FIGURE_PREFIX.replace('fig_', 'notebook_')}_summary.txt"
summary_path = SUMMARY_DIR / summary_filename

# Calculs supplémentaires pour le résumé
n_timesteps = len(ref_aligned.time)
n_lat = len(ref_aligned.latitude)
n_lon = len(ref_aligned.longitude)
lat_min = ref_aligned.latitude.min().item()
lat_max = ref_aligned.latitude.max().item()
lon_min = ref_aligned.longitude.min().item()
lon_max = ref_aligned.longitude.max().item()

# Amélioration relative apportée par le transport
rmse_improvement = (
    (metrics_no_trans["RMSE"] - metrics_trans["RMSE"]) / metrics_no_trans["RMSE"]
) * 100
nrmse_improvement = (
    (metrics_no_trans["NRMSE"] - metrics_trans["NRMSE"]) / metrics_no_trans["NRMSE"]
) * 100
mape_improvement = (
    (metrics_no_trans["MAPE"] - metrics_trans["MAPE"]) / metrics_no_trans["MAPE"]
) * 100

# Biomasse moyenne
ref_mean_biomass = ref_aligned.mean().item()
trans_mean_biomass = dag_trans_aligned.mean().item()
no_trans_mean_biomass = dag_no_trans_aligned.mean().item()

with open(summary_path, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("NOTEBOOK 05F: COMPARAISON SEAPOPYM DAG vs SEAPODYM-LMTL (PACIFIQUE)\n")
    f.write("=" * 80 + "\n\n")

    f.write("DATE: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")

    f.write("OBJECTIF:\n")
    f.write("-" * 80 + "\n")
    f.write("Valider le modèle SeapoPym DAG (architecture Python) contre Seapodym-LMTL\n")
    f.write("(modèle de référence C++/Fortran) sur une simulation réaliste du Pacifique.\n")
    f.write("Quantifier l'impact du module de transport (advection + diffusion) sur\n")
    f.write("la distribution de biomasse micronectonique.\n\n")

    f.write("DONNÉES UTILISÉES:\n")
    f.write("-" * 80 + "\n")
    f.write("Référence (Seapodym-LMTL)    : seapodym_lmtl_output_pacific_ref.zarr\n")
    f.write("SeapoPym avec transport      : seapopym_pacific_transport_optimized.zarr\n")
    f.write("SeapoPym sans transport      : seapopym_pacific_no_transport_optimized.zarr\n\n")

    f.write("CONFIGURATION SPATIALE:\n")
    f.write("-" * 80 + "\n")
    f.write("Région                       : Pacifique\n")
    f.write(f"Grille                       : {n_lat} × {n_lon} points\n")
    f.write(f"Latitude                     : [{lat_min:.1f}°, {lat_max:.1f}°]\n")
    f.write(f"Longitude                    : [{lon_min:.1f}°, {lon_max:.1f}°]\n\n")

    f.write("CONFIGURATION TEMPORELLE:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Période d'analyse            : {TIME_START} à {TIME_END}\n")
    f.write("Période de spin-up exclue    : 1998-1999 (2 ans)\n")
    f.write(f"Nombre de pas de temps       : {n_timesteps} jours\n")
    f.write("Résolution temporelle:\n")
    f.write("  - Référence (Seapodym)     : journalier (1D)\n")
    f.write("  - SeapoPym                 : 3-horaire → moyenné à journalier\n\n")

    f.write("MÉTRIQUES DE VALIDATION:\n")
    f.write("-" * 80 + "\n")
    f.write("\n1. DAG TRANSPORT vs RÉFÉRENCE (Configuration cible):\n")
    f.write(f"   RMSE                      : {metrics_trans['RMSE']:.4f} g/m²\n")
    f.write(f"   NRMSE (normalisé par std) : {metrics_trans['NRMSE']:.4f}\n")
    f.write(f"   MAPE (erreur relative)    : {metrics_trans['MAPE']:.2f}%\n\n")

    f.write("2. DAG NO-TRANSPORT vs RÉFÉRENCE (Baseline sans transport):\n")
    f.write(f"   RMSE                      : {metrics_no_trans['RMSE']:.4f} g/m²\n")
    f.write(f"   NRMSE (normalisé par std) : {metrics_no_trans['NRMSE']:.4f}\n")
    f.write(f"   MAPE (erreur relative)    : {metrics_no_trans['MAPE']:.2f}%\n\n")

    f.write("IMPACT DU TRANSPORT:\n")
    f.write("-" * 80 + "\n")
    f.write("Le module de transport améliore significativement les résultats:\n")
    f.write(f"   Réduction RMSE            : {rmse_improvement:+.1f}%\n")
    f.write(f"   Réduction NRMSE           : {nrmse_improvement:+.1f}%\n")
    f.write(f"   Réduction MAPE            : {mape_improvement:+.1f}%\n\n")

    f.write("INTERPRÉTATION:\n")
    f.write("-" * 80 + "\n")
    if rmse_improvement > 0:
        f.write("✅ Le transport AMÉLIORE la correspondance avec le modèle de référence.\n")
        f.write("   Le transport de biomasse (advection par courants + diffusion) capture\n")
        f.write("   des processus physiques importants pour la distribution spatiale.\n\n")
    else:
        f.write(
            "⚠️ Le transport ne semble pas améliorer les résultats dans cette configuration.\n\n"
        )

    f.write("STATISTIQUES DE BIOMASSE:\n")
    f.write("-" * 80 + "\n")
    f.write("Biomasse moyenne (g/m²):\n")
    f.write(f"   Référence (Seapodym)      : {ref_mean_biomass:.4f}\n")
    f.write(f"   DAG Transport             : {trans_mean_biomass:.4f}\n")
    f.write(f"   DAG No-Transport          : {no_trans_mean_biomass:.4f}\n\n")

    f.write("ANALYSE PAR ZONE LATITUDINALE:\n")
    f.write("-" * 80 + "\n")
    for zone_name, (lat_min_z, lat_max_z) in zones.items():
        ref_zone = ref_aligned.sel(latitude=slice(lat_min_z, lat_max_z))
        trans_zone = dag_trans_aligned.sel(latitude=slice(lat_min_z, lat_max_z))
        no_trans_zone = dag_no_trans_aligned.sel(latitude=slice(lat_min_z, lat_max_z))

        zone_metrics_trans = compute_metrics(ref_zone, trans_zone)
        zone_metrics_no_trans = compute_metrics(ref_zone, no_trans_zone)

        f.write(f"\n{zone_name}:\n")
        f.write(
            f"   Transport - RMSE={zone_metrics_trans['RMSE']:.2f}, NRMSE={zone_metrics_trans['NRMSE']:.2f}\n"
        )
        f.write(
            f"   No-Trans  - RMSE={zone_metrics_no_trans['RMSE']:.2f}, NRMSE={zone_metrics_no_trans['NRMSE']:.2f}\n"
        )

    f.write("\n\nFICHIERS GÉNÉRÉS:\n")
    f.write("-" * 80 + "\n")
    f.write("- fig_05e_spatial_rmse.png          : Cartes RMSE spatiales\n")
    f.write("- fig_05e_pacific_timeseries_zones.png : Séries temporelles par zone\n")
    f.write(f"- {summary_filename}  : Ce fichier résumé\n\n")

    f.write("CONCLUSION:\n")
    f.write("-" * 80 + "\n")
    if metrics_trans["NRMSE"] < 0.5:
        f.write("✅ VALIDATION RÉUSSIE\n")
        f.write("   Le modèle SeapoPym DAG avec transport reproduit avec précision\n")
        f.write(
            f"   les sorties du modèle Seapodym-LMTL (NRMSE = {metrics_trans['NRMSE']:.2f} < 0.5).\n"
        )
        f.write("   L'architecture Python DAG est validée pour des simulations réalistes.\n")
    else:
        f.write("⚠️ VALIDATION PARTIELLE\n")
        f.write(f"   NRMSE = {metrics_trans['NRMSE']:.2f} (seuil = 0.5)\n")
        f.write("   Des écarts significatifs existent, analyse supplémentaire requise.\n")

    f.write("\n" + "=" * 80 + "\n")

print(f"✅ Résumé sauvegardé : {summary_path}")

# %% Affichage console
print("\n" + "=" * 70)
print("RÉSUMÉ")
print("=" * 70)
print(f"Période d'analyse : {TIME_START} à {TIME_END}")
print(f"Grille : {n_lat} x {n_lon}")
print("\nDAG Transport vs Référence:")
print(f"  RMSE  : {metrics_trans['RMSE']:.2f}")
print(f"  NRMSE : {metrics_trans['NRMSE']:.2f}")
print(f"  MAPE  : {metrics_trans['MAPE']:.2f} %")
print("\nDAG No-Transport vs Référence:")
print(f"  RMSE  : {metrics_no_trans['RMSE']:.2f}")
print(f"  NRMSE : {metrics_no_trans['NRMSE']:.2f}")
print(f"  MAPE  : {metrics_no_trans['MAPE']:.2f} %")
print(f"\nImpact Transport: RMSE {rmse_improvement:+.1f}%, NRMSE {nrmse_improvement:+.1f}%")
print("=" * 70)
