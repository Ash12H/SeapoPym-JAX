#!/usr/bin/env python
"""Comparaison SeapoPym JAX vs Seapodym-LMTL (Pacifique).

Ce script analyse les résultats des simulations Pacifique pour valider
le modèle SeapoPym avec backend JAX.

Objectifs :
1. Validation : Comparer la simulation JAX (avec transport) avec la référence Seapodym-LMTL.
2. Impact du Transport : Comparer la simulation avec transport vs sans transport.

Fichiers attendus :
- Référence : seapodym_lmtl_output_pacific_ref.zarr (journalier)
- Transport : seapopym_pacific_transport_jax.zarr (3-horaire)
- No-Transport : seapopym_pacific_no_transport_jax.zarr (3-horaire)
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA_DIR = BASE_DIR.parent / "data"
FIGURES_DIR = BASE_DIR.parent / "figures"
SUMMARY_DIR = BASE_DIR.parent / "summary"
FIGURES_DIR.mkdir(exist_ok=True)
SUMMARY_DIR.mkdir(exist_ok=True)

# Data files (JAX versions)
FILE_REF = DATA_DIR / "seapodym_lmtl_output_pacific_ref.zarr"
FILE_TRANS = DATA_DIR / "seapopym_pacific_transport_jax.zarr"
FILE_NO_TRANS = DATA_DIR / "seapopym_pacific_no_transport_jax.zarr"

# Analysis period (exclude spin-up)
TIME_START = "2000-01-01"
TIME_END = "2019-12-31"

# Latitudinal zones for analysis
ZONES = {
    "Nord (> 20)": (20, 60),
    "Tropicale (-20 a 20)": (-20, 20),
    "Sud (< -20)": (-60, -20),
}

# =============================================================================
# FILE VERIFICATION
# =============================================================================

print("=" * 70)
print("VERIFICATION DES FICHIERS")
print("=" * 70)
print(f"Reference    : {FILE_REF}")
print(f"  Existe     : {FILE_REF.exists()}")
print(f"Transport    : {FILE_TRANS}")
print(f"  Existe     : {FILE_TRANS.exists()}")
print(f"No-Transport : {FILE_NO_TRANS}")
print(f"  Existe     : {FILE_NO_TRANS.exists()}")
print("=" * 70)

missing = []
if not FILE_REF.exists():
    missing.append("Reference")
if not FILE_TRANS.exists():
    missing.append("Transport")
if not FILE_NO_TRANS.exists():
    missing.append("No-Transport")

if missing:
    print(f"\nFichiers manquants: {missing}")
    print("Veuillez d'abord executer article_05b_simulation_pacific.py")
    raise FileNotFoundError(f"Missing files: {missing}")

# =============================================================================
# LOAD DATA
# =============================================================================

# %%
print("\nCHARGEMENT DES DONNEES")
print("-" * 70)

# Reference (Seapodym original)
ds_ref = xr.open_zarr(FILE_REF)
ref = ds_ref["zooplankton"].load()
print(f"Reference   : {ref.shape}, dims={list(ref.dims)}")
print(f"  Period    : {pd.to_datetime(ref.time.values).min()} -> {pd.to_datetime(ref.time.values).max()}")

# Transport (SeapoPym JAX)
ds_trans = xr.open_zarr(FILE_TRANS)
dag_trans = ds_trans["biomass"].load()
print(f"Transport   : {dag_trans.shape}, dims={list(dag_trans.dims)}")
print(f"  Period    : {pd.to_datetime(dag_trans.T.values).min()} -> {pd.to_datetime(dag_trans.T.values).max()}")

# No-Transport (SeapoPym JAX)
ds_no_trans = xr.open_zarr(FILE_NO_TRANS)
dag_no_trans = ds_no_trans["biomass"].load()
print(f"No-Transport: {dag_no_trans.shape}, dims={list(dag_no_trans.dims)}")
print(f"  Period    : {pd.to_datetime(dag_no_trans.T.values).min()} -> {pd.to_datetime(dag_no_trans.T.values).max()}")

# =============================================================================
# STANDARDIZE COORDINATES
# =============================================================================

# %%
print("\nSTANDARDISATION DES COORDONNEES")
print("-" * 70)

# JAX uses Y/X/T, Seapodym uses latitude/longitude/time
# Standardize to time/latitude/longitude
rename_to_standard = {"Y": "latitude", "X": "longitude", "T": "time", "y": "latitude", "x": "longitude"}

ref = ref.rename({k: v for k, v in rename_to_standard.items() if k in ref.dims})
dag_trans = dag_trans.rename({k: v for k, v in rename_to_standard.items() if k in dag_trans.dims})
dag_no_trans = dag_no_trans.rename({k: v for k, v in rename_to_standard.items() if k in dag_no_trans.dims})

print(f"Ref coords       : {list(ref.dims)}")
print(f"Transport coords : {list(dag_trans.dims)}")
print(f"No-Trans coords  : {list(dag_no_trans.dims)}")

# Grid verification
lat_ref = ref.latitude.values
lat_trans = dag_trans.latitude.values
lon_ref = ref.longitude.values
lon_trans = dag_trans.longitude.values

print(
    f"\nGrille Reference   : lat=[{lat_ref.min():.1f}, {lat_ref.max():.1f}], lon=[{lon_ref.min():.1f}, {lon_ref.max():.1f}]"
)
print(
    f"Grille Transport   : lat=[{lat_trans.min():.1f}, {lat_trans.max():.1f}], lon=[{lon_trans.min():.1f}, {lon_trans.max():.1f}]"
)

if len(lat_ref) == len(lat_trans) and len(lon_ref) == len(lon_trans):
    print("Grilles spatiales de meme taille")
else:
    print("Grilles spatiales de tailles differentes!")

# =============================================================================
# TEMPORAL ALIGNMENT
# =============================================================================

# %%
print("\nALIGNEMENT TEMPOREL")
print("-" * 70)

print(f"Periode d'analyse : {TIME_START} -> {TIME_END}")

# Temporal selection
ref_cut = ref.sel(time=slice(TIME_START, TIME_END))
dag_trans_cut = dag_trans.sel(time=slice(TIME_START, TIME_END))
dag_no_trans_cut = dag_no_trans.sel(time=slice(TIME_START, TIME_END))

print("\nApres selection temporelle:")
print(f"  Reference    : {len(ref_cut.time)} pas de temps")
print(f"  Transport    : {len(dag_trans_cut.time)} pas de temps")
print(f"  No-Transport : {len(dag_no_trans_cut.time)} pas de temps")

if len(dag_trans_cut.time) == 0:
    print("\nERREUR: Transport n'a pas de donnees dans la periode!")
    raise ValueError("No transport data in the selected period")

# Resample SeapoPym data to daily (reference is daily, SeapoPym is 3-hourly)
print("\nRESAMPLING A LA JOURNEE")
print("-" * 70)

dag_trans_daily = dag_trans_cut.resample(time="1D").mean()
dag_no_trans_daily = dag_no_trans_cut.resample(time="1D").mean()

print(f"Transport (daily)    : {len(dag_trans_daily.time)} jours")
print(f"No-Transport (daily) : {len(dag_no_trans_daily.time)} jours")

# Normalize timestamps to 00:00
ref_cut["time"] = ref_cut.time.dt.floor("D")
dag_trans_daily["time"] = dag_trans_daily.time.dt.floor("D")
dag_no_trans_daily["time"] = dag_no_trans_daily.time.dt.floor("D")

# Align grids (intersection)
ref_aligned, dag_trans_aligned = xr.align(ref_cut, dag_trans_daily, join="inner")
_, dag_no_trans_aligned = xr.align(ref_aligned, dag_no_trans_daily, join="inner")

print("\nApres alignement:")
print(f"  Periode alignee : {len(ref_aligned.time)} jours")
print(f"  Grille alignee  : {len(ref_aligned.latitude)} x {len(ref_aligned.longitude)}")

# =============================================================================
# OCEAN MASK
# =============================================================================

# %%
print("\nAPPLICATION DU MASQUE TERRE/MER")
print("-" * 70)

# Reference uses NaN for land, JAX models use 0
mask_ocean = ~np.isnan(ref_aligned.isel(time=0))

# Apply mask
dag_trans_aligned = dag_trans_aligned.where(mask_ocean)
dag_no_trans_aligned = dag_no_trans_aligned.where(mask_ocean)

n_total = mask_ocean.sizes["latitude"] * mask_ocean.sizes["longitude"]
n_ocean = mask_ocean.sum().item()
pct_ocean = (n_ocean / n_total) * 100

print(f"Points ocean : {n_ocean} / {n_total} ({pct_ocean:.1f}%)")
print("Masque applique aux modeles JAX")

# =============================================================================
# VALIDATION METRICS
# =============================================================================

# %%
print("\n" + "=" * 70)
print("METRIQUES DE VALIDATION")
print("=" * 70)


def compute_metrics(da_ref, da_model):
    """Compute RMSE, NRMSE (normalized by std) and MAPE."""
    diff = da_model - da_ref

    # RMSE
    rmse = np.sqrt((diff**2).mean()).item()

    # NRMSE (normalized by reference std)
    std_ref = da_ref.std().item()
    nrmse = rmse / std_ref if std_ref > 0 else np.nan

    # MAPE
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.abs(diff / da_ref).where(da_ref > 1e-6).mean().item() * 100

    return {"RMSE": rmse, "NRMSE": nrmse, "MAPE": mape}


# JAX Transport vs Ref
metrics_trans = compute_metrics(ref_aligned, dag_trans_aligned)
print("\nJAX Transport vs Reference:")
print(f"  RMSE  : {metrics_trans['RMSE']:.2f}")
print(f"  NRMSE : {metrics_trans['NRMSE']:.2f}")
print(f"  MAPE  : {metrics_trans['MAPE']:.2f} %")

# JAX No-Transport vs Ref
metrics_no_trans = compute_metrics(ref_aligned, dag_no_trans_aligned)
print("\nJAX No-Transport vs Reference:")
print(f"  RMSE  : {metrics_no_trans['RMSE']:.2f}")
print(f"  NRMSE : {metrics_no_trans['NRMSE']:.2f}")
print(f"  MAPE  : {metrics_no_trans['MAPE']:.2f} %")

# =============================================================================
# FIGURES
# =============================================================================

# %%
print("\nGENERATION DES FIGURES")
print("-" * 70)

# Spatial RMSE
diff_trans = dag_trans_aligned - ref_aligned
rmse_trans = np.sqrt((diff_trans**2).mean(dim="time"))

diff_no_trans = dag_no_trans_aligned - ref_aligned
rmse_no_trans = np.sqrt((diff_no_trans**2).mean(dim="time"))

vmax_rmse = float(ref_aligned.quantile(0.75))

# RMSE maps
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax1 = axes[0]
im1 = rmse_trans.plot(ax=ax1, cmap="viridis_r", vmin=0, vmax=vmax_rmse, add_colorbar=False)
ax1.set_title("RMSE - JAX Transport vs Reference", fontsize=12, fontweight="bold")
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
plt.colorbar(im1, ax=ax1, label="RMSE (g/m2)")

ax2 = axes[1]
im2 = rmse_no_trans.plot(ax=ax2, cmap="viridis_r", vmin=0, vmax=vmax_rmse, add_colorbar=False)
ax2.set_title("RMSE - JAX No-Transport vs Reference", fontsize=12, fontweight="bold")
ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")
plt.colorbar(im2, ax=ax2, label="RMSE (g/m2)")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig_05f_spatial_rmse.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: fig_05f_spatial_rmse.png")

# Time series by latitudinal zone
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

for ax, (zone_name, (lat_min, lat_max)) in zip(axes, ZONES.items(), strict=False):
    ref_zone = ref_aligned.sel(latitude=slice(lat_min, lat_max))
    trans_zone = dag_trans_aligned.sel(latitude=slice(lat_min, lat_max))
    no_trans_zone = dag_no_trans_aligned.sel(latitude=slice(lat_min, lat_max))

    ts_ref = ref_zone.mean(["latitude", "longitude"])
    ts_trans = trans_zone.mean(["latitude", "longitude"])
    ts_no_trans = no_trans_zone.mean(["latitude", "longitude"])

    ts_ref.plot(ax=ax, label="Seapodym (Ref)", linewidth=2, color="black")
    ts_trans.plot(ax=ax, label="JAX (Transport)", linewidth=1.5, linestyle="--", color="blue")
    ts_no_trans.plot(ax=ax, label="JAX (No Transport)", linewidth=1, linestyle=":", color="grey", alpha=0.7)

    ax.set_title(f"Biomasse Moyenne - Zone {zone_name}")
    ax.set_xlabel("Temps")
    ax.set_ylabel("Biomasse (g/m2)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig_05f_pacific_timeseries_zones.png", dpi=150)
plt.show()
print("Saved: fig_05f_pacific_timeseries_zones.png")

# =============================================================================
# SUMMARY
# =============================================================================

# %%
n_timesteps = len(ref_aligned.time)
n_lat = len(ref_aligned.latitude)
n_lon = len(ref_aligned.longitude)
lat_min_grid = ref_aligned.latitude.min().item()
lat_max_grid = ref_aligned.latitude.max().item()
lon_min_grid = ref_aligned.longitude.min().item()
lon_max_grid = ref_aligned.longitude.max().item()

# Transport improvement
rmse_improvement = ((metrics_no_trans["RMSE"] - metrics_trans["RMSE"]) / metrics_no_trans["RMSE"]) * 100
nrmse_improvement = ((metrics_no_trans["NRMSE"] - metrics_trans["NRMSE"]) / metrics_no_trans["NRMSE"]) * 100
mape_improvement = ((metrics_no_trans["MAPE"] - metrics_trans["MAPE"]) / metrics_no_trans["MAPE"]) * 100

# Mean biomass
ref_mean_biomass = ref_aligned.mean().item()
trans_mean_biomass = dag_trans_aligned.mean().item()
no_trans_mean_biomass = dag_no_trans_aligned.mean().item()

summary_path = SUMMARY_DIR / "notebook_05f_comparison_pacific_summary.txt"

with open(summary_path, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("NOTEBOOK 05F: COMPARAISON SEAPOPYM JAX vs SEAPODYM-LMTL (PACIFIQUE)\n")
    f.write("=" * 80 + "\n\n")
    f.write("DATE: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
    f.write("OBJECTIF:\n")
    f.write("-" * 80 + "\n")
    f.write("Valider le modele SeapoPym JAX contre Seapodym-LMTL (reference C++).\n\n")
    f.write("CONFIGURATION:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Grille: {n_lat} x {n_lon}\n")
    f.write(f"Latitude: [{lat_min_grid:.1f}, {lat_max_grid:.1f}]\n")
    f.write(f"Longitude: [{lon_min_grid:.1f}, {lon_max_grid:.1f}]\n")
    f.write(f"Periode: {TIME_START} a {TIME_END}\n")
    f.write(f"Pas de temps: {n_timesteps} jours\n\n")
    f.write("METRIQUES:\n")
    f.write("-" * 80 + "\n")
    f.write(f"JAX Transport vs Ref: RMSE={metrics_trans['RMSE']:.2f}, NRMSE={metrics_trans['NRMSE']:.2f}\n")
    f.write(f"JAX No-Trans vs Ref:  RMSE={metrics_no_trans['RMSE']:.2f}, NRMSE={metrics_no_trans['NRMSE']:.2f}\n")
    f.write(f"Impact Transport: RMSE {rmse_improvement:+.1f}%\n\n")
    f.write("BIOMASSE MOYENNE (g/m2):\n")
    f.write("-" * 80 + "\n")
    f.write(f"Reference:     {ref_mean_biomass:.4f}\n")
    f.write(f"JAX Transport: {trans_mean_biomass:.4f}\n")
    f.write(f"JAX No-Trans:  {no_trans_mean_biomass:.4f}\n\n")
    f.write("=" * 80 + "\n")

print(f"Resume sauvegarde : {summary_path}")

# Console summary
print("\n" + "=" * 70)
print("RESUME")
print("=" * 70)
print(f"Periode d'analyse : {TIME_START} a {TIME_END}")
print(f"Grille : {n_lat} x {n_lon}")
print("\nJAX Transport vs Reference:")
print(f"  RMSE  : {metrics_trans['RMSE']:.2f}")
print(f"  NRMSE : {metrics_trans['NRMSE']:.2f}")
print("\nJAX No-Transport vs Reference:")
print(f"  RMSE  : {metrics_no_trans['RMSE']:.2f}")
print(f"  NRMSE : {metrics_no_trans['NRMSE']:.2f}")
print(f"\nImpact Transport: RMSE {rmse_improvement:+.1f}%")
print("=" * 70)
