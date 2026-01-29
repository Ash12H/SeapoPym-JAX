"""Notebook 06 BONUS: Animation Comparative Pacifique (2015-2020).

**Objectif**: Générer une animation GIF montrant l'évolution temporelle de la biomasse
micronectonique dans le Pacifique pour trois configurations :
1. SEAPODYM-LMTL (référence C++)
2. SeapoPym DAG avec transport
3. SeapoPym DAG sans transport

**Configuration**:
- Période: 2015-2020 (5 ans)
- Résolution temporelle: 1 image par semaine (moyenne hebdomadaire)
- Format de sortie: GIF animé
- Échelle: logarithmique (fixe)
- Projection: PlateCarree (cartopy)
"""

# %% Imports
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

# === CONFIGURATION DES CHEMINS ===
BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA_DIR = BASE_DIR.parent / "data"
FIGURES_DIR = BASE_DIR.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Fichiers de données
FILE_REF = DATA_DIR / "seapodym_lmtl_output_pacific_ref.zarr"
FILE_TRANS = DATA_DIR / "seapopym_pacific_transport_optimized.zarr"
FILE_NO_TRANS = DATA_DIR / "seapopym_pacific_no_transport_optimized.zarr"

print(f"Répertoire de base : {BASE_DIR}")
print(f"Répertoire données : {DATA_DIR}")
print(f"Répertoire figures : {FIGURES_DIR}")
print("✅ Imports et configuration des chemins réussis")

# %% [markdown]
# ## Configuration de l'Animation

# %%
# ============================================================================
# CONFIGURATION - Modifiez ces paramètres pour ajuster l'animation
# ============================================================================

# --- Période temporelle ---
TIME_START = "2015-01-01"
TIME_END = "2020-12-31"
RESAMPLE_FREQ = "1W"  # 1 semaine (weekly average)

# --- Paramètres de visualisation ---
VMIN_LOG = -2  # log10(biomasse) minimum (0.01 g/m²)
VMAX_LOG = 1  # log10(biomasse) maximum (10 g/m²)
CMAP = "viridis"
LAND_COLOR = "#DDDDDD"
OCEAN_COLOR = "#F0F0F0"

# --- Paramètres d'animation ---
FPS = 5  # Frames per second
DPI = 150  # Résolution de la figure
FIGURE_WIDTH = 18  # Largeur en inches
FIGURE_HEIGHT = 5  # Hauteur en inches

# --- Nom de sortie ---
FIGURE_PREFIX = "fig_06_bonus_animation_pacific"
OUTPUT_GIF = FIGURES_DIR / f"{FIGURE_PREFIX}.gif"

# ============================================================================

print("=" * 80)
print("CONFIGURATION DE L'ANIMATION")
print("=" * 80)
print(f"Période                      : {TIME_START} → {TIME_END}")
print(f"Résolution temporelle        : {RESAMPLE_FREQ} (moyenne hebdomadaire)")
print(f"Échelle de couleurs          : log10([{10**VMIN_LOG:.2f}, {10**VMAX_LOG:.2f}]) g/m²")
print(f"Colormap                     : {CMAP}")
print(f"FPS                          : {FPS}")
print(f"DPI                          : {DPI}")
print(f"Sortie                       : {OUTPUT_GIF}")
print("=" * 80)

# %% [markdown]
# ## Chargement et Préparation des Données

# %%
print("\n📊 CHARGEMENT DES DONNÉES")
print("-" * 80)

# Vérifier que tous les fichiers existent
missing = []
if not FILE_REF.exists():
    missing.append(str(FILE_REF))
if not FILE_TRANS.exists():
    missing.append(str(FILE_TRANS))
if not FILE_NO_TRANS.exists():
    missing.append(str(FILE_NO_TRANS))

if missing:
    print("\n❌ Fichiers manquants:")
    for f in missing:
        print(f"   - {f}")
    print("\nVeuillez d'abord exécuter:")
    print("  - article_05b_simulation_pacific.py")
    print("  - article_05d_benchmark_seapodym_original.py")
    raise FileNotFoundError("Missing data files")

# Référence (Seapodym original)
ds_ref = xr.open_zarr(FILE_REF)
ref = ds_ref["zooplankton"].load()
print(f"✅ Référence (SEAPODYM-LMTL) : {ref.shape}")

# Transport (SeapoPym DAG)
ds_trans = xr.open_zarr(FILE_TRANS)
dag_trans = ds_trans["biomass"].load()
print(f"✅ Transport (SeapoPym DAG)  : {dag_trans.shape}")

# No-Transport (SeapoPym DAG)
ds_no_trans = xr.open_zarr(FILE_NO_TRANS)
dag_no_trans = ds_no_trans["biomass"].load()
print(f"✅ No-Transport (SeapoPym)   : {dag_no_trans.shape}")

# %% Standardisation des coordonnées
print("\n📊 STANDARDISATION DES COORDONNÉES")
print("-" * 80)

# SeapoPym utilise y/x, Seapodym utilise latitude/longitude
rename_dict = {"y": "latitude", "x": "longitude"}

ref = ref.rename({k: v for k, v in rename_dict.items() if k in ref.dims})
dag_trans = dag_trans.rename({k: v for k, v in rename_dict.items() if k in dag_trans.dims})
dag_no_trans = dag_no_trans.rename({k: v for k, v in rename_dict.items() if k in dag_no_trans.dims})

print(f"Coordonnées standardisées : {list(ref.dims)}")

# %% Alignement temporel et resampling
print("\n📊 ALIGNEMENT TEMPOREL ET RESAMPLING")
print("-" * 80)

# Sélection de la période
ref_cut = ref.sel(time=slice(TIME_START, TIME_END))
dag_trans_cut = dag_trans.sel(time=slice(TIME_START, TIME_END))
dag_no_trans_cut = dag_no_trans.sel(time=slice(TIME_START, TIME_END))

print(f"Période sélectionnée : {TIME_START} → {TIME_END}")
print(f"  Référence    : {len(ref_cut.time)} timesteps")
print(f"  Transport    : {len(dag_trans_cut.time)} timesteps")
print(f"  No-Transport : {len(dag_no_trans_cut.time)} timesteps")

# Resampling hebdomadaire
print(f"\nResampling à {RESAMPLE_FREQ} (moyenne hebdomadaire)...")
ref_weekly = ref_cut.resample(time=RESAMPLE_FREQ).mean()
dag_trans_weekly = dag_trans_cut.resample(time=RESAMPLE_FREQ).mean()
dag_no_trans_weekly = dag_no_trans_cut.resample(time=RESAMPLE_FREQ).mean()

print(f"  Référence    : {len(ref_weekly.time)} weeks")
print(f"  Transport    : {len(dag_trans_weekly.time)} weeks")
print(f"  No-Transport : {len(dag_no_trans_weekly.time)} weeks")

# Alignement final (même grille temporelle)
ref_aligned, dag_trans_aligned = xr.align(ref_weekly, dag_trans_weekly, join="inner")
_, dag_no_trans_aligned = xr.align(ref_aligned, dag_no_trans_weekly, join="inner")

n_frames = len(ref_aligned.time)
print(f"\n✅ Nombre de frames pour l'animation : {n_frames}")
print(f"   Durée estimée du GIF : {n_frames / FPS:.1f} secondes")

# %% Configuration Matplotlib et Cartopy
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": DPI,
    }
)

print("\n✅ Configuration matplotlib et cartopy")

# %% [markdown]
# ## Génération de l'Animation

# %%
# Conversion des longitudes pour centrer sur le Pacifique
print("\n🌏 CONVERSION DES LONGITUDES POUR CENTRAGE PACIFIQUE")
print("-" * 80)


def convert_lon_to_pacific(lon):
    """Convertit les longitudes [-180, 180] en [0, 360] pour centrer sur le Pacifique."""
    return np.where(lon < 0, lon + 360, lon)


# Convertir les coordonnées
ref_aligned["longitude"] = convert_lon_to_pacific(ref_aligned.longitude)
dag_trans_aligned["longitude"] = convert_lon_to_pacific(dag_trans_aligned.longitude)
dag_no_trans_aligned["longitude"] = convert_lon_to_pacific(dag_no_trans_aligned.longitude)

# Trier les longitudes
ref_aligned = ref_aligned.sortby("longitude")
dag_trans_aligned = dag_trans_aligned.sortby("longitude")
dag_no_trans_aligned = dag_no_trans_aligned.sortby("longitude")

print(f"Longitude range : [{ref_aligned.longitude.min().values:.1f}, {ref_aligned.longitude.max().values:.1f}]")
print(f"Latitude range  : [{ref_aligned.latitude.min().values:.1f}, {ref_aligned.latitude.max().values:.1f}]")

# %% Création de l'animation
print("\n🎬 CRÉATION DE L'ANIMATION")
print("-" * 80)

# Créer la figure avec projection centrée sur le Pacifique
fig, axes = plt.subplots(
    1,
    3,
    figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
    subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
)

# Limites géographiques
lons = ref_aligned.longitude.values
lats = ref_aligned.latitude.values
extent = [lons.min(), lons.max(), lats.min(), lats.max()]

# Pré-configurer les axes (ne change pas entre frames)
for idx, ax in enumerate(axes):
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Land et coastlines
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, edgecolor="none", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black", zorder=2)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--", zorder=3)
    gl.top_labels = False
    gl.right_labels = False
    if idx > 0:
        gl.left_labels = False

# Titres des panneaux
titles = ["SEAPODYM-LMTL (Reference)", "SeapoPym with Transport", "SeapoPym without Transport"]

for ax, title in zip(axes, titles, strict=True):
    ax.set_title(title, fontsize=11, fontweight="bold")

# Initialiser les pcolormesh (seront mis à jour à chaque frame)
meshes = []
for ax in axes:
    mesh = ax.pcolormesh(
        lons,
        lats,
        np.zeros((len(lats), len(lons))),
        transform=ccrs.PlateCarree(),
        cmap=CMAP,
        vmin=VMIN_LOG,
        vmax=VMAX_LOG,
        shading="auto",
        zorder=0,
    )
    meshes.append(mesh)

# Créer la colorbar une seule fois
cbar = fig.colorbar(meshes[0], ax=axes, orientation="horizontal", pad=0.05, aspect=40, shrink=0.8)
cbar.set_label(r"$\log_{10}$(Biomass) [g/m²]", fontsize=10)

# Titre global (sera mis à jour)
title_text = fig.suptitle("", fontsize=13, fontweight="bold", y=0.98)


# Fonction d'initialisation
def init():
    """Initialise l'animation."""
    for mesh in meshes:
        mesh.set_array(np.zeros((len(lats), len(lons))).ravel())
    title_text.set_text("Initializing...")
    return meshes + [title_text]


# Fonction de mise à jour
def update(frame_idx):
    """Met à jour le frame."""
    # Données du timestep actuel
    time_current = ref_aligned.time.isel(time=frame_idx)
    time_str = pd.to_datetime(time_current.values).strftime("%Y-%m-%d")

    ref_frame = ref_aligned.isel(time=frame_idx)
    trans_frame = dag_trans_aligned.isel(time=frame_idx)
    no_trans_frame = dag_no_trans_aligned.isel(time=frame_idx)

    # Log transform
    ref_log = np.log10(ref_frame.where(ref_frame > 0))
    trans_log = np.log10(trans_frame.where(trans_frame > 0))
    no_trans_log = np.log10(no_trans_frame.where(no_trans_frame > 0))

    ref_log = ref_frame.where(ref_frame > 0)
    trans_log = trans_frame.where(trans_frame > 0)
    no_trans_log = no_trans_frame.where(no_trans_frame > 0)

    # Mettre à jour les données des meshes
    datasets = [ref_log, trans_log, no_trans_log]
    for mesh, data in zip(meshes, datasets, strict=True):
        mesh.set_array(data.values.ravel())

    # Mettre à jour le titre
    title_text.set_text(f"Micronekton Biomass Evolution - Week of {time_str}")

    return meshes + [title_text]


# Créer l'animation
print(f"Génération de {n_frames} frames...")
anim = FuncAnimation(
    fig,
    update,
    init_func=init,
    frames=tqdm(range(n_frames), desc="Rendering frames"),
    blit=True,
    repeat=True,
)

# Sauvegarder en GIF
print(f"\n💾 Sauvegarde du GIF : {OUTPUT_GIF}")
writer = PillowWriter(fps=FPS)
anim.save(OUTPUT_GIF, writer=writer, dpi=DPI)

print(f"✅ Animation sauvegardée : {OUTPUT_GIF}")
print(f"   Taille du fichier : {OUTPUT_GIF.stat().st_size / (1024**2):.1f} MB")
print(f"   Nombre de frames  : {n_frames}")
print(f"   Durée             : {n_frames / FPS:.1f} secondes")
print(f"   FPS               : {FPS}")

plt.close(fig)

# %% [markdown]
# ## Résumé

# %%
print("\n" + "=" * 80)
print("RÉSUMÉ - ANIMATION GÉNÉRÉE")
print("=" * 80)
print(f"Période             : {TIME_START} → {TIME_END}")
print(f"Résolution temporelle : {RESAMPLE_FREQ} (hebdomadaire)")
print(f"Nombre de frames    : {n_frames}")
print(f"FPS                 : {FPS}")
print(f"Durée               : {n_frames / FPS:.1f} secondes")
print(f"Échelle             : log₁₀([{10**VMIN_LOG:.2f}, {10**VMAX_LOG:.2f}]) g/m²")
print(f"Fichier de sortie   : {OUTPUT_GIF}")
print("=" * 80)
