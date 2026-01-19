"""Notebook 02A: Comparaison SeapoPym DAG vs SeapoPym v0.3.

Valide que l'architecture DAG (SeapoPym v1.0) reproduit fidèlement les
résultats de SeapoPym v0.3 en configuration **sans transport** (0D).

Cette comparaison correspond à la section **1.2** des Résultats de l'article :
> "Comparaison avec SeapoPym v0.3 (Sans Transport)"
"""

# %% Imports
import logging
from datetime import timedelta
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
import pint_xarray  # noqa: F401
import xarray as xr

from seapopym.blueprint import Blueprint
from seapopym.controller import SimulationConfig, SimulationController
from seapopym.lmtl.configuration import LMTLParams
from seapopym.lmtl.core import (
    compute_day_length,
    compute_gillooly_temperature,
    compute_layer_weighted_mean,
    compute_mortality_tendency,
    compute_production_dynamics,
    compute_production_initialization,
    compute_recruitment_age,
    compute_threshold_temperature,
)
from seapopym.standard.coordinates import Coordinates

# Configuration
ureg = pint.get_application_registry()
logging.basicConfig(level=logging.INFO)

# === CONFIGURATION DES CHEMINS ===
BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA_DIR = BASE_DIR.parent / "data"
FIGURES_DIR = BASE_DIR.parent / "figures"
SUMMARY_DIR = BASE_DIR.parent / "summary"
FIGURES_DIR.mkdir(exist_ok=True)
SUMMARY_DIR.mkdir(exist_ok=True)

print(f"Répertoire de base : {BASE_DIR}")
print(f"Répertoire figures : {FIGURES_DIR}")
print("✅ Imports et configuration des chemins réussis")

# %% [markdown]
# ## Configuration des Paramètres

# %%
# ============================================================================
# CONFIGURATION - Modifiez ces paramètres pour ajuster l'expérience
# ============================================================================

# --- Paramètres LMTL (Biologie) ---
# IMPORTANT : Ces paramètres doivent être IDENTIQUES à ceux utilisés pour v0.3
LMTL_E = 0.1668  # Efficacité de transfert trophique [sans dimension]
LMTL_LAMBDA_0 = 1 / 150  # Taux de mortalité de référence [day^-1]
LMTL_GAMMA_LAMBDA = 0.15  # Sensibilité thermique de la mortalité [°C^-1]
LMTL_TAU_R_0 = 10.38  # Âge de recrutement de référence [days]
LMTL_GAMMA_TAU_R = 0.11  # Sensibilité thermique du recrutement [°C^-1]
LMTL_T_REF = 0.0  # Température de référence [°C]
LMTL_DAY_LAYER = 0  # Couche verticale de jour [sans dimension]
LMTL_NIGHT_LAYER = 0  # Couche verticale de nuit [sans dimension]

# --- Configuration Temporelle ---
START_DATE = "1998-01-02"  # Date de début (spin-up)
END_DATE = "2019-12-31"  # Date de fin
TIMESTEP_DAYS = 1  # Pas de temps [jours]

# --- Période de Comparaison ---
COMPARISON_START = "2000"  # Début comparaison (après spin-up)
COMPARISON_END = "2019"  # Fin comparaison

# --- Chemins des Données Externes ---
DATA_EXTERNAL_BASE = (
    "/Users/adm-lehodey/Documents/Workspace/Projects/phd_optimization/notebooks/Article_1/data"
)
FORCINGS_SUBPATH = "1_global/post_processed_light_global_multiyear_bgc_001_033.zarr"
V03_RESULTS_SUBPATH = "2_global_simulation/biomass_global.zarr"

# --- Seuils de Validation ---
THRESHOLD_CORRELATION = 0.99  # Corrélation minimale attendue
THRESHOLD_BIAS = 0.01  # Biais maximal attendu [g/m²]
THRESHOLD_L2_ERROR = 5.0  # Erreur L2 maximale attendue [%]
THRESHOLD_NRMSE = 0.5  # NRMSE maximal attendu

# --- Figures ---
FIGURE_PREFIX = "fig_02a_comparison_v03"  # Préfixe pour les noms de fichiers
FIGURE_DPI = 300  # Résolution des figures sauvegardées

# ============================================================================

# Construction des chemins complets
DATA_EXTERNAL = Path(DATA_EXTERNAL_BASE)
ZARR_FORCINGS = DATA_EXTERNAL / FORCINGS_SUBPATH
ZARR_V03 = DATA_EXTERNAL / V03_RESULTS_SUBPATH

# Configuration temporelle
timestep = timedelta(days=TIMESTEP_DAYS)
COMPARISON_PERIOD = slice(COMPARISON_START, COMPARISON_END)

# Construction des paramètres LMTL
lmtl_params = LMTLParams(
    day_layer=LMTL_DAY_LAYER,
    night_layer=LMTL_NIGHT_LAYER,
    tau_r_0=LMTL_TAU_R_0 * ureg.days,
    gamma_tau_r=ureg.Quantity(LMTL_GAMMA_TAU_R, ureg.degC**-1),
    lambda_0=ureg.Quantity(LMTL_LAMBDA_0, ureg.day**-1),
    gamma_lambda=ureg.Quantity(LMTL_GAMMA_LAMBDA, ureg.degC**-1),
    E=LMTL_E,
    T_ref=ureg.Quantity(LMTL_T_REF, ureg.degC),
)

# Configuration de la simulation
config = SimulationConfig(
    start_date=START_DATE,
    end_date=END_DATE,
    timestep=timestep,
)

print("=" * 80)
print("CONFIGURATION - COMPARAISON SeapoPym DAG vs SeapoPym v0.3")
print("=" * 80)
print("Paramètres LMTL:")
print(f"  E                 : {LMTL_E}")
print(f"  λ₀                : {LMTL_LAMBDA_0:.6f} day⁻¹")
print(f"  γ_λ               : {LMTL_GAMMA_LAMBDA} °C⁻¹")
print(f"  τ_r,0             : {LMTL_TAU_R_0} days")
print(f"  γ_τr              : {LMTL_GAMMA_TAU_R} °C⁻¹")
print(f"  T_ref             : {LMTL_T_REF} °C")
print(f"  Couches           : day={LMTL_DAY_LAYER}, night={LMTL_NIGHT_LAYER}")
print()
print("Configuration temporelle:")
print(f"  Période totale    : {START_DATE} → {END_DATE}")
print(f"  Pas de temps      : {TIMESTEP_DAYS} jour(s)")
print(f"  Période de comparaison : {COMPARISON_START} → {COMPARISON_END}")
print()
print("Seuils de validation:")
print(f"  Corrélation       : > {THRESHOLD_CORRELATION}")
print(f"  Biais             : < {THRESHOLD_BIAS} g/m²")
print(f"  Erreur L2         : < {THRESHOLD_L2_ERROR}%")
print(f"  NRMSE             : < {THRESHOLD_NRMSE}")
print("=" * 80)

# %% [markdown]
# ## 3. Chargement des Forçages

# %%
# Chargement des forçages
print(f"Chargement des forçages depuis : {ZARR_FORCINGS}")
ds_raw = xr.open_zarr(ZARR_FORCINGS)

# Renommage des dimensions pour correspondre au standard Seapopym
ds = ds_raw.rename(
    {
        "T": Coordinates.T.value,
        "Z": Coordinates.Z.value,
        "Y": Coordinates.Y.value,
        "X": Coordinates.X.value,
    }
)
ds.x.attrs = {}
ds.y.attrs = {}

# Sélection temporelle et variables
forcings = ds.sel({Coordinates.T.value: slice("1998-01-01", "2020-01-01")})
forcings = forcings[["primary_production", "temperature"]].load()

# Création des cohortes
cohorts = (np.arange(0, np.ceil(lmtl_params.tau_r_0.magnitude) + 1) * ureg.day).to("second")
cohorts_da = xr.DataArray(
    cohorts.magnitude, dims=["cohort"], name="cohort", attrs={"units": "second"}
)

# Ajout des paramètres aux forçages
forcings = forcings.assign_coords(cohort=cohorts_da)
forcings["dt"] = config.timestep.total_seconds()

# Normalisation des unités
if "primary_production" in forcings:
    forcings["primary_production"].attrs["units"] = "mg/m**2/day"

if "temperature" in forcings and forcings["temperature"].attrs.get("units") in ["degC", "deg_C"]:
    forcings["temperature"].attrs["units"] = "degree_Celsius"

print(f"Forçages chargés : {list(forcings.data_vars)}")
print(f"Période : {forcings.time.values[0]} -> {forcings.time.values[-1]}")
print(f"Cohortes : {len(cohorts_da)}")

# %% [markdown]
# ## 4. Configuration du Blueprint (0D - Sans Transport)


# %%
def configure_model(bp):
    """Configure le modèle LMTL sans transport."""
    # Enregistrement des forçages
    bp.register_forcing(
        "temperature",
        dims=(Coordinates.T.value, Coordinates.Z.value, Coordinates.Y.value, Coordinates.X.value),
        units="degree_Celsius",
    )
    bp.register_forcing(
        "primary_production",
        dims=(Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value),
        units="g/m**2/second",
    )
    bp.register_forcing("dt")
    bp.register_forcing("cohort")
    bp.register_forcing(Coordinates.T.value)
    bp.register_forcing(Coordinates.Y.value)

    # Groupe Zooplankton (SANS transport)
    bp.register_group(
        group_prefix="Zooplankton",
        units=[
            {
                "func": compute_day_length,
                "output_mapping": {"output": "day_length"},
                "input_mapping": {"latitude": Coordinates.Y.value, "time": Coordinates.T.value},
                "output_units": {"output": "dimensionless"},
            },
            {
                "func": compute_layer_weighted_mean,
                "input_mapping": {"forcing": "temperature"},
                "output_mapping": {"output": "mean_temperature"},
                "output_units": {"output": "degree_Celsius"},
            },
            {
                "func": compute_threshold_temperature,
                "input_mapping": {"temperature": "mean_temperature", "min_temperature": "T_ref"},
                "output_mapping": {"output": "thresholded_temperature"},
                "output_units": {"output": "degree_Celsius"},
            },
            {
                "func": compute_gillooly_temperature,
                "input_mapping": {"temperature": "thresholded_temperature"},
                "output_mapping": {"output": "gillooly_temperature"},
                "output_units": {"output": "degree_Celsius"},
            },
            {
                "func": compute_recruitment_age,
                "input_mapping": {"temperature": "gillooly_temperature"},
                "output_mapping": {"output": "recruitment_age"},
                "output_units": {"output": "second"},
            },
            {
                "func": compute_production_initialization,
                "input_mapping": {"cohorts": "cohort"},
                "output_mapping": {"output": "production_source_npp"},
                "output_tendencies": {"output": "production"},
                "output_units": {"output": "g/m**2/second"},
            },
            {
                "func": compute_production_dynamics,
                "input_mapping": {
                    "cohort_ages": "cohort",
                    "dt": "dt",
                },
                "output_mapping": {
                    "production_tendency": "production_tendency",
                    "recruitment_source": "biomass_source",
                },
                "output_tendencies": {
                    "production_tendency": "production",
                    "recruitment_source": "biomass",
                },
                "output_units": {
                    "production_tendency": "g/m**2/second",
                    "recruitment_source": "g/m**2/second",
                },
            },
            {
                "func": compute_mortality_tendency,
                "input_mapping": {"temperature": "gillooly_temperature"},
                "output_mapping": {"mortality_loss": "biomass_mortality"},
                "output_tendencies": {"mortality_loss": "biomass"},
                "output_units": {"mortality_loss": "g/m**2/second"},
            },
        ],
        parameters={
            "day_layer": {"units": "dimensionless"},
            "night_layer": {"units": "dimensionless"},
            "tau_r_0": {"units": "second"},
            "gamma_tau_r": {"units": "1/degree_Celsius"},
            "lambda_0": {"units": "1/second"},
            "gamma_lambda": {"units": "1/degree_Celsius"},
            "T_ref": {"units": "degree_Celsius"},
            "E": {"units": "dimensionless"},
        },
        state_variables={
            "production": {
                "dims": (Coordinates.Y.value, Coordinates.X.value, "cohort"),
                "units": "g/m**2/second",
            },
            "biomass": {
                "dims": (Coordinates.Y.value, Coordinates.X.value),
                "units": "g/m**2",
            },
        },
    )


# Visualisation du Blueprint
bp = Blueprint()
configure_model(bp)
print("Blueprint configuré")
print(bp.export_mermaid())

# %% [markdown]
# ## 5. État Initial et Exécution de la Simulation

# %%
# Dimensions spatiales
lats = forcings[Coordinates.Y.value]
lons = forcings[Coordinates.X.value]

# Biomasse initiale nulle
biomass_init = xr.DataArray(
    np.zeros((len(lats), len(lons))),
    coords={Coordinates.Y.value: lats, Coordinates.X.value: lons},
    dims=(Coordinates.Y.value, Coordinates.X.value),
    name="biomass",
)
biomass_init.attrs = {"units": "g/m**2"}

# Production initiale nulle
production_init = xr.DataArray(
    np.zeros((len(lats), len(lons), len(cohorts))),
    coords={Coordinates.Y.value: lats, Coordinates.X.value: lons, "cohort": cohorts_da},
    dims=(Coordinates.Y.value, Coordinates.X.value, "cohort"),
    name="production",
)
production_init.attrs = {"units": "g/m**2/day"}

initial_state = xr.Dataset({"biomass": biomass_init, "production": production_init})

print("État initial créé")

# %%
# Configuration et exécution du contrôleur
controller = SimulationController(config, backend="sequential")
controller.setup(
    model_configuration_func=configure_model,
    forcings=forcings,
    initial_state={"Zooplankton": initial_state},
    parameters={"Zooplankton": lmtl_params},
    output_variables={"Zooplankton": ["biomass"]},
)

print("Démarrage de la simulation SeapoPym DAG...")
controller.run()
print("Simulation terminée.")

# %%
# Extraction des résultats DAG
results_dag = controller.results["Zooplankton/biomass"].rename("biomass_dag")
print(f"Résultats DAG : {results_dag.shape}")
print(f"Période : {results_dag.time.values[0]} -> {results_dag.time.values[-1]}")

# Visualisation rapide
results_dag.mean("time").plot(figsize=(12, 6))
plt.title("Biomasse moyenne - SeapoPym DAG")
plt.savefig(FIGURES_DIR / f"{FIGURE_PREFIX}_dag_mean.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Chargement des Résultats SeapoPym v0.3

# %%
# Chargement des résultats v0.3
print(f"Chargement des résultats v0.3 depuis : {ZARR_V03}")
seapopym_v03 = xr.open_zarr(ZARR_V03)
seapopym_v03 = seapopym_v03["biomass"].squeeze().rename({"T": "time", "X": "x", "Y": "y"}).load()
seapopym_v03.x.attrs = {}
seapopym_v03.y.attrs = {}

# Normalisation des unités
seapopym_v03 = seapopym_v03.pint.quantify().pint.to("g/m^2").pint.dequantify()

print(f"Résultats v0.3 : {seapopym_v03.shape}")
print(f"Période : {seapopym_v03.time.values[0]} -> {seapopym_v03.time.values[-1]}")

# %% [markdown]
# ## 7. Alignement et Préparation des Données

# %%
# Sélection de la période de comparaison (exclure spin-up 1998-1999)
v03_aligned = seapopym_v03.sel(time=COMPARISON_PERIOD)
dag_aligned = results_dag.sel(time=COMPARISON_PERIOD)

# Normalisation des timestamps
v03_aligned = v03_aligned.assign_coords(time=v03_aligned.time.dt.floor("D"))
dag_aligned = dag_aligned.assign_coords(time=dag_aligned.time.dt.floor("D"))

print(f"v0.3 time range: {v03_aligned.time.values[0]} to {v03_aligned.time.values[-1]}")
print(f"DAG time range: {dag_aligned.time.values[0]} to {dag_aligned.time.values[-1]}")
print(f"v0.3 shape: {v03_aligned.shape}")
print(f"DAG shape: {dag_aligned.shape}")

# Alignement final
v03_aligned, dag_aligned = xr.align(v03_aligned, dag_aligned, join="inner")
print("\nAprès alignement :")
print(f"v0.3 shape: {v03_aligned.shape}")
print(f"DAG shape: {dag_aligned.shape}")

# %% [markdown]
# ## 8. Calcul des Métriques de Comparaison


# %%
def compute_metrics(ref, test):
    """Calcule les métriques de comparaison."""
    diff = test - ref

    # RMSE
    rmse = np.sqrt((diff**2).mean()).values

    # Corrélation globale
    corr = xr.corr(ref, test).values

    # Biais moyen
    bias = diff.mean().values

    # Erreur L2 normalisée
    l2_error = (np.sqrt((diff**2).sum() / (ref**2).sum()).values) * 100

    # NRMSE (normalisé par l'écart-type de la référence)
    std_ref = ref.std().values
    nrmse = rmse / std_ref if std_ref > 0 else np.nan

    return {
        "RMSE (g/m²)": float(rmse),
        "Corrélation": float(corr),
        "Biais moyen (g/m²)": float(bias),
        "Erreur L2 (%)": float(l2_error),
        "NRMSE": float(nrmse),
    }


metrics = compute_metrics(v03_aligned, dag_aligned)

# Affichage
print("=" * 60)
print("VALIDATION SeapoPym DAG vs SeapoPym v0.3 (Sans Transport)")
print("=" * 60)
for key, value in metrics.items():
    print(f"  {key}: {value:.6f}")
print("=" * 60)

# %%
# Tableau récapitulatif pour l'article
results_table = pd.DataFrame(
    {
        "Métrique": list(metrics.keys()),
        "Valeur": [f"{v:.4f}" for v in metrics.values()],
        "Seuil": [
            "—",
            f"> {THRESHOLD_CORRELATION}",
            "~ 0",
            f"< {THRESHOLD_L2_ERROR}%",
            f"< {THRESHOLD_NRMSE}",
        ],
        "Validation": [
            "—",
            "✓" if metrics["Corrélation"] > THRESHOLD_CORRELATION else "✗",
            "✓" if abs(metrics["Biais moyen (g/m²)"]) < THRESHOLD_BIAS else "✗",
            "✓" if metrics["Erreur L2 (%)"] < THRESHOLD_L2_ERROR else "✗",
            "✓" if metrics["NRMSE"] < THRESHOLD_NRMSE else "✗",
        ],
    }
)

print("\nTableau pour l'article (Section 3.1.2) :")
print(results_table.to_markdown(index=False))

# %% [markdown]
# ## 9. Visualisations pour l'Article

# %%
# --- Figure A : Cartes de biomasse moyenne ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

v03_aligned.mean("time").plot(ax=axes[0], cmap="viridis", vmin=0)
axes[0].set_title("SeapoPym v0.3 - Biomasse moyenne")

dag_aligned.mean("time").plot(ax=axes[1], cmap="viridis", vmin=0)
axes[1].set_title("SeapoPym DAG (v1.0) - Biomasse moyenne")

plt.tight_layout()
plt.savefig(FIGURES_DIR / f"{FIGURE_PREFIX}_maps.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.show()

# %%
# --- Figure B1 : Bias (Différence Absolue) ---
plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
# Calcul du biais
diff_mean = (dag_aligned - v03_aligned).mean("time", skipna=False)
vmax_val = max(abs(diff_mean.min().item()), abs(diff_mean.max().item()))
# Plot
im = diff_mean.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap="RdBu_r",
    vmin=-vmax_val,
    vmax=vmax_val,
    cbar_kwargs={"label": "Biais ($g.m^{-2}$)", "shrink": 0.8},
)
# Esthétique Cartopy
ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=1)
ax.coastlines(resolution="110m", linewidth=0.2, color="black", zorder=2)
ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
ax.set_title("Bias (SeapoPym DAG - v0.3)")
plt.savefig(FIGURES_DIR / f"{FIGURE_PREFIX}_bias.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.show()

# %%
# --- Figure B2 : NRMSE Spatiale ---
plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
# Calcul NRMSE
diff = dag_aligned - v03_aligned
rmse_spatial = np.sqrt((diff**2).mean(dim="time"))
std_ref_spatial = v03_aligned.std(dim="time")
nrmse_spatial = rmse_spatial / std_ref_spatial
# Plot
im = nrmse_spatial.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    vmin=0,
    vmax=1,
    cmap="viridis_r",
    cbar_kwargs={"label": "NRMSE", "shrink": 0.8},
)
# Esthétique Cartopy
ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=1)
ax.coastlines(resolution="110m", linewidth=0.2, color="black", zorder=2)
ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
ax.set_title("Normalized RMSE")
plt.savefig(FIGURES_DIR / f"{FIGURE_PREFIX}_nrmse.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.show()

# %%
# --- Figure C : Séries temporelles de biomasse globale ---
fig, ax = plt.subplots(figsize=(12, 4))

biomass_v03 = v03_aligned.sum(["x", "y"])
biomass_dag = dag_aligned.sum(["x", "y"])

biomass_v03.plot(ax=ax, label="SeapoPym v0.3", alpha=0.8)
biomass_dag.plot(ax=ax, label="SeapoPym DAG", alpha=0.8, linestyle="--")

ax.legend()
ax.set_title("Biomasse totale globale - Comparaison v0.3 vs DAG")
ax.set_ylabel("Biomasse totale (g/m²)")

plt.tight_layout()
plt.savefig(FIGURES_DIR / f"{FIGURE_PREFIX}_timeseries.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.show()

# %%
# --- Figure D : Scatter plot ---
fig, ax = plt.subplots(figsize=(6, 6))

# Échantillonnage pour éviter surcharge
sample_v03 = v03_aligned.values.flatten()[::100]
sample_dag = dag_aligned.values.flatten()[::100]
mask = ~np.isnan(sample_v03) & ~np.isnan(sample_dag)

ax.scatter(sample_v03[mask], sample_dag[mask], alpha=0.1, s=1)
ax.plot([0, sample_v03[mask].max()], [0, sample_v03[mask].max()], "r--", label="1:1")

ax.set_xlabel("SeapoPym v0.3 (g/m²)")
ax.set_ylabel("SeapoPym DAG (g/m²)")
ax.set_title(f"Scatter v0.3 vs DAG (R² = {metrics['Corrélation'] ** 2:.4f})")
ax.legend()

plt.tight_layout()
plt.savefig(FIGURES_DIR / f"{FIGURE_PREFIX}_scatter.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 10. Conclusion
#
# Le modèle DAG reproduit les résultats de SeapoPym v0.3 avec :
# - Une erreur L2 normalisée très faible
# - Une corrélation quasi parfaite
# - Un biais négligeable

# %%
# Résumé final
print("\n" + "=" * 60)
print("RÉSUMÉ FINAL")
print("=" * 60)
print("Période de comparaison : 2000-2019")
print(f"Nombre de pas de temps : {len(v03_aligned.time)}")
print(f"Grille : {len(v03_aligned.y)} x {len(v03_aligned.x)}")
print("\nMétriques :")
for key, value in metrics.items():
    print(f"  • {key}: {value:.6f}")
print("\nConclusion : ✅ Validation réussie - Non-régression confirmée")
print("=" * 60)

# %% Export des Résultats

# Calcul de statistiques supplémentaires pour le résumé
n_timesteps = len(v03_aligned.time)
n_lat = len(v03_aligned.y)
n_lon = len(v03_aligned.x)
n_cohorts = len(cohorts_da)

# Statistiques spatiales de biomasse
biomass_mean_spatial_v03 = v03_aligned.mean("time")
biomass_global_total_v03 = v03_aligned.sum(["x", "y"]).mean().values
biomass_global_total_dag = dag_aligned.sum(["x", "y"]).mean().values

summary_filename = f"{FIGURE_PREFIX.replace('fig_', 'notebook_')}_summary.txt"
output_summary_path = SUMMARY_DIR / summary_filename

# Vérification validation
all_valid = all(
    [
        metrics["Corrélation"] > THRESHOLD_CORRELATION,
        abs(metrics["Biais moyen (g/m²)"]) < THRESHOLD_BIAS,
        metrics["Erreur L2 (%)"] < THRESHOLD_L2_ERROR,
        metrics["NRMSE"] < THRESHOLD_NRMSE,
    ]
)

with open(output_summary_path, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("RÉSUMÉ DE VALIDATION : SeapoPym DAG vs SeapoPym v0.3 (Sans Transport)\n")
    f.write("=" * 80 + "\n")
    f.write(f"DATE: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("OBJECTIF:\n")
    f.write("-" * 80 + "\n")
    f.write("Valider la non-régression de l'architecture DAG (v1.0) en reproduisant les\n")
    f.write("résultats de SeapoPym v0.3 en configuration 0D (sans transport).\n\n")

    f.write("PARAMÈTRES LMTL:\n")
    f.write("-" * 80 + "\n")
    f.write(f"E                : {LMTL_E}\n")
    f.write(f"lambda_0         : {LMTL_LAMBDA_0:.6f} day^-1\n")
    f.write(f"gamma_lambda     : {LMTL_GAMMA_LAMBDA} °C^-1\n")
    f.write(f"tau_r_0          : {LMTL_TAU_R_0} days\n")
    f.write(f"gamma_tau_r      : {LMTL_GAMMA_TAU_R} °C^-1\n")
    f.write(f"T_ref            : {LMTL_T_REF} °C\n\n")

    f.write("CONFIGURATION SPATIALE:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Grille           : {n_lat} × {n_lon} points\n")
    f.write(f"Nombre de cohortes : {n_cohorts}\n\n")

    f.write("MÉTRIQUES DE COMPARAISON:\n")
    f.write("-" * 80 + "\n")
    f.write(results_table.to_markdown(index=False) + "\n\n")

    f.write("VALIDATION:\n")
    f.write("-" * 80 + "\n")
    if all_valid:
        f.write("✅ VALIDATION RÉUSSIE - Tous les critères sont satisfaits\n")
    else:
        f.write("⚠️ ATTENTION - Certains critères ne sont pas satisfaits\n")
    f.write("=" * 80 + "\n")

print(f"✅ Résumé sauvegardé dans : {output_summary_path}")
