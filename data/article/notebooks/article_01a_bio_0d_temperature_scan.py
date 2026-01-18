"""Notebook 01A: Validation Biologie 0D - Scan en Température.

**Objectif**: Valider la convergence asymptotique des processus biologiques
sur une plage continue de températures (0°C à 35°C).

**Approche Adaptative**:
La dynamique biologique dépend fortement de la température :
- À basse température, la mortalité (λ) est faible → convergence lente → simulation longue.
- À haute température, la mortalité est élevée → convergence rapide → simulation courte.

Nous adaptons dynamiquement la durée de simulation et le pas de temps pour chaque
température afin d'optimiser le coût de calcul tout en garantissant la précision.
"""

# %% Imports
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
import xarray as xr

from seapopym.controller import SimulationConfig, SimulationController
from seapopym.lmtl.configuration import LMTLParams
from seapopym.lmtl.core import (
    compute_gillooly_temperature,
    compute_mortality_tendency,
    compute_production_dynamics,
    compute_production_initialization,
    compute_recruitment_age,
    compute_threshold_temperature,
)

ureg = pint.get_application_registry()

# === CONFIGURATION DES CHEMINS ===
BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA_DIR = BASE_DIR.parent / "data"
FIGURES_DIR = BASE_DIR.parent / "figures"
SUMMARY_DIR = BASE_DIR.parent / "summary"
FIGURES_DIR.mkdir(exist_ok=True)
SUMMARY_DIR.mkdir(exist_ok=True)

print("✅ Imports réussis")

# %% [markdown]
# ## Configuration des Paramètres

# %%
# ============================================================================
# CONFIGURATION - Modifiez ces paramètres pour ajuster l'expérience
# ============================================================================

# --- Paramètres LMTL (Biologie) ---
LMTL_E = 0.1668  # Efficacité de transfert trophique [sans dimension]
LMTL_LAMBDA_0 = 1 / 150  # Taux de mortalité de référence [day^-1]
LMTL_GAMMA_LAMBDA = 0.15  # Sensibilité thermique de la mortalité [°C^-1]
LMTL_TAU_R_0 = 10.38  # Âge de recrutement de référence [days]
LMTL_GAMMA_TAU_R = 0.11  # Sensibilité thermique du recrutement [°C^-1]
LMTL_T_REF = 0.0  # Température de référence [°C]
LMTL_DAY_LAYER = 0  # Couche verticale de jour [sans dimension]
LMTL_NIGHT_LAYER = 0  # Couche verticale de nuit [sans dimension]

# --- Forçage NPP ---
NPP_MG_M2_DAY = 300.0  # Production primaire nette [mg/m²/day]

# --- Scan en Température ---
TEMP_MIN = 0  # Température minimale [°C]
TEMP_MAX = 35  # Température maximale [°C]
TEMP_STEP = 1  # Pas de température [°C]

# --- Approche Adaptative (Critères de Simulation) ---
DURATION_FACTOR = 15.0  # Durée = FACTOR × (1/λ) pour convergence > 99.9%
DT_FACTOR = 100.0  # Pas de temps = (1/λ) / FACTOR pour résolution fine
DT_MIN_SECONDS = 1.0  # Pas de temps minimum [s]
DT_MAX_SECONDS = 12 * 3600.0  # Pas de temps maximum [s] (12 heures)

# --- Configuration Temporelle ---
START_DATE = "2000-01-01"  # Date de début des simulations

# --- Figures ---
FIGURE_PREFIX = "fig_01a_temperature_scan"  # Préfixe pour les noms de fichiers
FIGURE_FORMATS = ["png"]  # Formats de sauvegarde

# ============================================================================

# Conversion NPP en unités SI
NPP_g_m2_s = (NPP_MG_M2_DAY * 1e-3) / 86400.0

# Plage de températures
TEMPERATURES = np.arange(TEMP_MIN, TEMP_MAX + 1, TEMP_STEP)

# Construction des paramètres LMTL
lmtl_params = LMTLParams(
    day_layer=ureg.Quantity(LMTL_DAY_LAYER, ureg.dimensionless),
    night_layer=ureg.Quantity(LMTL_NIGHT_LAYER, ureg.dimensionless),
    tau_r_0=ureg.Quantity(LMTL_TAU_R_0, ureg.day),
    gamma_tau_r=ureg.Quantity(LMTL_GAMMA_TAU_R, ureg.degC**-1),
    lambda_0=ureg.Quantity(LMTL_LAMBDA_0, ureg.day**-1),
    gamma_lambda=ureg.Quantity(LMTL_GAMMA_LAMBDA, ureg.degC**-1),
    E=ureg.Quantity(LMTL_E, ureg.dimensionless),
    T_ref=ureg.Quantity(LMTL_T_REF, ureg.degC),
)

print("=" * 80)
print("CONFIGURATION - VALIDATION BIOLOGIE 0D - SCAN EN TEMPÉRATURE")
print("=" * 80)
print(f"Efficacité E              : {LMTL_E}")
print(f"λ₀                        : {LMTL_LAMBDA_0:.6f} day⁻¹")
print(f"γ_λ                       : {LMTL_GAMMA_LAMBDA} °C⁻¹")
print(f"τ_r,0                     : {LMTL_TAU_R_0} days")
print(f"γ_τr                      : {LMTL_GAMMA_TAU_R} °C⁻¹")
print(f"T_ref                     : {LMTL_T_REF} °C")
print(f"NPP                       : {NPP_g_m2_s:.2e} g/m²/s ({NPP_MG_M2_DAY} mg/m²/day)")
print(f"Plage de températures     : {TEMP_MIN}°C à {TEMP_MAX}°C (pas de {TEMP_STEP}°C)")
print(f"Nombre de simulations     : {len(TEMPERATURES)}")
print(f"Critère durée             : {DURATION_FACTOR} × (1/λ)")
print(f"Critère dt                : (1/λ) / {DT_FACTOR}")
print("=" * 80)

# %% Configuration Matplotlib
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

COLORS = {
    "blue": "#0077BB",
    "orange": "#EE7733",
    "green": "#009988",
    "red": "#CC3311",
    "purple": "#AA3377",
}


def save_figure(fig, name, formats=FIGURE_FORMATS):
    """Sauvegarde une figure dans les formats requis."""
    for fmt in formats:
        filepath = FIGURES_DIR / f"{name}.{fmt}"
        fig.savefig(filepath, format=fmt, dpi=300, bbox_inches="tight")
        print(f"✅ Saved: {filepath}")


# %% [markdown]
# ## Configuration du Blueprint 0D


# %%
def configure_0d_biology_model(bp):
    """Configure un Blueprint 0D (biologie seule)."""
    # Forcings
    bp.register_forcing("temperature", dims=(), units="degree_Celsius")
    bp.register_forcing("primary_production", dims=(), units="g/m**2/second")
    bp.register_forcing("dt", units="second")
    bp.register_forcing("cohort", dims=("cohort",), units="second")

    # Groupe fonctionnel
    bp.register_group(
        group_prefix="Zooplankton",
        units=[
            {
                "func": compute_threshold_temperature,
                "input_mapping": {"temperature": "temperature", "min_temperature": "T_ref"},
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
                "input_mapping": {"cohort_ages": "cohort", "dt": "dt"},
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
            "production": {"dims": ("cohort",), "units": "g/m**2/second"},
            "biomass": {"dims": (), "units": "g/m**2"},
        },
    )


# %% [markdown]
# ## 2. Calcul Théorique et Paramètres de Simulation
#
# Nous calculons d'abord les valeurs théoriques (λ, B_eq) pour chaque température.


# %%
def compute_theory_and_sim_params(T, NPP, params):
    """Calcule les valeurs théoriques et paramètres de simulation adaptatifs."""
    T_thresh = max(T, params.T_ref.magnitude)
    T_gillooly = T_thresh / (1 + T_thresh / 273.0)

    # Mortalité
    lambda_T = params.lambda_0.to("1/second").magnitude * np.exp(
        params.gamma_lambda.magnitude * T_gillooly
    )

    # Âge recrutement
    tau_r = params.tau_r_0.to("second").magnitude * np.exp(
        -params.gamma_tau_r.magnitude * T_gillooly
    )

    # Equilibre
    R = params.E.magnitude * NPP
    B_eq = R / lambda_T

    # Paramètres de simulation adaptatifs
    time_scale = 1.0 / lambda_T
    duration_seconds = DURATION_FACTOR * time_scale
    dt_seconds = time_scale / DT_FACTOR
    dt_seconds = min(dt_seconds, DT_MAX_SECONDS)
    dt_seconds = max(dt_seconds, DT_MIN_SECONDS)

    return {
        "B_eq": B_eq,
        "lambda": lambda_T,
        "tau_r": tau_r,
        "time_scale_days": time_scale / 86400.0,
        "duration_days": duration_seconds / 86400.0,
        "dt_seconds": dt_seconds,
    }


# Pré-calcul
scan_config = {}
print(
    f"{'T [°C]':<8} {'λ [1/d]':<12} {'1/λ [d]':<12} {'Durée [d]':<12} {'dt [s]':<12} {'B_eq':<10}"
)
print("-" * 70)

for T in TEMPERATURES:
    cfg = compute_theory_and_sim_params(T, NPP_g_m2_s, lmtl_params)
    scan_config[T] = cfg

    if T % 5 == 0:  # Afficher tous les 5 degrés
        print(
            f"{T:<8} {cfg['lambda'] * 86400:<12.4f} {cfg['time_scale_days']:<12.1f} "
            f"{cfg['duration_days']:<12.1f} {cfg['dt_seconds']:<12.0f} {cfg['B_eq']:<10.4f}"
        )

# %% [markdown]
# ## 3. Exécution des Simulations

# %%
results_data = []

print("Démarrage du scan en température...")

for i, T in enumerate(TEMPERATURES):
    # Paramètres spécifiques
    cfg = scan_config[T]
    dt_val = cfg["dt_seconds"]
    duration_days = cfg["duration_days"]

    # Configuration temporelle
    start = datetime.fromisoformat(START_DATE)
    end = start + timedelta(days=duration_days)
    n_steps = int(np.ceil((end - start).total_seconds() / dt_val))
    timestep_delta = timedelta(seconds=dt_val)

    config = SimulationConfig(
        start_date=START_DATE,
        end_date=(start + n_steps * timestep_delta).isoformat(),
        timestep=timestep_delta,
    )

    # Cohortes adaptées à tau_r
    tau_r_0_days = lmtl_params.tau_r_0.to("day").magnitude
    cohorts = (np.arange(0, np.ceil(tau_r_0_days) + 1) * ureg.day).to("second")
    cohorts_da = xr.DataArray(cohorts.magnitude, dims=["cohort"], attrs={"units": "second"})

    # Forçages
    time_da = xr.DataArray(
        pd.date_range(start=START_DATE, periods=n_steps, freq=timestep_delta),
        dims=["time"],
    )

    forcings = xr.Dataset(
        {
            "temperature": xr.DataArray(
                np.full(n_steps, T), dims=["time"], attrs={"units": "degC"}
            ),
            "primary_production": xr.DataArray(
                np.full(n_steps, NPP_g_m2_s), dims=["time"], attrs={"units": "g/m**2/s"}
            ),
            "dt": dt_val,
            "cohort": cohorts_da,
        },
        coords={"time": time_da},
    )

    # État initial nul
    initial_state = xr.Dataset(
        {
            "biomass": xr.DataArray(0.0),
            "production": xr.DataArray(
                np.zeros(len(cohorts)), coords={"cohort": cohorts_da}, dims=["cohort"]
            ),
        }
    )

    # Controller
    controller = SimulationController(config, backend="sequential")
    controller.setup(
        configure_0d_biology_model,
        forcings=forcings,
        initial_state={"Zooplankton": initial_state},
        parameters={"Zooplankton": asdict(lmtl_params)},
        output_variables={"Zooplankton": ["biomass"]},
    )

    # Run
    if i % 10 == 0:
        print(f"  Simulation T={T}°C (dt={dt_val:.0f}s, Durée={duration_days:.1f}d)...")
    controller.run()

    # Analyse
    biomass_final = controller.results["Zooplankton/biomass"].values[-1]
    B_theory = cfg["B_eq"]
    error = 100 * abs(biomass_final - B_theory) / B_theory

    results_data.append(
        {
            "Temperature": T,
            "Simulated": biomass_final,
            "Theoretical": B_theory,
            "Error_Percent": error,
            "dt": dt_val,
            "duration": duration_days,
        }
    )

df_results = pd.DataFrame(results_data)
print("✅ Scan terminé.")

# %% [markdown]
# ## 4. Analyse et Visualisation

# %%
# Figure 1 : Comparaison Simulation vs Théorie
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.9, 7), sharex=True)

# Panel A: Biomasse absolue
ax1.plot(
    df_results["Temperature"], df_results["Theoretical"], "k--", label="Théorie", linewidth=1.5
)
ax1.plot(
    df_results["Temperature"],
    df_results["Simulated"],
    "o",
    color=COLORS["blue"],
    markersize=4,
    label="Simulation",
    alpha=0.8,
)
ax1.set_ylabel("Biomasse à l'équilibre [g/m²]")
ax1.set_yscale("log")
ax1.set_title("A. Biomasse d'Équilibre vs Température")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel B: Erreur Relative
ax2.bar(
    df_results["Temperature"],
    df_results["Error_Percent"],
    color=COLORS["red"],
    alpha=0.7,
    width=0.8,
)
ax2.axhline(0, color="black", linewidth=0.5)
ax2.set_xlabel("Température [°C]")
ax2.set_ylabel("Erreur Relative [%]")
ax2.set_title("B. Précision de la Simulation")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, FIGURE_PREFIX)
plt.show()

# %% [markdown]
# ## 5. Conclusion

# %%
print("Statistiques d'erreur :")
print(f"  Max Error: {df_results['Error_Percent'].max():.4f}%")
print(f"  Mean Error: {df_results['Error_Percent'].mean():.4f}%")
print(
    f"  Température Max Error: {df_results.loc[df_results['Error_Percent'].idxmax(), 'Temperature']}°C"
)

if df_results["Error_Percent"].max() < 0.1:
    print("\n✅ VALIDATION RÉUSSIE : L'erreur est négligeable sur toute la plage.")
else:
    print("\n⚠️ ATTENTION : Certaines erreurs dépassent 0.1%.")

# %% Génération du résumé
# Calcul du NRMSE
rmse = np.sqrt(((df_results["Simulated"] - df_results["Theoretical"]) ** 2).mean())
nrmse_val = rmse / df_results["Theoretical"].mean()

# Extraction des valeurs min/max pour le résumé
lambda_values = np.array([scan_config[T]["lambda"] for T in TEMPERATURES])
beq_values = df_results["Theoretical"].values
time_scale_values = np.array([scan_config[T]["time_scale_days"] for T in TEMPERATURES])
dt_values = df_results["dt"].values
duration_values = df_results["duration"].values

# Génération du fichier résumé
summary_filename = f"{FIGURE_PREFIX.replace('fig_', 'notebook_')}_summary.txt"
summary_path = SUMMARY_DIR / summary_filename

with open(summary_path, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("NOTEBOOK 1A: VALIDATION BIOLOGIE 0D - SCAN EN TEMPÉRATURE\n")
    f.write("=" * 80 + "\n\n")

    f.write("DATE: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")

    f.write("OBJECTIF:\n")
    f.write("-" * 80 + "\n")
    f.write("Valider la convergence asymptotique des processus biologiques sur une plage\n")
    f.write(f"de températures ({TEMP_MIN}°C à {TEMP_MAX}°C) avec pas de temps adaptatif.\n\n")

    f.write("CONFIGURATION DE L'EXPÉRIENCE:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Plage de températures : {TEMP_MIN}°C à {TEMP_MAX}°C\n")
    f.write(f"Pas de température    : {TEMP_STEP}°C\n")
    f.write(f"Nombre de simulations : {len(TEMPERATURES)}\n")
    f.write(f"Nombre de cohortes    : {len(cohorts)}\n\n")

    f.write("PARAMÈTRES DU MODÈLE:\n")
    f.write("-" * 80 + "\n")
    f.write(f"E                : {LMTL_E}\n")
    f.write(f"lambda_0         : {LMTL_LAMBDA_0:.6f} day^-1\n")
    f.write(f"                   ({LMTL_LAMBDA_0 / 86400:.2e} s^-1)\n")
    f.write(f"gamma_lambda     : {LMTL_GAMMA_LAMBDA} °C^-1\n")
    f.write(f"tau_r_0          : {LMTL_TAU_R_0} days\n")
    f.write(f"gamma_tau_r      : {LMTL_GAMMA_TAU_R} °C^-1\n")
    f.write(f"T_ref            : {LMTL_T_REF} °C\n")
    f.write(f"NPP              : {NPP_g_m2_s:.2e} g/m²/s ({NPP_MG_M2_DAY} mg/m²/day)\n\n")

    f.write("VARIABILITÉ DU SYSTÈME:\n")
    f.write("-" * 80 + "\n")
    f.write("λ(T) - Taux de mortalité:\n")
    f.write(f"  Min (0°C)      : {lambda_values.min() * 86400:.4f} day^-1\n")
    f.write(f"  Max ({TEMP_MAX}°C)     : {lambda_values.max() * 86400:.4f} day^-1\n")
    f.write(f"  Ratio max/min  : {lambda_values.max() / lambda_values.min():.1f}\n\n")
    f.write("1/λ(T) - Échelle de temps caractéristique:\n")
    f.write(f"  Min ({TEMP_MAX}°C)     : {time_scale_values.min():.2f} days\n")
    f.write(f"  Max (0°C)      : {time_scale_values.max():.1f} days\n")
    f.write(f"  Ratio max/min  : {time_scale_values.max() / time_scale_values.min():.1f}\n\n")
    f.write("B_eq(T) - Biomasse d'équilibre:\n")
    f.write(f"  Min ({TEMP_MAX}°C)     : {beq_values.min():.4f} g/m²\n")
    f.write(f"  Max (0°C)      : {beq_values.max():.4f} g/m²\n")
    f.write(f"  Ratio max/min  : {beq_values.max() / beq_values.min():.1f}\n\n")

    f.write("APPROCHE ADAPTATIVE:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Critère durée : {DURATION_FACTOR} × (1/λ) pour convergence > 99.9%\n")
    f.write(f"Critère dt    : (1/λ) / {DT_FACTOR} pour résolution temporelle fine\n")
    f.write("Durée simulation:\n")
    f.write(f"  Min ({TEMP_MAX}°C)     : {duration_values.min():.2f} days\n")
    f.write(f"  Max (0°C)      : {duration_values.max():.1f} days\n")
    f.write("Pas de temps (dt):\n")
    f.write(
        f"  Min ({TEMP_MAX}°C)     : {dt_values.min():.0f} s ({dt_values.min() / 3600:.2f} h)\n"
    )
    f.write(f"  Max (0°C)      : {dt_values.max():.0f} s ({dt_values.max() / 3600:.2f} h)\n\n")

    f.write("RÉSULTATS - STATISTIQUES D'ERREUR:\n")
    f.write("-" * 80 + "\n")
    f.write(f"NRMSE (Normalised)    : {nrmse_val:.6f}\n")
    f.write(f"Erreur maximale       : {df_results['Error_Percent'].max():.4f}%\n")
    f.write(f"Erreur moyenne        : {df_results['Error_Percent'].mean():.4f}%\n")
    f.write(f"Écart-type            : {df_results['Error_Percent'].std():.4f}%\n")
    f.write(
        f"Température max error : {df_results.loc[df_results['Error_Percent'].idxmax(), 'Temperature']}°C\n\n"
    )

    f.write("VALIDATION:\n")
    f.write("-" * 80 + "\n")
    if df_results["Error_Percent"].max() < 1.0:
        f.write("✅ VALIDATION RÉUSSIE\n")
    else:
        f.write("⚠️  ATTENTION : Certaines erreurs dépassent 1%\n")

    f.write("\n")
    f.write("FICHIERS GÉNÉRÉS:\n")
    f.write("-" * 80 + "\n")
    for fmt in FIGURE_FORMATS:
        f.write(f"- {FIGURE_PREFIX}.{fmt}\n")
    f.write(f"- {summary_filename} (ce fichier)\n\n")

    f.write("=" * 80 + "\n")

print(f"✅ Résumé sauvegardé : {summary_path}")
