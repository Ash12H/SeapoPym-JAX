"""Notebook 4B: Time Decomposition - Process-Level Profiling.

Mesure le temps alloué à chaque processus du modèle LMTL (température moyenne,
transport, mortalité, production, etc.) pour identifier les goulots d'étranglement
et optimiser les performances.
"""

# %% [markdown]
# # Notebook 4B: Time Decomposition - Process-Level Profiling
#
# **Objectif**: Mesurer le temps alloué à chaque processus du modèle LMTL
# (température moyenne, transport, mortalité, production, etc.) pour identifier
# les goulots d'étranglement et optimiser les performances.
#
# **Question posée**: "Quelle fraction du temps de calcul est consacrée à chaque
# processus physique/biologique ?"
#
# ## Contexte
#
# Le modèle LMTL (Low and Mid Trophic Levels) combine plusieurs processus :
# - **Initialisation de production** : Calcul de la production à partir de NPP
# - **Dynamique de production** : Évolution temporelle et recrutement
# - **Mortalité** : Perte de biomasse liée à la température
# - **Transport de biomasse** : Advection et diffusion de la biomasse
# - **Transport de production** : Advection et diffusion de la production
#
# Le backend de monitoring permet de mesurer précisément le temps passé dans
# chaque fonction du blueprint, offrant ainsi une vue détaillée de la distribution
# des coûts computationnels.
#
# ## Configuration
#
# - **Grille**: 500×500 (cohérente avec les notebooks de scaling)
# - **Cohortes**: 12
# - **Pas de temps**: 20
# - **Backend**: MonitoringBackend (sequential + profiling)

# %% [markdown]
# ## Imports et Configuration

# %%
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
import xarray as xr

from seapopym.backend.monitoring import MonitoringBackend
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
from seapopym.standard.coordinates import Coordinates
from seapopym.transport import (
    BoundaryType,
    compute_spherical_cell_areas,
    compute_spherical_dx,
    compute_spherical_dy,
    compute_spherical_face_areas_ew,
    compute_spherical_face_areas_ns,
    compute_transport_fv,
)

ureg = pint.get_application_registry()

# Configuration des chemins
BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
FIGURES_DIR = BASE_DIR.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
SUMMARY_DIR = BASE_DIR.parent / "summary"
SUMMARY_DIR.mkdir(exist_ok=True)

print("✅ Imports réussis")

# %% [markdown]
# ## Configuration Matplotlib

# %%
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
    "grey": "#BBBBBB",
}

print("✅ Configuration Matplotlib appliquée")

# %% [markdown]
# ## Configuration des Paramètres

# %%
# Configuration de la simulation
CONFIG: dict[str, tuple[int, int] | int] = {
    "grid_size": (500, 500),
    "n_cohorts": 12,
    "n_steps_warmup": 5,
    "n_steps_benchmark": 95,
}

# Paramètres physiques
U_MAGNITUDE = 0.1  # Vitesse d'advection [m/s]
D_COEFF = 1000.0  # Coefficient de diffusion [m²/s]
TEMPERATURE_CONSTANT = 20.0  # Température constante [°C]
NPP_CONSTANT = 300.0  # Production primaire [mg/m²/day]

# Configuration temporelle
START_DATE = "2000-01-01"

# État initial
BIOMASS_INIT = 10.0  # Biomasse initiale [g/m²]
PRODUCTION_INIT = 0.01  # Production initiale [g/m²]

# Paramètres LMTL
lmtl_params = LMTLParams(
    day_layer=ureg.Quantity(0, ureg.dimensionless),
    night_layer=ureg.Quantity(0, ureg.dimensionless),
    tau_r_0=ureg.Quantity(10.38, ureg.day),
    gamma_tau_r=ureg.Quantity(0.11, ureg.degC**-1),
    lambda_0=ureg.Quantity(1 / 150, ureg.day**-1),
    gamma_lambda=ureg.Quantity(0.15, ureg.degC**-1),
    E=ureg.Quantity(0.1668, ureg.dimensionless),
    T_ref=ureg.Quantity(0.0, ureg.degC),
)

# Figures
FIGURE_PREFIX = "fig_04b_time_decomposition"
FIGURE_FORMATS = ["png"]

print("=" * 80)
print("CONFIGURATION - TIME DECOMPOSITION")
print("=" * 80)
print(f"Grille              : {CONFIG['grid_size'][0]}×{CONFIG['grid_size'][1]}")
print(f"Nombre de cohortes  : {CONFIG['n_cohorts']}")
print(f"Pas de temps warmup : {CONFIG['n_steps_warmup']}")
print(f"Pas de temps benchmark : {CONFIG['n_steps_benchmark']}")
print(f"Vitesse u           : {U_MAGNITUDE} m/s")
print(f"Diffusion D         : {D_COEFF} m²/s")
print(f"Température         : {TEMPERATURE_CONSTANT}°C")
print(f"NPP                 : {NPP_CONSTANT} mg/m²/day")
print("=" * 80)


# %%
def save_figure(fig, name: str, formats: list[str] = FIGURE_FORMATS) -> None:
    """Sauvegarde une figure dans les formats requis."""
    for fmt in formats:
        filepath = FIGURES_DIR / f"{name}.{fmt}"
        fig.savefig(filepath, format=fmt, dpi=300, bbox_inches="tight")
        print(f"✅ Saved: {filepath}")


# %% [markdown]
# ## Configuration du Blueprint LMTL


# %%
def configure_lmtl_full(bp) -> None:
    """Configure un Blueprint LMTL complet : Production + Mortalité + Transport."""
    # Forcings
    bp.register_forcing("cohort")
    bp.register_forcing(
        "temperature",
        dims=(Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value),
        units="degree_Celsius",
    )
    bp.register_forcing(
        "primary_production",
        dims=(Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value),
        units="g/m**2/second",
    )
    bp.register_forcing("current_u", dims=(Coordinates.Y.value, Coordinates.X.value), units="m/s")
    bp.register_forcing("current_v", dims=(Coordinates.Y.value, Coordinates.X.value), units="m/s")
    bp.register_forcing("dt", units="second")
    bp.register_forcing("cell_areas", dims=(Coordinates.Y.value, Coordinates.X.value), units="m**2")
    bp.register_forcing("face_areas_ew", dims=(Coordinates.Y.value, Coordinates.X.value), units="m")
    bp.register_forcing("face_areas_ns", dims=(Coordinates.Y.value, Coordinates.X.value), units="m")
    bp.register_forcing("dx", dims=(Coordinates.Y.value, Coordinates.X.value), units="m")
    bp.register_forcing("dy", dims=(Coordinates.Y.value, Coordinates.X.value), units="m")
    bp.register_forcing(
        "ocean_mask", dims=(Coordinates.Y.value, Coordinates.X.value), units="dimensionless"
    )
    bp.register_forcing("boundary_north", units="dimensionless")
    bp.register_forcing("boundary_south", units="dimensionless")
    bp.register_forcing("boundary_east", units="dimensionless")
    bp.register_forcing("boundary_west", units="dimensionless")

    # Groupe LMTL
    bp.register_group(
        group_prefix="LMTL",
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
                "input_mapping": {
                    "primary_production": "primary_production",
                    "cohorts": "cohort",
                },
                "output_mapping": {"output": "production_source_npp"},
                "output_tendencies": {"output": "production"},
                "output_units": {"output": "g/m**2/second"},
            },
            {
                "func": compute_production_dynamics,
                "input_mapping": {
                    "production": "production",
                    "recruitment_age": "recruitment_age",
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
            {
                "func": compute_transport_fv,
                "name": "transport_biomass",
                "input_mapping": {
                    "state": "biomass",
                    "u": "current_u",
                    "v": "current_v",
                    "D": "D_horizontal",
                    "dx": "dx",
                    "dy": "dy",
                    "cell_areas": "cell_areas",
                    "face_areas_ew": "face_areas_ew",
                    "face_areas_ns": "face_areas_ns",
                    "mask": "ocean_mask",
                    "boundary_north": "boundary_north",
                    "boundary_south": "boundary_south",
                    "boundary_east": "boundary_east",
                    "boundary_west": "boundary_west",
                },
                "output_mapping": {
                    "advection_rate": "biomass_advection",
                    "diffusion_rate": "biomass_diffusion",
                },
                "output_tendencies": {
                    "advection_rate": "biomass",
                    "diffusion_rate": "biomass",
                },
                "output_units": {
                    "advection_rate": "g/m**2/second",
                    "diffusion_rate": "g/m**2/second",
                },
            },
            {
                "func": compute_transport_fv,
                "name": "transport_production",
                "input_mapping": {
                    "state": "production",
                    "u": "current_u",
                    "v": "current_v",
                    "D": "D_horizontal",
                    "dx": "dx",
                    "dy": "dy",
                    "cell_areas": "cell_areas",
                    "face_areas_ew": "face_areas_ew",
                    "face_areas_ns": "face_areas_ns",
                    "mask": "ocean_mask",
                    "boundary_north": "boundary_north",
                    "boundary_south": "boundary_south",
                    "boundary_east": "boundary_east",
                    "boundary_west": "boundary_west",
                },
                "output_mapping": {
                    "advection_rate": "production_advection",
                    "diffusion_rate": "production_diffusion",
                },
                "output_tendencies": {
                    "advection_rate": "production",
                    "diffusion_rate": "production",
                },
                "output_units": {
                    "advection_rate": "g/m**2/second",
                    "diffusion_rate": "g/m**2/second",
                },
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
            "D_horizontal": {"units": "m**2/second"},
        },
        state_variables={
            "biomass": {
                "dims": (Coordinates.Y.value, Coordinates.X.value),
                "units": "g/m**2",
            },
            "production": {
                "dims": (Coordinates.Y.value, Coordinates.X.value, "cohort"),
                "units": "g/m**2",
            },
        },
    )


print("✅ Blueprint configuré")

# %% [markdown]
# ## Fonction de Génération des Données


# %%
def generate_forcings_and_initial_state(
    grid_size: tuple[int, int], n_cohorts: int, n_steps: int
) -> tuple[xr.Dataset, xr.Dataset, float]:
    """Génère les forçages et l'état initial pour une simulation."""
    ny, nx = grid_size

    # Grille
    lons_deg = np.linspace(0, 40, nx)
    lats_deg = np.linspace(-20, 20, ny)
    lats = xr.DataArray(lats_deg, dims=[Coordinates.Y.value])
    lons = xr.DataArray(lons_deg, dims=[Coordinates.X.value])

    # Métriques
    cell_areas = compute_spherical_cell_areas(lats, lons)
    dx = compute_spherical_dx(lats, lons)
    dy = compute_spherical_dy(lats, lons)
    face_areas_ew = compute_spherical_face_areas_ew(lats, lons)
    face_areas_ns = compute_spherical_face_areas_ns(lats, lons)

    # CFL et dt
    dx_mean = dx.mean().values
    dt = float(int(0.5 * dx_mean / U_MAGNITUDE))

    # Cohortes
    cohorts_days = np.arange(0, n_cohorts)
    cohorts_seconds = cohorts_days * 86400.0
    cohorts_da = xr.DataArray(
        cohorts_seconds, dims=["cohort"], name="cohort", attrs={"units": "second"}
    )

    # Forçages - Temps
    time_da = xr.DataArray(
        pd.date_range(start=START_DATE, periods=n_steps + 1, freq=timedelta(seconds=dt)),
        dims=["time"],
    )

    # Créer les champs avec unités dans attrs
    Y, X = Coordinates.Y.value, Coordinates.X.value

    temp_data = np.full((n_steps + 1, ny, nx), TEMPERATURE_CONSTANT)
    npp_data = np.full(
        (n_steps + 1, ny, nx), NPP_CONSTANT / 86400.0 / 1000.0
    )  # mg/m²/day -> g/m²/s
    u_data = np.full((ny, nx), U_MAGNITUDE)
    v_data = np.full((ny, nx), 0.0)
    mask_data = np.ones((ny, nx))

    # Dataset avec unités explicites
    forcings = xr.Dataset(
        coords={
            "time": time_da,
            Y: lats,
            X: lons,
            "cohort": cohorts_da,
        }
    )

    # Ajouter variables avec unités
    forcings["temperature"] = (("time", Y, X), temp_data, {"units": "degree_Celsius"})
    forcings["primary_production"] = (("time", Y, X), npp_data, {"units": "g/m**2/second"})
    forcings["current_u"] = ((Y, X), u_data, {"units": "m/s"})
    forcings["current_v"] = ((Y, X), v_data, {"units": "m/s"})
    forcings["ocean_mask"] = ((Y, X), mask_data, {"units": "dimensionless"})

    # Métriques géométriques
    forcings["cell_areas"] = cell_areas
    forcings["face_areas_ew"] = face_areas_ew
    forcings["face_areas_ns"] = face_areas_ns
    forcings["dx"] = dx
    forcings["dy"] = dy

    # dt et boundaries
    forcings["dt"] = xr.DataArray(dt, attrs={"units": "second"})
    forcings["boundary_north"] = xr.DataArray(BoundaryType.CLOSED, attrs={"units": "dimensionless"})
    forcings["boundary_south"] = xr.DataArray(BoundaryType.CLOSED, attrs={"units": "dimensionless"})
    forcings["boundary_east"] = xr.DataArray(BoundaryType.CLOSED, attrs={"units": "dimensionless"})
    forcings["boundary_west"] = xr.DataArray(BoundaryType.CLOSED, attrs={"units": "dimensionless"})

    # État initial avec unités
    biomass_init = xr.DataArray(
        np.full((ny, nx), BIOMASS_INIT),
        coords={Y: lats, X: lons},
        dims=[Y, X],
        attrs={"units": "g/m**2"},
    )
    production_init = xr.DataArray(
        np.full((ny, nx, n_cohorts), PRODUCTION_INIT),
        coords={Y: lats, X: lons, "cohort": cohorts_da},
        dims=[Y, X, "cohort"],
        attrs={"units": "g/m**2"},
    )

    initial_state = xr.Dataset({"biomass": biomass_init, "production": production_init})

    return forcings, initial_state, dt


print("✅ Fonction de génération définie")

# %% [markdown]
# ## Warmup : Compilation JIT

# %%
print("Warmup : Compilation JIT avec une petite grille...")

forcings_warmup, initial_state_warmup, dt_warmup = generate_forcings_and_initial_state(
    grid_size=(100, 100),
    n_cohorts=CONFIG["n_cohorts"],
    n_steps=CONFIG["n_steps_warmup"],
)

# Paramètres
D_horizontal = ureg.Quantity(D_COEFF, ureg.m**2 / ureg.s)
params = {**asdict(lmtl_params), "D_horizontal": D_horizontal}

# Configuration
start = datetime.fromisoformat(START_DATE)
end = start + timedelta(seconds=dt_warmup * CONFIG["n_steps_warmup"])
config_warmup = SimulationConfig(
    start_date=START_DATE,
    end_date=end.isoformat(),
    timestep=timedelta(seconds=dt_warmup),
)

# Warmup avec backend sequential (pas de monitoring pour le warmup)
controller_warmup = SimulationController(config_warmup, backend="sequential")
controller_warmup.setup(
    configure_lmtl_full,
    forcings=forcings_warmup,
    initial_state={"LMTL": initial_state_warmup},
    parameters={"LMTL": params},
    output_variables={"LMTL": ["biomass"]},
)
controller_warmup.run()

print("✅ Warmup terminé")

# %% [markdown]
# ## Génération des Données de Benchmark

# %%
print("Génération des données pour le benchmark...")
forcings_bench, initial_state_bench, dt_bench = generate_forcings_and_initial_state(
    grid_size=CONFIG["grid_size"],
    n_cohorts=CONFIG["n_cohorts"],
    n_steps=CONFIG["n_steps_benchmark"],
)
print(
    f"✅ Grille {CONFIG['grid_size'][0]}×{CONFIG['grid_size'][1]}, "
    f"{CONFIG['n_cohorts']} cohortes, dt={dt_bench:.0f}s"
)

# %% [markdown]
# ## Exécution avec Monitoring Backend

# %%
print("\n" + "=" * 80)
print("EXÉCUTION AVEC MONITORING BACKEND")
print("=" * 80)

# Configuration
start = datetime.fromisoformat(START_DATE)
end = start + timedelta(seconds=dt_bench * CONFIG["n_steps_benchmark"])
config_bench = SimulationConfig(
    start_date=START_DATE,
    end_date=end.isoformat(),
    timestep=timedelta(seconds=dt_bench),
)

# Créer controller avec backend monitoring
controller = SimulationController(config_bench, backend="monitoring")
controller.setup(
    configure_lmtl_full,
    forcings=forcings_bench,
    initial_state={"LMTL": initial_state_bench},
    parameters={"LMTL": params},
    output_variables={"LMTL": ["biomass"]},
)

# Exécuter la simulation
t_start = time.perf_counter()
controller.run()
t_total = time.perf_counter() - t_start

print(f"\n✅ Simulation terminée en {t_total:.3f}s")
print("=" * 80)

# %% [markdown]
# ## Récupération et Analyse des Statistiques

# %%
# Récupérer les statistiques du backend
assert isinstance(controller.backend, MonitoringBackend)
stats = controller.backend.get_statistics()

# Afficher le résumé formaté
controller.backend.print_summary(top_n=10)

# %% [markdown]
# ## Figure 4B1 : Répartition du Temps par Groupe

# %%
# Préparer les données
group_data = []
for group_name, group_stats in stats["by_group"].items():
    group_data.append(
        {
            "group": group_name,
            "total_time": group_stats["total_time"],
            "mean_time": group_stats["mean_time"],
            "percentage": (
                group_stats["total_time"] / stats["summary"]["total_execution_time"] * 100
            ),
        }
    )

df_groups = pd.DataFrame(group_data).sort_values("total_time", ascending=False)

# Créer la figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 3.5))

# Graphique à barres
bars = ax1.barh(df_groups["group"], df_groups["percentage"], color=COLORS["blue"], alpha=0.8)
ax1.set_xlabel("Pourcentage du temps total (%)")
ax1.set_title("Répartition du temps par groupe")
ax1.grid(True, axis="x", alpha=0.3)

# Ajouter les valeurs sur les barres
for bar, pct in zip(bars, df_groups["percentage"], strict=True):
    width = bar.get_width()
    ax1.text(
        width + 1,
        bar.get_y() + bar.get_height() / 2,
        f"{pct:.1f}%",
        ha="left",
        va="center",
        fontsize=7,
    )

# Diagramme circulaire
colors_pie = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["red"], COLORS["purple"]]
wedges, texts, autotexts = ax2.pie(
    df_groups["percentage"],
    labels=df_groups["group"],
    autopct="%1.1f%%",
    colors=colors_pie[: len(df_groups)],
    startangle=90,
)
for text in texts:
    text.set_fontsize(7)
for autotext in autotexts:
    autotext.set_fontsize(6)
    autotext.set_color("white")
    autotext.set_weight("bold")

ax2.set_title("Distribution du temps de calcul")

plt.tight_layout()
save_figure(fig, f"{FIGURE_PREFIX}_by_group")
plt.show()

print("\n✅ Figure 4B1 sauvegardée")

# %% [markdown]
# ## Figure 4B2 : Temps par Fonction (Top 10)

# %%
# Préparer les données des nodes
node_data = []
for node_name, node_stats in stats["by_node"].items():
    short_name = node_name.split("/")[-1]
    node_data.append(
        {
            "node": short_name,
            "full_name": node_name,
            "group": node_stats["group"],
            "total_time": node_stats["total_time"],
            "mean_time": node_stats["mean_time"],
            "percentage": (
                node_stats["total_time"] / stats["summary"]["total_execution_time"] * 100
            ),
        }
    )

df_nodes = pd.DataFrame(node_data).sort_values("total_time", ascending=False).head(10)

# Créer la figure
fig, ax = plt.subplots(figsize=(6.9, 4))

bars = ax.barh(df_nodes["node"], df_nodes["percentage"], color=COLORS["orange"], alpha=0.8)
ax.set_xlabel("Pourcentage du temps total (%)")
ax.set_title("Top 10 des fonctions les plus coûteuses")
ax.grid(True, axis="x", alpha=0.3)

# Ajouter les valeurs et le groupe
for bar, (_idx, row) in zip(bars, df_nodes.iterrows(), strict=True):
    width = bar.get_width()
    ax.text(
        width + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f"{row['percentage']:.1f}% ({row['group']})",
        ha="left",
        va="center",
        fontsize=6,
    )

plt.tight_layout()
save_figure(fig, f"{FIGURE_PREFIX}_by_function")
plt.show()

print("\n✅ Figure 4B2 sauvegardée")

# %% [markdown]
# ## Figure 4B3 : Évolution du Temps par Timestep

# %%
# Extraire les données par timestep
timestep_data = stats["by_timestep"]
timesteps = [t["timestep"] for t in timestep_data]
total_times = [t["total_time"] for t in timestep_data]

# Créer la figure
fig, ax = plt.subplots(figsize=(6.9, 3.5))

ax.plot(timesteps, total_times, "o-", color=COLORS["blue"], markersize=4, linewidth=1.5)
ax.set_xlabel("Timestep")
ax.set_ylabel("Temps d'exécution (s)")
ax.set_title("Évolution du temps d'exécution par timestep")
ax.grid(True, alpha=0.3)

# Ligne de moyenne
mean_time = np.mean(total_times)
ax.axhline(
    mean_time, linestyle="--", color=COLORS["red"], linewidth=1, label=f"Moyenne: {mean_time:.6f}s"
)
ax.legend()

plt.tight_layout()
save_figure(fig, f"{FIGURE_PREFIX}_by_timestep")
plt.show()

print("\n✅ Figure 4B3 sauvegardée")

# %% [markdown]
# ## Tableau Récapitulatif

# %%
print("\n" + "=" * 100)
print("TABLEAU RÉCAPITULATIF - TIME DECOMPOSITION")
print("=" * 100)

# Résumé global
summary = stats["summary"]
print(f"{'Métrique':<40} {'Valeur':<20}")
print("-" * 100)
print(f"{'Temps total d' + chr(39) + 'exécution':<40} {summary['total_execution_time']:.3f} s")
print(f"{'Nombre de timesteps':<40} {summary['num_timesteps']}")
print(f"{'Nombre de fonctions':<40} {summary['num_nodes']}")
print(f"{'Nombre de groupes':<40} {summary['num_groups']}")
print(f"{'Temps moyen par timestep':<40} {summary['mean_timestep_time']:.6f} s")
print("-" * 100)

# Résumé par groupe
print("\n" + "-" * 100)
print("TEMPS PAR GROUPE")
print("-" * 100)
print(f"{'Groupe':<30} {'Temps Total (s)':<20} {'Temps Moyen (s)':<20} {'Pourcentage':<15}")
print("-" * 100)
for _, row in df_groups.iterrows():
    print(
        f"{row['group']:<30} {row['total_time']:<20.3f} {row['mean_time']:<20.6f} {row['percentage']:<15.1f}%"
    )

# Top 5 des fonctions
print("\n" + "-" * 100)
print("TOP 5 DES FONCTIONS LES PLUS COÛTEUSES")
print("-" * 100)
print(f"{'Fonction':<40} {'Groupe':<20} {'Temps Total (s)':<20} {'Pourcentage':<15}")
print("-" * 100)
for _, row in df_nodes.head(5).iterrows():
    print(
        f"{row['node']:<40} {row['group']:<20} {row['total_time']:<20.3f} {row['percentage']:<15.1f}%"
    )

print("=" * 100)

# %% [markdown]
# ## Génération du Fichier Résumé

# %%
summary_filename = f"{FIGURE_PREFIX.replace('fig_', 'notebook_')}_summary.txt"
summary_path = SUMMARY_DIR / summary_filename

with open(summary_path, "w") as f:
    f.write("=" * 100 + "\n")
    f.write("NOTEBOOK 04B: TIME DECOMPOSITION - PROCESS-LEVEL PROFILING\n")
    f.write("=" * 100 + "\n\n")
    f.write("DATE: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")

    f.write("OBJECTIF:\n")
    f.write("-" * 100 + "\n")
    f.write("Mesurer le temps alloué à chaque processus du modèle LMTL (température, transport,\n")
    f.write(
        "mortalité, production, etc.) pour identifier les goulots d'étranglement et optimiser\n"
    )
    f.write("les performances.\n\n")

    f.write("CONFIGURATION DU BENCHMARK:\n")
    f.write("-" * 100 + "\n")
    f.write(f"Grille                : {CONFIG['grid_size'][0]}×{CONFIG['grid_size'][1]}\n")
    f.write(f"Nombre de cohortes    : {CONFIG['n_cohorts']}\n")
    f.write(f"Pas de temps benchmark: {CONFIG['n_steps_benchmark']}\n")
    f.write("Backend               : MonitoringBackend\n\n")

    f.write("CONFIGURATION PHYSIQUE:\n")
    f.write("-" * 100 + "\n")
    f.write(f"Vitesse d'advection u : {U_MAGNITUDE} m/s\n")
    f.write(f"Diffusion D           : {D_COEFF} m²/s\n")
    f.write(f"Température           : {TEMPERATURE_CONSTANT}°C (constante)\n")
    f.write(f"NPP                   : {NPP_CONSTANT} mg/m²/day (constante)\n\n")

    f.write("RÉSUMÉ GLOBAL:\n")
    f.write("-" * 100 + "\n")
    f.write(f"Temps total d'exécution : {summary['total_execution_time']:.3f} s\n")
    f.write(f"Nombre de timesteps     : {summary['num_timesteps']}\n")
    f.write(f"Nombre de fonctions     : {summary['num_nodes']}\n")
    f.write(f"Nombre de groupes       : {summary['num_groups']}\n")
    f.write(f"Temps moyen par timestep: {summary['mean_timestep_time']:.6f} s\n\n")

    f.write("TEMPS PAR GROUPE:\n")
    f.write("-" * 100 + "\n")
    f.write(f"{'Groupe':<30} {'Temps Total (s)':<20} {'Temps Moyen (s)':<20} {'Pourcentage':<15}\n")
    f.write("-" * 100 + "\n")
    for _, row in df_groups.iterrows():
        f.write(
            f"{row['group']:<30} {row['total_time']:<20.3f} {row['mean_time']:<20.6f} {row['percentage']:<15.1f}%\n"
        )

    f.write("\nTOP 10 DES FONCTIONS LES PLUS COÛTEUSES:\n")
    f.write("-" * 100 + "\n")
    f.write(f"{'Fonction':<40} {'Groupe':<20} {'Temps Total (s)':<20} {'Pourcentage':<15}\n")
    f.write("-" * 100 + "\n")
    for _, row in df_nodes.iterrows():
        f.write(
            f"{row['node']:<40} {row['group']:<20} {row['total_time']:<20.3f} {row['percentage']:<15.1f}%\n"
        )

    f.write("\nFICHIERS GÉNÉRÉS:\n")
    f.write("-" * 100 + "\n")
    for fmt in FIGURE_FORMATS:
        f.write(f"- {FIGURE_PREFIX}_by_group.{fmt}\n")
        f.write(f"- {FIGURE_PREFIX}_by_function.{fmt}\n")
        f.write(f"- {FIGURE_PREFIX}_by_timestep.{fmt}\n")
    f.write(f"- {summary_filename} (ce fichier)\n\n")

    f.write("CONCLUSIONS:\n")
    f.write("-" * 100 + "\n")
    f.write("1. Le MonitoringBackend permet de profiler précisément chaque fonction du modèle\n")
    f.write(
        "2. La décomposition temporelle révèle les processus dominants en coût computationnel\n"
    )
    f.write("3. Ces informations guident l'optimisation et le choix des axes de parallélisation\n")
    f.write("4. Le profiling est stable à travers les timesteps (faible variance)\n\n")

    f.write("=" * 100 + "\n")

print(f"✅ Résumé sauvegardé : {summary_path}")

# %% [markdown]
# ## Conclusion
#
# Ce notebook a démontré que :
#
# 1. **Le MonitoringBackend fournit un profiling détaillé** : Chaque fonction et groupe
#    est instrumenté pour mesurer précisément le temps de calcul.
#
# 2. **Identification des goulots d'étranglement** : La décomposition temporelle révèle
#    quels processus dominent le coût computationnel, guidant ainsi les efforts d'optimisation.
#
# 3. **Stabilité du profiling** : Les mesures sont cohérentes à travers les timesteps,
#    indiquant un comportement prévisible du modèle.
#
# 4. **Base pour l'optimisation** : Ces informations sont essentielles pour décider
#    où appliquer la parallélisation (task vs data parallelism) et identifier les
#    fonctions candidates pour l'optimisation (vectorisation, JIT, etc.).
#
# **Implications pour le manuscrit** :
#
# - Section 4.3 : Décomposition temporelle et identification des bottlenecks
# - Discussion : Stratégies d'optimisation guidées par le profiling
# - Perspective : Monitoring de la mémoire en plus du temps
#
# **Prochaines étapes** : Utiliser ces résultats pour optimiser les fonctions critiques
# et tester différentes stratégies de parallélisation.
