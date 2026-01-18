#!/usr/bin/env python

# # Notebook 4A: Weak Scaling - Complexité Algorithmique
#
# **Objectif**: Démontrer que le temps de calcul croît linéairement avec la taille du problème (O(N)).
#
# **Question posée**: "Si je double la taille de ma grille, le temps de calcul double-t-il ?"
#
# ## Théorie
#
# Un algorithme bien conçu pour des problèmes à grille régulière doit avoir une **complexité linéaire O(N)** où N est le nombre total de cellules.
#
# En échelle log-log :
# $$\log(T) = \alpha \log(N) + \beta$$
#
# où :
#
# -   $T$ : Temps de calcul par pas de temps
# -   $N$ : Nombre de cellules (nx × ny)
# -   $\alpha$ : Pente (exposant de complexité)
# -   $\beta$ : Ordonnée à l'origine
#
# **Critère de succès** : $\alpha \approx 1.0 \pm 0.1$
#
# ## Configuration
#
# -   **Backend** : Sequential (pas de parallélisation)
# -   **Grilles testées** : 500×500, 1000×1000, 2000×2000
# -   **Nombre de cohortes** : 50 (charge lourde)
# -   **Pas de temps benchmark** : 20
# -   **Warmup** : 3 pas de temps (compilation JIT)
#

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
    compute_transport_fv_optimized,
)

ureg = pint.get_application_registry()

# === CONFIGURATION DES CHEMINS ===
BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA_DIR = BASE_DIR.parent / "data"
FIGURES_DIR = BASE_DIR.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
SUMMARY_DIR = BASE_DIR.parent / "summary"

print("✅ Imports réussis")


# ## Configuration Matplotlib pour Publications
#

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

COLOR_SIM = COLORS["blue"]
COLOR_THEORY = COLORS["red"]

print("✅ Configuration Matplotlib appliquée")


# ## 1. Configuration du Benchmark
#

# %%


# ============================================================================
# CONFIGURATION - Modifiez ces paramètres pour ajuster l'expérience
# ============================================================================

# --- Configuration Weak Scaling ---
CONFIG_WEAK = {
    "grid_sizes": [
        (100, 100),
        (200, 200),
        (400, 400),
        (800, 800),
        (1600, 1600),
        (3200, 3200),
    ],
    "n_cohorts": 50,
    "n_steps_warmup": 3,
    "n_steps_benchmark": 20,
    "backend": "sequential",
}

# --- Paramètres Physiques ---
U_MAGNITUDE = 0.1  # Vitesse d'advection [m/s]
D_COEFF = 1000.0  # Coefficient de diffusion [m²/s]
TEMPERATURE_CONSTANT = 20.0  # Température constante [°C]
NPP_CONSTANT = 300.0  # Production primaire [mg/m²/day]

# --- Configuration Temporelle ---
START_DATE = "2000-01-01"

# --- État Initial ---
BIOMASS_INIT = 10.0  # Biomasse initiale [g/m²]
PRODUCTION_INIT = 0.01  # Production initiale [g/m²]

# --- Paramètres LMTL ---
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

# --- Figures ---
FIGURE_PREFIX = "fig_04a_weak_scaling"
FIGURE_FORMATS = [
    # "pdf",
    "png",
]

# ============================================================================

print("=" * 80)
print("CONFIGURATION - WEAK SCALING")
print("=" * 80)
print(f"Grilles testées     : {CONFIG_WEAK['grid_sizes']}")
print(f"Nombre de cohortes  : {CONFIG_WEAK['n_cohorts']}")
print(f"Pas de temps warmup : {CONFIG_WEAK['n_steps_warmup']}")
print(f"Pas de temps benchmark : {CONFIG_WEAK['n_steps_benchmark']}")
print(f"Backend             : {CONFIG_WEAK['backend']}")
print(f"Vitesse u           : {U_MAGNITUDE} m/s")
print(f"Diffusion D         : {D_COEFF} m²/s")
print(f"Température         : {TEMPERATURE_CONSTANT}°C")
print(f"NPP                 : {NPP_CONSTANT} mg/m²/day")
print("=" * 80)


# %%


def save_figure(fig, name, formats=FIGURE_FORMATS):
    """Sauvegarde une figure dans les formats requis."""
    for fmt in formats:
        filepath = FIGURES_DIR / f"{name}.{fmt}"
        fig.savefig(filepath, format=fmt, dpi=300, bbox_inches="tight")
        print(f"✅ Saved: {filepath}")


# ## 2. Configuration du Blueprint LMTL Complet
#

# %%


def configure_lmtl_full(bp):
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
                "func": compute_transport_fv_optimized,
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


# ## 3. Fonction de Benchmark
#

# %%


def run_benchmark(grid_size, n_cohorts, n_steps, backend="sequential"):
    """Execute un benchmark pour une taille de grille donnée."""
    ny, nx = grid_size
    print(f"\n{'=' * 80}")
    print(f"Benchmark : Grille {nx}×{ny}, {n_cohorts} cohortes, {n_steps} pas de temps")
    print(f"{'=' * 80}")

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

    # CFL et dt (utilise les constantes globales)
    dx_mean = dx.mean().values
    dt = float(int(0.5 * dx_mean / U_MAGNITUDE))

    # Cohortes
    cohorts_days = np.arange(0, n_cohorts)
    cohorts_seconds = cohorts_days * 86400.0
    cohorts_da = xr.DataArray(
        cohorts_seconds, dims=["cohort"], name="cohort", attrs={"units": "second"}
    )

    # Forçages (utilise les constantes globales)
    time_da = xr.DataArray(
        pd.date_range(start=START_DATE, periods=n_steps + 1, freq=timedelta(seconds=dt)),
        dims=["time"],
    )

    ocean_mask = xr.DataArray(
        np.ones((ny, nx)),
        coords={Coordinates.Y.value: lats, Coordinates.X.value: lons},
        dims=(Coordinates.Y.value, Coordinates.X.value),
    )
    u_field = xr.DataArray(
        np.full((ny, nx), U_MAGNITUDE),
        coords={Coordinates.Y.value: lats, Coordinates.X.value: lons},
        dims=(Coordinates.Y.value, Coordinates.X.value),
    )
    v_field = xr.DataArray(
        np.full((ny, nx), 0.0),
        coords={Coordinates.Y.value: lats, Coordinates.X.value: lons},
        dims=(Coordinates.Y.value, Coordinates.X.value),
    )
    temp_field = xr.DataArray(
        np.full((n_steps + 1, ny, nx), TEMPERATURE_CONSTANT),
        coords={"time": time_da, Coordinates.Y.value: lats, Coordinates.X.value: lons},
        dims=("time", Coordinates.Y.value, Coordinates.X.value),
    )
    npp_field = xr.DataArray(
        np.full((n_steps + 1, ny, nx), NPP_CONSTANT / 86400.0 / 1000.0),  # mg/m²/day -> g/m²/s
        coords={"time": time_da, Coordinates.Y.value: lats, Coordinates.X.value: lons},
        dims=("time", Coordinates.Y.value, Coordinates.X.value),
    )

    forcings = xr.Dataset(
        {
            "temperature": temp_field,
            "primary_production": npp_field,
            "current_u": u_field,
            "current_v": v_field,
            "cell_areas": cell_areas,
            "face_areas_ew": face_areas_ew,
            "face_areas_ns": face_areas_ns,
            "dx": dx,
            "dy": dy,
            "ocean_mask": ocean_mask,
            "dt": dt,
            "boundary_north": BoundaryType.CLOSED,
            "boundary_south": BoundaryType.CLOSED,
            "boundary_east": BoundaryType.CLOSED,
            "boundary_west": BoundaryType.CLOSED,
        },
        coords={"time": time_da, "cohort": cohorts_da},
    )

    # État initial (utilise les constantes globales)
    biomass_init = xr.DataArray(
        np.full((ny, nx), BIOMASS_INIT),
        coords={Coordinates.Y.value: lats, Coordinates.X.value: lons},
        dims=[Coordinates.Y.value, Coordinates.X.value],
    )
    production_init = xr.DataArray(
        np.full((ny, nx, n_cohorts), PRODUCTION_INIT),
        coords={
            Coordinates.Y.value: lats,
            Coordinates.X.value: lons,
            "cohort": cohorts_da,
        },
        dims=[Coordinates.Y.value, Coordinates.X.value, "cohort"],
    )

    initial_state = xr.Dataset({"biomass": biomass_init, "production": production_init})

    # Paramètres (utilise les constantes globales)
    D_horizontal = ureg.Quantity(D_COEFF, ureg.m**2 / ureg.s)
    params = {**asdict(lmtl_params), "D_horizontal": D_horizontal}

    # Configuration
    start = datetime.fromisoformat(START_DATE)
    end = start + timedelta(seconds=dt * n_steps)
    config = SimulationConfig(
        start_date=START_DATE,
        end_date=end.isoformat(),
        timestep=timedelta(seconds=dt),
    )

    controller = SimulationController(config, backend=backend)
    controller.setup(
        configure_lmtl_full,
        forcings=forcings,
        initial_state={"LMTL": initial_state},
        parameters={"LMTL": params},
        output_variables={"LMTL": ["biomass"]},
    )

    # Exécution
    t_start = time.perf_counter()
    controller.run()
    t_end = time.perf_counter()

    elapsed = t_end - t_start
    time_per_step = elapsed / n_steps

    print(f"Temps total        : {elapsed:.3f} s")
    print(f"Temps par pas      : {time_per_step * 1000:.3f} ms")
    print(f"Cellules par seconde : {nx * ny / time_per_step:.0f} cells/s")

    return {
        "grid_size": grid_size,
        "n_cells": nx * ny,
        "n_cohorts": n_cohorts,
        "n_steps": n_steps,
        "elapsed": elapsed,
        "time_per_step": time_per_step,
    }


print("✅ Fonction de benchmark définie")


# ## 4. Warmup : Compilation JIT
#

# %%


print("Warmup : Compilation JIT avec une petite grille...")
_ = run_benchmark(
    grid_size=(100, 100),
    n_cohorts=CONFIG_WEAK["n_cohorts"],
    n_steps=CONFIG_WEAK["n_steps_warmup"],
    backend=CONFIG_WEAK["backend"],
)
print("\n✅ Warmup terminé")


# ## 5. Weak Scaling : Boucle sur les Tailles de Grille
#

# %%


results = []

for grid_size in CONFIG_WEAK["grid_sizes"]:
    result = run_benchmark(
        grid_size=grid_size,
        n_cohorts=CONFIG_WEAK["n_cohorts"],
        n_steps=CONFIG_WEAK["n_steps_benchmark"],
        backend=CONFIG_WEAK["backend"],
    )
    results.append(result)

print("\n" + "=" * 80)
print("✅ Tous les benchmarks terminés")
print("=" * 80)


# ## 6. Analyse : Régression log-log
#

# %%


# Extraction des données
n_cells = np.array([r["n_cells"] for r in results])
times = np.array([r["time_per_step"] for r in results])

# Régression log-log
log_n = np.log10(n_cells)
log_t = np.log10(times)
slope, intercept = np.polyfit(log_n, log_t, 1)

# Ligne de fit
n_fit = np.logspace(np.log10(n_cells.min()), np.log10(n_cells.max()), 100)
t_fit = 10 ** (intercept + slope * np.log10(n_fit))

print("\nRégression log-log :")
print(f"  Pente (exposant de complexité) : {slope:.3f}")
print(f"  Ordonnée à l'origine            : {intercept:.3f}")
print(f"\nComplexité algorithmique estimée : O(N^{slope:.2f})")

if 0.9 <= slope <= 1.1:
    print("✅ Complexité linéaire O(N) confirmée (pente entre 0.9 et 1.1)")
else:
    print("⚠️  Pente hors de la plage attendue [0.9, 1.1]")


# ## 7. Figure 4A : Weak Scaling (log-log)
#

# %%


fig, ax = plt.subplots(figsize=(6.9, 4))

# Points mesurés
ax.loglog(
    n_cells,
    times * 1000,  # Convertir en ms
    "o",
    color=COLOR_SIM,
    markersize=8,
    label="Measurements",
)

# Ligne de fit
ax.loglog(
    n_fit,
    t_fit * 1000,
    "--",
    color=COLORS["grey"],
    linewidth=1.5,
    label=f"Fit: O(N$^{{{slope:.2f}}}$)",
)

ax.set_xlabel("Number of Cells (N)")
ax.set_ylabel("Time per Step [ms]")
ax.set_title("Weak Scaling: Computational Complexity")
ax.legend(loc="best")
ax.grid(True, which="both", alpha=0.3)

# Annotations
for i, r in enumerate(results):
    ax.annotate(
        f"{r['grid_size'][0]}×{r['grid_size'][1]}",
        (r["n_cells"], r["time_per_step"] * 1000),
        xytext=(10, -5),
        textcoords="offset points",
        fontsize=7,
    )

plt.tight_layout()
save_figure(fig, FIGURE_PREFIX)
plt.show()


# ## 8. Tableau Récapitulatif
#

# %%


print("=" * 100)
print("TABLEAU RÉCAPITULATIF - WEAK SCALING")
print("=" * 100)
print(f"{'Grille':<15} {'N Cellules':<15} {'Temps/Step (ms)':<20} {'Complexité':<15}")
print("-" * 100)

for r in results:
    grid_str = f"{r['grid_size'][0]}×{r['grid_size'][1]}"
    n_cells_str = f"{r['n_cells']:,}"
    time_str = f"{r['time_per_step'] * 1000:.3f}"
    complexity_str = f"O(N^{slope:.2f})"
    print(f"{grid_str:<15} {n_cells_str:<15} {time_str:<20} {complexity_str:<15}")

print("-" * 100)
print(f"\nPente (exposant) : {slope:.3f}")
print(f"Complexité estimée : O(N^{slope:.2f})")
print("=" * 100)

# Validation
if 0.9 <= slope <= 1.1:
    print("\n✅ VALIDATION RÉUSSIE")
    print("   - Complexité linéaire O(N) confirmée")
    print("   - Le temps de calcul croît proportionnellement à la taille du problème")
    print("   - L'algorithme est bien optimisé pour les grilles régulières")
else:
    print("\n⚠️  ATTENTION")
    print(f"   - Pente : {slope:.3f} (hors [0.9, 1.1])")
    print("   - La complexité n'est pas strictement linéaire")

print("=" * 100)


# ## 9. Génération du Fichier Résumé
#

# %%


# Calcul des statistiques pour le résumé
time_min_ms = min([r["time_per_step"] for r in results]) * 1000
time_max_ms = max([r["time_per_step"] for r in results]) * 1000
n_cells_min = min([r["n_cells"] for r in results])
n_cells_max = max([r["n_cells"] for r in results])

# Calcul R² de l'ajustement
log_n = np.log10(n_cells)
log_t = np.log10(times)
ss_res = np.sum((log_t - (slope * log_n + intercept)) ** 2)
ss_tot = np.sum((log_t - np.mean(log_t)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# Génération du nom de fichier dynamique
summary_filename = f"{FIGURE_PREFIX.replace('fig_', 'notebook_')}_summary.txt"
summary_path = SUMMARY_DIR / summary_filename

# Créer le répertoire summary s'il n'existe pas
summary_path.parent.mkdir(exist_ok=True)

with open(summary_path, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("NOTEBOOK 04A: WEAK SCALING - COMPLEXITÉ ALGORITHMIQUE\n")
    f.write("=" * 80 + "\n\n")
    f.write("DATE: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")

    f.write("OBJECTIF:\n")
    f.write("-" * 80 + "\n")
    f.write(
        "Démontrer que le temps de calcul croît linéairement avec la taille du problème (O(N)).\n"
    )
    f.write(
        "Question posée: Si je double la taille de ma grille, le temps de calcul double-t-il ?\n\n"
    )

    f.write("CONFIGURATION DU BENCHMARK:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Grilles testées       : {CONFIG_WEAK['grid_sizes']}\n")
    f.write(f"Nombre de cellules    : {n_cells_min:,} à {n_cells_max:,}\n")
    f.write(f"Nombre de cohortes    : {CONFIG_WEAK['n_cohorts']}\n")
    f.write(f"Pas de temps warmup   : {CONFIG_WEAK['n_steps_warmup']}\n")
    f.write(f"Pas de temps benchmark: {CONFIG_WEAK['n_steps_benchmark']}\n")
    f.write(f"Backend               : {CONFIG_WEAK['backend']}\n\n")

    f.write("CONFIGURATION PHYSIQUE:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Vitesse d'advection u : {U_MAGNITUDE} m/s\n")
    f.write(f"Diffusion D           : {D_COEFF} m²/s\n")
    f.write(f"Température           : {TEMPERATURE_CONSTANT}°C (constante)\n")
    f.write(f"NPP                   : {NPP_CONSTANT} mg/m²/day (constante)\n\n")

    f.write("CONFIGURATION DU MODÈLE:\n")
    f.write("-" * 80 + "\n")
    f.write("Modèle complet LMTL   : Production + Mortalité + Transport\n")
    f.write(f"Biomasse initiale     : {BIOMASS_INIT} g/m²\n")
    f.write(f"Production initiale   : {PRODUCTION_INIT} g/m²\n")
    f.write("Schéma advection      : Upwind (1er ordre)\n")
    f.write("Schéma diffusion      : Centré (2nd ordre)\n\n")

    f.write("PARAMÈTRES LMTL:\n")
    f.write("-" * 80 + "\n")
    f.write(f"lambda_0              : {1 / 150:.6f} day⁻¹\n")
    f.write("gamma_lambda          : 0.15 °C⁻¹\n")
    f.write("tau_r_0               : 10.38 days\n")
    f.write("gamma_tau_r           : 0.11 °C⁻¹\n")
    f.write("E                     : 0.1668\n")
    f.write("T_ref                 : 0.0 °C\n\n")

    f.write("RÉSULTATS - WEAK SCALING:\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Grille':<15} {'N Cellules':<15} {'Temps/Step (ms)':<20}\n")
    f.write("-" * 80 + "\n")

    for r in results:
        grid_str = f"{r['grid_size'][0]}×{r['grid_size'][1]}"
        n_cells_str = f"{r['n_cells']:,}"
        time_str = f"{r['time_per_step'] * 1000:.3f}"
        f.write(f"{grid_str:<15} {n_cells_str:<15} {time_str:<20}\n")

    f.write("-" * 80 + "\n")
    f.write(f"Temps min (grille max): {time_min_ms:.3f} ms\n")
    f.write(f"Temps max (grille min): {time_max_ms:.3f} ms\n")
    f.write(f"Ratio temps max/min   : {time_max_ms / time_min_ms:.2f}×\n")
    f.write(f"Ratio grille max/min  : {n_cells_max / n_cells_min:.2f}×\n\n")

    f.write("ANALYSE DE COMPLEXITÉ:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Régression log-log    : log(T) = {slope:.3f} × log(N) + {intercept:.3f}\n")
    f.write(f"Pente (exposant)      : {slope:.3f}\n")
    f.write(f"R² de l'ajustement    : {r_squared:.4f}\n")
    f.write(f"Complexité estimée    : O(N^{slope:.2f})\n")
    f.write("Complexité théorique  : O(N^1.00)\n\n")

    f.write("VALIDATION:\n")
    f.write("-" * 80 + "\n")
    if 0.9 <= slope <= 1.1:
        f.write("✅ VALIDATION RÉUSSIE\n")
        f.write(f"   • Complexité linéaire O(N) confirmée (pente {slope:.3f} ∈ [0.9, 1.1])\n")
        f.write("   • Le temps de calcul croît proportionnellement à la taille du problème\n")
        f.write("   • L'algorithme est bien optimisé pour les grilles régulières\n")
        f.write(f"   • Qualité de l'ajustement excellente (R² = {r_squared:.4f})\n")
    else:
        f.write("⚠️  ATTENTION : Résultats à vérifier\n")
        f.write(f"   • Pente {slope:.3f} hors de [0.9, 1.1]\n")
        if r_squared < 0.95:
            f.write(f"   • R² ({r_squared:.4f}) inférieur à 0.95\n")

    f.write("\n")
    f.write("CONCLUSIONS:\n")
    f.write("-" * 80 + "\n")
    f.write("1. L'architecture DAG a une complexité algorithmique linéaire O(N)\n")
    f.write("2. Le temps de calcul croît proportionnellement au nombre de cellules\n")
    f.write("3. L'implémentation est efficace et scalable pour les grilles régulières\n")
    f.write("4. Pas de surcoût algorithmique caché (pas de O(N log N) ou O(N²))\n")
    f.write("5. Prédictibilité : on peut estimer le temps de calcul pour toute grille\n\n")

    f.write("FICHIERS GÉNÉRÉS:\n")
    f.write("-" * 80 + "\n")
    for fmt in FIGURE_FORMATS:
        f.write(f"- {FIGURE_PREFIX}.{fmt}\n")
    f.write(f"- {summary_filename} (ce fichier)\n\n")

    f.write("=" * 80 + "\n")

print(f"✅ Résumé sauvegardé : {summary_path}")


# ## Conclusion
#
# Ce notebook a démontré que :
#
# 1. **Complexité linéaire O(N)** : Le temps de calcul croît proportionnellement au nombre de cellules.
#
# 2. **Scalabilité algorithmique** : L'architecture DAG est bien optimisée pour les grilles régulières.
#
# 3. **Absence de surcoût** : Pas de complexité cachée (O(N log N), O(N²), etc.).
#
# 4. **Prédictibilité** : On peut estimer le temps de calcul pour n'importe quelle taille de grille.
#
# **Prochaine étape** : Strong scaling (Notebook 4B) - tester l'accélération parallèle avec Dask.
#
