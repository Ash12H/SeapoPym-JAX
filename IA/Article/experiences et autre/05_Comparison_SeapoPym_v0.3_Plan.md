# Plan de Comparaison SeapoPym DAG vs SeapoPym v0.3

## Objectif

Créer un **notebook unique** qui :

1. Exécute une simulation SeapoPym DAG (v1.0) en configuration **0D** (sans transport)
2. Compare les résultats avec SeapoPym v0.3
3. Génère les métriques et figures pour l'article

Cette comparaison correspond à la section **1.2** des Résultats de l'article :

> "Comparaison avec SeapoPym v0.3 (Sans Transport)"

---

## 1. Notebook Cible

**Nom** : `data/notebook/article_02_comparison_seapopym_v0.3.ipynb`

**Structure proposée** :

```
1. Configuration et imports
2. Chargement des données et paramètres
3. Simulation SeapoPym DAG (0D)
4. Chargement des résultats SeapoPym v0.3
5. Comparaison et métriques
6. Visualisations pour l'article
7. Sauvegarde des figures
```

---

## 2. Sources de Données

| Source        | Chemin                                                                        | Description                                           |
| ------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------- |
| Forçages      | `phd_optimization/.../post_processed_light_global_multiyear_bgc_001_033.zarr` | Température + NPP                                     |
| SeapoPym v0.3 | `phd_optimization/.../biomass_global.zarr`                                    | Résultats de la version Python précédente (référence) |

---

## 3. Structure Détaillée du Notebook

### Section 1 : Imports et Configuration

```python
# Imports standard
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
import xarray as xr

# Imports Seapopym
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
```

### Section 2 : Paramètres LMTL (identiques à v0.3)

```python
# Paramètres LMTL - DOIVENT être identiques à ceux utilisés pour v0.3
lmtl_params = LMTLParams(
    day_layer=0,
    night_layer=0,
    tau_r_0=10.38 * ureg.days,
    gamma_tau_r=ureg.Quantity(0.11, ureg.degC**-1),
    lambda_0=ureg.Quantity(1 / 150, ureg.day**-1),
    gamma_lambda=ureg.Quantity(0.15, ureg.degC**-1),
    E=0.1668,
    T_ref=ureg.Quantity(0, ureg.degC),
)

# Configuration simulation
start_date = "1998-01-02"
end_date = "2019-12-31"
timestep = timedelta(days=1)

config = SimulationConfig(
    start_date=start_date,
    end_date=end_date,
    timestep=timestep,
)
```

### Section 3 : Chargement des Forçages

```python
# Chemin vers le fichier Zarr des forçages
zarr_path = "/Users/adm-lehodey/Documents/Workspace/Projects/phd_optimization/notebooks/Article_1/data/1_global/post_processed_light_global_multiyear_bgc_001_033.zarr"

# Chargement et renommage des dimensions
ds_raw = xr.open_zarr(zarr_path)
ds = ds_raw.rename({
    "T": Coordinates.T.value,
    "Z": Coordinates.Z.value,
    "Y": Coordinates.Y.value,
    "X": Coordinates.X.value,
})
ds.x.attrs = {}
ds.y.attrs = {}

# Préparation des forçages
forcings = ds.sel({Coordinates.T.value: slice("1998-01-01", "2020-01-01")})
forcings = forcings[["primary_production", "temperature"]].load()

# Création des cohortes
cohorts = (np.arange(0, np.ceil(lmtl_params.tau_r_0.magnitude) + 1) * ureg.day).to("second")
cohorts_da = xr.DataArray(cohorts.magnitude, dims=["cohort"], name="cohort", attrs={"units": "second"})

forcings = forcings.assign_coords(cohort=cohorts_da)
forcings["dt"] = config.timestep.total_seconds()

# Normalisation des unités
if "primary_production" in forcings:
    forcings["primary_production"].attrs["units"] = "mg/m**2/day"
if "temperature" in forcings and forcings["temperature"].attrs.get("units") in ["degC", "deg_C"]:
    forcings["temperature"].attrs["units"] = "degree_Celsius"
```

### Section 4 : Configuration du Blueprint (0D - sans transport)

```python
def configure_model(bp):
    """Configure le modèle LMTL sans transport."""

    # Forçages
    bp.register_forcing("temperature",
        dims=(Coordinates.T.value, Coordinates.Z.value, Coordinates.Y.value, Coordinates.X.value),
        units="degree_Celsius")
    bp.register_forcing("primary_production",
        dims=(Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value),
        units="g/m**2/second")
    bp.register_forcing("dt")
    bp.register_forcing("cohort")
    bp.register_forcing(Coordinates.T.value)
    bp.register_forcing(Coordinates.Y.value)

    # Groupe Zooplankton (SANS transport)
    bp.register_group(
        group_prefix="Zooplankton",
        units=[
            {"func": compute_day_length,
             "output_mapping": {"output": "day_length"},
             "input_mapping": {"latitude": Coordinates.Y.value, "time": Coordinates.T.value},
             "output_units": {"output": "dimensionless"}},
            {"func": compute_layer_weighted_mean,
             "input_mapping": {"forcing": "temperature"},
             "output_mapping": {"output": "mean_temperature"},
             "output_units": {"output": "degree_Celsius"}},
            {"func": compute_threshold_temperature,
             "input_mapping": {"temperature": "mean_temperature", "min_temperature": "T_ref"},
             "output_mapping": {"output": "thresholded_temperature"},
             "output_units": {"output": "degree_Celsius"}},
            {"func": compute_gillooly_temperature,
             "input_mapping": {"temperature": "thresholded_temperature"},
             "output_mapping": {"output": "gillooly_temperature"},
             "output_units": {"output": "degree_Celsius"}},
            {"func": compute_recruitment_age,
             "input_mapping": {"temperature": "gillooly_temperature"},
             "output_mapping": {"output": "recruitment_age"},
             "output_units": {"output": "second"}},
            {"func": compute_production_initialization,
             "input_mapping": {"cohorts": "cohort"},
             "output_mapping": {"output": "production_source_npp"},
             "output_tendencies": {"output": "production"},
             "output_units": {"output": "g/m**2/second"}},
            {"func": compute_production_dynamics,
             "input_mapping": {"cohort_ages": "cohort", "dt": "dt"},
             "output_mapping": {"production_tendency": "production_tendency", "recruitment_source": "biomass_source"},
             "output_tendencies": {"production_tendency": "production", "recruitment_source": "biomass"},
             "output_units": {"production_tendency": "g/m**2/second", "recruitment_source": "g/m**2/second"}},
            {"func": compute_mortality_tendency,
             "input_mapping": {"temperature": "gillooly_temperature"},
             "output_mapping": {"mortality_loss": "biomass_mortality"},
             "output_tendencies": {"mortality_loss": "biomass"},
             "output_units": {"mortality_loss": "g/m**2/second"}},
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
            "production": {"dims": (Coordinates.Y.value, Coordinates.X.value, "cohort"), "units": "g/m**2/second"},
            "biomass": {"dims": (Coordinates.Y.value, Coordinates.X.value), "units": "g/m**2"},
        },
    )
```

### Section 5 : État Initial et Exécution

```python
# État initial (zéro)
lats = forcings[Coordinates.Y.value]
lons = forcings[Coordinates.X.value]

biomass_init = xr.DataArray(
    np.zeros((len(lats), len(lons))),
    coords={Coordinates.Y.value: lats, Coordinates.X.value: lons},
    dims=(Coordinates.Y.value, Coordinates.X.value),
    name="biomass",
)
biomass_init.attrs = {"units": "g/m**2"}

production_init = xr.DataArray(
    np.zeros((len(lats), len(lons), len(cohorts))),
    coords={Coordinates.Y.value: lats, Coordinates.X.value: lons, "cohort": cohorts_da},
    dims=(Coordinates.Y.value, Coordinates.X.value, "cohort"),
    name="production",
)
production_init.attrs = {"units": "g/m**2/day"}

initial_state = xr.Dataset({"biomass": biomass_init, "production": production_init})

# Exécution
controller = SimulationController(config)
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

# Extraction des résultats
results_dag = controller.results["Zooplankton/biomass"]
```

### Section 6 : Chargement des Résultats SeapoPym v0.3

```python
# Chargement des résultats v0.3
v03_path = "/Users/adm-lehodey/Documents/Workspace/Projects/phd_optimization/notebooks/Article_1/data/2_global_simulation/biomass_global.zarr"

seapopym_v03 = xr.open_zarr(v03_path)
seapopym_v03 = (
    seapopym_v03["biomass"]
    .squeeze()
    .rename({"T": "time", "X": "x", "Y": "y"})
    .load()
)
seapopym_v03.x.attrs = {}
seapopym_v03.y.attrs = {}

# Normalisation des unités (si nécessaire)
import pint_xarray
seapopym_v03 = seapopym_v03.pint.quantify().pint.to("g/m^2").pint.dequantify()

# Alignement temporel
comparison_period = slice("2000", "2019")  # Exclure spin-up
seapopym_v03_aligned = seapopym_v03.sel(time=comparison_period)
results_dag_aligned = results_dag.sel(time=comparison_period)

# Normalisation des timestamps
seapopym_v03_aligned = seapopym_v03_aligned.assign_coords(time=seapopym_v03_aligned.time.dt.floor("D"))
results_dag_aligned = results_dag_aligned.assign_coords(time=results_dag_aligned.time.dt.floor("D"))
```

### Section 7 : Calcul des Métriques

```python
def compute_metrics(ref, test):
    """Calcule les métriques de comparaison."""
    # Alignement
    ref, test = xr.align(ref, test, join="inner")

    diff = test - ref

    # RMSE
    rmse = np.sqrt((diff ** 2).mean()).values

    # Corrélation globale
    corr = xr.corr(ref, test).values

    # Biais moyen
    bias = diff.mean().values

    # Erreur L2 normalisée
    l2_error = (np.sqrt((diff ** 2).sum() / (ref ** 2).sum()).values) * 100

    # MAPE
    mape = (np.abs(diff / ref) * 100).mean().values

    return {
        'RMSE (g/m²)': rmse,
        'Corrélation': corr,
        'Biais moyen (g/m²)': bias,
        'Erreur L2 (%)': l2_error,
        'MAPE (%)': mape,
    }

metrics = compute_metrics(seapopym_v03_aligned, results_dag_aligned)

# Affichage
print("=" * 60)
print("VALIDATION SeapoPym DAG vs SeapoPym v0.3 (Sans Transport)")
print("=" * 60)
for key, value in metrics.items():
    print(f"  {key}: {value:.6f}")
print("=" * 60)
```

### Section 8 : Figures pour l'Article

```python
import matplotlib.pyplot as plt

fig_dir = "/Users/adm-lehodey/Documents/Workspace/Projects/seapopym-message/data/article/figures/"

# --- Figure A : Cartes de biomasse moyenne ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

seapopym_v03_aligned.mean("time").plot(ax=axes[0], cmap="viridis", vmin=0)
axes[0].set_title("SeapoPym v0.3 - Biomasse moyenne")

results_dag_aligned.mean("time").plot(ax=axes[1], cmap="viridis", vmin=0)
axes[1].set_title("SeapoPym DAG (v1.0) - Biomasse moyenne")

plt.tight_layout()
plt.savefig(f"{fig_dir}fig_01c_comparison_v03_maps.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Figure B : Différences ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

diff_mean = (results_dag_aligned - seapopym_v03_aligned).mean("time")
diff_mean.plot(ax=axes[0], cmap="RdBu_r", center=0)
axes[0].set_title("Différence absolue (DAG - v0.3)")

mape_spatial = (np.abs(results_dag_aligned - seapopym_v03_aligned) / seapopym_v03_aligned * 100).mean("time")
mape_spatial.plot(ax=axes[1], vmax=1, cmap="Reds")
axes[1].set_title("Erreur relative moyenne (%)")

plt.tight_layout()
plt.savefig(f"{fig_dir}fig_01c_comparison_v03_diff.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Figure C : Séries temporelles de biomasse globale ---
fig, ax = plt.subplots(figsize=(12, 4))

biomass_v03 = seapopym_v03_aligned.sum(["x", "y"])
biomass_dag = results_dag_aligned.sum(["x", "y"])

biomass_v03.plot(ax=ax, label="SeapoPym v0.3", alpha=0.8)
biomass_dag.plot(ax=ax, label="SeapoPym DAG", alpha=0.8, linestyle="--")

ax.legend()
ax.set_title("Biomasse totale globale - Comparaison v0.3 vs DAG")
ax.set_ylabel("Biomasse totale (g/m²)")

plt.tight_layout()
plt.savefig(f"{fig_dir}fig_01c_comparison_v03_timeseries.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Figure D : Scatter plot ---
fig, ax = plt.subplots(figsize=(6, 6))

# Échantillonnage pour éviter surcharge
sample_v03 = seapopym_v03_aligned.values.flatten()[::100]
sample_dag = results_dag_aligned.values.flatten()[::100]
mask = ~np.isnan(sample_v03) & ~np.isnan(sample_dag)

ax.scatter(sample_v03[mask], sample_dag[mask], alpha=0.1, s=1)
ax.plot([0, sample_v03[mask].max()], [0, sample_v03[mask].max()], 'r--', label="1:1")

ax.set_xlabel("SeapoPym v0.3 (g/m²)")
ax.set_ylabel("SeapoPym DAG (g/m²)")
ax.set_title(f"Scatter v0.3 vs DAG (R² = {metrics['Corrélation']**2:.4f})")
ax.legend()

plt.tight_layout()
plt.savefig(f"{fig_dir}fig_01c_comparison_v03_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Section 9 : Tableau Récapitulatif

```python
# Tableau pour le manuscrit
import pandas as pd

results_table = pd.DataFrame({
    "Métrique": list(metrics.keys()),
    "Valeur": [f"{v:.4f}" for v in metrics.values()],
    "Seuil": ["—", "> 0.99", "~ 0", "< 1%", "< 1%"],
    "Validation": ["—", "✓" if metrics['Corrélation'] > 0.99 else "✗",
                   "✓" if abs(metrics['Biais moyen (g/m²)']) < 0.01 else "✗",
                   "✓" if metrics['Erreur L2 (%)'] < 1 else "✗",
                   "✓" if metrics['MAPE (%)'] < 1 else "✗"],
})

print("\nTableau pour l'article (Section 3.1.2) :")
print(results_table.to_markdown(index=False))
```

---

## 4. Résultats Attendus

### Métriques de validation

| Métrique             | Seuil de validation |
| -------------------- | ------------------- |
| Erreur L2 normalisée | < 1%                |
| Corrélation spatiale | > 0.99              |
| Biais moyen          | ~ 0                 |
| MAPE moyen           | < 1%                |

### Figures générées

| Figure  | Fichier                                 | Description                       |
| ------- | --------------------------------------- | --------------------------------- |
| Maps    | `fig_01c_comparison_v03_maps.png`       | Cartes côte à côte v0.3 vs DAG    |
| Diff    | `fig_01c_comparison_v03_diff.png`       | Différences absolues et relatives |
| Time    | `fig_01c_comparison_v03_timeseries.png` | Séries temporelles                |
| Scatter | `fig_01c_comparison_v03_scatter.png`    | Corrélation                       |

### Conclusion attendue

> "Le modèle DAG reproduit les résultats de SeapoPym v0.3 avec une erreur < 1%, confirmant la **non-régression** de l'implémentation Python."

---

## 5. Checklist d'Implémentation

-   [ ] Créer le notebook `article_02_comparison_seapopym_v0.3.ipynb`
-   [ ] Implémenter les sections 1-9
-   [ ] Exécuter le notebook complet
-   [ ] Vérifier que les métriques satisfont les seuils
-   [ ] Générer les figures finales
-   [ ] Mettre à jour la section 1.2 de `03_Resultats.md` avec les résultats

---

## Notes Techniques

### Points de vigilance

1. **Paramètres LMTL identiques** : Les paramètres doivent être exactement les mêmes que ceux utilisés pour générer les résultats v0.3.

2. **Période de spin-up** : Exclure 1998-1999 de la comparaison.

3. **Alignement temporel** : Normaliser les timestamps avec `dt.floor("D")`.

4. **Unités** : S'assurer que les deux sorties sont en `g/m²`.

5. **Masques** : Vérifier que les NaN (terres) sont aux mêmes positions.
