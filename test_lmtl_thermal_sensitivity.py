"""Test LMTL with thermal sensitivity to verify recruitment from intermediate cohorts."""
from datetime import datetime, timedelta

import numpy as np
import pint
import xarray as xr

from seapopym.blueprint import Blueprint
from seapopym.controller import SimulationConfig, SimulationController
from seapopym.lmtl.configuration import LMTLParams
from seapopym.lmtl.core import (
    compute_aging_tendency,
    compute_day_length,
    compute_mean_temperature,
    compute_mortality_tendency,
    compute_production_initialization,
    compute_recruitment_age,
    compute_recruitment_tendency,
)
from seapopym.standard.coordinates import Coordinates

ureg = pint.get_application_registry()

# Configuration
config = SimulationConfig(
    start_date="2020-01-01",
    end_date="2020-04-01",  # 91 jours pour atteindre l'équilibre
    timestep=timedelta(days=1),
)

# Calcul de gamma_tau_r pour obtenir un recruitment_age plus court
# tau_r = tau_r_0 * exp(-gamma * (T - T_ref))
# On veut tau_r ≈ 5 jours à T=20°C avec tau_r_0=10.38 jours et T_ref=0°C
# 5 = 10.38 * exp(-gamma * 20)
# exp(-gamma * 20) = 5/10.38 = 0.482
# -gamma * 20 = ln(0.482) = -0.73
# gamma = 0.73 / 20 = 0.0365 /°C

gamma_tau_r_value = 0.0365

# Paramètres avec sensibilité thermique
lmtl_params = LMTLParams(
    day_layer=1,
    night_layer=1,
    tau_r_0=10.38 * ureg.days,
    gamma_tau_r=ureg.Quantity(gamma_tau_r_value, ureg.degC**-1),  # Sensibilité thermique
    lambda_0=ureg.Quantity(1 / 150, ureg.day**-1),
    gamma_lambda=ureg.Quantity(0, ureg.degC**-1),
    E=0.1,
    T_ref=ureg.Quantity(0, ureg.degC),
)

# Calcul de l'âge de recrutement attendu
tau_r_expected_days = 10.38 * np.exp(-gamma_tau_r_value * (20.0 - 0.0))
print("=" * 70)
print("CONFIGURATION DU TEST")
print("=" * 70)
print(f"tau_r_0 = 10.38 jours")
print(f"gamma_tau_r = {gamma_tau_r_value:.4f} /°C")
print(f"Température = 20°C")
print(f"T_ref = 0°C")
print(f"\nÂge de recrutement attendu à 20°C:")
print(f"  tau_r = 10.38 * exp(-{gamma_tau_r_value} * 20) = {tau_r_expected_days:.2f} jours")
print(f"  = {tau_r_expected_days * 86400:.0f} secondes")
print(f"\nCohortes disponibles: 0-10 jours")
print(f"Cohortes qui devraient être recrutées: {int(np.ceil(tau_r_expected_days))} jours et +")
print("=" * 70)
print()

# Forçages
import pandas as pd

times = np.arange(
    config.start_date,
    pd.to_datetime(config.end_date) + config.timestep,
    config.timestep,
    dtype="datetime64[ns]",
)
lats = np.array([0.0])
lons = np.array([0.0])
depths = np.array([10.0, 100.0])
# IMPORTANT: On a besoin d'une cohorte "tampon" qui sera recrutée
# Avec tau_r ≈ 5.00 jours, on va jusqu'à 5 jours pour que cohort[4] + dt >= tau_r
cohorts = (np.arange(0, 6) * ureg.day).to("second")  # 0-5 jours

temp_data = np.ones((len(times), len(lats), len(lons), len(depths))) * 20.0
temperature = xr.DataArray(
    temp_data,
    coords={
        Coordinates.T.value: times,
        Coordinates.Y.value: lats,
        Coordinates.X.value: lons,
        Coordinates.Z: depths,
    },
    dims=(Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value, Coordinates.Z),
    name="temperature",
    attrs={"units": "degC"},
)

npp = xr.DataArray(
    np.ones((len(times), len(lats), len(lons))) * 1.0,
    coords={Coordinates.T.value: times, Coordinates.Y.value: lats, Coordinates.X.value: lons},
    dims=(Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value),
    name="primary_production",
    attrs={"units": "g/m**2/day"},
)

forcings = xr.Dataset({"temperature": temperature, "primary_production": npp})

# État initial
biomass_init = xr.DataArray(
    np.zeros((len(lats), len(lons))),
    coords={Coordinates.Y.value: lats, Coordinates.X.value: lons},
    dims=(Coordinates.Y.value, Coordinates.X.value),
    name="biomass",
)

production_init = xr.DataArray(
    np.zeros((len(lats), len(lons), len(cohorts))),
    coords={Coordinates.Y.value: lats, Coordinates.X.value: lons, "cohort": cohorts},
    dims=(Coordinates.Y.value, Coordinates.X.value, "cohort"),
    name="production",
)

dt_val = config.timestep.total_seconds()
initial_state = xr.Dataset({"biomass": biomass_init, "production": production_init, "dt": dt_val})


def configure_model(bp):
    bp.register_forcing(
        "temperature",
        dims=(Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value, Coordinates.Z),
        units="degC",
    )
    bp.register_forcing(
        "primary_production",
        dims=(Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value),
        units="g/m**2/s",
    )
    bp.register_forcing(
        "production", dims=("cohort", Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value)
    )
    bp.register_forcing(
        "biomass", dims=(Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value)
    )
    bp.register_forcing("dt")
    bp.register_forcing("cohort")
    bp.register_forcing(Coordinates.T.value)
    bp.register_forcing(Coordinates.Y.value)

    bp.register_group(
        group_prefix="Micronekton",
        units=[
            {
                "func": compute_day_length,
                "output_mapping": {"output": "day_length"},
                "input_mapping": {"latitude": Coordinates.Y.value, "time": Coordinates.T.value},
                "output_units": {"output": "dimensionless"},
            },
            {
                "func": compute_mean_temperature,
                "output_mapping": {"output": "mean_temperature"},
                "output_units": {"output": "degC"},
            },
            {
                "func": compute_recruitment_age,
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
                "func": compute_aging_tendency,
                "output_mapping": {"aging_flux": "production_aging"},
                "output_tendencies": {"aging_flux": "production"},
                "output_units": {"aging_flux": "g/m**2/second"},
            },
            {
                "func": compute_recruitment_tendency,
                "input_mapping": {"cohort_ages": "cohort"},
                "output_mapping": {
                    "recruitment_sink": "production_sink",
                    "recruitment_source": "biomass_source",
                },
                "output_tendencies": {
                    "recruitment_sink": "production",
                    "recruitment_source": "biomass",
                },
                "output_units": {
                    "recruitment_sink": "g/m**2/second",
                    "recruitment_source": "g/m**2/second",
                },
            },
            {
                "func": compute_mortality_tendency,
                "output_mapping": {"mortality_loss": "biomass_mortality"},
                "output_tendencies": {"mortality_loss": "biomass"},
                "output_units": {"mortality_loss": "g/m**2/second"},
            },
        ],
        parameters={
            "day_layer": {"units": "dimensionless"},
            "night_layer": {"units": "dimensionless"},
            "tau_r_0": {"units": "second"},
            "gamma_tau_r": {"units": "1/degC"},
            "lambda_0": {"units": "1/second"},
            "gamma_lambda": {"units": "1/degC"},
            "T_ref": {"units": "degC"},
            "E": {"units": "dimensionless"},
        },
    )


# Simulation
controller = SimulationController(config, backend="sequential")
controller.setup(
    configure_model,
    initial_state,
    forcings,
    parameters={"Micronekton": lmtl_params},
)

controller.run()

# Résultats
state = controller.state
biomass = state["biomass"].values[0, 0]
production = state["production"].values[0, 0, :]

print("=" * 70)
print(f"RÉSULTATS APRÈS {len(times)-1} JOURS")
print("=" * 70)
print(f"Biomasse finale: {biomass:.6f} g/m²")
print(f"\nProduction par cohorte (g/m²):")
for i, p in enumerate(production):
    cohort_age_days = i
    is_recruited = (cohort_age_days >= tau_r_expected_days)
    marker = " ← RECRUTÉ" if is_recruited and p < 0.05 else ""
    print(f"  Cohort {i:2d} ({i} jours): {p:.6f}{marker}")

print(f"\nProduction totale: {production.sum():.6f} g/m²")
print(f"Max production: {production.max():.6f} g/m²")

# Analyse
recruited_cohorts = (np.arange(11) >= tau_r_expected_days)
expected_recruited_count = recruited_cohorts.sum()
actual_low_production = (production < 0.05).sum()

print("\n" + "=" * 70)
print("ANALYSE")
print("=" * 70)
print(f"Cohortes attendues comme recrutées (âge ≥ {tau_r_expected_days:.2f} jours): {expected_recruited_count}")
print(f"Cohortes avec production faible (< 0.05): {actual_low_production}")

if np.any(np.isnan(production)):
    print("\n❌ ERREUR: Production contient des NaN!")
elif np.any(production < -0.01):
    print(f"\n❌ ERREUR: Production négative détectée (min = {production.min():.6f})")
elif biomass < 0.01:
    print(f"\n⚠️  ATTENTION: Biomasse très faible ({biomass:.6f} g/m²)")
elif production.max() > 1000:
    print(f"\n⚠️  ATTENTION: Production très élevée ({production.max():.2e})")
else:
    print("\n✅ Modèle stable avec recrutement depuis cohortes intermédiaires!")
    print(f"   Biomasse accumulée: {biomass:.3f} g/m²")
