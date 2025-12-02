"""Quick test to verify LMTL model with units."""

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import pint
import xarray as xr

from seapopym.controller import SimulationConfig, SimulationController
from seapopym.lmtl.configuration import LMTLParams
from seapopym.lmtl.core import (
    compute_day_length,
    compute_mean_temperature,
    compute_mortality_tendency,
    compute_production_dynamics,
    compute_production_initialization,
    compute_recruitment_age,
)
from seapopym.standard.coordinates import Coordinates

ureg = pint.get_application_registry()

# Configuration
config = SimulationConfig(
    start_date="2020-01-01",
    end_date="2021-01-01",  # 1 an pour atteindre l'équilibre
    timestep=timedelta(days=1),
)

# Paramètres
lmtl_params = LMTLParams(
    day_layer=1,
    night_layer=1,
    tau_r_0=10.38 * ureg.days,
    gamma_tau_r=ureg.Quantity(0, ureg.degC**-1),
    lambda_0=ureg.Quantity(1 / 150, ureg.day**-1),
    gamma_lambda=ureg.Quantity(0, ureg.degC**-1),
    E=0.1,
    T_ref=ureg.Quantity(0, ureg.degC),
)

# Forçages
times = np.arange(
    config.start_date,
    pd.to_datetime(config.end_date) + config.timestep,
    config.timestep,
    dtype="datetime64[ns]",
)
lats = np.array([0.0])
lons = np.array([0.0])
depths = np.array([10.0, 100.0])
cohorts = (np.arange(0, 11) * ureg.day).to("second")  # 0-10 jours (ne pas dépasser tau_r)

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

dt_val = config.timestep.total_seconds()  # dt en secondes, pas en jours!
initial_state = xr.Dataset({"biomass": biomass_init, "production": production_init, "dt": dt_val})


def configure_model(bp: Any) -> None:
    bp.register_forcing(
        "temperature",
        dims=(Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value, Coordinates.Z),
        units="degC",
    )
    bp.register_forcing(
        "primary_production",
        dims=(Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value),
        units="g/m**2/s",  # Blueprint attend des unités SI (par seconde)
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
                "func": compute_production_dynamics,
                "input_mapping": {"cohort_ages": "cohort"},
                "output_mapping": {
                    "production_tendency": "production_dynamics",
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
assert state is not None, "State should not be None after simulation"
biomass = state["biomass"].values[0, 0]
production = state["production"].values[0, 0, :]

print("=" * 60)
print(f"Après {len(times) - 1} jours de simulation:")
print("=" * 60)
print(f"Biomasse finale: {biomass:.6e} g/m²")
print("Production par cohorte (g/m²):")
for i, p in enumerate(production):
    print(f"  Cohort {i:2d} ({i} jours): {p:.6e}")
print(f"\nProduction totale: {production.sum():.6e} g/m²")
print(f"Max production: {production.max():.6e} g/m²")

# Vérification
if np.any(np.isnan(production)):
    print("\n❌ ERREUR: Production contient des NaN!")
elif np.any(np.isinf(production)):
    print("\n❌ ERREUR: Production contient des Inf!")
elif production.max() > 1000:
    print(f"\n⚠️  ATTENTION: Production très élevée ({production.max():.2e})")
elif biomass < 0.001 and len(times) > 20:
    print(f"\n⚠️  ATTENTION: Biomasse très faible après {len(times) - 1} jours")
else:
    print("\n✅ Valeurs réalistes!")
