"""Test LMTL with thermal sensitivity to verify recruitment from intermediate cohorts."""

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
    compute_layer_weighted_mean,
    compute_mortality_tendency,
    compute_production_dynamics,
    compute_production_initialization,
    compute_recruitment_age,
)
from seapopym.standard.coordinates import Coordinates

ureg = pint.get_application_registry()


def configure_model(bp: Any) -> None:
    bp.register_forcing(
        "temperature",
        dims=(Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value, Coordinates.Z.value),
        units="degC",
    )
    bp.register_forcing(
        "primary_production",
        dims=(Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value),
        units="g/m**2/s",
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
                "func": compute_layer_weighted_mean,
                "input_mapping": {"forcing": "temperature"},
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
                "input_mapping": {"temperature": "mean_temperature"},
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


def test_thermal_sensitivity():
    """Test thermal sensitivity affecting recruitment age."""
    # Configuration
    config = SimulationConfig(
        start_date="2020-01-01",
        end_date="2020-04-01",  # 91 jours pour atteindre l'équilibre
        timestep=timedelta(days=1),
    )

    gamma_tau_r_value = 0.0365

    # Paramètres avec sensibilité thermique
    lmtl_params = LMTLParams(
        day_layer=1,
        night_layer=1,
        tau_r_0=10.38 * ureg.days,
        gamma_tau_r=ureg.Quantity(gamma_tau_r_value, ureg.degC**-1),
        lambda_0=ureg.Quantity(1 / 150, ureg.day**-1),
        gamma_lambda=ureg.Quantity(0, ureg.degC**-1),
        E=0.1,
        T_ref=ureg.Quantity(0, ureg.degC),
    )

    # Calcul de l'âge de recrutement attendu (pour verification)
    tau_r_expected_days = 10.38 * np.exp(-gamma_tau_r_value * (20.0 - 0.0))

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

    cohorts = (np.arange(0, 11) * ureg.day).to("second")  # 0-10 jours

    temp_data = np.ones((len(times), len(lats), len(lons), len(depths))) * 20.0
    temperature = xr.DataArray(
        temp_data,
        coords={
            Coordinates.T.value: times,
            Coordinates.Y.value: lats,
            Coordinates.X.value: lons,
            Coordinates.Z.value: depths,
        },
        dims=(Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value, Coordinates.Z.value),
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

    dt_val = config.timestep.total_seconds()
    forcings = xr.Dataset({"temperature": temperature, "primary_production": npp, "dt": dt_val})

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

    initial_state = xr.Dataset({"biomass": biomass_init, "production": production_init})

    # Simulation
    controller = SimulationController(config, backend="sequential")
    controller.setup(
        configure_model,
        {"Micronekton": initial_state},
        forcings,
        parameters={"Micronekton": lmtl_params},
    )

    controller.run()

    # Résultats
    state = controller.state
    assert state is not None, "State should not be None after simulation"

    biomass = (
        state["Micronekton/biomass"].isel({Coordinates.Y.value: 0, Coordinates.X.value: 0}).values
    )
    production = (
        state["Micronekton/production"]
        .isel({Coordinates.Y.value: 0, Coordinates.X.value: 0})
        .values
    )

    # Assertions
    assert not np.any(np.isnan(production)), "Production contains NaNs"
    assert float(np.min(production)) >= -0.01, f"Negative production detected: {np.min(production)}"
    assert float(np.mean(biomass)) > 0.01, f"Biomass too low: {biomass}"
    assert float(np.max(production)) < 1000, f"Production too high: {np.max(production)}"

    # Check recruitment from intermediate cohorts
    # Cohorts older than tau_r should be mostly empty (recruited)
    production_vals = production  # Already numpy
    for i, p in enumerate(production_vals):
        cohort_age_days = i
        if cohort_age_days >= tau_r_expected_days + 1.5:
            # Should be low because it's recruited
            assert (
                float(np.max(p)) < 0.05
            ), f"Cohort {i} (age {i} > tau_r {tau_r_expected_days:.2f}) should be recruited but has biomass {p}"
