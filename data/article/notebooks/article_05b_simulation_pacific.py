"""Simulation SeapoPym DAG - Pacifique (Transport vs No-Transport) - OPTIMIZED.

Version optimisée utilisant:
- compute_transport_fv_optimized (kernel Numba unifié)
- compute_production_dynamics_optimized (Numba guvectorize)

Cette version est ~2-3x plus rapide que la version originale.
"""

# %%
import logging
import os
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pint
import xarray as xr

from seapopym.blueprint import Blueprint
from seapopym.controller import SimulationConfig, SimulationController
from seapopym.lmtl.configuration import LMTLParams
from seapopym.lmtl.core import (
    compute_day_length,
    compute_gillooly_temperature,
    compute_layer_weighted_mean,
    compute_mortality_tendency,
    compute_production_dynamics_optimized,  # OPTIMIZED
    compute_production_initialization,
    compute_recruitment_age,
    compute_threshold_temperature,
)
from seapopym.standard.coordinates import Coordinates

# Import Transport OPTIMIZED
from seapopym.transport import (
    BoundaryType,
    check_diffusion_stability,
    compute_advection_cfl,
    compute_spherical_cell_areas,
    compute_spherical_dx,
    compute_spherical_dy,
    compute_spherical_face_areas_ew,
    compute_spherical_face_areas_ns,
    compute_transport_fv_optimized,  # OPTIMIZED
)

logging.basicConfig(level=logging.INFO)
ureg = pint.get_application_registry()

# Chemins
DATA_DIR = (
    Path(__file__).parent.parent / "data" if "__file__" in globals() else Path.cwd().parent / "data"
)
INPUT_ZARR = DATA_DIR / "seapodym_lmtl_forcings_pacific.zarr"
OUTPUT_TRANSPORT = DATA_DIR / "seapopym_pacific_transport_optimized.zarr"
OUTPUT_NO_TRANSPORT = DATA_DIR / "seapopym_pacific_no_transport_optimized.zarr"

print(f"✅ Imports OK")
print(f"📂 Data dir: {DATA_DIR}")

# %% [markdown]
# ## 1. Chargement et Configuration

# %%
# Chargement des forçages préparés
ds = xr.open_zarr(INPUT_ZARR).load()
print(f"Forçages chargés : {ds.dims}")

# Paramètres LMTL
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

# Cohortes
cohorts = (np.arange(0, np.ceil(lmtl_params.tau_r_0.magnitude) + 1) * ureg.day).to("second")
cohorts_da = xr.DataArray(
    cohorts.magnitude, dims=["cohort"], name="cohort", attrs={"units": "second"}
)

# Ajout aux forçages
ds = ds.assign_coords(cohort=cohorts_da)

# Coefficient de diffusion horizontale (paramètre de transport)
D_horizontal = ureg.Quantity(500, ureg.m**2 / ureg.s)

# Normalisation unités
if "primary_production" in ds:
    ds["primary_production"].attrs["units"] = "mg/m**2/day"
if "temperature" in ds:
    ds["temperature"].attrs["units"] = "degC"

# %%
# Standardisation des noms de dimensions/coordonnées
rename_mapping = {
    "time": Coordinates.T.value,
    "depth": Coordinates.Z.value,
    "latitude": Coordinates.Y.value,
    "longitude": Coordinates.X.value,
}
ds = ds.rename({k: v for k, v in rename_mapping.items() if k in ds.dims or k in ds.coords})

# Grid metrics et ocean mask
lat = ds[Coordinates.Y.value]
lon = ds[Coordinates.X.value]

cell_areas = compute_spherical_cell_areas(lat, lon)
face_areas_ew = compute_spherical_face_areas_ew(lat, lon)
face_areas_ns = compute_spherical_face_areas_ns(lat, lon)
dx = compute_spherical_dx(lat, lon)
dy = compute_spherical_dy(lat, lon)

ds["cell_areas"] = cell_areas
ds["face_areas_ew"] = face_areas_ew
ds["face_areas_ns"] = face_areas_ns
ds["dx"] = dx
ds["dy"] = dy

# Create ocean mask from temperature
ocean_mask = xr.where(
    ds["temperature"].isel({Coordinates.T.value: 0, Coordinates.Z.value: 0}).notnull(),
    1.0,
    0.0,
)
ocean_mask.attrs["units"] = "dimensionless"
ds["ocean_mask"] = ocean_mask

# Boundary conditions
ds["boundary_north"] = BoundaryType.CLOSED
ds["boundary_south"] = BoundaryType.CLOSED
ds["boundary_east"] = BoundaryType.CLOSED
ds["boundary_west"] = BoundaryType.CLOSED

ds["dt"] = None  # Will be set later

print(f"✅ Grid: {len(lat)} × {len(lon)}")

# %% [markdown]
# ## 2. Calcul du Pas de Temps (CFL)

# %%
# Vitesse max
u_max = abs(ds.current_u).max().compute().item()
v_max = abs(ds.current_v).max().compute().item()
vel_max = max(u_max, v_max)

# Résolution min
lat_max_rad = np.radians(60)
dx_min_m = 111_000 * np.cos(lat_max_rad)

# CFL Target = 0.5
dt_target_s = (0.5 * dx_min_m) / vel_max
dt_days = dt_target_s / 86400

print(f"Vitesse Max : {vel_max:.2f} m/s")
print(f"Dx Min (60°) : {dx_min_m / 1000:.1f} km")
print(f"Dt Optimal (CFL=0.5) : {dt_target_s:.0f} s ({dt_days:.2f} jours)")

timestep = timedelta(hours=3)
print(f"Timestep choisi : {timestep}")

config = SimulationConfig(
    start_date="1998-01-01",
    end_date="2020-01-01",  # 3 mois pour test rapide
    timestep=timestep,
)

ds["dt"] = float(config.timestep.total_seconds())

# Vérification de stabilité diffusion
stability_diffusion = check_diffusion_stability(
    D=D_horizontal.magnitude,
    dx=ds["dx"],
    dy=ds["dy"],
    dt=config.timestep.total_seconds(),
)

print(f"\n=== Vérification Stabilité Diffusion ===")
print(f"  Coefficient de diffusion : D = {stability_diffusion['D_max']:.2f} m²/s")
print(f"  CFL diffusion            : {stability_diffusion['cfl_diffusion']:.6f} (limite = 0.25)")
print(f"  Marge de sécurité        : {stability_diffusion['margin']:.2f}x")

# %% [markdown]
# ## 3. Définition des Blueprints (OPTIMIZED)


# %%
def configure_common(bp, with_transport=False):
    """Base commune LMTL avec fonctions OPTIMISÉES."""
    # Forçages
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
    bp.register_forcing(
        "ocean_mask", dims=(Coordinates.Y.value, Coordinates.X.value), units="dimensionless"
    )
    bp.register_forcing("dt")
    bp.register_forcing("cohort")
    bp.register_forcing(Coordinates.T.value)
    bp.register_forcing(Coordinates.Y.value)

    if with_transport:
        bp.register_forcing(
            "current_u",
            dims=(
                Coordinates.T.value,
                Coordinates.Z.value,
                Coordinates.Y.value,
                Coordinates.X.value,
            ),
            units="m/s",
        )
        bp.register_forcing(
            "current_v",
            dims=(
                Coordinates.T.value,
                Coordinates.Z.value,
                Coordinates.Y.value,
                Coordinates.X.value,
            ),
            units="m/s",
        )
        bp.register_forcing(
            "cell_areas", dims=(Coordinates.Y.value, Coordinates.X.value), units="m**2"
        )
        bp.register_forcing(
            "face_areas_ew", dims=(Coordinates.Y.value, Coordinates.X.value), units="m"
        )
        bp.register_forcing(
            "face_areas_ns", dims=(Coordinates.Y.value, Coordinates.X.value), units="m"
        )
        bp.register_forcing("dx", dims=(Coordinates.Y.value, Coordinates.X.value), units="m")
        bp.register_forcing("dy", dims=(Coordinates.Y.value, Coordinates.X.value), units="m")
        bp.register_forcing("boundary_north", units="dimensionless")
        bp.register_forcing("boundary_south", units="dimensionless")
        bp.register_forcing("boundary_east", units="dimensionless")
        bp.register_forcing("boundary_west", units="dimensionless")

    # Units Communes
    units = [
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
            # OPTIMIZED: compute_production_dynamics_optimized
            "func": compute_production_dynamics_optimized,
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
    ]

    if with_transport:
        # Ajouter le calcul des moyennes verticales des courants
        units.insert(
            2,
            {
                "func": compute_layer_weighted_mean,
                "input_mapping": {"forcing": "current_u"},
                "output_mapping": {"output": "mean_current_u"},
                "output_units": {"output": "m/s"},
            },
        )
        units.insert(
            3,
            {
                "func": compute_layer_weighted_mean,
                "input_mapping": {"forcing": "current_v"},
                "output_mapping": {"output": "mean_current_v"},
                "output_units": {"output": "m/s"},
            },
        )

        # OPTIMIZED: Transport unifié pour biomass (advection + diffusion)
        units.append(
            {
                "func": compute_transport_fv_optimized,
                "name": "transport_biomass",
                "input_mapping": {
                    "state": "biomass",
                    "u": "mean_current_u",
                    "v": "mean_current_v",
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
                    "advection_rate": "biomass_advection_tendency",
                    "diffusion_rate": "biomass_diffusion_tendency",
                },
                "output_tendencies": {
                    "advection_rate": "biomass",
                    "diffusion_rate": "biomass",
                },
                "output_units": {
                    "advection_rate": "g/m**2/second",
                    "diffusion_rate": "g/m**2/second",
                },
            }
        )

        # OPTIMIZED: Transport unifié pour production (advection + diffusion)
        units.append(
            {
                "func": compute_transport_fv_optimized,
                "name": "transport_production",
                "input_mapping": {
                    "state": "production",
                    "u": "mean_current_u",
                    "v": "mean_current_v",
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
                    "advection_rate": "production_advection_tendency",
                    "diffusion_rate": "production_diffusion_tendency",
                },
                "output_tendencies": {
                    "advection_rate": "production",
                    "diffusion_rate": "production",
                },
                "output_units": {
                    "advection_rate": "g/m**2/second",
                    "diffusion_rate": "g/m**2/second",
                },
            }
        )

    # Paramètres
    parameters = {
        "day_layer": {"units": "dimensionless"},
        "night_layer": {"units": "dimensionless"},
        "tau_r_0": {"units": "second"},
        "gamma_tau_r": {"units": "1/degree_Celsius"},
        "lambda_0": {"units": "1/second"},
        "gamma_lambda": {"units": "1/degree_Celsius"},
        "T_ref": {"units": "degree_Celsius"},
        "E": {"units": "dimensionless"},
    }

    if with_transport:
        parameters["D_horizontal"] = {"units": "m**2/second"}

    bp.register_group(
        group_prefix="Zooplankton",
        units=units,
        parameters=parameters,
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

    # Diagnostic CFL pour transport
    if with_transport:

        def check_cfl_advection(u, v, dx, dy, dt):
            """Vérifie que la CFL d'advection reste <= 1.0."""
            stability = compute_advection_cfl(u, v, dx, dy, dt)
            if not stability["is_stable"]:
                raise ValueError(
                    f"Advection CFL instability detected! Max CFL = {stability['cfl_max']:.4f} (> 1.0)"
                )
            return {}

        bp.register_diagnostic(
            check_cfl_advection,
            input_mapping={
                "u": "Zooplankton/mean_current_u",
                "v": "Zooplankton/mean_current_v",
                "dx": "dx",
                "dy": "dy",
                "dt": "dt",
            },
            name="check_stability_cfl",
        )


def configure_no_transport(bp):
    configure_common(bp, with_transport=False)


def configure_transport(bp):
    configure_common(bp, with_transport=True)


print("✅ Blueprint configuré (OPTIMIZED)")

# %%
# Initialisation État Zéro
lats = ds[Coordinates.Y.value]
lons = ds[Coordinates.X.value]

biomass_init = xr.DataArray(
    np.zeros((len(lats), len(lons))),
    coords={Coordinates.Y.value: lats, Coordinates.X.value: lons},
    dims=(Coordinates.Y.value, Coordinates.X.value),
    name="biomass",
    attrs={"units": "g/m**2"},
)
production_init = xr.DataArray(
    np.zeros((len(lats), len(lons), len(cohorts))),
    coords={Coordinates.Y.value: lats, Coordinates.X.value: lons, "cohort": cohorts_da},
    dims=(Coordinates.Y.value, Coordinates.X.value, "cohort"),
    name="production",
    attrs={"units": "g/m**2/day"},
)
initial_state = xr.Dataset({"biomass": biomass_init, "production": production_init})

print("✅ État initial créé")

# %% [markdown]
# ## 4. Exécution de la Simulation (No-Transport)

# %%
# Paramètres pour No-Transport (biologie uniquement)
zooplankton_params_no_transport = asdict(lmtl_params)

# Run No-Transport
print("\n--- Démarrage Simulation NO-TRANSPORT ---")
import time

t0 = time.perf_counter()

ctrl_no = SimulationController(config)
ctrl_no.setup(
    model_configuration_func=configure_no_transport,
    forcings=ds,
    initial_state={"Zooplankton": initial_state},
    parameters={"Zooplankton": zooplankton_params_no_transport},
    output_variables={"Zooplankton": ["biomass"]},
)
ctrl_no.run()

t_no_transport = time.perf_counter() - t0
print(f"✅ Simulation NO-TRANSPORT terminée en {t_no_transport:.1f}s")

print(f"Sauvegarde No-Transport vers {OUTPUT_NO_TRANSPORT}")
ctrl_no.results["Zooplankton/biomass"].rename("biomass").to_zarr(OUTPUT_NO_TRANSPORT, mode="w")

# %% [markdown]
# ## 5. Exécution de la Simulation (Transport)

# %%
# Paramètres pour Transport (biologie + advection + diffusion)
zooplankton_params_transport = {**asdict(lmtl_params), "D_horizontal": D_horizontal}

# Run Transport
print("\n--- Démarrage Simulation TRANSPORT ---")
t0 = time.perf_counter()

ctrl_tr = SimulationController(config)
ctrl_tr.setup(
    model_configuration_func=configure_transport,
    forcings=ds,
    initial_state={"Zooplankton": initial_state},
    parameters={"Zooplankton": zooplankton_params_transport},
    output_variables={"Zooplankton": ["biomass"]},
)
ctrl_tr.run()

t_transport = time.perf_counter() - t0
print(f"✅ Simulation TRANSPORT terminée en {t_transport:.1f}s")

print(f"Sauvegarde Transport vers {OUTPUT_TRANSPORT}")
ctrl_tr.results["Zooplankton/biomass"].rename("biomass").to_zarr(OUTPUT_TRANSPORT, mode="w")

# %% [markdown]
# ## 6. Résumé

# %%
print("\n" + "=" * 60)
print("RÉSUMÉ - SIMULATIONS OPTIMISÉES")
print("=" * 60)
print(f"Période: {config.start_date} à {config.end_date}")
print(f"Timestep: {config.timestep}")
print(f"")
print(f"NO-TRANSPORT: {t_no_transport:.1f}s")
print(f"TRANSPORT:    {t_transport:.1f}s")
print("=" * 60)
