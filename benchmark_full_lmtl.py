#!/usr/bin/env python3
"""Benchmark des Backends Seapodym (Modèle LMTL Complet).

Ce script évalue les performances des différentes architectures d'exécution (Backends)
de `seapopym` sur une simulation complète de type LMTL (Low Mid Trophic Levels).

Il intègre toute la chaîne de traitement :
1. Biologie : Température -> Métabolisme -> Recrutement -> Production -> Mortalité
2. Transport : Advection + Diffusion (unifié)
"""

import logging
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any

import dask
import dask.array as da
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
    compute_transport_numba,
)

# Suppress Dask warnings
logging.getLogger("distributed").setLevel(logging.WARNING)
logging.getLogger("dask").setLevel(logging.WARNING)

# Initialiser le registre d'unités
ureg = pint.get_application_registry()

GRID_SIZE = 1000  # Taille modérée pour le test
TAU_R_0 = 50

# ==============================================================================
# Configuration Dask (Scheduler threadé - mémoire partagée)
# ==============================================================================
NUM_THREADS = 12  # Nombre de threads pour le calcul parallèle


# ==============================================================================
# 1. Génération de Données Synthétiques
# ==============================================================================
def generate_synthetic_data(grid_size: int = 50, days: int = 5) -> xr.Dataset:
    """Génère un jeu de données synthétique complet compatible avec le modèle LMTL."""

    # 1. Coordonnées
    lat = np.linspace(-40, 40, grid_size)
    lon = np.linspace(-180, 180, grid_size)
    times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(days)]

    # Enum Values
    T, Z, Y, X = Coordinates.T.value, Coordinates.Z.value, Coordinates.Y.value, Coordinates.X.value

    # 2. Variables Physiques (4D: T, Z, Y, X)
    shape_4d = (len(times), 1, grid_size, grid_size)
    shape_3d = (len(times), grid_size, grid_size)
    shape_2d = (grid_size, grid_size)

    # Température (Gradient Nord-Sud)
    temp_data = np.zeros(shape_4d)
    lat_grid, _ = np.meshgrid(lat, lon, indexing="ij")
    for t in range(len(times)):
        temp_data[t, 0, :, :] = 25 - 0.5 * np.abs(lat_grid)

    # Courants (U zonal constant, V nul)
    u_data = np.full(shape_4d, 0.5)
    v_data = np.zeros(shape_4d)

    # PP (Production Primaire)
    pp_data = np.full(shape_3d, 100.0)

    # 3. Dataset
    ds = xr.Dataset(
        coords={
            T: times,
            Z: [0],
            Y: lat,
            X: lon,
        }
    )

    ds["temperature"] = ((T, Z, Y, X), temp_data, {"units": "degree_Celsius"})
    ds["current_u"] = ((T, Z, Y, X), u_data, {"units": "m/s"})
    ds["current_v"] = ((T, Z, Y, X), v_data, {"units": "m/s"})
    ds["primary_production"] = ((T, Y, X), pp_data, {"units": "g/m**2/second"})

    # 4. Métriques
    lat_da = ds[Y]
    lon_da = ds[X]

    ds["cell_areas"] = compute_spherical_cell_areas(lat_da, lon_da)
    ds["face_areas_ew"] = compute_spherical_face_areas_ew(lat_da, lon_da)
    ds["face_areas_ns"] = compute_spherical_face_areas_ns(lat_da, lon_da)
    ds["dx"] = compute_spherical_dx(lat_da, lon_da)
    ds["dy"] = compute_spherical_dy(lat_da, lon_da)
    ds["ocean_mask"] = ((Y, X), np.ones(shape_2d), {"units": "dimensionless"})

    # Limites
    ds["boundary_north"] = xr.DataArray(BoundaryType.CLOSED, attrs={"units": "dimensionless"})
    ds["boundary_south"] = xr.DataArray(BoundaryType.CLOSED, attrs={"units": "dimensionless"})
    ds["boundary_east"] = xr.DataArray(BoundaryType.PERIODIC, attrs={"units": "dimensionless"})
    ds["boundary_west"] = xr.DataArray(BoundaryType.PERIODIC, attrs={"units": "dimensionless"})

    return ds


# ==============================================================================
# 2. Configuration du Modèle (Blueprint)
# ==============================================================================
def configure_model(bp: Blueprint) -> None:
    """Configuration standard du modèle LMTL."""

    def check_cfl_advection(_u: Any, _v: Any, _dx: Any, _dy: Any, _dt: Any) -> dict:
        return {}

    # Enum Values
    T, Z, Y, X = Coordinates.T.value, Coordinates.Z.value, Coordinates.Y.value, Coordinates.X.value

    bp.register_forcing("temperature", dims=(T, Z, Y, X), units="degree_Celsius")
    bp.register_forcing("primary_production", dims=(T, Y, X), units="g/m**2/second")
    bp.register_forcing("ocean_mask", dims=(Y, X), units="dimensionless")
    bp.register_forcing("dt")
    bp.register_forcing("cohort")
    bp.register_forcing(T)
    bp.register_forcing(Y)
    bp.register_forcing("current_u", dims=(T, Z, Y, X), units="m/s")
    bp.register_forcing("current_v", dims=(T, Z, Y, X), units="m/s")
    bp.register_forcing("cell_areas", dims=(Y, X), units="m**2")
    bp.register_forcing("face_areas_ew", dims=(Y, X), units="m")
    bp.register_forcing("face_areas_ns", dims=(Y, X), units="m")
    bp.register_forcing("dx", dims=(Y, X), units="m")
    bp.register_forcing("dy", dims=(Y, X), units="m")
    bp.register_forcing("boundary_north", units="dimensionless")
    bp.register_forcing("boundary_south", units="dimensionless")
    bp.register_forcing("boundary_east", units="dimensionless")
    bp.register_forcing("boundary_west", units="dimensionless")

    bp.register_group(
        group_prefix="Zooplankton",
        units=[
            {
                "func": compute_day_length,
                "output_mapping": {"output": "day_length"},
                "input_mapping": {"latitude": Y, "time": T},
                "output_units": {"output": "dimensionless"},
            },
            {
                "func": compute_layer_weighted_mean,
                "input_mapping": {"forcing": "temperature"},
                "output_mapping": {"output": "mean_temperature"},
                "output_units": {"output": "degree_Celsius"},
            },
            {
                "func": compute_layer_weighted_mean,
                "input_mapping": {"forcing": "current_u"},
                "output_mapping": {"output": "mean_current_u"},
                "output_units": {"output": "m/s"},
            },
            {
                "func": compute_layer_weighted_mean,
                "input_mapping": {"forcing": "current_v"},
                "output_mapping": {"output": "mean_current_v"},
                "output_units": {"output": "m/s"},
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
            {
                "func": compute_transport_numba,
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
            },
            {
                "func": compute_transport_numba,
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
            "production": {"dims": (Y, X, "cohort"), "units": "g/m**2/second"},
            "biomass": {"dims": (Y, X), "units": "g/m**2"},
        },
    )

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


# ==============================================================================
# 3. Main Benchmark Logic
# ==============================================================================


def main() -> None:
    """Execute LMTL backend benchmark."""
    print("================================================================================")
    print("SEAPODYM-LMTL BACKEND BENCHMARK")
    print("================================================================================")

    # Enum Values
    T, Y, X = Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value

    # 1. Configuration

    DAYS = 5  # Durée courte

    print("Configuration:")
    print(f"  Grid Size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Duration:  {DAYS} days")

    print("\nGénération des données...")
    forcings = generate_synthetic_data(grid_size=GRID_SIZE, days=DAYS)

    # Paramètres LMTL
    lmtl_params = LMTLParams(
        day_layer=ureg.Quantity(0, ureg.dimensionless),
        night_layer=ureg.Quantity(0, ureg.dimensionless),
        tau_r_0=ureg.Quantity(TAU_R_0, ureg.day),
        gamma_tau_r=ureg.Quantity(0, ureg.degC**-1),
        lambda_0=ureg.Quantity(1 / 150, ureg.day**-1),
        gamma_lambda=ureg.Quantity(0.15, ureg.degC**-1),
        E=ureg.Quantity(0.1668, ureg.dimensionless),
        T_ref=ureg.Quantity(0, ureg.degC),
    )
    D_horizontal = ureg.Quantity(100.0, ureg.m**2 / ureg.s)
    zooplankton_params = {**asdict(lmtl_params), "D_horizontal": D_horizontal}

    # Cohortes
    cohorts = (np.arange(0, np.ceil(lmtl_params.tau_r_0.magnitude) + 1) * ureg.day).to("second")
    cohorts_da = xr.DataArray(
        cohorts.magnitude, dims=["cohort"], name="cohort", attrs={"units": "second"}
    )
    forcings = forcings.assign_coords(cohort=cohorts_da)
    forcings["dt"] = (forcings[T][1] - forcings[T][0]).dt.total_seconds().item()

    # État Initial
    lats = forcings[Y]
    lons = forcings[X]
    biomass_init = xr.DataArray(
        np.zeros((len(lats), len(lons))),
        coords={Y: lats, X: lons},
        dims=(Y, X),
        attrs={"units": "g/m**2"},
    )
    production_init = xr.DataArray(
        np.zeros((len(lats), len(lons), len(cohorts))),
        coords={Y: lats, X: lons, "cohort": cohorts_da},
        dims=(Y, X, "cohort"),
        attrs={"units": "g/m**2/second"},
    )
    initial_state = xr.Dataset({"biomass": biomass_init, "production": production_init})

    # ==============================================================================
    # Configuration Dask
    # ==============================================================================
    print("\nConfiguration Dask:")
    print("  Scheduler: Threads (mémoire partagée)")
    print(f"  Threads:   {NUM_THREADS}")

    # Configurer le scheduler threadé Dask
    dask.config.set(scheduler="threads", num_workers=NUM_THREADS)

    # Benchmark Function
    def run_benchmark(backend_name: str, chunks: dict | None = None) -> dict[str, float]:
        """Run benchmark for a given backend."""
        print(f"\n--- Testing Backend: {backend_name.upper()} ---")

        config = SimulationConfig(
            start_date=forcings[T].values[0],
            end_date=forcings[T].values[-1],
            timestep=timedelta(seconds=forcings["dt"].item()),
        )

        try:
            # Setup
            t0 = time.time()
            controller = SimulationController(config, backend=backend_name)
            controller.setup(
                model_configuration_func=configure_model,
                forcings=forcings,
                initial_state={"Zooplankton": initial_state},
                parameters={"Zooplankton": zooplankton_params},
                chunks=chunks,
            )
            t_setup = time.time() - t0
            print(f"  Setup Time: {t_setup:.4f} s")

            # Run
            t0 = time.time()
            controller.run()

            # Force evaluation for data_parallel backend (lazy evaluation)
            # Without this, we only measure graph construction time, not actual computation
            if backend_name == "data_parallel" and controller.state is not None:
                # Access results via the state or forcing manager depending on implementation
                # In current seapopym, results are typically updated in-place or stored in controller
                # Let's force computation on the state variables if they are dask arrays
                for _name, data in controller.state.items():
                    if hasattr(data, "compute") and isinstance(data.data, da.Array):
                        data.compute()

            t_run = time.time() - t0
            print(f"  Run Time:   {t_run:.4f} s")

            return {"setup": t_setup, "run": t_run, "total": t_setup + t_run}

        except Exception as e:
            print(f"  ERROR: {e}")
            # print full stack trace for debugging
            import traceback

            traceback.print_exc()
            return {"setup": np.nan, "run": np.nan, "total": np.nan}

    # Run Tests
    results = {}

    # 1. Sequential
    results["Sequential"] = run_benchmark("sequential")

    # 2. Task Parallel
    results["Task Parallel"] = run_benchmark("task_parallel")

    # 3. Data Parallel
    # On chunke uniquement la dimension 'cohort' pour paralléliser le transport des cohortes
    # On évite de chunker Y et X car le transport nécessite l'intégrité spatiale
    chunks = {Y: -1, X: -1, "cohort": 1}
    results["Data Parallel"] = run_benchmark("data_parallel", chunks=chunks)

    # Summary
    print("\n\n" + "=" * 80)
    print(
        f"{'BACKEND':<20} | {'SETUP (s)':<10} | {'RUN (s)':<10} | {'TOTAL (s)':<10} | {'SPEEDUP':<10}"
    )
    print("-" * 80)

    ref_time = results["Sequential"]["total"]

    for name, res in results.items():
        speedup = "N/A" if np.isnan(res["total"]) else f"{ref_time / res['total']:.2f}x"
        print(
            f"{name:<20} | {res['setup']:<10.4f} | {res['run']:<10.4f} | {res['total']:<10.4f} | {speedup}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
