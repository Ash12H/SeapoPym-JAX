"""Global LMTL benchmark for performance testing with distributed backend."""

import logging
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Literal

import dask.array as da
import numpy as np
import pint
import xarray as xr
from dask.distributed import Client, LocalCluster

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
from seapopym.standard.coordinates import Coordinates, GridPosition
from seapopym.transport import BoundaryType, compute_transport_numba

# --- CONFIGURATION ---

# 1. Grid & Time
# Pour un test rapide mais significatif
LAT_SIZE = 180 * 5  # Global 1 deg (approx)
LON_SIZE = 360 * 5
DEPTH_SIZE = 1  # Nombre de couches verticales
START_DATE = datetime(2000, 1, 1)
TIMESTEPS = 2  # Nombre de pas de temps
TIMESTEP_DURATION = timedelta(hours=3)
# Client Dask
DASK_WORKERS = 1  # Reduced to give more RAM per worker
DASK_THREADS = 2
DASK_MEMORY = "42GB"
# 2. Model Parameters
NB_COHORTS_DAYS = 527  # tau_r_0 en jours (détermine la dimension 'cohort')

# 3. Computing
BACKEND_TYPE: Literal["distributed", "sequential"] = "sequential"
# Chunks Dask (seulement utilisé si distributed)
# IMPORTANT: 'lat' and 'lon' must be -1 (not chunked) because they are CORE dimensions.
# 'cohort' must be -1 to match the broadcasted structure of non-cohort variables (u, v) inside apply_ufunc.
CHUNKS = {
    "depth": DEPTH_SIZE,
    "cohort": NB_COHORTS_DAYS,
    "lat": LAT_SIZE,
    "lon": LON_SIZE,
    "time": TIMESTEPS,
}


# 4. Output
OUTPUT_VARIABLES = {"Zooplankton": ["biomass"]}
OUTPUT_PATH = None  # None = MemoryWriter, "output.zarr" = ZarrWriter

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# Silence verbose logs
logging.getLogger("distributed").setLevel(logging.WARNING)
logging.getLogger("seapopym.backend.validation").setLevel(logging.WARNING)

# Enable logging for forcing manager to track rechunks
logging.getLogger("seapopym.forcing.core").setLevel(logging.DEBUG)

logger = logging.getLogger("GlobalLMTL")


for k, v in CHUNKS.items():
    if v == -1:
        logger.warning(f"Chunk for dimension {k} is set to -1. This may cause issues with Dask.")


def log_chunks(name: str, ds: xr.Dataset) -> None:
    """Log the chunk sizes of all variables in a dataset."""
    logger.info(f"=== Chunks for {name} ===")
    for var_name, var in ds.data_vars.items():
        if hasattr(var.data, "chunks"):
            logger.info(f"  {var_name}: dims={var.dims}, chunks={var.chunks}")
        else:
            logger.info(f"  {var_name}: dims={var.dims}, chunks=None (not dask)")
    logger.info("=========================")


def generate_synthetic_forcings(
    dates: list, lat: np.ndarray, lon: np.ndarray, depth: list
) -> xr.Dataset:
    """Génère des forçages synthétiques compatibles Dask."""

    # Dimensions
    nT = len(dates)
    nX = len(lon)
    nY = len(lat)
    nZ = len(depth)

    # Grid Face Dimensions (Staggered)
    dim_x = Coordinates.X
    dim_y = Coordinates.Y
    x_face_dim = GridPosition.get_face_dim(dim_x, GridPosition.LEFT)
    y_face_dim = GridPosition.get_face_dim(dim_y, GridPosition.LEFT)

    print(
        f"DEBUG: Generating synth data with spatial chunks: Lat={CHUNKS.get('lat', -1)}, Lon={CHUNKS.get('lon', -1)}"
        f"\nDEBUG: Generating synth data with other chunks: Time={CHUNKS.get('time', -1)}, Depth={CHUNKS.get('depth', -1)}"
    )

    # Order: (T, Z, Y, X) - Standard geospatial convention
    chunks_4d = (
        CHUNKS.get("time", -1),
        CHUNKS.get("depth", -1),
        CHUNKS.get("lat", -1),
        CHUNKS.get("lon", -1),
    )
    chunks_3d = (
        CHUNKS.get("time", -1),
        CHUNKS.get("lat", -1),
        CHUNKS.get("lon", -1),
    )
    chunks_2d = (
        CHUNKS.get("lat", -1),
        CHUNKS.get("lon", -1),
    )

    # Coordonnées
    coords = {
        Coordinates.T.value: dates,
        Coordinates.Z.value: depth,
        Coordinates.Y.value: lat,
        Coordinates.X.value: lon,
    }

    # Temperature (4D) - Order: (T, Z, Y, X)
    temp_data = da.random.uniform(10, 30, size=(nT, nZ, nY, nX), chunks=chunks_4d)

    # Currents (4D) - Order: (T, Z, Y, X)
    u_data = da.random.uniform(-0.5, 0.5, size=(nT, nZ, nY, nX), chunks=chunks_4d)
    v_data = da.random.uniform(-0.5, 0.5, size=(nT, nZ, nY, nX), chunks=chunks_4d)

    # Primary Production (3D) - Order: (T, Y, X)
    pp_data = da.random.uniform(0, 1e-5, size=(nT, nY, nX), chunks=chunks_3d)

    # Grid Metrics - Order: (Y, X)
    dx_val = 111000.0
    dy_val = 111000.0
    area_val = dx_val * dy_val

    cell_areas = da.ones((nY, nX), chunks=chunks_2d, dtype=np.float32) * area_val
    dx = da.ones((nY, nX), chunks=chunks_2d, dtype=np.float32) * dx_val
    dy = da.ones((nY, nX), chunks=chunks_2d, dtype=np.float32) * dy_val

    # Face Areas (Staggered) - Order: (Y, X)
    # face_areas_ew: (Y, X_face) -> (nY, nX+1)
    face_areas_ew_data = (
        da.ones(
            (nY, nX + 1),
            chunks=(CHUNKS.get("lat", -1), CHUNKS.get("lon", -1) + 1),
        )
        * dy_val
    )

    # face_areas_ns: (Y_face, X) -> (nY+1, nX)
    face_areas_ns_data = (
        da.ones(
            (nY + 1, nX),
            chunks=(CHUNKS.get("lat", -1) + 1, CHUNKS.get("lon", -1)),
        )
        * dx_val
    )

    ocean_mask = da.ones((nY, nX), chunks=chunks_2d)

    ds = xr.Dataset(
        {
            "temperature": (
                (
                    Coordinates.T.value,
                    Coordinates.Z.value,
                    Coordinates.Y.value,
                    Coordinates.X.value,
                ),
                temp_data,
            ),
            "current_u": (
                (
                    Coordinates.T.value,
                    Coordinates.Z.value,
                    Coordinates.Y.value,
                    Coordinates.X.value,
                ),
                u_data,
            ),
            "current_v": (
                (
                    Coordinates.T.value,
                    Coordinates.Z.value,
                    Coordinates.Y.value,
                    Coordinates.X.value,
                ),
                v_data,
            ),
            "primary_production": (
                (Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value),
                pp_data,
            ),
            "cell_areas": ((Coordinates.Y.value, Coordinates.X.value), cell_areas),
            # STAGGERED VARIABLES
            "face_areas_ew": ((Coordinates.Y.value, x_face_dim), face_areas_ew_data),
            "face_areas_ns": ((y_face_dim, Coordinates.X.value), face_areas_ns_data),
            "dx": ((Coordinates.Y.value, Coordinates.X.value), dx),
            "dy": ((Coordinates.Y.value, Coordinates.X.value), dy),
            "ocean_mask": ((Coordinates.Y.value, Coordinates.X.value), ocean_mask),
            # Boundary Conditions
            "boundary_north": BoundaryType.CLOSED,
            "boundary_south": BoundaryType.CLOSED,
            "boundary_east": BoundaryType.CLOSED,
            "boundary_west": BoundaryType.CLOSED,
        },
        coords=coords,
    )

    # Attributes
    ds["temperature"].attrs["units"] = "degree_Celsius"
    ds["current_u"].attrs["units"] = "m/s"
    ds["current_v"].attrs["units"] = "m/s"
    ds["primary_production"].attrs["units"] = "g/m**2/second"
    ds["ocean_mask"].attrs["units"] = "dimensionless"
    ds["cell_areas"].attrs["units"] = "m**2"
    ds["dx"].attrs["units"] = "m"
    ds["dy"].attrs["units"] = "m"
    ds["face_areas_ew"].attrs["units"] = "m"
    ds["face_areas_ns"].attrs["units"] = "m"

    ds["dt"] = TIMESTEP_DURATION.total_seconds()

    return ds


def configure_model(bp: Blueprint) -> None:
    """Configuration identique au Notebook."""

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
    bp.register_forcing(Coordinates.X.value)
    bp.register_forcing(
        "current_u",
        dims=(Coordinates.T.value, Coordinates.Z.value, Coordinates.Y.value, Coordinates.X.value),
        units="m/s",
    )
    bp.register_forcing(
        "current_v",
        dims=(Coordinates.T.value, Coordinates.Z.value, Coordinates.Y.value, Coordinates.X.value),
        units="m/s",
    )
    bp.register_forcing("cell_areas", dims=(Coordinates.Y.value, Coordinates.X.value), units="m**2")
    bp.register_forcing("face_areas_ew", dims=(Coordinates.Y.value, Coordinates.X.value), units="m")
    bp.register_forcing("face_areas_ns", dims=(Coordinates.Y.value, Coordinates.X.value), units="m")
    bp.register_forcing("dx", dims=(Coordinates.Y.value, Coordinates.X.value), units="m")
    bp.register_forcing("dy", dims=(Coordinates.Y.value, Coordinates.X.value), units="m")

    # Register boundary conditions as inputs
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
            # Transport unifié pour biomass (advection + diffusion)
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


def run_benchmark() -> None:
    """Run the global LMTL benchmark with specified backend configuration."""
    ureg = pint.get_application_registry()

    print(f"\n--- 1. Starting Benchmark ({BACKEND_TYPE}) ---")

    if BACKEND_TYPE == "distributed":
        cluster = LocalCluster(
            n_workers=DASK_WORKERS, threads_per_worker=DASK_THREADS, memory_limit=DASK_MEMORY
        )
        client = Client(cluster)
        print(f"Dask Dashboard: {client.dashboard_link}")

    try:
        # --- PREPARE ---
        print("\n--- 2. Generating Synthetic Data ---")
        lat = np.linspace(-90, 90, LAT_SIZE)
        lon = np.linspace(0, 360, LON_SIZE)
        depth = [0]
        dates = [
            START_DATE + i * TIMESTEP_DURATION for i in range(TIMESTEP_DURATION.days + TIMESTEPS)
        ]

        forcings = generate_synthetic_forcings(dates, lat, lon, depth)

        cohorts = (np.arange(0, NB_COHORTS_DAYS) * ureg.day).to("second")
        cohorts_da = xr.DataArray(
            cohorts.magnitude, dims=["cohort"], name="cohort", attrs={"units": "second"}
        )
        forcings["cohort"] = cohorts_da

        # Persist forcings in memory to avoid recomputation
        # IMPORTANT: Must reassign the result for persist() to take effect
        logger.info("Persisting forcings in Dask distributed memory...")
        forcings = forcings.persist()
        logger.info("Forcings persisted.")

        # Params
        lmtl_params = LMTLParams(
            day_layer=ureg.Quantity(0, ureg.dimensionless),
            night_layer=ureg.Quantity(0, ureg.dimensionless),
            tau_r_0=ureg.Quantity(NB_COHORTS_DAYS, ureg.day),
            gamma_tau_r=ureg.Quantity(0.11, ureg.degC**-1),
            lambda_0=ureg.Quantity(1 / 150, ureg.day**-1),
            gamma_lambda=ureg.Quantity(0.15, ureg.degC**-1),
            E=ureg.Quantity(0.1668, ureg.dimensionless),
            T_ref=ureg.Quantity(0, ureg.degC),
        )
        D_horizontal = ureg.Quantity(100.0, ureg.m**2 / ureg.s)
        zooplankton_params = {**asdict(lmtl_params), "D_horizontal": D_horizontal}

        # Initial State
        # Order: (Y, X) for biomass, (Y, X, cohort) for production
        biomass_data = da.zeros(
            (LAT_SIZE, LON_SIZE), chunks=(CHUNKS.get("lat", LAT_SIZE), CHUNKS.get("lon", LON_SIZE))
        )

        prod_data = da.zeros(
            (LAT_SIZE, LON_SIZE, NB_COHORTS_DAYS),
            chunks=(CHUNKS.get("lat", -1), CHUNKS.get("lon", -1), CHUNKS.get("cohort", -1)),
        )

        zooplankton_ds = xr.Dataset(
            {
                "biomass": ((Coordinates.Y.value, Coordinates.X.value), biomass_data),
                "production": ((Coordinates.Y.value, Coordinates.X.value, "cohort"), prod_data),
            }
        )
        zooplankton_ds["biomass"].attrs["units"] = "g/m**2"
        zooplankton_ds["production"].attrs["units"] = "g/m**2/second"

        # Persist initial state in memory
        logger.info("Persisting initial state in Dask distributed memory...")
        zooplankton_ds = zooplankton_ds.persist()
        logger.info("Initial state persisted.")

        initial_state = {"Zooplankton": zooplankton_ds}

        # Log chunks before setup
        logger.info("\n--- Pre-Setup Chunk Analysis ---")
        log_chunks("Forcings", forcings)
        log_chunks("Initial State (Zooplankton)", initial_state["Zooplankton"])

        # --- SETUP ---
        logger.info("\n--- 3. Setup Controller ---")
        config = SimulationConfig(
            start_date=START_DATE,
            end_date=START_DATE + TIMESTEPS * TIMESTEP_DURATION,
            timestep=TIMESTEP_DURATION,
        )

        from seapopym.backend import DistributedBackend, SequentialBackend

        backend_obj = DistributedBackend() if BACKEND_TYPE == "distributed" else SequentialBackend()

        controller = SimulationController(config, backend=backend_obj)

        controller.setup(
            configure_model,
            initial_state=initial_state,
            forcings=forcings,
            parameters={"Zooplankton": zooplankton_params},
            output_variables=OUTPUT_VARIABLES,
            output_path=OUTPUT_PATH,
        )

        if controller.forcing_manager is not None:
            controller.forcing_manager.method = "ffill"
            log_chunks("After Setup - Forcings", controller.forcing_manager.forcings)

        # --- RUN ---
        print(f"\n--- 4. Running Simulation ({TIMESTEPS} steps) ---")
        start_run = time.time()

        controller.run()

        duration = time.time() - start_run
        print(f"\nDone in {duration:.2f}s ({(duration / TIMESTEPS):.3f} s/step)")

        # --- CHECK ---
        results = controller.results
        print(f"Results variables: {list(results.data_vars)}")

        if "Zooplankton/biomass" in results.data_vars:
            val = results["Zooplankton/biomass"].mean().compute().item()
            print(f"✅ Final Mean Biomass: {val}")
        else:
            print("❌ Biomass missing from results")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()

    finally:
        client.wait_for_workers(DASK_WORKERS)
        if BACKEND_TYPE == "distributed" and "client" in locals():
            client.close()
            cluster.close()


if __name__ == "__main__":
    run_benchmark()
