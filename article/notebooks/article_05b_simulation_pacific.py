#!/usr/bin/env python
"""Simulation SeapoPym - Pacifique (Transport vs No-Transport).

Version JAX utilisant:
- Blueprint/Config/compile_model/Runner architecture
- seapopym.functions.lmtl pour la biologie
- seapopym.functions.transport pour l'advection/diffusion

Compare deux configurations:
1. NO-TRANSPORT: biologie uniquement (0D à chaque point de grille)
2. TRANSPORT: biologie + advection + diffusion horizontale
"""

# %%
import time
from pathlib import Path

import numpy as np
import xarray as xr

# Import LMTL and transport functions (registers them with @functional decorator)
import seapopym.functions.lmtl  # noqa: F401
import seapopym.functions.transport  # noqa: F401
from seapopym.blueprint import Blueprint, Config
from seapopym.compiler import compile_model
from seapopym.engine import Runner

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA_DIR = BASE_DIR.parent / "data"
INPUT_ZARR = DATA_DIR / "seapodym_lmtl_forcings_pacific.zarr"
OUTPUT_NO_TRANSPORT = DATA_DIR / "seapopym_pacific_no_transport_jax.zarr"
OUTPUT_TRANSPORT = DATA_DIR / "seapopym_pacific_transport_jax.zarr"

# Simulation Period
START_DATE = "1998-01-01"
END_DATE = "2000-12-31"
DT = "3h"

# LMTL Biological Parameters
LMTL_E = 0.1668
LMTL_LAMBDA_0 = 1 / 150  # 1/day
LMTL_GAMMA_LAMBDA = 0.15  # 1/degC
LMTL_TAU_R_0 = 10.38  # days
LMTL_GAMMA_TAU_R = 0.11  # 1/degC
LMTL_T_REF = 0.0  # degC

# Transport Parameters
D_HORIZONTAL = 500.0  # m²/s

print(f"Data dir: {DATA_DIR}")
print(f"Input: {INPUT_ZARR}")

# =============================================================================
# 1. LOAD DATA
# =============================================================================

# %%
print("\n=== Loading forcings ===")
ds = xr.open_zarr(INPUT_ZARR).load()
print(f"Forcings loaded: {ds.dims}")

# Rename dimensions to match blueprint conventions
rename_map = {
    "time": "T",
    "depth": "Z",
    "latitude": "Y",
    "longitude": "X",
}
ds = ds.rename({k: v for k, v in rename_map.items() if k in ds.dims or k in ds.coords})

# Extract coordinates
lat = ds["Y"]
lon = ds["X"]
ny = len(lat)
nx = len(lon)

print(f"Grid: {ny} x {nx}")

# =============================================================================
# 2. PREPARE GRID METRICS
# =============================================================================

# %%
print("\n=== Computing grid metrics ===")

# Earth radius in meters
R_EARTH = 6371000.0

# Cell dimensions in degrees
dlat = float(lat.diff("Y").mean())
dlon = float(lon.diff("X").mean())

# Convert to meters (approximate for lat/lon grid)
# dy is constant (latitude spacing)
dy_m = np.abs(dlat) * np.pi / 180.0 * R_EARTH

# dx varies with latitude (longitude spacing)
lat_rad = np.deg2rad(lat.values)
dx_m = np.abs(dlon) * np.pi / 180.0 * R_EARTH * np.cos(lat_rad)

# Create 2D arrays
dx_2d = np.broadcast_to(dx_m[:, None], (ny, nx))
dy_2d = np.full((ny, nx), dy_m)

# Cell area
cell_area = dx_2d * dy_2d

# For lat/lon grid: face_height = dy, face_width = dx
face_height = dy_2d
face_width = dx_2d

# Create DataArrays
dx_da = xr.DataArray(dx_2d, dims=["Y", "X"], coords={"Y": lat, "X": lon})
dy_da = xr.DataArray(dy_2d, dims=["Y", "X"], coords={"Y": lat, "X": lon})
cell_area_da = xr.DataArray(cell_area, dims=["Y", "X"], coords={"Y": lat, "X": lon})
face_height_da = xr.DataArray(face_height, dims=["Y", "X"], coords={"Y": lat, "X": lon})
face_width_da = xr.DataArray(face_width, dims=["Y", "X"], coords={"Y": lat, "X": lon})

# Ocean mask (from temperature data)
ocean_mask = xr.where(
    ds["temperature"].isel(T=0, Z=0).notnull(),
    1.0,
    0.0,
)

# Diffusion coefficient (scalar constant)

print(f"dx range: {dx_2d.min():.0f} - {dx_2d.max():.0f} m")
print(f"dy: {dy_m:.0f} m")

# =============================================================================
# 3. COHORT CONFIGURATION
# =============================================================================

# %%
max_age_days = int(np.ceil(LMTL_TAU_R_0))
cohort_ages_days = np.arange(0, max_age_days + 1)
cohort_ages_sec = cohort_ages_days * 86400.0
n_cohorts = len(cohort_ages_sec)

print(f"Cohorts: {n_cohorts} (0 to {max_age_days} days)")

# =============================================================================
# 4. PREPARE FORCINGS
# =============================================================================

# %%
# Select surface layer (Z=0) for temperature
# Primary production should already be 2D (T, Y, X)
temperature = ds["temperature"].isel(Z=0)
primary_production = ds["primary_production"]

# Convert units if needed
# Temperature: already in degC
# NPP: from mg/m²/day to g/m²/s
if "units" in primary_production.attrs:
    if "mg" in primary_production.attrs["units"]:
        primary_production = primary_production / 1000.0  # mg -> g
    if "day" in primary_production.attrs["units"]:
        primary_production = primary_production / 86400.0  # /day -> /s

# Currents (for transport)
current_u = ds["current_u"].isel(Z=0)
current_v = ds["current_v"].isel(Z=0)

print(f"Temperature shape: {temperature.shape}")
print(f"NPP shape: {primary_production.shape}")

# =============================================================================
# 5. BLUEPRINT DEFINITIONS
# =============================================================================


def create_blueprint_no_transport():
    """Create blueprint for LMTL without transport."""
    return Blueprint.from_dict(
        {
            "id": "lmtl-pacific-no-transport",
            "version": "1.0",
            "declarations": {
                "state": {
                    "biomass": {"units": "g/m^2", "dims": ["Y", "X"]},
                    "production": {"units": "g/m^2", "dims": ["Y", "X", "C"]},
                },
                "parameters": {
                    "lambda_0": {"units": "1/s"},
                    "gamma_lambda": {"units": "1/delta_degC"},
                    "tau_r_0": {"units": "s"},
                    "gamma_tau_r": {"units": "1/delta_degC"},
                    "t_ref": {"units": "degC"},
                    "efficiency": {"units": "dimensionless"},
                    "cohort_ages": {"units": "s", "dims": ["C"]},
                },
                "forcings": {
                    "temperature": {"units": "degC", "dims": ["T", "Y", "X"]},
                    "primary_production": {"units": "g/m^2/s", "dims": ["T", "Y", "X"]},
                },
            },
            "process": [
                # Temperature processing
                {
                    "func": "lmtl:gillooly_temperature",
                    "inputs": {"temp": "forcings.temperature"},
                    "outputs": {"return": {"target": "derived.temp_norm", "type": "derived"}},
                },
                {
                    "func": "lmtl:recruitment_age",
                    "inputs": {
                        "temp": "derived.temp_norm",
                        "tau_r_0": "parameters.tau_r_0",
                        "gamma": "parameters.gamma_tau_r",
                        "t_ref": "parameters.t_ref",
                    },
                    "outputs": {"return": {"target": "derived.rec_age", "type": "derived"}},
                },
                # LMTL dynamics
                {
                    "func": "lmtl:npp_injection",
                    "inputs": {
                        "npp": "forcings.primary_production",
                        "efficiency": "parameters.efficiency",
                        "production": "state.production",
                    },
                    "outputs": {"return": {"target": "tendencies.production", "type": "tendency"}},
                },
                {
                    "func": "lmtl:aging_flow",
                    "inputs": {
                        "production": "state.production",
                        "cohort_ages": "parameters.cohort_ages",
                        "rec_age": "derived.rec_age",
                    },
                    "outputs": {"return": {"target": "tendencies.production", "type": "tendency"}},
                },
                {
                    "func": "lmtl:recruitment_flow",
                    "inputs": {
                        "production": "state.production",
                        "cohort_ages": "parameters.cohort_ages",
                        "rec_age": "derived.rec_age",
                    },
                    "outputs": {
                        "prod_loss": {"target": "tendencies.production", "type": "tendency"},
                        "biomass_gain": {"target": "tendencies.biomass", "type": "tendency"},
                    },
                },
                {
                    "func": "lmtl:mortality",
                    "inputs": {
                        "biomass": "state.biomass",
                        "temp": "derived.temp_norm",
                        "lambda_0": "parameters.lambda_0",
                        "gamma": "parameters.gamma_lambda",
                        "t_ref": "parameters.t_ref",
                    },
                    "outputs": {"return": {"target": "tendencies.biomass", "type": "tendency"}},
                },
            ],
        }
    )


def create_blueprint_transport():
    """Create blueprint for LMTL with transport."""
    return Blueprint.from_dict(
        {
            "id": "lmtl-pacific-transport",
            "version": "1.0",
            "declarations": {
                "state": {
                    "biomass": {"units": "g/m^2", "dims": ["Y", "X"]},
                    "production": {"units": "g/m^2", "dims": ["Y", "X", "C"]},
                },
                "parameters": {
                    "lambda_0": {"units": "1/s"},
                    "gamma_lambda": {"units": "1/delta_degC"},
                    "tau_r_0": {"units": "s"},
                    "gamma_tau_r": {"units": "1/delta_degC"},
                    "t_ref": {"units": "degC"},
                    "efficiency": {"units": "dimensionless"},
                    "cohort_ages": {"units": "s", "dims": ["C"]},
                    # Transport parameters
                    "D": {"units": "m^2/s", "dims": ["Y", "X"]},
                    "dx": {"units": "m", "dims": ["Y", "X"]},
                    "dy": {"units": "m", "dims": ["Y", "X"]},
                    "face_height": {"units": "m", "dims": ["Y", "X"]},
                    "face_width": {"units": "m", "dims": ["Y", "X"]},
                    "cell_area": {"units": "m^2", "dims": ["Y", "X"]},
                    "mask": {"units": "dimensionless", "dims": ["Y", "X"]},
                    "bc_north": {"units": "dimensionless"},
                    "bc_south": {"units": "dimensionless"},
                    "bc_east": {"units": "dimensionless"},
                    "bc_west": {"units": "dimensionless"},
                },
                "forcings": {
                    "temperature": {"units": "degC", "dims": ["T", "Y", "X"]},
                    "primary_production": {"units": "g/m^2/s", "dims": ["T", "Y", "X"]},
                    "current_u": {"units": "m/s", "dims": ["T", "Y", "X"]},
                    "current_v": {"units": "m/s", "dims": ["T", "Y", "X"]},
                },
            },
            "process": [
                # Temperature processing
                {
                    "func": "lmtl:gillooly_temperature",
                    "inputs": {"temp": "forcings.temperature"},
                    "outputs": {"return": {"target": "derived.temp_norm", "type": "derived"}},
                },
                {
                    "func": "lmtl:recruitment_age",
                    "inputs": {
                        "temp": "derived.temp_norm",
                        "tau_r_0": "parameters.tau_r_0",
                        "gamma": "parameters.gamma_tau_r",
                        "t_ref": "parameters.t_ref",
                    },
                    "outputs": {"return": {"target": "derived.rec_age", "type": "derived"}},
                },
                # LMTL dynamics
                {
                    "func": "lmtl:npp_injection",
                    "inputs": {
                        "npp": "forcings.primary_production",
                        "efficiency": "parameters.efficiency",
                        "production": "state.production",
                    },
                    "outputs": {"return": {"target": "tendencies.production", "type": "tendency"}},
                },
                {
                    "func": "lmtl:aging_flow",
                    "inputs": {
                        "production": "state.production",
                        "cohort_ages": "parameters.cohort_ages",
                        "rec_age": "derived.rec_age",
                    },
                    "outputs": {"return": {"target": "tendencies.production", "type": "tendency"}},
                },
                {
                    "func": "lmtl:recruitment_flow",
                    "inputs": {
                        "production": "state.production",
                        "cohort_ages": "parameters.cohort_ages",
                        "rec_age": "derived.rec_age",
                    },
                    "outputs": {
                        "prod_loss": {"target": "tendencies.production", "type": "tendency"},
                        "biomass_gain": {"target": "tendencies.biomass", "type": "tendency"},
                    },
                },
                {
                    "func": "lmtl:mortality",
                    "inputs": {
                        "biomass": "state.biomass",
                        "temp": "derived.temp_norm",
                        "lambda_0": "parameters.lambda_0",
                        "gamma": "parameters.gamma_lambda",
                        "t_ref": "parameters.t_ref",
                    },
                    "outputs": {"return": {"target": "tendencies.biomass", "type": "tendency"}},
                },
                # Transport for biomass
                {
                    "func": "phys:transport_tendency",
                    "inputs": {
                        "state": "state.biomass",
                        "u": "forcings.current_u",
                        "v": "forcings.current_v",
                        "D": "parameters.D",
                        "dx": "parameters.dx",
                        "dy": "parameters.dy",
                        "face_height": "parameters.face_height",
                        "face_width": "parameters.face_width",
                        "cell_area": "parameters.cell_area",
                        "mask": "parameters.mask",
                        "bc_north": "parameters.bc_north",
                        "bc_south": "parameters.bc_south",
                        "bc_east": "parameters.bc_east",
                        "bc_west": "parameters.bc_west",
                    },
                    "outputs": {
                        "advection_rate": {"target": "tendencies.biomass", "type": "tendency"},
                        "diffusion_rate": {"target": "tendencies.biomass", "type": "tendency"},
                    },
                },
                # Transport for production (vmapped over cohort dimension C)
                {
                    "func": "phys:transport_tendency",
                    "inputs": {
                        "state": "state.production",
                        "u": "forcings.current_u",
                        "v": "forcings.current_v",
                        "D": "parameters.D",
                        "dx": "parameters.dx",
                        "dy": "parameters.dy",
                        "face_height": "parameters.face_height",
                        "face_width": "parameters.face_width",
                        "cell_area": "parameters.cell_area",
                        "mask": "parameters.mask",
                        "bc_north": "parameters.bc_north",
                        "bc_south": "parameters.bc_south",
                        "bc_east": "parameters.bc_east",
                        "bc_west": "parameters.bc_west",
                    },
                    "outputs": {
                        "advection_rate": {"target": "tendencies.production", "type": "tendency"},
                        "diffusion_rate": {"target": "tendencies.production", "type": "tendency"},
                    },
                },
            ],
        }
    )


# =============================================================================
# 6. CREATE CONFIGS
# =============================================================================

# %%
print("\n=== Creating configurations ===")

# Common parameters
common_params = {
    "lambda_0": {"value": LMTL_LAMBDA_0 / 86400.0},  # 1/day -> 1/s
    "gamma_lambda": {"value": LMTL_GAMMA_LAMBDA},
    "tau_r_0": {"value": LMTL_TAU_R_0 * 86400.0},  # days -> s
    "gamma_tau_r": {"value": LMTL_GAMMA_TAU_R},
    "t_ref": {"value": LMTL_T_REF},
    "efficiency": {"value": LMTL_E},
    "cohort_ages": {"value": cohort_ages_sec.tolist()},
}

# Common initial state
initial_state = {
    "biomass": xr.DataArray(np.zeros((ny, nx)), dims=["Y", "X"], coords={"Y": lat, "X": lon}),
    "production": xr.DataArray(np.zeros((ny, nx, n_cohorts)), dims=["Y", "X", "C"], coords={"Y": lat, "X": lon}),
}

# Common execution config
execution_config = {
    "time_start": START_DATE,
    "time_end": END_DATE,
    "dt": DT,
    "forcing_interpolation": "linear",
    "batch_size": 100,
}

# Config for NO-TRANSPORT
config_no_transport = Config.from_dict(
    {
        "parameters": common_params,
        "forcings": {
            "temperature": temperature,
            "primary_production": primary_production,
        },
        "initial_state": initial_state,
        "execution": execution_config,
    }
)

# Config for TRANSPORT
transport_params = {
    **common_params,
    "D": D_HORIZONTAL,
    "dx": dx_da,
    "dy": dy_da,
    "face_height": face_height_da,
    "face_width": face_width_da,
    "cell_area": cell_area_da,
    "mask": ocean_mask,
    "bc_north": {"value": 0},  # CLOSED
    "bc_south": {"value": 0},  # CLOSED
    "bc_east": {"value": 0},  # CLOSED
    "bc_west": {"value": 0},  # CLOSED
}

config_transport = Config.from_dict(
    {
        "parameters": transport_params,
        "forcings": {
            "temperature": temperature,
            "primary_production": primary_production,
            "current_u": current_u,
            "current_v": current_v,
        },
        "initial_state": initial_state,
        "execution": execution_config,
    }
)

print("Configurations created.")

# =============================================================================
# 7. RUN SIMULATIONS
# =============================================================================

# %%
if __name__ == "__main__":
    # --- NO-TRANSPORT ---
    print("\n" + "=" * 60)
    print("SIMULATION NO-TRANSPORT")
    print("=" * 60)

    blueprint_no = create_blueprint_no_transport()
    print("Compiling model (no transport)...")
    model_no = compile_model(blueprint_no, config_no_transport)

    runner_no = Runner.simulation()
    print(f"Running simulation ({START_DATE} to {END_DATE}, dt={DT})...")
    t0 = time.perf_counter()
    state_no, outputs_no = runner_no.run(model_no, export_variables=["biomass"])
    t_no = time.perf_counter() - t0
    print(f"Simulation completed in {t_no:.1f}s")

    # Save results
    print(f"Saving to {OUTPUT_NO_TRANSPORT}...")
    outputs_no["biomass"].to_zarr(OUTPUT_NO_TRANSPORT, mode="w")

    # --- TRANSPORT ---
    print("\n" + "=" * 60)
    print("SIMULATION WITH TRANSPORT")
    print("=" * 60)

    blueprint_tr = create_blueprint_transport()
    print("Compiling model (with transport)...")
    model_tr = compile_model(blueprint_tr, config_transport)

    runner_tr = Runner.simulation()
    print(f"Running simulation ({START_DATE} to {END_DATE}, dt={DT})...")
    t0 = time.perf_counter()
    state_tr, outputs_tr = runner_tr.run(model_tr, export_variables=["biomass"])
    t_tr = time.perf_counter() - t0
    print(f"Simulation completed in {t_tr:.1f}s")

    # Save results
    print(f"Saving to {OUTPUT_TRANSPORT}...")
    outputs_tr["biomass"].to_zarr(OUTPUT_TRANSPORT, mode="w")

    # =============================================================================
    # 8. SUMMARY
    # =============================================================================

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Timestep: {DT}")
    print(f"Grid: {ny} x {nx}")
    print("")
    print(f"NO-TRANSPORT: {t_no:.1f}s")
    print(f"TRANSPORT:    {t_tr:.1f}s")
    print(f"Ratio:        {t_tr / t_no:.2f}x")
    print("")

    # Final biomass statistics
    biomass_no_final = outputs_no["biomass"].isel(T=-1).mean().item()
    biomass_tr_final = outputs_tr["biomass"].isel(T=-1).mean().item()
    print(f"Mean final biomass (no transport): {biomass_no_final:.4f} g/m²")
    print(f"Mean final biomass (transport):    {biomass_tr_final:.4f} g/m²")

    print("=" * 60)
