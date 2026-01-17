# %% [markdown]
"""
# Comparison: Original vs Optimized Transport

Quick benchmark comparing compute_transport_fv vs compute_transport_fv_optimized
with real Pacific data.
"""

# %%
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pint
import xarray as xr

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
    compute_transport_fv,
    compute_transport_fv_optimized,
)

ureg = pint.get_application_registry()
print("✅ Imports loaded")

# %%
# Configuration
BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA_DIR = BASE_DIR.parent / "data"
INPUT_ZARR = DATA_DIR / "seapodym_lmtl_forcings_pacific.zarr"

N_COHORTS = 12
N_STEPS = 8
N_RUNS = 3
START_DATE = "2012-01-01"
END_DATE = "2012-01-02"
TIMESTEP = timedelta(hours=3)
D_HORIZONTAL = 500.0

print(f"✅ Config: {N_COHORTS} cohorts, {N_STEPS} steps, {N_RUNS} runs")

# %%
# Load data
print("📂 Loading Pacific data...")
ds_raw = xr.open_zarr(INPUT_ZARR)
ds_period = ds_raw.sel(time=slice(START_DATE, END_DATE))
ds = ds_period.load()

T, Z, Y, X = Coordinates.T.value, Coordinates.Z.value, Coordinates.Y.value, Coordinates.X.value
rename_map = {"time": T, "depth": Z, "latitude": Y, "longitude": X}
ds = ds.rename({k: v for k, v in rename_map.items() if k in ds.dims})

if "primary_production" in ds:
    ds["primary_production"].attrs["units"] = "mg/m**2/day"
if "temperature" in ds:
    ds["temperature"].attrs["units"] = "degC"

lat = ds[Y]
lon = ds[X]

ds["cell_areas"] = compute_spherical_cell_areas(lat, lon)
ds["face_areas_ew"] = compute_spherical_face_areas_ew(lat, lon)
ds["face_areas_ns"] = compute_spherical_face_areas_ns(lat, lon)
ds["dx"] = compute_spherical_dx(lat, lon)
ds["dy"] = compute_spherical_dy(lat, lon)

temp_t0 = ds["temperature"].isel({T: 0, Z: 0})
ocean_mask = xr.where(temp_t0.notnull(), 1.0, 0.0).compute()
ocean_mask.attrs["units"] = "dimensionless"
ds["ocean_mask"] = ocean_mask

ds["boundary_north"] = xr.DataArray(BoundaryType.CLOSED, attrs={"units": "dimensionless"})
ds["boundary_south"] = xr.DataArray(BoundaryType.CLOSED, attrs={"units": "dimensionless"})
ds["boundary_east"] = xr.DataArray(BoundaryType.CLOSED, attrs={"units": "dimensionless"})
ds["boundary_west"] = xr.DataArray(BoundaryType.CLOSED, attrs={"units": "dimensionless"})
ds["dt"] = xr.DataArray(TIMESTEP.total_seconds(), attrs={"units": "second"})

cohorts_seconds = np.arange(0, N_COHORTS) * 86400.0
cohorts_da = xr.DataArray(cohorts_seconds, dims=["cohort"], attrs={"units": "second"})
ds["cohort"] = cohorts_da

print(f"✅ Grid: {len(lat)} × {len(lon)}")

# %%
# Initial state
biomass_init = xr.DataArray(
    np.full((len(lat), len(lon)), 1.0),
    coords={Y: lat, X: lon},
    dims=[Y, X],
    attrs={"units": "g/m**2"},
)
production_init = xr.DataArray(
    np.full((len(lat), len(lon), N_COHORTS), 0.01),
    coords={Y: lat, X: lon, "cohort": cohorts_da},
    dims=[Y, X, "cohort"],
    attrs={"units": "g/m**2"},
)
initial_state = xr.Dataset({"biomass": biomass_init, "production": production_init})

# %%
# Params
tau_r_0_value = N_COHORTS - 0.5
lmtl_params = LMTLParams(
    day_layer=0,
    night_layer=0,
    tau_r_0=ureg.Quantity(tau_r_0_value, ureg.day),
    gamma_tau_r=ureg.Quantity(0.11, ureg.degC**-1),
    lambda_0=ureg.Quantity(1 / 150, ureg.day**-1),
    gamma_lambda=ureg.Quantity(0.15, ureg.degC**-1),
    E=0.1668,
    T_ref=ureg.Quantity(0, ureg.degC),
)
params = {**asdict(lmtl_params), "D_horizontal": ureg.Quantity(D_HORIZONTAL, ureg.m**2 / ureg.s)}


# %%
def create_blueprint_func(transport_func, name):
    """Create a blueprint configuration function with specified transport."""

    def configure(bp):
        bp.register_forcing("temperature", dims=(T, Z, Y, X), units="degree_Celsius")
        bp.register_forcing("primary_production", dims=(T, Y, X), units="g/m**2/second")
        bp.register_forcing("current_u", dims=(T, Z, Y, X), units="m/s")
        bp.register_forcing("current_v", dims=(T, Z, Y, X), units="m/s")
        bp.register_forcing("ocean_mask", dims=(Y, X), units="dimensionless")
        bp.register_forcing("cell_areas", dims=(Y, X), units="m**2")
        bp.register_forcing("face_areas_ew", dims=(Y, X), units="m")
        bp.register_forcing("face_areas_ns", dims=(Y, X), units="m")
        bp.register_forcing("dx", dims=(Y, X), units="m")
        bp.register_forcing("dy", dims=(Y, X), units="m")
        bp.register_forcing("dt")
        bp.register_forcing("cohort")
        bp.register_forcing(T)
        bp.register_forcing(Y)
        bp.register_forcing("boundary_north", units="dimensionless")
        bp.register_forcing("boundary_south", units="dimensionless")
        bp.register_forcing("boundary_east", units="dimensionless")
        bp.register_forcing("boundary_west", units="dimensionless")

        units = [
            {
                "func": compute_day_length,
                "input_mapping": {"latitude": Y, "time": T},
                "output_mapping": {"output": "day_length"},
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
                "func": transport_func,
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
                    "advection_rate": "biomass_advection",
                    "diffusion_rate": "biomass_diffusion",
                },
                "output_tendencies": {"advection_rate": "biomass", "diffusion_rate": "biomass"},
                "output_units": {
                    "advection_rate": "g/m**2/second",
                    "diffusion_rate": "g/m**2/second",
                },
            },
            {
                "func": transport_func,
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
        ]
        bp.register_group(
            group_prefix="LMTL",
            units=units,
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
                "biomass": {"dims": (Y, X), "units": "g/m**2"},
                "production": {"dims": (Y, X, "cohort"), "units": "g/m**2"},
            },
        )

    return configure


# %%
def run_benchmark(transport_func, name):
    """Run benchmark with specified transport function."""
    print(f"\n{'=' * 60}")
    print(f"🔧 Benchmarking: {name}")
    print(f"{'=' * 60}")

    configure_func = create_blueprint_func(transport_func, name)

    start = datetime.fromisoformat(START_DATE)
    end = start + TIMESTEP * N_STEPS
    config = SimulationConfig(start_date=START_DATE, end_date=end.isoformat(), timestep=TIMESTEP)

    # Warmup
    controller = SimulationController(config, backend="sequential")
    controller.setup(
        configure_func,
        forcings=ds.copy(),
        initial_state={"LMTL": initial_state.copy(deep=True)},
        parameters={"LMTL": params},
        output_variables={"LMTL": ["biomass"]},
    )
    controller.run()

    # Benchmark
    times = []
    for run in range(N_RUNS):
        controller = SimulationController(config, backend="sequential")
        controller.setup(
            configure_func,
            forcings=ds.copy(),
            initial_state={"LMTL": initial_state.copy(deep=True)},
            parameters={"LMTL": params},
            output_variables={"LMTL": ["biomass"]},
        )

        t0 = time.perf_counter()
        controller.run()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"   Run {run + 1}/{N_RUNS}: {elapsed:.3f}s")

    mean_time = np.mean(times)
    std_time = np.std(times)
    time_per_step = mean_time / N_STEPS

    print(f"   Mean: {mean_time:.3f} ± {std_time:.3f}s")
    print(f"   Per step: {time_per_step * 1000:.1f} ms")

    return {
        "name": name,
        "mean_time": mean_time,
        "std_time": std_time,
        "time_per_step": time_per_step,
    }


# %%
print("\n" + "=" * 80)
print("COMPARISON: ORIGINAL vs OPTIMIZED TRANSPORT")
print("=" * 80)
print(f"Grid: {len(lat)} × {len(lon)}")
print(f"Cohorts: {N_COHORTS}")
print(f"Steps: {N_STEPS}")

results = []
results.append(run_benchmark(compute_transport_fv, "Original (compute_transport_fv)"))
results.append(
    run_benchmark(compute_transport_fv_optimized, "Optimized (compute_transport_fv_optimized)")
)

# %%
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

original = results[0]
optimized = results[1]

speedup = original["mean_time"] / optimized["mean_time"]

print(f"\n{'Metric':<30} {'Original':<15} {'Optimized':<15} {'Speedup':<10}")
print("-" * 70)
print(
    f"{'Total time (s)':<30} {original['mean_time']:.3f}           {optimized['mean_time']:.3f}           {speedup:.2f}x"
)
print(
    f"{'Per step (ms)':<30} {original['time_per_step'] * 1000:.1f}            {optimized['time_per_step'] * 1000:.1f}            {speedup:.2f}x"
)

print(f"\n✅ Optimized version is {speedup:.2f}x faster!")
