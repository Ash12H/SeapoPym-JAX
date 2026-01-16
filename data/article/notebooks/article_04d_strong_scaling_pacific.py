# %% [markdown]
"""
# Notebook 04D: Strong Scaling with Pacific Configuration

This notebook measures strong scaling using the REAL Pacific data configuration:
- Dimension Z présente (même si depth=1)
- `compute_layer_weighted_mean` pour moyenner sur Z
- `compute_day_length` pour le cycle jour/nuit
- Vraies données zarr (chargées en mémoire)

Compare this with 04c (synthetic data) to see the overhead of real data processing.
"""

# %%
import logging
import os
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

import dask
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
import xarray as xr
from dask.distributed import Client, LocalCluster

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
)

logging.basicConfig(level=logging.WARNING)  # Reduce noise
ureg = pint.get_application_registry()

print("✅ Imports loaded")

# %% [markdown]
"""
## Configuration
"""

# %%
# === PATHS ===
BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA_DIR = BASE_DIR.parent / "data"
INPUT_ZARR = DATA_DIR / "seapodym_lmtl_forcings_pacific.zarr"
FIGURES_DIR = BASE_DIR.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
SUMMARY_DIR = BASE_DIR.parent / "summary"
SUMMARY_DIR.mkdir(exist_ok=True)

# === BENCHMARK CONFIGURATION ===
N_COHORTS_LIST = [12]  # Test with 12 cohorts only
N_STEPS_BENCHMARK = 4  # Steps per benchmark
N_RUNS = 1  # Runs per configuration
WORKERS_LIST = [1, 2, 4, 8, 12]  # 1 worker = sequential baseline

# === DASK ===
USE_PROCESSES = False  # Threads for notebook compatibility

# === SIMULATION ===
# Data is DAILY, but we interpolate to 3h timestep
# Need at least 2 days of data for interpolation
START_DATE = "2012-01-01"
END_DATE = "2012-01-02"  # At least 2 days for interpolation
TIMESTEP = timedelta(hours=3)  # 8 steps per day

# === PHYSICAL ===
D_HORIZONTAL = 500.0  # m²/s

FIGURE_PREFIX = "fig_04d_strong_scaling_pacific"

# Adjust workers based on CPU count
MAX_CPUS = os.cpu_count() or 4
WORKERS_LIST = [n for n in WORKERS_LIST if n <= MAX_CPUS]
if MAX_CPUS not in WORKERS_LIST:
    WORKERS_LIST.append(MAX_CPUS)
if 1 not in WORKERS_LIST:
    WORKERS_LIST.insert(0, 1)
WORKERS_LIST = sorted(list(set(WORKERS_LIST)))

print(f"✅ Configuration")
print(f"   Cohorts: {N_COHORTS_LIST}")
print(f"   Workers: {WORKERS_LIST}")
print(f"   Steps: {N_STEPS_BENCHMARK}")

# %% [markdown]
"""
## Load Pacific Data

Load a small subset of real Pacific data.
This includes 4D fields (time, depth, lat, lon).
"""

# %%
print("📂 Loading Pacific forcing data...")

# Open zarr
ds_raw = xr.open_zarr(INPUT_ZARR)
print(f"   Full dims: {dict(ds_raw.sizes)}")

# Select at least 2 days for interpolation
# Day 1 (START_DATE) and Day 2 (END_DATE) allow interpolation between them
ds_period = ds_raw.sel(time=slice(START_DATE, END_DATE))
print(f"   Selected dims: {dict(ds_period.sizes)}")

# Load to memory
ds_loaded = ds_period.load()
print("✅ Data loaded to memory")

# %%
# Standardize dimension names
rename_mapping = {
    "time": Coordinates.T.value,
    "depth": Coordinates.Z.value,
    "latitude": Coordinates.Y.value,
    "longitude": Coordinates.X.value,
}
ds = ds_loaded.rename({k: v for k, v in rename_mapping.items() if k in ds_loaded.dims})

# Units
if "primary_production" in ds:
    ds["primary_production"].attrs["units"] = "mg/m**2/day"
if "temperature" in ds:
    ds["temperature"].attrs["units"] = "degC"

# Grid metrics
lat = ds[Coordinates.Y.value]
lon = ds[Coordinates.X.value]

cell_areas = compute_spherical_cell_areas(lat, lon)
face_areas_ew = compute_spherical_face_areas_ew(lat, lon)
face_areas_ns = compute_spherical_face_areas_ns(lat, lon)
dx = compute_spherical_dx(lat, lon)
dy = compute_spherical_dy(lat, lon)

# Ocean mask
temp_t0 = ds["temperature"].isel({Coordinates.T.value: 0, Coordinates.Z.value: 0})
ocean_mask = xr.where(temp_t0.notnull(), 1.0, 0.0).compute()
ocean_mask.attrs["units"] = "dimensionless"

# Add to dataset
ds["cell_areas"] = cell_areas
ds["face_areas_ew"] = face_areas_ew
ds["face_areas_ns"] = face_areas_ns
ds["dx"] = dx
ds["dy"] = dy
ds["ocean_mask"] = ocean_mask

# Boundaries
ds["boundary_north"] = xr.DataArray(BoundaryType.CLOSED, attrs={"units": "dimensionless"})
ds["boundary_south"] = xr.DataArray(BoundaryType.CLOSED, attrs={"units": "dimensionless"})
ds["boundary_east"] = xr.DataArray(BoundaryType.CLOSED, attrs={"units": "dimensionless"})
ds["boundary_west"] = xr.DataArray(BoundaryType.CLOSED, attrs={"units": "dimensionless"})

# dt
dt_seconds = TIMESTEP.total_seconds()
ds["dt"] = xr.DataArray(dt_seconds, attrs={"units": "second"})

print(f"✅ Data prepared")
print(f"   Grid: {len(lat)} x {len(lon)}")

# %% [markdown]
"""
## Blueprint (Pacific Configuration)

Uses the same structure as article_05b:
- compute_layer_weighted_mean for Z averaging
- compute_day_length for day/night
- Full transport (advection + diffusion)
"""


# %%
def create_lmtl_params(n_cohorts):
    """Create LMTL params with tau_r_0 adapted to n_cohorts."""
    tau_r_0_value = n_cohorts - 0.5
    return LMTLParams(
        day_layer=0,
        night_layer=0,
        tau_r_0=ureg.Quantity(tau_r_0_value, ureg.day),
        gamma_tau_r=ureg.Quantity(0.11, ureg.degC**-1),
        lambda_0=ureg.Quantity(1 / 150, ureg.day**-1),
        gamma_lambda=ureg.Quantity(0.15, ureg.degC**-1),
        E=0.1668,
        T_ref=ureg.Quantity(0, ureg.degC),
    )


def configure_transport_pacific(bp):
    """Configure LMTL Blueprint with Pacific structure (includes Z dimension)."""
    Y, X, T, Z = Coordinates.Y.value, Coordinates.X.value, Coordinates.T.value, Coordinates.Z.value

    # === FORCINGS (4D fields for temp/currents) ===
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

    # === UNITS ===
    units = [
        # Day length (for vertical migration weighting)
        {
            "func": compute_day_length,
            "input_mapping": {"latitude": Y, "time": T},
            "output_mapping": {"output": "day_length"},
            "output_units": {"output": "dimensionless"},
        },
        # Vertical mean of temperature (Z averaging)
        {
            "func": compute_layer_weighted_mean,
            "input_mapping": {"forcing": "temperature"},
            "output_mapping": {"output": "mean_temperature"},
            "output_units": {"output": "degree_Celsius"},
        },
        # Vertical mean of currents (Z averaging)
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
        # Temperature processing
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
        # Recruitment age
        {
            "func": compute_recruitment_age,
            "input_mapping": {"temperature": "gillooly_temperature"},
            "output_mapping": {"output": "recruitment_age"},
            "output_units": {"output": "second"},
        },
        # Production from NPP
        {
            "func": compute_production_initialization,
            "input_mapping": {"cohorts": "cohort"},
            "output_mapping": {"output": "production_source_npp"},
            "output_tendencies": {"output": "production"},
            "output_units": {"output": "g/m**2/second"},
        },
        # Production dynamics
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
        # Mortality
        {
            "func": compute_mortality_tendency,
            "input_mapping": {"temperature": "gillooly_temperature"},
            "output_mapping": {"mortality_loss": "biomass_mortality"},
            "output_tendencies": {"mortality_loss": "biomass"},
            "output_units": {"mortality_loss": "g/m**2/second"},
        },
        # Transport for biomass
        {
            "func": compute_transport_fv,
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
            "output_tendencies": {
                "advection_rate": "biomass",
                "diffusion_rate": "biomass",
            },
            "output_units": {
                "advection_rate": "g/m**2/second",
                "diffusion_rate": "g/m**2/second",
            },
        },
        # Transport for production
        {
            "func": compute_transport_fv,
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

    # === PARAMETERS ===
    parameters = {
        "day_layer": {"units": "dimensionless"},
        "night_layer": {"units": "dimensionless"},
        "tau_r_0": {"units": "second"},
        "gamma_tau_r": {"units": "1/degree_Celsius"},
        "lambda_0": {"units": "1/second"},
        "gamma_lambda": {"units": "1/degree_Celsius"},
        "T_ref": {"units": "degree_Celsius"},
        "E": {"units": "dimensionless"},
        "D_horizontal": {"units": "m**2/second"},
    }

    # === STATE VARIABLES ===
    # cohort FIRST, then Y, X (core dims last) to avoid transpose
    bp.register_group(
        group_prefix="Zooplankton",
        units=units,
        parameters=parameters,
        state_variables={
            "biomass": {"dims": (Y, X), "units": "g/m**2"},
            "production": {"dims": ("cohort", Y, X), "units": "g/m**2/second"},
        },
    )


print("✅ Blueprint function defined")

# %% [markdown]
"""
## Benchmark Function
"""


# %%
def run_benchmark_iteration(ds, n_cohorts, n_steps, n_workers):
    """Run a single benchmark iteration."""
    mode_str = "processes" if USE_PROCESSES else "threads"
    print(f"    Workers: {n_workers} ({mode_str})...")

    # Create cluster
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        processes=USE_PROCESSES,
        dashboard_address=None,
    )
    client = Client(cluster)

    try:
        # Create cohorts
        cohort_ages = np.arange(n_cohorts) * 86400.0
        cohorts_da = xr.DataArray(cohort_ages, dims=["cohort"], attrs={"units": "second"})

        # Add cohorts to forcings and CHUNK them
        ds_bench = ds.assign_coords(cohort=cohorts_da)
        # Chunk forcings for distributed backend (cohort=1 for parallelism)
        ds_bench = ds_bench.chunk({"cohort": 1})

        # LMTL params
        lmtl_params = create_lmtl_params(n_cohorts)
        D_horizontal = ureg.Quantity(D_HORIZONTAL, ureg.m**2 / ureg.s)
        params = {**asdict(lmtl_params), "D_horizontal": D_horizontal}
        print(f"    tau_r_0 = {lmtl_params.tau_r_0.magnitude:.1f} days")

        # Initial state (cohort, Y, X order)
        Y, X = Coordinates.Y.value, Coordinates.X.value
        lats, lons = ds_bench[Y], ds_bench[X]

        biomass_init = xr.DataArray(
            np.zeros((len(lats), len(lons)), dtype=np.float32),
            coords={Y: lats, X: lons},
            dims=(Y, X),
            attrs={"units": "g/m**2"},
        )
        production_init = xr.DataArray(
            np.zeros((n_cohorts, len(lats), len(lons)), dtype=np.float32),
            coords={"cohort": cohorts_da, Y: lats, X: lons},
            dims=("cohort", Y, X),
            attrs={"units": "g/m**2/second"},
        )

        initial_state = xr.Dataset({"biomass": biomass_init, "production": production_init})
        initial_state_chunked = initial_state.chunk({"cohort": 1})

        # Config
        end_date = datetime.fromisoformat(START_DATE) + timedelta(seconds=dt_seconds * n_steps)
        config = SimulationConfig(
            start_date=START_DATE,
            end_date=end_date.isoformat(),
            timestep=TIMESTEP,
        )

        # Setup - use sequential backend for 1 worker (true sequential)
        use_backend = "sequential" if n_workers == 1 else "distributed"
        controller = SimulationController(config, backend=use_backend)
        if n_workers == 1:
            print(f"    Using sequential backend (true sequential)")
        controller.setup(
            model_configuration_func=configure_transport_pacific,
            forcings=ds_bench,
            initial_state={"Zooplankton": initial_state_chunked},
            parameters={"Zooplankton": params},
            output_variables={"Zooplankton": ["biomass"]},
        )

        # Run with timing
        t_start = time.perf_counter()
        controller.run()

        # Force computation
        if controller.state is not None:
            for _, data in controller.state.items():
                if isinstance(data.data, dask.array.Array):
                    data.data.compute()

        t_end = time.perf_counter()

        return (t_end - t_start) / n_steps

    finally:
        client.close()
        cluster.close()
        time.sleep(1)


# %% [markdown]
"""
## Run Benchmark
"""

# %%
print("🚀 Starting Strong Scaling Benchmark (Pacific Config)")
print(f"   Grid: {len(lat)} x {len(lon)}")
print(f"   Cohorts: {N_COHORTS_LIST}")
print(f"   Workers: {WORKERS_LIST}")
print()

results = []

for n_cohorts in N_COHORTS_LIST:
    print(f"=== {n_cohorts} Cohorts ===")

    for n_workers in WORKERS_LIST:
        run_times = []
        for run_idx in range(N_RUNS):
            t_step = run_benchmark_iteration(ds, n_cohorts, N_STEPS_BENCHMARK, n_workers)
            run_times.append(t_step)
            print(f"      Run {run_idx + 1}: {t_step:.3f} s/step")

        avg_time = np.mean(run_times)
        std_time = np.std(run_times) if N_RUNS > 1 else 0.0
        results.append(
            {
                "n_cohorts": n_cohorts,
                "workers": n_workers,
                "time_per_step": avg_time,
                "time_std": std_time,
            }
        )

print("\n✅ Benchmark complete")

# %%
# Compute metrics
df = pd.DataFrame(results)

for nc in N_COHORTS_LIST:
    mask = df["n_cohorts"] == nc
    baseline = df[(mask) & (df["workers"] == 1)]["time_per_step"].values[0]
    df.loc[mask, "speedup"] = baseline / df.loc[mask, "time_per_step"]
    df.loc[mask, "efficiency"] = df.loc[mask, "speedup"] / df.loc[mask, "workers"]
    df.loc[mask, "baseline_time"] = baseline

print("📊 Results:")
print(df.to_string(index=False))

# Save
csv_path = SUMMARY_DIR / f"{FIGURE_PREFIX}.csv"
df.to_csv(csv_path, index=False)
print(f"\n💾 Saved: {csv_path}")

# %%
# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(N_COHORTS_LIST)))

for idx, nc in enumerate(N_COHORTS_LIST):
    subset = df[df["n_cohorts"] == nc]
    ax1.plot(
        subset["workers"],
        subset["speedup"],
        "o-",
        color=colors[idx],
        linewidth=2,
        label=f"{nc} Cohorts",
    )
    ax2.plot(
        subset["workers"],
        subset["efficiency"],
        "o-",
        color=colors[idx],
        linewidth=2,
        label=f"{nc} Cohorts",
    )

ax1.plot(WORKERS_LIST, WORKERS_LIST, "k--", alpha=0.5, label="Ideal")
ax2.axhline(1.0, color="k", linestyle="--", alpha=0.5)

ax1.set_xlabel("Workers")
ax1.set_ylabel("Speedup")
ax1.set_title("Strong Scaling: Pacific Config")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xlabel("Workers")
ax2.set_ylabel("Efficiency")
ax2.set_title("Parallel Efficiency")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.2)

plt.tight_layout()

fig_path = FIGURES_DIR / f"{FIGURE_PREFIX}.png"
plt.savefig(fig_path, dpi=150)
print(f"📈 Saved: {fig_path}")
plt.show()

print("🎉 Done!")
