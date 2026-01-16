# %% [markdown]
"""
# Notebook 04C: Strong Scaling Analysis

This notebook measures the **strong scaling** performance of the LMTL model.
Strong scaling tests how the execution time decreases when we increase the number of CPUs
while keeping the problem size fixed.

## Key Concepts

- **Strong Scaling**: Fixed problem size, increasing parallelism
- **Speedup**: `T(1) / T(N)` where `T(N)` is the execution time with N workers
- **Efficiency**: `Speedup / N` (ideal efficiency = 1.0)

## Parallelism Strategy

The LMTL model uses a Directed Acyclic Graph (DAG) Blueprint with an explicit solver.
The transport module (`compute_transport_fv`) does NOT support chunking on spatial
dimensions (X, Y) due to stencil operations requiring neighbor data.

We achieve parallelism by chunking the **cohort** dimension, as cohorts are independent
and can be processed in parallel. This is Data Parallelism.

## Dask Configuration

Two execution modes are available:

### Notebook Mode (USE_PROCESSES=False, default)
- Uses threads instead of processes
- Compatible with interactive notebook execution
- NumPy/Numba operations release the GIL, so parallelism still works for compute
- Some overhead from GIL for Python-level operations

### Script Mode (USE_PROCESSES=True)
- Uses separate processes for true parallelism (no GIL)
- Requires running as a script: `python article_04c_strong_scaling.py`
- More accurate representation of real HPC scaling
- NOT compatible with interactive notebook execution on macOS (spawn issue)

"""

# %%
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
from dask.distributed import Client, LocalCluster, performance_report

from seapopym.controller import SimulationConfig, SimulationController
from seapopym.lmtl.configuration import LMTLParams
from seapopym.lmtl.core import (
    compute_gillooly_temperature,
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

ureg = pint.get_application_registry()

print("✅ Imports loaded successfully")

# %% [markdown]
"""
## Configuration

All configurable parameters are defined here. Adjust these values to change
the benchmark behavior without modifying the rest of the code.

### Benchmark Parameters
- `GRID_SIZE`: Spatial resolution (ny, nx). Use (100, 100) for quick tests, (500, 500) for article.
- `N_COHORTS_LIST`: List of cohort counts to test. 12 and 528 show different scaling behaviors.
- `N_STEPS_BENCHMARK`: Number of simulation steps per benchmark run.
- `N_RUNS`: Number of repetitions per configuration (for statistical averaging).
- `WORKERS_LIST`: Number of workers/CPUs to test.

### Profiling Options
- `ENABLE_DASK_PROFILING`: If True, generates HTML performance reports for each run.
"""

# %%
# === BENCHMARK CONFIGURATION ===
# Grid size: (100, 100) for quick tests, (500, 500) for production benchmarks
GRID_SIZE = (100, 100)

# Cohort configurations to compare
N_COHORTS_LIST = [12, 120]

# Simulation steps (more steps = more accurate timing, but slower)
N_STEPS_WARMUP = 2  # Steps for JIT warmup (excluded from timing)
N_STEPS_BENCHMARK = 4  # Steps for actual benchmark

# Number of runs per configuration (for averaging)
N_RUNS = 1

# Worker counts to test (will be filtered based on available CPUs)
WORKERS_LIST = [1, 2, 4, 8, 12]

# Profiling options
ENABLE_DASK_PROFILING = True  # Set True to generate HTML performance reports

# === EXECUTION MODE ===
# USE_PROCESSES=False: Uses threads (notebook-compatible, GIL-limited for Python code)
# USE_PROCESSES=True: Uses processes (script-only, true parallelism, no GIL)
# For interactive notebook execution on macOS, keep this False.
# For production benchmarks, run as script with USE_PROCESSES=True.
USE_PROCESSES = True

# === PHYSICAL PARAMETERS ===
U_MAGNITUDE = 0.1  # m/s - advection velocity
D_COEFF = 1000.0  # m²/s - horizontal diffusion coefficient
TEMPERATURE_CONSTANT = 0.0  # °C - constant temperature field
NPP_CONSTANT = 300.0  # mg C/m²/day - net primary production
START_DATE = "2000-01-01"
BIOMASS_INIT = 10.0  # g/m² - initial biomass
PRODUCTION_INIT = 0.01  # g/m² - initial production

# === LMTL MODEL PARAMETERS ===
# Note: tau_r_0 is computed dynamically based on n_cohorts
# For N cohorts (each 1 day apart), tau_r_0 must be in range (N-1, N) days
# to ensure recruitment occurs within the cohort range.


def create_lmtl_params(n_cohorts):
    """Create LMTL parameters with tau_r_0 adapted to the number of cohorts.

    Parameters
    ----------
    n_cohorts : int
        Number of cohorts in the simulation

    Returns
    -------
    LMTLParams
        LMTL parameters with tau_r_0 set to (n_cohorts - 0.5) days
    """
    # tau_r_0 must be in range (N-1, N) days for N cohorts
    # We use N - 0.5 as a sensible middle value
    tau_r_0_value = n_cohorts - 0.5

    return LMTLParams(
        day_layer=ureg.Quantity(0, ureg.dimensionless),
        night_layer=ureg.Quantity(0, ureg.dimensionless),
        tau_r_0=ureg.Quantity(tau_r_0_value, ureg.day),
        gamma_tau_r=ureg.Quantity(0.11, ureg.degC**-1),
        lambda_0=ureg.Quantity(1 / 150, ureg.day**-1),
        gamma_lambda=ureg.Quantity(0.15, ureg.degC**-1),
        E=ureg.Quantity(0.1668, ureg.dimensionless),
        T_ref=ureg.Quantity(0.0, ureg.degC),
    )


# === OUTPUT PATHS ===
BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
FIGURES_DIR = BASE_DIR.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
SUMMARY_DIR = BASE_DIR.parent / "summary"
SUMMARY_DIR.mkdir(exist_ok=True)
PROFILING_DIR = SUMMARY_DIR / "profiling"
PROFILING_DIR.mkdir(exist_ok=True)

FIGURE_PREFIX = "fig_04c_strong_scaling"

# %%
# === ADJUST WORKERS LIST BASED ON AVAILABLE CPUS ===
MAX_CPUS = os.cpu_count() or 4
WORKERS_LIST = [n for n in WORKERS_LIST if n <= MAX_CPUS]

# Ensure we test up to max CPUs
if MAX_CPUS not in WORKERS_LIST:
    WORKERS_LIST.append(MAX_CPUS)

# Ensure sequential baseline (1 worker) is included
if 1 not in WORKERS_LIST:
    WORKERS_LIST.insert(0, 1)

WORKERS_LIST = sorted(list(set(WORKERS_LIST)))

print(f"✅ Configuration ready")
print(f"   Grid size: {GRID_SIZE}")
print(f"   Cohorts to test: {N_COHORTS_LIST}")
print(f"   Workers to test: {WORKERS_LIST}")
print(f"   Available CPUs: {MAX_CPUS}")
print(f"   Dask profiling: {'Enabled' if ENABLE_DASK_PROFILING else 'Disabled'}")

# %% [markdown]
"""
## Blueprint Configuration

The Blueprint defines the LMTL model structure as a DAG (Directed Acyclic Graph).

### Model Components

1. **Temperature Processing**:
   - `compute_threshold_temperature`: Apply minimum temperature threshold
   - `compute_gillooly_temperature`: Gillooly-based temperature scaling

2. **Recruitment & Production**:
   - `compute_recruitment_age`: Age at which cohorts recruit to biomass
   - `compute_production_initialization`: NPP-based production source
   - `compute_production_dynamics`: Aging and recruitment processes

3. **Mortality**:
   - `compute_mortality_tendency`: Temperature-dependent mortality rate

4. **Transport** (applied to both biomass and production):
   - `compute_transport_fv`: Finite-volume advection-diffusion scheme

### State Variables
- `biomass`: Total biomass density (2D: Y, X)
- `production`: Cohort-resolved production (3D: Y, X, cohort)
"""


# %%
def configure_lmtl_full(bp):
    """Configure complete LMTL Blueprint with transport.

    This function registers all forcings, parameters, and computational units
    for the LMTL model including horizontal transport (advection + diffusion).

    Parameters
    ----------
    bp : Blueprint
        Blueprint instance to configure.
    """
    # === FORCINGS ===
    bp.register_forcing("cohort")
    bp.register_forcing(
        "temperature",
        dims=(Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value),
        units="degree_Celsius",
    )
    bp.register_forcing(
        "primary_production",
        dims=(Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value),
        units="g/m**2/second",
    )
    bp.register_forcing("current_u", dims=(Coordinates.Y.value, Coordinates.X.value), units="m/s")
    bp.register_forcing("current_v", dims=(Coordinates.Y.value, Coordinates.X.value), units="m/s")
    bp.register_forcing("dt", units="second")
    bp.register_forcing("cell_areas", dims=(Coordinates.Y.value, Coordinates.X.value), units="m**2")
    bp.register_forcing("face_areas_ew", dims=(Coordinates.Y.value, Coordinates.X.value), units="m")
    bp.register_forcing("face_areas_ns", dims=(Coordinates.Y.value, Coordinates.X.value), units="m")
    bp.register_forcing("dx", dims=(Coordinates.Y.value, Coordinates.X.value), units="m")
    bp.register_forcing("dy", dims=(Coordinates.Y.value, Coordinates.X.value), units="m")
    bp.register_forcing(
        "ocean_mask", dims=(Coordinates.Y.value, Coordinates.X.value), units="dimensionless"
    )
    bp.register_forcing("boundary_north", units="dimensionless")
    bp.register_forcing("boundary_south", units="dimensionless")
    bp.register_forcing("boundary_east", units="dimensionless")
    bp.register_forcing("boundary_west", units="dimensionless")

    # === LMTL GROUP ===
    bp.register_group(
        group_prefix="LMTL",
        units=[
            # Temperature processing
            {
                "func": compute_threshold_temperature,
                "input_mapping": {"temperature": "temperature", "min_temperature": "T_ref"},
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
            # Production initialization from NPP
            {
                "func": compute_production_initialization,
                "input_mapping": {"primary_production": "primary_production", "cohorts": "cohort"},
                "output_mapping": {"output": "production_source_npp"},
                "output_tendencies": {"output": "production"},
                "output_units": {"output": "g/m**2/second"},
            },
            # Production dynamics (aging + recruitment to biomass)
            {
                "func": compute_production_dynamics,
                "input_mapping": {
                    "production": "production",
                    "recruitment_age": "recruitment_age",
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
                "name": "transport_biomass",
                "input_mapping": {
                    "state": "biomass",
                    "u": "current_u",
                    "v": "current_v",
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
            # Transport for production
            {
                "func": compute_transport_fv,
                "name": "transport_production",
                "input_mapping": {
                    "state": "production",
                    "u": "current_u",
                    "v": "current_v",
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
            "biomass": {"dims": (Coordinates.Y.value, Coordinates.X.value), "units": "g/m**2"},
            "production": {
                "dims": (Coordinates.Y.value, Coordinates.X.value, "cohort"),
                "units": "g/m**2",
            },
        },
    )


print("✅ Blueprint configuration function defined")

# %% [markdown]
"""
## Synthetic Data Generation

For the strong scaling benchmark, we generate synthetic forcing data with:
- Constant temperature field (20°C)
- Constant NPP field (300 mg C/m²/day)
- Constant velocity field (0.1 m/s eastward)
- Spherical grid metrics for the given lat/lon domain

The grid spans 0-40°E longitude and -20° to 20°N latitude.
"""


# %%
def generate_inputs(grid_size, n_cohorts, n_steps):
    """Generate synthetic forcing data for benchmarking.

    Parameters
    ----------
    grid_size : tuple
        (ny, nx) grid dimensions
    n_cohorts : int
        Number of cohorts
    n_steps : int
        Number of simulation time steps

    Returns
    -------
    forcings : xr.Dataset
        Forcing data (temperature, NPP, currents, grid metrics)
    initial_state : xr.Dataset
        Initial state (biomass, production)
    dt : float
        Timestep in seconds (CFL-constrained)
    """
    ny, nx = grid_size
    lons_deg = np.linspace(0, 40, nx)
    lats_deg = np.linspace(-20, 20, ny)
    lats = xr.DataArray(lats_deg, dims=[Coordinates.Y.value])
    lons = xr.DataArray(lons_deg, dims=[Coordinates.X.value])

    # Spherical grid metrics
    cell_areas = compute_spherical_cell_areas(lats, lons)
    dx = compute_spherical_dx(lats, lons)
    dy = compute_spherical_dy(lats, lons)
    face_areas_ew = compute_spherical_face_areas_ew(lats, lons)
    face_areas_ns = compute_spherical_face_areas_ns(lats, lons)

    # CFL-constrained timestep (CFL ~ 0.5)
    dx_mean = dx.mean().values
    dt = float(int(0.5 * dx_mean / U_MAGNITUDE))

    # Coordinates
    cohorts_da = xr.DataArray(
        np.arange(n_cohorts) * 86400.0,  # 1 day per cohort
        dims=["cohort"],
        name="cohort",
        attrs={"units": "second"},
    )
    time_da = xr.DataArray(
        pd.date_range(start=START_DATE, periods=n_steps + 1, freq=timedelta(seconds=dt)),
        dims=["time"],
    )

    Y, X = Coordinates.Y.value, Coordinates.X.value
    forcings = xr.Dataset(coords={"time": time_da, Y: lats, X: lons, "cohort": cohorts_da})

    # Temperature (constant)
    forcings["temperature"] = (
        ("time", Y, X),
        np.full((n_steps + 1, ny, nx), TEMPERATURE_CONSTANT),
        {"units": "degree_Celsius"},
    )

    # Primary production (constant, converted to g/m²/s)
    forcings["primary_production"] = (
        ("time", Y, X),
        np.full((n_steps + 1, ny, nx), NPP_CONSTANT / 8.64e7),  # mg C/m²/day -> g/m²/s
        {"units": "g/m**2/second"},
    )

    # Currents (constant eastward flow)
    forcings["current_u"] = ((Y, X), np.full((ny, nx), U_MAGNITUDE), {"units": "m/s"})
    forcings["current_v"] = ((Y, X), np.full((ny, nx), 0.0), {"units": "m/s"})

    # Ocean mask (all ocean)
    forcings["ocean_mask"] = ((Y, X), np.ones((ny, nx)), {"units": "dimensionless"})

    # Grid metrics
    forcings["cell_areas"] = cell_areas
    forcings["face_areas_ew"] = face_areas_ew
    forcings["face_areas_ns"] = face_areas_ns
    forcings["dx"] = dx
    forcings["dy"] = dy
    forcings["dt"] = xr.DataArray(dt, attrs={"units": "second"})

    # Boundary conditions (closed on all sides)
    for side in ["north", "south", "east", "west"]:
        forcings[f"boundary_{side}"] = xr.DataArray(
            BoundaryType.CLOSED, attrs={"units": "dimensionless"}
        )

    # Initial state
    biomass_init = xr.DataArray(
        np.full((ny, nx), BIOMASS_INIT),
        coords={Y: lats, X: lons},
        dims=[Y, X],
        attrs={"units": "g/m**2"},
    )
    production_init = xr.DataArray(
        np.full((ny, nx, n_cohorts), PRODUCTION_INIT),
        coords={Y: lats, X: lons, "cohort": cohorts_da},
        dims=[Y, X, "cohort"],
        attrs={"units": "g/m**2"},
    )

    initial_state = xr.Dataset({"biomass": biomass_init, "production": production_init})

    return forcings, initial_state, dt


print("✅ Data generation function defined")

# %% [markdown]
"""
## Benchmark Core Function

This function runs a single benchmark iteration with a specific Dask cluster configuration.

### Key Steps:
1. Create a `LocalCluster` with the specified number of workers
2. Apply chunking strategy: `cohort=1` for parallelism, `X=-1, Y=-1` (no spatial chunking)
3. Set up the `SimulationController` with the distributed backend
4. Run the simulation and force computation of results
5. Measure and return the time per step

### Why `processes=True`?
Using separate processes (vs threads) ensures true parallelism by avoiding the GIL.
Each worker has its own Python interpreter and memory space, which is closer to
real distributed computing scenarios (e.g., HPC clusters).
"""


# %%
def run_benchmark_iteration(
    forcings, initial_state, dt, n_steps, n_workers, chunks, run_id=0, n_cohorts=0
):
    """Run a single benchmark iteration with a specific cluster configuration.

    Parameters
    ----------
    forcings : xr.Dataset
        Forcing data
    initial_state : xr.Dataset
        Initial state
    dt : float
        Timestep in seconds
    n_steps : int
        Number of simulation steps
    n_workers : int
        Number of Dask workers
    chunks : dict
        Chunking specification
    run_id : int
        Run identifier (for profiling file naming)
    n_cohorts : int
        Number of cohorts (for profiling file naming)

    Returns
    -------
    float
        Average time per step in seconds
    """
    mode_str = "processes" if USE_PROCESSES else "threads"
    print(f"    Starting LocalCluster with {n_workers} workers ({mode_str} mode)...")

    # Create LocalCluster
    # processes=True: separate processes, true parallelism, no GIL (script mode)
    # processes=False: threads, GIL-limited but notebook-compatible
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        processes=USE_PROCESSES,
        dashboard_address=None,  # Disable dashboard to save resources
    )
    client = Client(cluster)

    try:
        # Model parameters with tau_r_0 adapted to n_cohorts
        lmtl_params = create_lmtl_params(n_cohorts)
        D_horizontal = ureg.Quantity(D_COEFF, ureg.m**2 / ureg.s)
        params = {**asdict(lmtl_params), "D_horizontal": D_horizontal}
        print(f"    tau_r_0 = {lmtl_params.tau_r_0.magnitude:.1f} days (for {n_cohorts} cohorts)")

        # Simulation config
        start = datetime.fromisoformat(START_DATE)
        end = start + timedelta(seconds=dt * n_steps)
        config = SimulationConfig(
            start_date=START_DATE,
            end_date=end.isoformat(),
            timestep=timedelta(seconds=dt),
        )

        # Apply chunking (only to dimensions that exist in each dataset)
        forcings_chunks = {k: v for k, v in chunks.items() if k in forcings.dims}
        forcings_chunked = forcings.chunk(forcings_chunks)

        init_chunks = {k: v for k, v in chunks.items() if k in initial_state.dims}
        initial_state_chunked = initial_state.chunk(init_chunks)

        # Setup controller
        # Use sequential backend for 1 worker (true sequential baseline)
        # Use distributed backend for multiple workers
        if n_workers == 1:
            print("    Using sequential backend (true sequential)")
            controller = SimulationController(config, backend="sequential")
        else:
            controller = SimulationController(config, backend="distributed")
        controller.setup(
            configure_lmtl_full,
            forcings=forcings_chunked,
            initial_state={"LMTL": initial_state_chunked},
            parameters={"LMTL": params},
        )

        # Optional Dask profiling
        profile_path = None
        if ENABLE_DASK_PROFILING:
            profile_path = PROFILING_DIR / f"profile_{n_cohorts}c_{n_workers}w_run{run_id}.html"

        # Benchmark with optional profiling
        if ENABLE_DASK_PROFILING and profile_path:
            with performance_report(filename=str(profile_path)):
                t_start = time.perf_counter()
                controller.run()

                # Force computation of all dask arrays
                if controller.state is not None:
                    for _, data in controller.state.items():
                        if isinstance(data.data, dask.array.Array):
                            data.data.compute()

                t_end = time.perf_counter()
            print(f"    Profiling report saved to: {profile_path}")
        else:
            t_start = time.perf_counter()
            controller.run()

            # Force computation of all dask arrays
            if controller.state is not None:
                for _, data in controller.state.items():
                    if isinstance(data.data, dask.array.Array):
                        data.data.compute()

            t_end = time.perf_counter()

        return (t_end - t_start) / n_steps

    finally:
        client.close()
        cluster.close()
        # Allow time for cleanup
        time.sleep(1)


print("✅ Benchmark function defined")

# %% [markdown]
"""
## Run Strong Scaling Benchmark

This section executes the benchmark for all configurations:
- For each cohort count (12, 528)
- For each worker count (1, 2, 4, ..., MAX_CPUS)
- Run N_RUNS repetitions and compute averages

### Chunking Strategy

```python
chunks = {"cohort": 1, "time": -1, Y: -1, X: -1}
```

- `cohort=1`: Each cohort is a separate Dask chunk → enables parallel processing
- `X=-1, Y=-1`: No spatial chunking → required by transport stencil operations
- `time=-1`: No temporal chunking → simulation steps are sequential

This strategy leverages Data Parallelism: independent cohorts are computed in parallel
across workers, while each worker processes the full spatial domain.
"""


# %%
def run_benchmark():
    """Execute the full strong scaling benchmark.

    This function is separated to allow proper multiprocessing on macOS
    when running as a script with USE_PROCESSES=True.

    Returns
    -------
    pd.DataFrame
        Results DataFrame with columns: n_cohorts, workers, time_per_step, time_std,
        speedup, efficiency, baseline_time
    """
    mode_str = "processes" if USE_PROCESSES else "threads"
    print(f"🚀 Starting Strong Scaling Benchmark")
    print(f"   Grid: {GRID_SIZE}")
    print(f"   Cohorts: {N_COHORTS_LIST}")
    print(f"   Workers: {WORKERS_LIST}")
    print(f"   Runs per config: {N_RUNS}")
    print(f"   Steps per run: {N_STEPS_BENCHMARK}")
    print(f"   Execution mode: {mode_str}")
    print()

    results = []

    for n_cohorts in N_COHORTS_LIST:
        print(f"=== Testing {n_cohorts} Cohorts ===")

        # Generate synthetic data
        forcings, init_state, dt = generate_inputs(GRID_SIZE, n_cohorts, N_STEPS_BENCHMARK)
        print(f"    Data generated (dt={dt:.0f}s)")

        # Chunking strategy: cohort=1 for parallelism, spatial dims unchunked
        chunks = {
            "cohort": 1,
            "time": -1,
            Coordinates.Y.value: -1,
            Coordinates.X.value: -1,
        }

        # Test each worker count
        for n_workers in WORKERS_LIST:
            print(f"\n  Workers: {n_workers}")
            run_times = []

            for run_idx in range(N_RUNS):
                t_step = run_benchmark_iteration(
                    forcings,
                    init_state,
                    dt,
                    N_STEPS_BENCHMARK,
                    n_workers,
                    chunks,
                    run_id=run_idx,
                    n_cohorts=n_cohorts,
                )
                run_times.append(t_step)
                print(f"    Run {run_idx + 1}/{N_RUNS}: {t_step:.3f} s/step")

            # Store results (speedup calculated later)
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
            print(f"    Average: {avg_time:.3f} ± {std_time:.3f} s/step")

    print("\n✅ Benchmark complete")

    # Convert to DataFrame and compute metrics
    df = pd.DataFrame(results)

    # Calculate speedup and efficiency for each cohort configuration
    for nc in N_COHORTS_LIST:
        mask = df["n_cohorts"] == nc
        baseline_time = df[(mask) & (df["workers"] == 1)]["time_per_step"].values[0]

        df.loc[mask, "speedup"] = baseline_time / df.loc[mask, "time_per_step"]
        df.loc[mask, "efficiency"] = df.loc[mask, "speedup"] / df.loc[mask, "workers"]
        df.loc[mask, "baseline_time"] = baseline_time

    return df


# %%
def plot_results(df):
    """Generate and save scaling plots.

    Parameters
    ----------
    df : pd.DataFrame
        Results from run_benchmark()

    Returns
    -------
    tuple
        (csv_path, fig_path) - paths to saved files
    """
    print("📊 Results Summary:")
    print(df.to_string(index=False))

    # Save to CSV
    csv_path = SUMMARY_DIR / f"{FIGURE_PREFIX}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Results saved to: {csv_path}")

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(N_COHORTS_LIST)))

    for idx, nc in enumerate(N_COHORTS_LIST):
        subset = df[df["n_cohorts"] == nc]
        color = colors[idx]

        # Speedup plot
        ax1.plot(
            subset["workers"],
            subset["speedup"],
            "o-",
            color=color,
            linewidth=2,
            markersize=8,
            label=f"{nc} Cohorts",
        )

        # Efficiency plot
        ax2.plot(
            subset["workers"],
            subset["efficiency"],
            "o-",
            color=color,
            linewidth=2,
            markersize=8,
            label=f"{nc} Cohorts",
        )

    # Ideal scaling reference
    ax1.plot(
        WORKERS_LIST,
        WORKERS_LIST,
        "k--",
        alpha=0.5,
        linewidth=1.5,
        label="Ideal (linear)",
    )
    ax2.axhline(1.0, color="k", linestyle="--", alpha=0.5, linewidth=1.5, label="Ideal (100%)")

    # Formatting
    ax1.set_xlabel("Number of Workers", fontsize=12)
    ax1.set_ylabel("Speedup", fontsize=12)
    ax1.set_title("Strong Scaling: Speedup", fontsize=14)
    ax1.legend(frameon=True, fancybox=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(WORKERS_LIST) + 1)
    ax1.set_ylim(0, max(WORKERS_LIST) + 1)

    ax2.set_xlabel("Number of Workers", fontsize=12)
    ax2.set_ylabel("Efficiency", fontsize=12)
    ax2.set_title("Strong Scaling: Parallel Efficiency", fontsize=14)
    ax2.legend(frameon=True, fancybox=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(WORKERS_LIST) + 1)
    ax2.set_ylim(0, 1.2)

    plt.tight_layout()

    # Save figure
    fig_path = FIGURES_DIR / f"{FIGURE_PREFIX}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"📈 Figure saved to: {fig_path}")

    plt.show()

    return csv_path, fig_path


# %% [markdown]
"""
## Run Benchmark

Execute the benchmark and generate plots.

**For notebook execution**: Run the cells below interactively.
The default `USE_PROCESSES=False` mode uses threads, which is compatible with notebooks.

**For script execution** (recommended for accurate results):
```bash
# Edit USE_PROCESSES = True in the configuration section, then:
python article_04c_strong_scaling.py
```
"""

# %%
# Run the benchmark
df = run_benchmark()

# %%
# Plot and save results
csv_path, fig_path = plot_results(df)

# %%
print("🎉 Notebook complete!")
print(f"\n📁 Output files:")
print(f"   - Results: {csv_path}")
print(f"   - Figure: {fig_path}")
if ENABLE_DASK_PROFILING:
    print(f"   - Profiles: {PROFILING_DIR}/")

# %% [markdown]
"""
## Analysis & Interpretation

### Expected Behavior

1. **More cohorts = Better scaling**:
   - With 528 cohorts and `cohort=1` chunking, we have 528 independent tasks
   - With 12 cohorts, we only have 12 tasks → limited parallelism

2. **Diminishing returns**:
   - As worker count increases, Dask overhead (scheduling, communication) grows
   - Efficiency typically decreases with more workers

3. **Optimal worker count**:
   - For 12 cohorts: ~12 workers (1 cohort per worker)
   - For 528 cohorts: can scale to more workers effectively

### Threads vs Processes Mode

- **Threads mode** (`USE_PROCESSES=False`): Suitable for notebooks but limited by GIL
  for pure Python code. NumPy/Numba operations still parallelize well.
- **Processes mode** (`USE_PROCESSES=True`): True parallelism, but requires script
  execution with proper `if __name__ == '__main__'` guard.

### Artifacts Saved

- **CSV results**: `summary/fig_04c_strong_scaling.csv`
- **Figure**: `figures/fig_04c_strong_scaling.png`
- **Dask profiles** (if enabled): `summary/profiling/profile_*.html`
"""

# %%
# For script execution with USE_PROCESSES=True
if __name__ == "__main__":
    # This block only runs when executing as a script, not when importing
    # It's required for multiprocessing with processes=True on macOS
    pass  # Benchmark already runs above; this is just for the guard
