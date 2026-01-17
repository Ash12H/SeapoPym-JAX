# %% [markdown]
"""
# Notebook 04D-bis: Strong Scaling with Numba Parallelism

Measures strong scaling by varying NUMBA_NUM_THREADS.
Each thread count runs in a separate subprocess to properly
initialize Numba with the correct thread count.
"""

# %%
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuration
BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
FIGURES_DIR = BASE_DIR.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
SUMMARY_DIR = BASE_DIR.parent / "summary"
SUMMARY_DIR.mkdir(exist_ok=True)

MAX_CPUS = os.cpu_count() or 4
THREADS_LIST = [1, 2, 4]
if MAX_CPUS >= 8:
    THREADS_LIST.append(8)
if MAX_CPUS >= 12:
    THREADS_LIST.append(12)
if MAX_CPUS not in THREADS_LIST:
    THREADS_LIST.append(MAX_CPUS)
THREADS_LIST = sorted(list(set(THREADS_LIST)))

FIGURE_PREFIX = "fig_04d_bis_strong_scaling_numba"

print(f"✅ Configuration")
print(f"   Threads to test: {THREADS_LIST}")

# %% [markdown]
"""
## Benchmark Script (run in subprocess)
"""

# %%
BENCHMARK_SCRIPT = """
import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

# Get thread count from argument
N_THREADS = int(sys.argv[1])
os.environ["NUMBA_NUM_THREADS"] = str(N_THREADS)

import numpy as np
import xarray as xr
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
import pint
import numba

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
    compute_transport_fv_optimized,
)

ureg = pint.get_application_registry()

# Config
N_COHORTS = 12
N_STEPS = 8
N_RUNS = 3
DATA_DIR = Path(sys.argv[2])
INPUT_ZARR = DATA_DIR / "seapodym_lmtl_forcings_pacific.zarr"
START_DATE = "2012-01-01"
END_DATE = "2012-01-02"
TIMESTEP = timedelta(hours=3)
D_HORIZONTAL = 500.0

# Load data
ds_raw = xr.open_zarr(INPUT_ZARR)
ds_period = ds_raw.sel(time=slice(START_DATE, END_DATE))
ds = ds_period.load()

# Rename dims to match Coordinates enum values
T, Z, Y, X = Coordinates.T.value, Coordinates.Z.value, Coordinates.Y.value, Coordinates.X.value
rename_map = {"time": T, "depth": Z, "latitude": Y, "longitude": X}
ds = ds.rename({k: v for k, v in rename_map.items() if k in ds.dims})

# Set units
if "primary_production" in ds:
    ds["primary_production"].attrs["units"] = "mg/m**2/day"
if "temperature" in ds:
    ds["temperature"].attrs["units"] = "degC"

lat = ds[Y]
lon = ds[X]

# Grid metrics
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

# Cohorts
cohorts_seconds = np.arange(0, N_COHORTS) * 86400.0
cohorts_da = xr.DataArray(cohorts_seconds, dims=["cohort"], attrs={"units": "second"})
ds["cohort"] = cohorts_da

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

# Params
tau_r_0_value = N_COHORTS - 0.5
lmtl_params = LMTLParams(
    day_layer=0, night_layer=0,
    tau_r_0=ureg.Quantity(tau_r_0_value, ureg.day),
    gamma_tau_r=ureg.Quantity(0.11, ureg.degC**-1),
    lambda_0=ureg.Quantity(1 / 150, ureg.day**-1),
    gamma_lambda=ureg.Quantity(0.15, ureg.degC**-1),
    E=0.1668,
    T_ref=ureg.Quantity(0, ureg.degC),
)
params = {**asdict(lmtl_params), "D_horizontal": ureg.Quantity(D_HORIZONTAL, ureg.m**2 / ureg.s)}

def configure_blueprint(bp):
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
        {"func": compute_day_length, "input_mapping": {"latitude": Y, "time": T}, "output_mapping": {"output": "day_length"}, "output_units": {"output": "dimensionless"}},
        {"func": compute_layer_weighted_mean, "input_mapping": {"forcing": "temperature"}, "output_mapping": {"output": "mean_temperature"}, "output_units": {"output": "degree_Celsius"}},
        {"func": compute_layer_weighted_mean, "input_mapping": {"forcing": "current_u"}, "output_mapping": {"output": "mean_current_u"}, "output_units": {"output": "m/s"}},
        {"func": compute_layer_weighted_mean, "input_mapping": {"forcing": "current_v"}, "output_mapping": {"output": "mean_current_v"}, "output_units": {"output": "m/s"}},
        {"func": compute_threshold_temperature, "input_mapping": {"temperature": "mean_temperature", "min_temperature": "T_ref"}, "output_mapping": {"output": "thresholded_temperature"}, "output_units": {"output": "degree_Celsius"}},
        {"func": compute_gillooly_temperature, "input_mapping": {"temperature": "thresholded_temperature"}, "output_mapping": {"output": "gillooly_temperature"}, "output_units": {"output": "degree_Celsius"}},
        {"func": compute_recruitment_age, "input_mapping": {"temperature": "gillooly_temperature"}, "output_mapping": {"output": "recruitment_age"}, "output_units": {"output": "second"}},
        {"func": compute_production_initialization, "input_mapping": {"cohorts": "cohort"}, "output_mapping": {"output": "production_source_npp"}, "output_tendencies": {"output": "production"}, "output_units": {"output": "g/m**2/second"}},
        {"func": compute_production_dynamics, "input_mapping": {"cohort_ages": "cohort", "dt": "dt"}, "output_mapping": {"production_tendency": "production_tendency", "recruitment_source": "biomass_source"}, "output_tendencies": {"production_tendency": "production", "recruitment_source": "biomass"}, "output_units": {"production_tendency": "g/m**2/second", "recruitment_source": "g/m**2/second"}},
        {"func": compute_mortality_tendency, "input_mapping": {"temperature": "gillooly_temperature"}, "output_mapping": {"mortality_loss": "biomass_mortality"}, "output_tendencies": {"mortality_loss": "biomass"}, "output_units": {"mortality_loss": "g/m**2/second"}},
        {"func": compute_transport_fv_optimized, "name": "transport_biomass", "input_mapping": {"state": "biomass", "u": "mean_current_u", "v": "mean_current_v", "D": "D_horizontal", "dx": "dx", "dy": "dy", "cell_areas": "cell_areas", "face_areas_ew": "face_areas_ew", "face_areas_ns": "face_areas_ns", "mask": "ocean_mask", "boundary_north": "boundary_north", "boundary_south": "boundary_south", "boundary_east": "boundary_east", "boundary_west": "boundary_west"}, "output_mapping": {"advection_rate": "biomass_advection", "diffusion_rate": "biomass_diffusion"}, "output_tendencies": {"advection_rate": "biomass", "diffusion_rate": "biomass"}, "output_units": {"advection_rate": "g/m**2/second", "diffusion_rate": "g/m**2/second"}},
        {"func": compute_transport_fv_optimized, "name": "transport_production", "input_mapping": {"state": "production", "u": "mean_current_u", "v": "mean_current_v", "D": "D_horizontal", "dx": "dx", "dy": "dy", "cell_areas": "cell_areas", "face_areas_ew": "face_areas_ew", "face_areas_ns": "face_areas_ns", "mask": "ocean_mask", "boundary_north": "boundary_north", "boundary_south": "boundary_south", "boundary_east": "boundary_east", "boundary_west": "boundary_west"}, "output_mapping": {"advection_rate": "production_advection", "diffusion_rate": "production_diffusion"}, "output_tendencies": {"advection_rate": "production", "diffusion_rate": "production"}, "output_units": {"advection_rate": "g/m**2/second", "diffusion_rate": "g/m**2/second"}},
    ]
    bp.register_group(
        group_prefix="LMTL", units=units,
        parameters={"day_layer": {"units": "dimensionless"}, "night_layer": {"units": "dimensionless"}, "tau_r_0": {"units": "second"}, "gamma_tau_r": {"units": "1/degree_Celsius"}, "lambda_0": {"units": "1/second"}, "gamma_lambda": {"units": "1/degree_Celsius"}, "T_ref": {"units": "degree_Celsius"}, "E": {"units": "dimensionless"}, "D_horizontal": {"units": "m**2/second"}},
        state_variables={"biomass": {"dims": (Y, X), "units": "g/m**2"}, "production": {"dims": (Y, X, "cohort"), "units": "g/m**2"}},
    )

# Simulation config
start = datetime.fromisoformat(START_DATE)
end = start + TIMESTEP * N_STEPS
config = SimulationConfig(start_date=START_DATE, end_date=end.isoformat(), timestep=TIMESTEP)

# Warmup
controller = SimulationController(config, backend="sequential")
controller.setup(configure_blueprint, forcings=ds.copy(), initial_state={"LMTL": initial_state.copy(deep=True)}, parameters={"LMTL": params}, output_variables={"LMTL": ["biomass"]})
controller.run()

# Benchmark
times = []
for run in range(N_RUNS):
    controller = SimulationController(config, backend="sequential")
    controller.setup(configure_blueprint, forcings=ds.copy(), initial_state={"LMTL": initial_state.copy(deep=True)}, parameters={"LMTL": params}, output_variables={"LMTL": ["biomass"]})
    t0 = time.perf_counter()
    controller.run()
    times.append(time.perf_counter() - t0)

result = {"n_threads": N_THREADS, "numba_threads": numba.get_num_threads(), "mean_time": float(np.mean(times)), "std_time": float(np.std(times)), "times": [float(t) for t in times]}
print(json.dumps(result))
"""

# %% [markdown]
"""
## Run Benchmarks
"""

# %%
# Save benchmark script
script_path = BASE_DIR / "_benchmark_numba_threads.py"
with open(script_path, "w") as f:
    f.write(BENCHMARK_SCRIPT)

DATA_DIR = BASE_DIR.parent / "data"
ROOT_DIR = BASE_DIR.parent.parent.parent  # seapopym-message root

print("\n" + "=" * 80)
print("STRONG SCALING BENCHMARK - NUMBA PARALLELISM")
print("=" * 80)

results = []
for n_threads in THREADS_LIST:
    print(f"\n{'=' * 60}")
    print(f"🔧 Testing {n_threads} threads")
    print(f"{'=' * 60}")

    result = subprocess.run(
        [sys.executable, str(script_path), str(n_threads), str(DATA_DIR)],
        capture_output=True,
        text=True,
        cwd=str(ROOT_DIR),
    )

    if result.returncode != 0:
        print(f"   ❌ Error: {result.stderr[-500:]}")
        continue

    # Parse JSON from last line
    for line in result.stdout.strip().split("\n"):
        if line.startswith("{"):
            data = json.loads(line)
            data["time_per_step"] = data["mean_time"] / 8
            results.append(data)
            print(f"   Numba threads: {data['numba_threads']}")
            print(f"   Mean: {data['mean_time']:.3f} ± {data['std_time']:.3f}s")
            print(f"   Per step: {data['time_per_step']:.4f}s")
            break

# Cleanup
script_path.unlink()

# %% [markdown]
"""
## Analysis
"""

# %%
if len(results) < 2:
    print("❌ Not enough results to analyze")
else:
    df = pd.DataFrame(results)

    # Calculate speedup
    baseline_time = df[df["n_threads"] == 1]["mean_time"].values[0]
    df["speedup"] = baseline_time / df["mean_time"]
    df["efficiency"] = df["speedup"] / df["n_threads"] * 100

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(
        df[["n_threads", "mean_time", "time_per_step", "speedup", "efficiency"]].to_string(
            index=False
        )
    )

    # %% Figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Execution time
    ax1 = axes[0]
    ax1.plot(df["n_threads"], df["mean_time"], "o-", linewidth=2, markersize=8)
    ax1.fill_between(
        df["n_threads"],
        df["mean_time"] - df["std_time"],
        df["mean_time"] + df["std_time"],
        alpha=0.2,
    )
    ax1.set_xlabel("Number of Numba Threads")
    ax1.set_ylabel("Execution Time (s)")
    ax1.set_title("Execution Time vs Threads")
    ax1.grid(True, alpha=0.3)

    # Speedup
    ax2 = axes[1]
    ax2.plot(df["n_threads"], df["speedup"], "o-", linewidth=2, markersize=8, color="green")
    ax2.plot(df["n_threads"], df["n_threads"], "--", color="gray", label="Ideal")
    ax2.set_xlabel("Number of Numba Threads")
    ax2.set_ylabel("Speedup")
    ax2.set_title("Strong Scaling Speedup")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Efficiency
    ax3 = axes[2]
    ax3.bar(df["n_threads"].astype(str), df["efficiency"], color="orange", alpha=0.8)
    ax3.axhline(100, color="gray", linestyle="--")
    ax3.set_xlabel("Number of Numba Threads")
    ax3.set_ylabel("Efficiency (%)")
    ax3.set_title("Parallel Efficiency")
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{FIGURE_PREFIX}.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ Figure saved: {FIGURES_DIR / f'{FIGURE_PREFIX}.png'}")

    # Summary
    summary_path = SUMMARY_DIR / f"{FIGURE_PREFIX.replace('fig_', 'notebook_')}_summary.txt"
    with open(summary_path, "w") as f:
        f.write("NOTEBOOK 04D-bis: STRONG SCALING - NUMBA PARALLELISM\n")
        f.write("=" * 60 + "\n\n")
        f.write(df[["n_threads", "mean_time", "speedup", "efficiency"]].to_string(index=False))
        f.write(f"\n\nMax speedup: {df['speedup'].max():.2f}x\n")
    print(f"✅ Summary saved: {summary_path}")

    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print(
        f"Best speedup: {df['speedup'].max():.2f}x with {df.loc[df['speedup'].idxmax(), 'n_threads']} threads"
    )
