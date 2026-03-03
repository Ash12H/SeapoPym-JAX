"""LMTL Model Example - 2D Grid Simulation (no transport).

This script demonstrates the LMTL (Low/Mid Trophic Level) ecosystem model
using a pre-defined blueprint from the model catalogue with JAX backend.

Processes (from LMTL_NO_TRANSPORT blueprint):
1. Day length (CBM photoperiod model)
2. DVM-weighted mean temperature (layer_weighted_mean)
3. Threshold temperature (max(T, T_ref))
4. Gillooly temperature normalization
5. Recruitment age (temperature-dependent)
6. NPP Injection -> Cohort 0
7. Aging Flux -> Transfer between cohorts (C -> C+1)
8. Recruitment Flux -> Transfer from eligible cohorts to Biomass
9. Natural Mortality -> Loss from Biomass
"""

import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Import LMTL functions (registers them with @functional decorator)
import seapopym.functions.lmtl  # noqa: F401
from seapopym.blueprint import Config
from seapopym.compiler import compile_model
from seapopym.engine import Runner
from seapopym.models import LMTL_NO_TRANSPORT

# =============================================================================
# 1. PARAMETERS
# =============================================================================

# LMTL Biological Parameters
LMTL_E = 0.1668
LMTL_LAMBDA_0 = 1 / 150  # 1/day
LMTL_GAMMA_LAMBDA = 0.15  # 1/degC
LMTL_TAU_R_0 = 10.38  # days
LMTL_GAMMA_TAU_R = 0.11  # 1/degC
LMTL_T_REF = 0.0  # degC
LATITUDE = 30.0  # degrees N

PLOT_FILE = "examples/images/01_lmtl_no_transport.png"

# =============================================================================
# 2. BLUEPRINT
# =============================================================================

blueprint = LMTL_NO_TRANSPORT

max_age_days = int(np.ceil(LMTL_TAU_R_0))
cohort_ages_sec = np.arange(0, max_age_days + 1) * 86400.0
n_cohorts = len(cohort_ages_sec)

# =============================================================================
# 3. CONFIGURATION (2D Grid Simulation)
# =============================================================================

# Simulation Time
start_date = "2000-01-01"
end_date = "2002-01-01"  # 2 years
dt = "3h"

# Generate dates covering [start, end] inclusive
start_pd = pd.to_datetime(start_date)
end_pd = pd.to_datetime(end_date)
n_days = (end_pd - start_pd).days + 5  # Add margin to cover end_date safely

dates = pd.date_range(start=start_pd, periods=n_days, freq="D")

# Grid (2D)
grid_size = (180, 360)
# grid_size = (1, 1)
ny, nx = grid_size
lat = np.arange(ny)
lon = np.arange(nx)

# Forcing Data Generation
day_of_year = dates.dayofyear.values

# Day of year forcing (T, Y, X)
doy_float = day_of_year.astype(float)
doy_3d = np.broadcast_to(doy_float[:, None, None], (len(dates), ny, nx))
doy_da = xr.DataArray(doy_3d, dims=["T", "Y", "X"], coords={"T": dates, "Y": lat, "X": lon})

# Temperature: 15C mean, +/- 5C seasonal amplitude (T, Z=1, Y, X)
temp_c = 15.0 + 5.0 * np.sin(2 * np.pi * day_of_year / 365.0)
temp_4d = np.broadcast_to(temp_c[:, None, None, None], (len(dates), 1, ny, nx))
temp_da = xr.DataArray(
    temp_4d,
    dims=["T", "Z", "Y", "X"],
    coords={"T": dates, "Z": np.arange(1), "Y": lat, "X": lon},
)

# NPP: 1.0 mean, +/- 0.5 seasonal amplitude (g/m^2/day)
npp_day = 1.0 + 0.5 * np.sin(2 * np.pi * day_of_year / 365.0)
npp_sec = npp_day / 86400.0
npp_3d = np.broadcast_to(npp_sec[:, None, None], (len(dates), ny, nx))
npp_da = xr.DataArray(npp_3d, dims=["T", "Y", "X"], coords={"T": dates, "Y": lat, "X": lon})


config = Config.from_dict(
    {
        "parameters": {
            "lambda_0": {"value": LMTL_LAMBDA_0 / 86400.0},
            "gamma_lambda": {"value": LMTL_GAMMA_LAMBDA},
            "tau_r_0": {"value": LMTL_TAU_R_0 * 86400.0},
            "gamma_tau_r": {"value": LMTL_GAMMA_TAU_R},
            "t_ref": {"value": LMTL_T_REF},
            "efficiency": {"value": LMTL_E},
            "cohort_ages": {"value": cohort_ages_sec.tolist()},
            "day_layer": {"value": [0]},
            "night_layer": {"value": [0]},
            "latitude": {"value": LATITUDE},
        },
        "forcings": {
            "temperature": temp_da,
            "primary_production": npp_da,
            "day_of_year": doy_da,
        },
        "initial_state": {
            "biomass": xr.DataArray(np.zeros((1, ny, nx)), dims=["F", "Y", "X"], coords={"Y": lat, "X": lon}),
            "production": xr.DataArray(
                np.zeros((1, ny, nx, n_cohorts)), dims=["F", "Y", "X", "C"], coords={"Y": lat, "X": lon}
            ),
        },
        "execution": {
            "time_start": start_date,
            "time_end": end_date,
            "dt": dt,
            "forcing_interpolation": "linear",
        },
    }
)

# =============================================================================
# 4. EXECUTION
# =============================================================================

if __name__ == "__main__":
    jax.config.update("jax_default_device", jax.devices("cpu")[0])
    print(f"Compiling model ({blueprint.id})...")
    model = compile_model(blueprint, config)

    runner = Runner.simulation(chunk_size=800)
    print(f"Running simulation on {grid_size} grid for {len(dates)} days...")
    t_start = time.time()
    state, outputs = runner.run(model, export_variables=["biomass"])
    t_end = time.time()
    print(f"Simulation completed in {t_end - t_start:.2f} seconds.")

    # =============================================================================
    # 5. VISUALIZATION
    # =============================================================================

    # Calculate mean biomass over the grid (using xarray dimensions)
    biomass_mean = outputs["biomass"].mean(dim=("Y", "X"))  # Mean over Y, X

    print(f"Number of timesteps: {len(biomass_mean)}")

    # Use time coordinates from the Dataset
    plot_dates = biomass_mean.coords["T"].values

    # Interpolate temperature data to match simulation timesteps
    # (temp_c is daily, simulation is at dt="3h")
    temp_da_plot = temp_da.isel(Z=0).interp(T=plot_dates, method="linear")
    temp_c_plot = temp_da_plot.mean(dim=("Y", "X")).values

    # Plot results
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = "tab:green"
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Mean Biomass (g/m²)", color=color)
    ax1.plot(plot_dates, biomass_mean, color=color, linewidth=2, label="Biomass")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Add Temperature on twin axis
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Temperature (°C)", color=color)
    ax2.plot(plot_dates, temp_c_plot, color=color, linestyle="--", alpha=0.5, label="Temp")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("LMTL Model - 2D Simulation Results (no transport)")
    fig.tight_layout()
    Path(PLOT_FILE).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{PLOT_FILE}")
    print(f"Plot saved to {PLOT_FILE}")
