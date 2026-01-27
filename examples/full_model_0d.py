"""0D LMTL Model Implementation (Decomposed Dynamics).

This script implements the full LMTL (Low/Mid Trophic Level) dynamics using a
decomposed approach where each physical process is a separate function.

Processes:
1. NPP Injection -> Cohort 0
2. Aging Flux -> Transfer between cohorts (C -> C+1)
3. Recruitment Flux -> Transfer from eligible cohorts to Biomass
4. Natural Mortality -> Loss from Biomass

This version supports 2D grid simulation.
"""

import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
import xarray as xr

from seapopym.blueprint import Blueprint, Config, functional
from seapopym.compiler import compile_model
from seapopym.engine import StreamingRunner

# Initialize Pint registry
ureg = pint.get_application_registry()

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


# =============================================================================
# 2. DECOMPOSED FUNCTIONS (JAX)
# =============================================================================


@functional(name="lmtl:gillooly_temperature", backend="jax", units={"temp": "degC", "return": "degC"})
def gillooly_temperature(temp):
    """Normalize temperature using Gillooly et al. (2001)."""
    return temp / (1.0 + temp / 273.0)


@functional(
    name="lmtl:recruitment_age",
    backend="jax",
    units={"temp": "degC", "tau_r_0": "s", "gamma": "1/delta_degC", "t_ref": "degC", "return": "s"},
)
def recruitment_age(temp, tau_r_0, gamma, t_ref):
    """Compute recruitment age (time to recruitment)."""
    return tau_r_0 * jnp.exp(-gamma * (temp - t_ref))


@functional(
    name="lmtl:mortality",
    backend="jax",
    units={
        "biomass": "g/m^2",
        "temp": "degC",
        "lambda_0": "1/s",
        "gamma": "1/delta_degC",
        "t_ref": "degC",
        "return": "g/m^2/s",
    },
)
def mortality_tendency(biomass, temp, lambda_0, gamma, t_ref):
    """Compute mortality loss for biomass."""
    rate = lambda_0 * jnp.exp(gamma * (temp - t_ref))
    return -rate * biomass


# --- Decomposed Production Dynamics ---


@functional(
    name="lmtl:npp_injection",
    backend="jax",
    core_dims={"production": ["C"]},
    units={
        "npp": "g/m^2/s",
        "efficiency": "dimensionless",
        "production": "g/m^2",
        "return": "g/m^2/s",
    },
)
def npp_injection(npp, efficiency, production):
    """Inject Primary Production into the first cohort (0)."""
    # npp is (Y, X) or (T, Y, X)
    # production is (C, Y, X)

    source_flux = npp * efficiency

    # Create tendencies tensor matching production shape
    tendency = jnp.zeros_like(production)

    # Add source flux to the first cohort
    # Use .at[0, ...].set() for JAX array update
    tendency = tendency.at[0, ...].set(source_flux)

    return tendency


@functional(
    name="lmtl:aging_flow",
    backend="jax",
    core_dims={"production": ["C"], "cohort_ages": ["C"]},
    units={"production": "g/m^2", "cohort_ages": "s", "rec_age": "s", "return": "g/m^2/s"},
)
def aging_flow(production, cohort_ages, rec_age):
    """Compute aging flux (transfer from C to C+1).

    Logic:
    - Calculates flow based on cohort duration.
    - If a cohort is recruited (Age > RecAge), it does NOT flow to C+1 (it goes to Biomass).
    - The last cohort does NOT flow out (accumulation/plus group).
    """
    # Handle Broadcasting: (C, Y, X)
    spatial_ndim = production.ndim - 1
    c_broadcast = (production.shape[0],) + (1,) * spatial_ndim

    # Cohort durations
    d_tau_raw = cohort_ages[1:] - cohort_ages[:-1]
    last_d_tau = d_tau_raw[-1:]
    d_tau = jnp.concatenate([d_tau_raw, last_d_tau])

    # Aging rate
    aging_coef = (1.0 / d_tau).reshape(c_broadcast)

    # Base Outflow
    base_outflow = production * aging_coef

    # 1. Recruitment Filter
    # If recruited, flow is diverted to biomass, so it's 0 for aging.
    cohort_ages_grid = cohort_ages.reshape(c_broadcast)
    is_recruited = cohort_ages_grid >= rec_age
    aging_outflow = jnp.where(is_recruited, 0.0, base_outflow)

    # 2. Last Cohort Filter (Accumulation)
    # Prevent outflow from the last cohort
    aging_outflow = aging_outflow.at[-1, ...].set(0.0)

    # 3. Balance (Loss + Gain from prev)
    loss = -aging_outflow

    # Gain for cohort i comes from outflow of i-1
    # Cohort 0 has no aging input
    gain = jnp.concatenate(
        [
            jnp.zeros((1,) + production.shape[1:]),
            aging_outflow[:-1],
        ],
        axis=0,
    )

    return loss + gain


@functional(
    name="lmtl:recruitment_flow",
    backend="jax",
    core_dims={"production": ["C"], "cohort_ages": ["C"]},
    outputs=["prod_loss", "biomass_gain"],
    units={
        "production": "g/m^2",
        "cohort_ages": "s",
        "rec_age": "s",
        "prod_loss": "g/m^2/s",
        "biomass_gain": "g/m^2/s",
    },
)
def recruitment_flow(production, cohort_ages, rec_age):
    """Compute recruitment flux (transfer from C to Biomass)."""
    spatial_ndim = production.ndim - 1
    c_broadcast = (production.shape[0],) + (1,) * spatial_ndim

    d_tau_raw = cohort_ages[1:] - cohort_ages[:-1]
    last_d_tau = d_tau_raw[-1:]
    d_tau = jnp.concatenate([d_tau_raw, last_d_tau])

    aging_coef = (1.0 / d_tau).reshape(c_broadcast)
    base_outflow = production * aging_coef

    # Filter: Keep ONLY recruited fluxes
    cohort_ages_grid = cohort_ages.reshape(c_broadcast)
    is_recruited = cohort_ages_grid >= rec_age

    flux_to_biomass = jnp.where(is_recruited, base_outflow, 0.0)

    # 1. Loss from Production
    prod_loss = -flux_to_biomass

    # 2. Gain to Biomass (Sum over all contributing cohorts)
    biomass_gain = jnp.sum(flux_to_biomass, axis=0)

    return prod_loss, biomass_gain


# =============================================================================
# 3. BLUEPRINT CONFIGURATION
# =============================================================================

max_age_days = int(np.ceil(LMTL_TAU_R_0))
cohort_ages_days = np.arange(0, max_age_days + 1)
cohort_ages_sec = cohort_ages_days * 86400.0
n_cohorts = len(cohort_ages_sec)

blueprint = Blueprint.from_dict(
    {
        "id": "lmtl-0d-decomposed",
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
            # 1. Derived Variables
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
            # 2. Dynamics (Decomposed)
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

# =============================================================================
# 4. CONFIGURATION (2D Grid Simulation)
# =============================================================================

# Simulation Time
start_date = "2000-01-01"
end_date = "2020-01-01"  # 2 years
dt = "3h"

# Generate dates covering [start, end] inclusive
start_pd = pd.to_datetime(start_date)
end_pd = pd.to_datetime(end_date)
n_days = (end_pd - start_pd).days + 5  # Add margin to cover end_date safely

dates = pd.date_range(start=start_pd, periods=n_days, freq="D")

# Grid (2D)
grid_size = (180, 360)
ny, nx = grid_size
lat = np.arange(ny)
lon = np.arange(nx)

# Forcing Data Generation
# Temperature: 20C mean, +/- 5C seasonal amplitude
day_of_year = dates.dayofyear.values
temp_c = 15.0 + 5.0 * np.sin(2 * np.pi * day_of_year / 365.0)
# Broadcast to (T, Y, X)
temp_3d = np.broadcast_to(temp_c[:, None, None], (len(dates), ny, nx))
temp_da = xr.DataArray(temp_3d, dims=["T", "Y", "X"], coords={"T": dates, "Y": lat, "X": lon})

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
            "cohort_ages": xr.DataArray(cohort_ages_sec, dims=["C"]),
        },
        "forcings": {"temperature": temp_da, "primary_production": npp_da},
        "initial_state": {
            "biomass": xr.DataArray(np.zeros((ny, nx)), dims=["Y", "X"], coords={"Y": lat, "X": lon}),
            "production": xr.DataArray(
                np.zeros((ny, nx, n_cohorts)), dims=["Y", "X", "C"], coords={"Y": lat, "X": lon}
            ),
        },
        "execution": {
            "time_start": start_date,
            "time_end": end_date,
            "dt": dt,
            "forcing_interpolation": "linear",
            "batch_size": 100,
        },
    }
)

# =============================================================================
# 5. EXECUTION
# =============================================================================

print("Compiling model (Decomposed LMTL 2D)...")
model = compile_model(blueprint, config, backend="jax")
print(f"Model compiled. Backend: {model.backend}")

runner = StreamingRunner(model)
print(f"Running simulation on {grid_size} grid for {len(dates)} days...")
t_start = time.time()
state, outputs = runner.run(export_variables=["biomass"])
t_end = time.time()
print(f"Simulation completed in {t_end - t_start:.2f} seconds.")

# =============================================================================
# 6. VISUALIZATION
# =============================================================================

# Calculate mean biomass over the grid (using xarray dimensions)
biomass_mean = outputs["biomass"].mean(dim=("Y", "X"))  # Mean over Y, X

print(f"Number of timestep : {len(biomass_mean)}")

# Use time coordinates from the Dataset
plot_dates = biomass_mean.coords["T"].values

# Interpolate temperature data to match simulation timesteps
# (temp_c is daily, simulation is at dt="3h")
temp_da_sim = temp_da.interp(T=plot_dates, method="linear")
temp_c_plot = temp_da_sim.mean(dim=("Y", "X")).values

# Plot results
fig, ax1 = plt.subplots(figsize=(10, 6))

color = "tab:green"
ax1.set_xlabel("Date")
ax1.set_ylabel("Mean Biomass (g/m^2)", color=color)
ax1.plot(plot_dates, biomass_mean, color=color, linewidth=2, label="Biomass")
ax1.tick_params(axis="y", labelcolor=color)
ax1.grid(True, alpha=0.3)

# Add Temperature on twin axis
ax2 = ax1.twinx()
color = "tab:red"
ax2.set_ylabel("Temperature (°C)", color=color)
ax2.plot(plot_dates, temp_c_plot, color=color, linestyle="--", alpha=0.5, label="Temp")
ax2.tick_params(axis="y", labelcolor=color)

plt.title("LMTL Decomposed Model - 2D Simulation Results")
fig.tight_layout()
plt.savefig("lmtl_decomposed_2d_results.png")
print("Plot saved to lmtl_decomposed_2d_results.png")
