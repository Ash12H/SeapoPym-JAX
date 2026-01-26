"""0D LMTL Model Implementation (Full Biological Complexity).

This script replicates the logic of 'article/notebooks/article_02a_comparison_seapopym_v0.3.py'
using the new Seapopym v1 Engine (JAX/NumPy).

It implements the full LMTL (Low/Mid Trophic Level) dynamics:
- Temperature-dependent development (Gillooly)
- Cohort-based production (Aging)
- Recruitment to biomass
- Mortality
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
import xarray as xr

from seapopym.blueprint import Blueprint, Config, functional
from seapopym.compiler import compile_model
from seapopym.engine import StreamingRunner

# Initialize Pint registry for unit definition
ureg = pint.get_application_registry()

# =============================================================================
# 1. PARAMETERS (From Notebook 02A)
# =============================================================================

# LMTL Biological Parameters
LMTL_E = 0.1668
LMTL_LAMBDA_0 = 1 / 150  # 1/day
LMTL_GAMMA_LAMBDA = 0.15  # 1/degC
LMTL_TAU_R_0 = 10.38  # days
LMTL_GAMMA_TAU_R = 0.11  # 1/degC
LMTL_T_REF = 0.0  # degC

# =============================================================================
# 2. FUNCTION DEFINITIONS (JAX/NumPy Compatible)
# =============================================================================


@functional(name="lmtl:gillooly_temperature", backend="jax", units={"temp": "degC", "return": "degC"})
def gillooly_temperature(temp):
    """Normalize temperature using Gillooly et al. (2001)."""
    # T_norm = T / (1 + T/273)
    return temp / (1.0 + temp / 273.0)


@functional(
    name="lmtl:recruitment_age",
    backend="jax",
    units={"temp": "degC", "tau_r_0": "s", "gamma": "1/delta_degC", "t_ref": "degC", "return": "s"},
)
def recruitment_age(temp, tau_r_0, gamma, t_ref):
    """Compute recruitment age (time to recruitment)."""

    # tau = tau_0 * exp(-gamma * (T - T_ref))
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
    """Compute mortality loss."""

    # rate = lambda_0 * exp(gamma * (T - T_ref))
    rate = lambda_0 * jnp.exp(gamma * (temp - t_ref))
    # Tendency = -rate * B
    return -rate * biomass


@functional(
    name="lmtl:production_dynamics",
    backend="jax",
    core_dims={"production": ["C"], "cohort_ages": ["C"]},
    outputs=["prod_tendency", "biomass_source"],
    units={
        "production": "g/m^2",
        "cohort_ages": "s",
        "rec_age": "s",
        "npp": "g/m^2/s",
        "efficiency": "dimensionless",
        "prod_tendency": "g/m^2/s",
        "biomass_source": "g/m^2/s",
    },
)
def production_dynamics(production, cohort_ages, rec_age, npp, efficiency):
    """Combined dynamics for Production cohorts.

    1. Input from NPP (into cohort 0)
    2. Aging (flux C -> C+1)
    3. Recruitment (flux C -> Biomass if age > rec_age)
    """
    # --- 1. Pre-compute Cohort Durations (d_tau) ---
    # diffs = ages[1:] - ages[:-1]
    # In Seapopym legacy, last cohort duration repeats previous
    d_tau_raw = cohort_ages[1:] - cohort_ages[:-1]
    last_d_tau = d_tau_raw[-1:]  # Keep dims
    d_tau = jnp.concatenate([d_tau_raw, last_d_tau])

    aging_rate = 1.0 / d_tau

    # --- 2. Outflow from each cohort (Aging) ---
    outflow = production * aging_rate

    # --- 3. Influx logic ---
    # Influx to [0]: From NPP
    # Influx to [i > 0]: From Outflow[i-1] * (1 - is_recruited[i-1])

    source_flux = npp * efficiency

    # Shift outflow to get influx from previous
    # [O0, O1, O2] -> [Source, O0, O1]
    influx_from_prev = jnp.concatenate(
        [
            jnp.array([source_flux]),  # Index 0 gets NPP source
            outflow[:-1],  # Indices 1..N get prev outflow
        ]
    )

    # --- 4. Recruitment Logic ---
    is_recruited = cohort_ages >= rec_age

    # If C-1 was recruited, it emptied into biomass, so it does NOT flow to C.
    # "is_recruited" is boolean vector.
    # Standard shift for 'prev_is_recruited': [False, rec[0], rec[1]...]
    # (Cohort 0 never has a "previous recruited" blocking it, since input is external NPP)

    # prev_is_recruited logic:
    # Vector [False] + is_recruited[:-1]
    prev_recruited = jnp.concatenate([jnp.array([False]), is_recruited[:-1]])

    # Filter influx: If prev was recruited, influx is 0 (it went to biomass instead)
    effective_influx = jnp.where(prev_recruited, 0.0, influx_from_prev)

    # Tendency = Influx - Outflow
    # Note: Outflow always happens (either to next cohort or to biomass/senescence)
    prod_tendency = effective_influx - outflow

    # --- 5. Biomass Source (Recruitment) ---
    # Sum of outflows from recruited cohorts
    # If a cohort is recruited, its outflow goes to Biomass.
    recruitment_flux = jnp.where(is_recruited, outflow, 0.0)
    biomass_source_val = jnp.sum(recruitment_flux)

    return prod_tendency, biomass_source_val


# =============================================================================
# 3. BLUEPRINT CONFIGURATION
# =============================================================================

# Define Cohorts
# Logic from legacy: np.arange(0, ceil(tau_r_0) + 1) * day
# MAX_AGE: LMTL_TAU_R_0 + 1 days.

max_age_days = int(np.ceil(LMTL_TAU_R_0) + 1)
cohort_ages_days = np.arange(0, max_age_days + 1)
cohort_ages_sec = cohort_ages_days * 86400.0
n_cohorts = len(cohort_ages_sec)

blueprint = Blueprint.from_dict(
    {
        "id": "lmtl-0d-full",
        "version": "1.0",
        "declarations": {
            "state": {"biomass": {"units": "g/m^2"}, "production": {"units": "g/m^2", "dims": ["C"]}},
            "parameters": {
                # LMTL params
                "lambda_0": {"units": "1/s"},
                "gamma_lambda": {"units": "1/delta_degC"},
                "tau_r_0": {"units": "s"},
                "gamma_tau_r": {"units": "1/delta_degC"},
                "t_ref": {"units": "degC"},
                "efficiency": {"units": "dimensionless"},
                # Grids
                "cohort_ages": {"units": "s", "dims": ["C"]},
            },
            "forcings": {
                "temperature": {"units": "degC", "dims": ["T"]},
                "primary_production": {"units": "g/m^2/s", "dims": ["T"]},
            },
        },
        "process": [
            # 1. Compute Temperature transformations
            # (Could be done inline inputs but explicit is nicer if re-used)
            {
                "func": "lmtl:gillooly_temperature",
                "inputs": {"temp": "forcings.temperature"},
                "outputs": {"return": {"target": "derived.temp_norm", "type": "derived"}},
            },
            # 2. Compute Dynamic Recruitment Age
            {
                "func": "lmtl:recruitment_age",
                "inputs": {
                    "temp": "derived.temp_norm",  # Use normalized T!
                    "tau_r_0": "parameters.tau_r_0",
                    "gamma": "parameters.gamma_tau_r",
                    "t_ref": "parameters.t_ref",
                },
                "outputs": {"return": {"target": "derived.rec_age", "type": "derived"}},
            },
            # 3. Production Dynamics (Aging + Recruitment)
            {
                "func": "lmtl:production_dynamics",
                "inputs": {
                    "production": "state.production",
                    "cohort_ages": "parameters.cohort_ages",
                    "rec_age": "derived.rec_age",
                    "npp": "forcings.primary_production",
                    "efficiency": "parameters.efficiency",
                },
                # Func returns (prod_tendency, biomass_source)
                "outputs": {
                    "prod_tendency": {"target": "tendencies.production", "type": "tendency"},
                    "biomass_source": {"target": "tendencies.biomass_recruitment", "type": "tendency"},
                },
            },
            # 4. Biomass Mortality
            {
                "func": "lmtl:mortality",
                "inputs": {
                    "biomass": "state.biomass",
                    "temp": "derived.temp_norm",
                    "lambda_0": "parameters.lambda_0",
                    "gamma": "parameters.gamma_lambda",
                    "t_ref": "parameters.t_ref",
                },
                "outputs": {"return": {"target": "tendencies.biomass_mortality", "type": "tendency"}},
            },
        ],
    }
)

# =============================================================================
# 4. CONFIGURATION (Data)
# =============================================================================

# Create dummy forcings (Sinusoidal seasonal cycle)
# Use real dates: 20 years of daily data
start_date = "2000-01-01"
end_date = "2020-01-01"  # 20 years exactly
dates = pd.date_range(start=start_date, periods=365 * 20, freq="D")
# Simulating 'day of year' for sine wave
day_of_year = dates.dayofyear.values

# Temperature: 20C mean, +/- 5C amplitude
temp_c = 20.0 + 5.0 * np.sin(2 * np.pi * day_of_year / 365.0)
temp_da = xr.DataArray(temp_c, dims=["T"], coords={"T": dates})

# NPP: 1.0 mean, +/- 0.5 amplitude (in g/m^2/day -> convert to /s)
npp_day = 1.0 + 0.5 * np.sin(2 * np.pi * day_of_year / 365.0)
npp_sec = npp_day / 86400.0
npp_da = xr.DataArray(npp_sec, dims=["T"], coords={"T": dates})

config = Config.from_dict(
    {
        "parameters": {
            "lambda_0": {"value": LMTL_LAMBDA_0 / 86400.0},  # Convert day^-1 to s^-1
            "gamma_lambda": {"value": LMTL_GAMMA_LAMBDA},
            "tau_r_0": {"value": LMTL_TAU_R_0 * 86400.0},  # Convert days to s
            "gamma_tau_r": {"value": LMTL_GAMMA_TAU_R},
            "t_ref": {"value": LMTL_T_REF},
            "efficiency": {"value": LMTL_E},
            "cohort_ages": xr.DataArray(cohort_ages_sec, dims=["C"]),
        },
        "forcings": {"temperature": temp_da, "primary_production": npp_da},
        "initial_state": {"biomass": xr.DataArray(0.0), "production": xr.DataArray(np.zeros(n_cohorts), dims=["C"])},
        "execution": {
            "time_start": start_date,
            "time_end": end_date,
            "dt": "0.05d",  # 20 timesteps per day
            "forcing_interpolation": "linear",  # Interpolate daily forcings to 0.05d resolution
        },
    }
)

# =============================================================================
# 5. EXECUTION
# =============================================================================

print("Compiling model (Full LMTL 0D)...")
model = compile_model(blueprint, config, backend="jax")
print(f"Model compiled. Backend: {model.backend}")

runner = StreamingRunner(model)
print("Running simulation...")
start_state, outputs = runner.run(output_path=None)

# =============================================================================
# 6. VISUALIZATION
# =============================================================================

biomass_ts = outputs["biomass"]

# Check if we have production output (if requested in blueprint... wait, outputs contains state generally)
# The StreamingRunner returns state variables.
# production_ts = outputs["production"] # Should be (T, C)

# Determine dt in seconds for plotting axis
dt_str = config.execution.dt
# Simple parsing logic mirroring compiler
if dt_str.endswith("d"):
    dt_seconds = float(dt_str[:-1]) * 86400
elif dt_str.endswith("h"):
    dt_seconds = float(dt_str[:-1]) * 3600
elif dt_str.endswith("m"):
    dt_seconds = float(dt_str[:-1]) * 60
elif dt_str.endswith("s"):
    dt_seconds = float(dt_str[:-1])
else:
    dt_seconds = float(dt_str)

# Robust time axis construction
start_ts = pd.Timestamp(start_date)
time_deltas = pd.to_timedelta(np.arange(len(biomass_ts)) * dt_seconds, unit="s")
time_axis = start_ts + time_deltas

fig, ax1 = plt.subplots(figsize=(10, 6))

color = "tab:red"
ax1.set_xlabel("Date")
ax1.set_ylabel("Biomass (g/m^2)", color=color)
ax1.plot(time_axis, biomass_ts, color=color, linewidth=2)
ax1.tick_params(axis="y", labelcolor=color)
ax1.grid(True, alpha=0.3)
# Rotate date labels
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = "tab:blue"
ax2.set_ylabel("Temperature (°C)", color=color)  # we already handled the x-label with ax1
# Plot temp on original daily axis (using dates from forcings)
ax2.plot(dates, temp_c, color=color, linestyle="--", alpha=0.6)
ax2.tick_params(axis="y", labelcolor=color)

plt.title("LMTL 0D Simulation: Biomass (dt=0.05d) & Daily Temperature")
plt.tight_layout()
plt.savefig("lmtl_0d_results_interpolated_dates.png")
print("Plot saved to lmtl_0d_results_interpolated_dates.png")
