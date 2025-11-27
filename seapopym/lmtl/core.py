"""Core functions for LMTL model."""

import numpy as np
import xarray as xr

from seapopym.standard.coordinates import Coordinates

# -------------------------------------------------------------------------
# Environment Functions
# -------------------------------------------------------------------------


def compute_day_length(latitude: xr.DataArray, time: xr.DataArray) -> dict[str, xr.DataArray]:
    """Compute day length fraction (0-1) based on latitude and day of year.

    Uses the CBM model (Forsythe et al., 1995).

    Parameters
    ----------
    latitude : xr.DataArray
        Latitude in degrees.
    time : xr.DataArray
        Time coordinate (datetime64).

    Returns
    -------
    dict
        {"output": day_length} where day_length is dimensionless [0.0-1.0].
    """
    # Ensure time is datetime64
    # We extract day of year
    # Note: time might be a coordinate or a DataArray.
    # If it's a DataArray without coords, .dt accessor works on values.
    doy = time.dt.dayofyear

    # Latitude in radians
    phi = np.deg2rad(latitude)

    # Declination of the sun
    # Declination of the sun
    # theta = 0.2163108 + 2 * np.arctan(0.9671396 * np.tan(0.00860 * (doy - 186)))

    # Hour angle
    # P = asin[ 0.39795 * cos(0.2163108 + 2 * atan(0.9671396 * tan(0.00860 * (J - 186)))) ]
    # We use a simplified approximation often used in ecological models
    # Brock model (1981)
    declination = 23.45 * np.sin(np.deg2rad(360 * (284 + doy) / 365))
    declination_rad = np.deg2rad(declination)

    # Sunset hour angle
    # cos(omega) = -tan(phi) * tan(delta)
    arg = -np.tan(phi) * np.tan(declination_rad)
    # Clamp to [-1, 1] for polar days/nights
    arg = np.clip(arg, -1.0, 1.0)
    hour_angle = np.arccos(arg)

    # Day length in hours = 24 * hour_angle / pi
    day_length_hours = 24.0 * hour_angle / np.pi

    return {"output": day_length_hours / 24.0}


def compute_mean_temperature(
    temperature: xr.DataArray,
    day_length: xr.DataArray,
    day_layer: float,
    night_layer: float,
) -> dict[str, xr.DataArray]:
    """Compute mean temperature experienced by vertically migrating organisms.

    T_mean = T_day * day_length + T_night * (1 - day_length)

    Parameters
    ----------
    temperature : xr.DataArray
        Temperature field [degC] with depth dimension.
    day_length : xr.DataArray
        Day length fraction [dimensionless, 0-1].
    day_layer : float
        Depth layer for daytime [m].
    night_layer : float
        Depth layer for nighttime [m].

    Returns
    -------
    dict
        {"output": mean_temperature} in [degC].
    """
    # Select temperature at specific layers
    # We assume 'depth' coordinate exists or we use method='nearest'
    t_day = temperature.sel({Coordinates.Z: day_layer}, method="nearest")
    t_night = temperature.sel({Coordinates.Z: night_layer}, method="nearest")

    return {"output": t_day * day_length + t_night * (1.0 - day_length)}


def compute_recruitment_age(
    mean_temperature: xr.DataArray,
    tau_r_0: float,
    gamma_tau_r: float,
    T_ref: float,
) -> dict[str, xr.DataArray]:
    """Compute minimum recruitment age based on temperature.

    tau_r = tau_r_0 * exp(-gamma * (T - T_ref))

    Parameters
    ----------
    mean_temperature : xr.DataArray
        Mean temperature [degC].
    tau_r_0 : float
        Base recruitment age [s] at reference temperature.
    gamma_tau_r : float
        Thermal sensitivity coefficient [1/degC].
    T_ref : float
        Reference temperature [degC].

    Returns
    -------
    dict
        {"output": recruitment_age} in [s].
    """
    return {"output": tau_r_0 * np.exp(-gamma_tau_r * (mean_temperature - T_ref))}


# -------------------------------------------------------------------------
# Dynamics / Tendency Functions
# -------------------------------------------------------------------------


def compute_production_initialization(
    primary_production: xr.DataArray,
    cohorts: xr.DataArray,
    E: float,
) -> dict[str, xr.DataArray]:
    """Compute production tendency for the first cohort (age 0).

    IMPORTANT: This function expects primary_production in SI units [g/m²/s].
    If your NPP data is in [g/m²/day], it must be converted before calling
    this function (the Controller handles this via unit conversion).

    Parameters
    ----------
    primary_production : xr.DataArray
        Net Primary Production [g/m²/s].
    cohorts : xr.DataArray
        Cohort ages coordinate [s].
    E : float
        Transfer efficiency from primary production [dimensionless].

    Returns
    -------
    dict
        {"output": production_tendency} in [g/m²/s] for cohort dimension.
        Only cohort 0 receives flux (E * NPP), others are zero.
    """
    # This assumes we can construct the full array or that xarray handles broadcasting.
    # But we don't know the cohort dimension size here easily without the state.
    # However, if 'primary_production' aligns with T, Y, X, we can add a cohort coord.

    # To be safe and efficient, we should probably handle this by returning a
    # specific variable "production_source" that is then mapped to "production"
    # but the TimeIntegrator logic sums everything.

    # Let's assume the caller (TimeIntegrator) handles broadcasting if we return
    # a DataArray with a single coordinate value for cohort=0.

    # Source is a rate (e.g. gC/m2/s)
    # The TimeIntegrator will multiply by dt (in seconds).
    tendency_rate = E * primary_production

    # We expand dims to include cohort=0
    tendency = tendency_rate.expand_dims(cohort=[0])

    # Reindex to match the full cohorts shape (filling other cohorts with 0)
    # We assume 'cohorts' is the coordinate DataArray for the cohort dimension.
    tendency = tendency.reindex(cohort=cohorts, fill_value=0.0)

    return {"output": tendency}


def compute_production_dynamics(
    production: xr.DataArray,
    recruitment_age: xr.DataArray,
    cohort_ages: xr.DataArray,
    dt: float,
) -> dict[str, xr.DataArray]:
    """Compute combined aging and recruitment dynamics.

    Handles:
    1. Aging flux (transport from C-1 to C).
    2. Outflow from C:
       - If recruited: Transfer to biomass (Recruitment).
       - If NOT recruited: Exit the model (Senescence/Outflow).

    Prevents double-counting of outflows for the last cohort.
    """
    # 1. Aging Influx (from previous cohort)
    # Flux entrant dans C = P[C-1] / dt
    # shift(cohort=1) décale vers la droite : la valeur à l'index C vient de C-1
    influx = production.shift(cohort=1, fill_value=0.0) / dt

    # 2. Identify Recruited Cohorts
    # cohort_ages doit être aligné avec production
    is_recruited = cohort_ages >= recruitment_age

    # 3. Calculate Outflows
    # Le contenu de la case P[C] sort au taux 1/dt
    total_outflow_rate = production / dt

    # Séparation des flux sortants (Exclusion Mutuelle)
    # Si recruté -> va dans recruitment_source
    # Si pas recruté -> va dans aging_out (perdu pour le système)
    recruitment_flux = total_outflow_rate.where(is_recruited, 0.0)
    # aging_out_flux = total_outflow_rate.where(~is_recruited, 0.0) # Implicite

    # 4. Tendency for Production (P)
    # dP/dt = Influx - Total_Outflow
    # On retire tout ce qui sort, peu importe où ça va (biomasse ou dehors)
    production_tendency = influx - total_outflow_rate

    # 5. Source for Biomass
    # Somme de tout ce qui est recruté sur toutes les cohortes
    biomass_source = recruitment_flux.sum(dim="cohort")

    return {
        "production_tendency": production_tendency,
        "recruitment_source": biomass_source,
    }


def compute_mortality_tendency(
    biomass: xr.DataArray,
    mean_temperature: xr.DataArray,
    lambda_0: float,
    gamma_lambda: float,
    T_ref: float,
) -> dict[str, xr.DataArray]:
    """Compute mortality tendency for biomass.

    rate = lambda_0 * exp(gamma * (T - T_ref))
    Tendency = -rate * biomass

    Parameters
    ----------
    biomass : xr.DataArray
        Biomass [g/m²].
    mean_temperature : xr.DataArray
        Mean temperature [degC].
    lambda_0 : float
        Base mortality rate [s^-1] at reference temperature.
    gamma_lambda : float
        Thermal sensitivity coefficient [1/degC].
    T_ref : float
        Reference temperature [degC].

    Returns
    -------
    dict
        {"mortality_loss": tendency} in [g/m²/s].
        Negative tendency representing biomass loss.
    """
    rate = lambda_0 * np.exp(gamma_lambda * (mean_temperature - T_ref))
    tendency = -rate * biomass

    return {"mortality_loss": tendency}
