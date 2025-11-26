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
    Returns a dimensionless fraction of the day (0.0 to 1.0).
    """
    # Ensure time is datetime64
    # We extract day of year
    # Note: time might be a coordinate or a DataArray.
    # If it's a DataArray without coords, .dt accessor works on values.
    doy = time.dt.dayofyear

    # Latitude in radians
    phi = np.deg2rad(latitude)

    # Declination of the sun
    theta = 0.2163108 + 2 * np.arctan(0.9671396 * np.tan(0.00860 * (doy - 186)))

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
    tau_r_0 : float
        Base recruitment age in seconds (SI units).
    ...

    Returns
    -------
    dict
        "output": Recruitment age in seconds.
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

    Source: E * NPP
    Tendency = Source / dt (since we add dt * tendency)

    Current TimeIntegrator expects full array tendency.
    So we must broadcast NPP to the production shape (with cohort dim)
    and mask everything except cohort 0.
    """
    # This assumes we can construct the full array or that xarray handles broadcasting.
    # But we don't know the cohort dimension size here easily without the state.
    # However, if 'primary_production' aligns with T, Y, X, we can add a cohort coord.

    # To be safe and efficient, we should probably handle this by returning a
    # specific variable "production_source" that is then mapped to "production"
    # but the TimeIntegrator logic sums everything.

    # Let's assume the caller (TimeIntegrator) handles broadcasting if we return
    # a DataArray with a single coordinate value for cohort=0.

    # Source is a rate (e.g. mgC/m2/s)
    # The TimeIntegrator will multiply by dt (in seconds).
    tendency_rate = E * primary_production

    # We expand dims to include cohort=0
    tendency = tendency_rate.expand_dims(cohort=[0])

    # Reindex to match the full cohorts shape (filling other cohorts with 0)
    # We assume 'cohorts' is the coordinate DataArray for the cohort dimension.
    tendency = tendency.reindex(cohort=cohorts, fill_value=0.0)

    return {"output": tendency}


def compute_aging_tendency(
    production: xr.DataArray,
    dt: float,
) -> dict[str, xr.DataArray]:
    """Compute aging tendency (advection in age).

    Tendency[c] = (production[c-1] - production[c]) / dt

    Parameters
    ----------
    dt : float
        Time step in seconds.
    """
    # Shift production to the right (c becomes c+1)
    # production.shift(cohort=1) means value at c comes from c-1.
    shifted = production.shift(cohort=1, fill_value=0.0)

    # The flux coming into c is p[c-1]
    # The flux leaving c is p[c]
    # Net change = In - Out
    tendency = (shifted - production) / dt

    # Note: For cohort 0, shifted is 0, so tendency is -p[0]/dt (outflux).
    # The influx for cohort 0 is handled by compute_production_initialization.

    return {"aging_flux": tendency}


def compute_recruitment_tendency(
    production: xr.DataArray,
    recruitment_age: xr.DataArray,
    cohort_ages: xr.DataArray,  # The values of the cohort coordinate (in seconds)
    dt: float,
) -> dict[str, xr.DataArray]:
    """Compute recruitment flux from production to biomass.

    Transfer all production where age >= recruitment_age.

    Parameters
    ----------
    recruitment_age : xr.DataArray
        Age at recruitment in seconds.
    cohort_ages : xr.DataArray
        Age of cohorts in seconds.
    dt : float
        Time step in seconds.
    """
    # Create a mask for recruited cohorts
    # cohort_ages must be broadcastable to production
    # recruitment_age varies in space/time (T, Y, X)

    # We ensure cohort_ages is properly aligned
    # If cohort_ages is just a coordinate, xarray handles comparison
    is_recruited = cohort_ages >= recruitment_age

    # Select recruited production
    recruited_production = production.where(is_recruited, 0.0)

    # Tendency for production (sink): remove all recruited
    # tendency = -amount / dt
    production_sink = -recruited_production / dt

    # Tendency for biomass (source): sum of recruited
    # We sum over the cohort dimension
    biomass_source = recruited_production.sum(dim="cohort") / dt

    return {"recruitment_sink": production_sink, "recruitment_source": biomass_source}


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
    lambda_0 : float
        Base mortality rate in s^-1.
    """
    rate = lambda_0 * np.exp(gamma_lambda * (mean_temperature - T_ref))
    tendency = -rate * biomass

    return {"mortality_loss": tendency}
