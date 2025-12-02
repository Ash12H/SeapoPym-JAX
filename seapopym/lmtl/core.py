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
    # Calculate Delta_tau (cohort duration)
    # We assume cohort_ages represents the age of the cohort.
    # If cohorts are [0, 1, 2], delta_tau is difference.
    # For the last cohort, we repeat the last delta.
    # Note: This assumes cohort_ages is sorted.

    # Calculate differences between consecutive cohorts
    # We use numpy gradient or diff. Since it's xarray, we can use diff but it reduces size.
    # Let's try to be robust.

    # Or we can assume constant step if appropriate, but variable support is requested.

    # Strategy: Use the mean diff if constant, or extend the last diff.
    # Since xarray padding is sometimes verbose, let's do it via numpy values if needed
    # or simply reindex.

    # Let's use xarray features:
    # shift(-1) to get next value, then subtract.
    # This gives Delta[i] = Age[i+1] - Age[i]
    # The last one will be NaN. We fill it with the previous Delta.

    # Ideally, we want the width of the bin.
    # If 'cohort_ages' are the lower bounds: Width[i] = Age[i+1] - Age[i]
    # If 'cohort_ages' are midpoints: Width[i] ~ (Age[i+1] - Age[i-1])/2

    # Given the user context (0, 1, 2 days), these look like lower bounds or indices.
    # Let's assume standard finite volume: Age[i+1] - Age[i].

    # Implementation using shift to keep size:
    # next_ages = cohort_ages.shift(cohort=-1)
    # d_tau = next_ages - cohort_ages
    # This leaves the last one as NaN. We fill it with the second to last value.

    d_tau = cohort_ages.diff("cohort")
    # If single cohort, d_tau is empty. Handle edge case?
    if d_tau.size == 0:
        # Fallback for single cohort: assume dt or some default?
        # But aging implies moving OUT of it.
        # If single cohort, maybe we assume width = dt? Or 1 unit?
        # Let's assume 1.0 * units if possible, or raise warning.
        # For now, let's assume at least 2 cohorts for aging to make sense.
        # If 1 cohort, it just flows out.
        # Let's use a safe fallback if size < 2.
        d_tau = xr.DataArray([dt], coords=cohort_ages.coords, dims=cohort_ages.dims)
    else:
        # Realign to original shape, filling the last one (which was lost in diff)
        # with the last calculated difference (nearest).
        d_tau = d_tau.reindex(cohort=cohort_ages.coords["cohort"], method="nearest")

    # 1. Aging Rate (Flux per unit mass) = 1 / Delta_tau
    # This is the speed at which mass leaves the cohort.
    aging_rate = 1.0 / d_tau

    # 2. Outflow Rate (Mass/Time) leaving cohort C
    # Flux_out = P[C] * aging_rate
    total_outflow_flux = production * aging_rate

    # 3. Influx (from previous cohort)
    # Flux_in[C] = Flux_out[C-1]
    influx_flux = total_outflow_flux.shift(cohort=1, fill_value=0.0)

    # 4. Identify Recruited Cohorts
    is_recruited = cohort_ages >= recruitment_age

    # 5. Separate Outflows
    # Si recruté -> va dans recruitment_source
    # Si pas recruté -> va dans aging_out (perdu pour le système ou cohorte suivante)
    # Note: The "aging_out" part that is NOT recruited is implicitly handled:
    # It leaves P[C] (via -total_outflow_flux) and enters P[C+1] (via +influx_flux).
    # Wait, if it is recruited, does it STILL go to next cohort?
    # Usually: Recruitment = removal from the population (transition to next stage).
    # So if recruited, it leaves the system (into biomass) and DOES NOT go to C+1.

    # Logic check:
    # If recruited: Outflow goes to Biomass.
    # If NOT recruited: Outflow goes to C+1.

    # Current code logic was:
    # production_tendency = influx - total_outflow_rate
    # This implies ALL outflow leaves the cohort C.
    # But Influx comes from C-1.
    # Does the outflow from C-1 go to C if C-1 was recruited?
    # If C-1 is recruited, its outflow goes to Biomass, NOT to C.

    # We need to filter the INFLUX based on whether the PREVIOUS cohort was recruited.
    # If C-1 was recruited, it emptied into Biomass, so Influx to C is 0.

    # Let's refine:
    # Outflow from C = P[C] / d_tau
    # Fraction Recruited from C = (1 if recruited else 0) * Outflow
    # Fraction Aging to C+1 from C = (0 if recruited else 1) * Outflow

    # So:
    # Influx to C = (Outflow from C-1) * (1 - is_recruited[C-1])

    # Let's compute the mask for the previous cohort
    prev_is_recruited = is_recruited.shift(cohort=1, fill_value=False)

    # Effective Influx to C
    # Only receive from C-1 if C-1 was NOT recruited.
    effective_influx = influx_flux.where(~prev_is_recruited, 0.0)

    # 6. Tendency for Production (P)
    # dP/dt = Influx_from_prev - Outflow_from_curr
    production_tendency = effective_influx - total_outflow_flux

    # 7. Source for Biomass
    # Sum of outflows from all cohorts that ARE recruited
    recruitment_flux = total_outflow_flux.where(is_recruited, 0.0)
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
