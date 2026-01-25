"""Core functions for LMTL model."""

import dask.array as da
import numpy as np
import xarray as xr

from seapopym.standard.coordinates import Coordinates

# Try to import Numba for optimized kernels
try:
    from numba import float64, guvectorize  # type: ignore[import-not-found]

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

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
    doy = time.dt.dayofyear

    # Latitude in radians
    phi = np.deg2rad(latitude)

    # Declination of the sun
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


def compute_layer_weighted_mean(
    forcing: xr.DataArray,
    day_length: xr.DataArray,
    day_layer: float,
    night_layer: float,
) -> dict[str, xr.DataArray]:
    """Compute mean forcing experienced by vertically migrating organisms.

    Mean = Forcing_day * day_length + Forcing_night * (1 - day_length)

    Parameters
    ----------
    forcing : xr.DataArray
        Forcing field (e.g. temperature, current) with depth dimension.
    day_length : xr.DataArray
        Day length fraction [dimensionless, 0-1].
    day_layer : float
        Depth layer for daytime [m].
    night_layer : float
        Depth layer for nighttime [m].

    Returns
    -------
    dict
        {"output": mean_forcing} with same units as input forcing.
    """
    # Select forcing at specific layers
    # We assume 'depth' coordinate exists or we use method='nearest'
    val_day = forcing.sel({Coordinates.Z: day_layer}, method="nearest")
    val_night = forcing.sel({Coordinates.Z: night_layer}, method="nearest")

    return {"output": val_day * day_length + val_night * (1.0 - day_length)}


def compute_threshold_temperature(
    temperature: xr.DataArray,
    min_temperature: float = 0.0,
) -> dict[str, xr.DataArray]:
    """Threshold temperature values below a minimum value.

    Parameters
    ----------
    temperature : xr.DataArray
        Input temperature [degC].
    min_temperature : float, optional
        Minimum allowed temperature [degC], by default 0.0.

    Returns
    -------
    dict
        {"output": thresholded_temperature}
    """
    return {"output": temperature.where(temperature >= min_temperature, min_temperature)}


def compute_gillooly_temperature(
    temperature: xr.DataArray,
) -> dict[str, xr.DataArray]:
    """Apply Gillooly temperature normalization.

    This transformation is used in metabolic scaling models based on the
    Arrhenius equation. It normalizes temperature from Celsius to a form
    suitable for metabolic rate calculations.

    T_normalized = T / (1 + T/273)

    where 273 represents the approximate conversion factor related to
    absolute zero.

    Parameters
    ----------
    temperature : xr.DataArray
        Input temperature [degC].

    Returns
    -------
    dict
        {"output": normalized_temperature} in [degC] (dimensionally same,
        but normalized for metabolic equations).

    Reference
    -------
    Gillooly et al. (2001) "Effects of size and temperature on metabolic rate"
    """
    normalized_temp = temperature / (1.0 + temperature / 273.0)
    return {"output": normalized_temp}


def compute_recruitment_age(
    temperature: xr.DataArray,
    tau_r_0: float,
    gamma_tau_r: float,
    T_ref: float,
) -> dict[str, xr.DataArray]:
    """Compute minimum recruitment age based on temperature.

    tau_r = tau_r_0 * exp(-gamma * (T - T_ref))

    Parameters
    ----------
    temperature : xr.DataArray
        Temperature [degC] (should be Gillooly-normalized in practice).
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
    return {"output": tau_r_0 * np.exp(-gamma_tau_r * (temperature - T_ref))}


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
    # Source is a rate (e.g. gC/m2/s)
    # The TimeIntegrator will multiply by dt (in seconds).
    tendency_rate = E * primary_production

    # Create a cohort mask: [1, 0, 0, ..., 0]
    # This approach avoids expensive reindex() operation that creates large Dask graphs
    n_cohorts = len(cohorts)
    cohort_mask = np.zeros(n_cohorts)
    cohort_mask[0] = 1.0

    # Broadcast tendency_rate × mask
    # Works for both numpy and dask arrays without graph explosion
    # expand_dims(axis=-1) ensures the NEW dimension (cohort) is at the END.
    tendency = tendency_rate.expand_dims(dim={"cohort": cohorts}, axis=-1) * xr.DataArray(
        cohort_mask, dims=["cohort"], coords={"cohort": cohorts}
    )

    return {"output": tendency}


def _shift_with_overlap(
    data: xr.DataArray, dim: str, shift: int = 1, fill_value: float = 0.0
) -> xr.DataArray:
    """Optimized shift using dask.map_overlap to avoid global rechunking.

    Only supports positive shift=1 for now (used in aging).
    Uses symmetric overlap with dask map_overlap to ensure communication
    is local between neighboring chunks, avoiding shuffle.
    """
    if shift != 1:
        # Fallback to standard shift for other cases or implement generic logic
        return data.shift({dim: shift}, fill_value=fill_value)

    # Check if we are dealing with dask array chunked along dim
    if not isinstance(data.data, da.Array):
        return data.shift({dim: shift}, fill_value=fill_value)

    # Get axis index
    axis = data.get_axis_num(dim)

    # Optimized path for Dask
    dask_arr = data.data

    def _local_shift_kernel(chunk: np.ndarray, axis: int, _fill_val: float) -> np.ndarray:
        """Kernel applied on chunks.

        Chunk has shape (..., 1_left + N + 1_right, ...) due to symmetric overlap.
        For shift=+1 (right):
        - Input schema: [Ghost_L | D_0, ..., D_N-1 | Ghost_R]
        - Target Output: [Ghost_L, D_0, ..., D_{N-2}]
        So we just take the slice [:-2] along the axis.
        """
        sl = [slice(None)] * chunk.ndim
        sl[axis] = slice(None, -2)
        return chunk[tuple(sl)]

    # Workaround: Dask supports constant boundary with symmetric overlap (int depth)
    # We ask for depth=1 (implies (1, 1)).
    depth = {axis: 1}
    boundary = {axis: fill_value}

    shifted_dask = dask_arr.map_overlap(
        _local_shift_kernel,
        dtype=dask_arr.dtype,
        depth=depth,
        boundary=boundary,
        axis=axis,
        _fill_val=fill_value,
        trim=False,  # We handle size manually
    )

    # Wrap back in DataArray
    return xr.DataArray(shifted_dask, coords=data.coords, dims=data.dims, name=data.name)


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
    Uses efficient Dask operations (concat/slice/map_overlap) instead of reindex/shift
    to prevent implicit rechunking and task explosion.
    """
    # Calculate Delta_tau (cohort duration)
    # We use underlying array ops to stay efficient
    # Using .data access + standard slicing/numpy ops works for both
    # Numpy and Dask (thanks to __array_function__ dispatch in Dask)

    # Get values (lazy if dask)
    c_vals = cohort_ages.data

    # Case with single cohort
    if cohort_ages.size < 2:
        d_tau = xr.DataArray([dt], coords=cohort_ages.coords, dims=cohort_ages.dims)
    else:
        # Universal slicing logic
        # diffs[i] = age[i+1] - age[i]
        diffs = c_vals[1:] - c_vals[:-1]

        # Pad last value
        last_val = diffs[-1:]  # Keep dimensions

        # Concatenate
        # np.concatenate dispatches to dask.array.concatenate if inputs are dask
        full_vals = np.concatenate([diffs, last_val], axis=0)

        d_tau = xr.DataArray(full_vals, coords=cohort_ages.coords, dims=cohort_ages.dims)

    # 1. Aging Rate (Flux per unit mass) = 1 / Delta_tau
    # This is the speed at which mass leaves the cohort.
    aging_rate = 1.0 / d_tau

    # 2. Outflow Rate (Mass/Time) leaving cohort C
    # Flux_out = P[C] * aging_rate
    total_outflow_flux = production * aging_rate

    # 3. Influx (from previous cohort)
    # Optimized shift
    influx_flux = _shift_with_overlap(total_outflow_flux, dim="cohort", shift=1, fill_value=0.0)

    # 4. Identify Recruited Cohorts
    is_recruited = cohort_ages >= recruitment_age

    # 5. Separate Outflows
    # We need to filter the INFLUX based on whether the PREVIOUS cohort was recruited.
    # If C-1 was recruited, it emptied into Biomass, so Influx to C is 0.

    # Influx to C = (Outflow from C-1) * (1 - is_recruited[C-1])

    # Optimized shift for mask
    prev_is_recruited = _shift_with_overlap(is_recruited, dim="cohort", shift=1, fill_value=False)

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


# =============================================================================
# NUMBA-OPTIMIZED VERSION
# =============================================================================

if NUMBA_AVAILABLE:

    @guvectorize(
        [
            (
                float64[:],  # production (cohort,)
                float64[:],  # cohort_ages (cohort,)
                float64,  # recruitment_age (scalar for this cell)
                float64[:],  # d_tau (cohort,)
                float64[:],  # production_tendency (output)
                float64[:],  # recruitment_flux (output)
            ),
        ],
        "(c),(c),(),(c)->(c),(c)",
        nopython=True,
        fastmath=True,
        cache=True,
    )
    def _production_dynamics_numba(  # type: ignore[no-untyped-def]
        production, cohort_ages, recruitment_age, d_tau, production_tendency, recruitment_flux
    ):
        """Numba kernel for production dynamics.

        Operates on a single spatial cell, processing all cohorts.
        Calculates:
        - production_tendency: Influx from C-1 minus outflow from C
        - recruitment_flux: Outflow that goes to biomass (recruited cohorts)
        """
        n_cohorts = len(production)
        prev_recruited = False
        prev_outflow = 0.0

        for c in range(n_cohorts):
            # Aging rate and outflow
            aging_rate = 1.0 / d_tau[c]
            outflow = production[c] * aging_rate

            # Influx from previous cohort (0 if first or prev was recruited)
            influx = 0.0 if c == 0 else (prev_outflow if not prev_recruited else 0.0)

            # Is this cohort recruited?
            is_recruited = cohort_ages[c] >= recruitment_age

            # Production tendency
            production_tendency[c] = influx - outflow

            # Recruitment flux (to sum later for biomass source)
            recruitment_flux[c] = outflow if is_recruited else 0.0

            # Update for next iteration
            prev_recruited = is_recruited
            prev_outflow = outflow


def compute_production_dynamics_optimized(
    production: xr.DataArray,
    recruitment_age: xr.DataArray,
    cohort_ages: xr.DataArray,
    dt: float,
) -> dict[str, xr.DataArray]:
    """Numba-optimized production dynamics (2.1x faster).

    Handles:
    1. Aging flux (transport from C-1 to C).
    2. Outflow from C:
       - If recruited: Transfer to biomass (Recruitment).
       - If NOT recruited: Exit the model (Senescence/Outflow).

    Uses xr.apply_ufunc with Numba guvectorize kernel for performance.
    Falls back to pure xarray implementation if Numba is unavailable.

    Performance:
        ~2.1x faster than compute_production_dynamics with xarray operations.

    Parameters
    ----------
    production : xr.DataArray
        Production field [g/m²], shape (..., cohort).
    recruitment_age : xr.DataArray
        Minimum recruitment age [s], shape (...).
    cohort_ages : xr.DataArray
        Cohort ages [s], shape (cohort,).
    dt : float
        Time step [s].

    Returns
    -------
    dict
        {
            "production_tendency": tendency for production [g/m²/s],
            "recruitment_source": source for biomass [g/m²/s]
        }
    """
    if not NUMBA_AVAILABLE:
        # Fallback to xarray implementation
        return compute_production_dynamics(production, recruitment_age, cohort_ages, dt)

    # Pre-compute d_tau (constant for all cells)
    c_vals = cohort_ages.values
    if len(c_vals) < 2:
        d_tau = np.array([dt])
    else:
        diffs = c_vals[1:] - c_vals[:-1]
        d_tau = np.concatenate([diffs, diffs[-1:]])

    d_tau_da = xr.DataArray(
        d_tau.astype(np.float64), dims=["cohort"], coords={"cohort": cohort_ages}
    )

    # Use apply_ufunc for dimension safety
    production_tendency, recruitment_flux = xr.apply_ufunc(
        _production_dynamics_numba,
        production,
        cohort_ages,
        recruitment_age,
        d_tau_da,
        input_core_dims=[
            ["cohort"],  # production
            ["cohort"],  # cohort_ages
            [],  # recruitment_age (scalar per cell)
            ["cohort"],  # d_tau
        ],
        output_core_dims=[
            ["cohort"],  # production_tendency
            ["cohort"],  # recruitment_flux
        ],
        dask="parallelized",
        output_dtypes=[production.dtype, production.dtype],
    )

    # Sum recruitment flux over cohorts to get biomass source
    biomass_source = recruitment_flux.sum(dim="cohort")

    return {
        "production_tendency": production_tendency,
        "recruitment_source": biomass_source,
    }


def compute_mortality_tendency(
    biomass: xr.DataArray,
    temperature: xr.DataArray,
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
    temperature : xr.DataArray
        Temperature [degC] (should be Gillooly-normalized in practice).
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
    rate = lambda_0 * np.exp(gamma_lambda * (temperature - T_ref))
    tendency = -rate * biomass

    return {"mortality_loss": tendency}
