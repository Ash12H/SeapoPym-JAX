# %% [markdown]
"""
# Notebook 05a: Optimizing compute_production_dynamics

Micro-benchmark and Numba optimization for the production dynamics function.
"""

# %%
import time

import numpy as np
import xarray as xr
from numba import guvectorize, float64, int32

print("✅ Imports loaded")

# %%
# Configuration
NY, NX = 121, 191
N_COHORTS = 12
N_RUNS = 50

print(f"✅ Config: {NY}×{NX} grid, {N_COHORTS} cohorts, {N_RUNS} runs")

# %%
# Create test data
np.random.seed(42)

# Cohort ages (in seconds, e.g., 0, 1 day, 2 days, ...)
cohort_ages = xr.DataArray(
    np.arange(N_COHORTS) * 86400.0,  # seconds
    dims=["cohort"],
    attrs={"units": "second"},
)

# Production field: (y, x, cohort)
production = xr.DataArray(
    np.random.rand(NY, NX, N_COHORTS).astype(np.float64) * 10.0,
    dims=["y", "x", "cohort"],
    coords={"cohort": cohort_ages},
    attrs={"units": "g/m**2"},
)

# Recruitment age: (y, x) - varies spatially
recruitment_age = xr.DataArray(
    np.random.uniform(5, 10, (NY, NX)).astype(np.float64) * 86400.0,  # 5-10 days in seconds
    dims=["y", "x"],
    attrs={"units": "second"},
)

# dt in seconds (3 hours)
dt = 3 * 3600.0

print(f"✅ Test data created")
print(f"   production: {production.shape}")
print(f"   recruitment_age: {recruitment_age.shape}")

# %% [markdown]
"""
## Reference Implementation (current)
"""


# %%
def compute_production_dynamics_ref(
    production: xr.DataArray,
    recruitment_age: xr.DataArray,
    cohort_ages: xr.DataArray,
    dt: float,
) -> dict[str, xr.DataArray]:
    """Reference implementation (current code)."""
    c_vals = cohort_ages.data

    if cohort_ages.size < 2:
        d_tau = xr.DataArray([dt], coords=cohort_ages.coords, dims=cohort_ages.dims)
    else:
        diffs = c_vals[1:] - c_vals[:-1]
        last_val = diffs[-1:]
        full_vals = np.concatenate([diffs, last_val], axis=0)
        d_tau = xr.DataArray(full_vals, coords=cohort_ages.coords, dims=cohort_ages.dims)

    aging_rate = 1.0 / d_tau
    total_outflow_flux = production * aging_rate
    influx_flux = total_outflow_flux.shift(cohort=1, fill_value=0.0)
    is_recruited = cohort_ages >= recruitment_age
    prev_is_recruited = is_recruited.shift(cohort=1, fill_value=False)
    effective_influx = influx_flux.where(~prev_is_recruited, 0.0)
    production_tendency = effective_influx - total_outflow_flux
    recruitment_flux = total_outflow_flux.where(is_recruited, 0.0)
    biomass_source = recruitment_flux.sum(dim="cohort")

    return {
        "production_tendency": production_tendency,
        "recruitment_source": biomass_source,
    }


# Warmup
_ = compute_production_dynamics_ref(production, recruitment_age, cohort_ages, dt)

# Benchmark
times_ref = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    result_ref = compute_production_dynamics_ref(production, recruitment_age, cohort_ages, dt)
    times_ref.append(time.perf_counter() - t0)

mean_ref = np.mean(times_ref) * 1000
std_ref = np.std(times_ref) * 1000
print(f"Reference: {mean_ref:.3f} ± {std_ref:.3f} ms")

# %% [markdown]
"""
## Numba Implementation
"""


# %%
@guvectorize(
    [
        (
            float64[:],  # production (cohort,)
            float64[:],  # cohort_ages (cohort,)
            float64,  # recruitment_age (scalar for this cell)
            float64[:],  # d_tau (cohort,)
            float64[:],  # production_tendency (output)
            float64[:],  # recruitment_flux (output per cohort, for summing)
        ),
    ],
    "(c),(c),(),(c)->(c),(c)",
    nopython=True,
    fastmath=True,
    cache=True,
)
def production_dynamics_numba(
    production, cohort_ages, recruitment_age, d_tau, production_tendency, recruitment_flux
):
    """Numba kernel for production dynamics (sequential).

    Operates on a single spatial cell, processing all cohorts.
    """
    n_cohorts = len(production)

    # Previous cohort was recruited? (for influx filtering)
    prev_recruited = False
    prev_outflow = 0.0

    for c in range(n_cohorts):
        # Aging rate and outflow
        aging_rate = 1.0 / d_tau[c]
        outflow = production[c] * aging_rate

        # Influx from previous cohort
        if c == 0:
            influx = 0.0
        else:
            # Only receive influx if previous cohort was NOT recruited
            influx = prev_outflow if not prev_recruited else 0.0

        # Is this cohort recruited?
        is_recruited = cohort_ages[c] >= recruitment_age

        # Production tendency
        production_tendency[c] = influx - outflow

        # Recruitment flux (to sum later for biomass source)
        recruitment_flux[c] = outflow if is_recruited else 0.0

        # Update for next iteration
        prev_recruited = is_recruited
        prev_outflow = outflow


# PARALLEL VERSION - same kernel but with target="parallel"
@guvectorize(
    [
        (
            float64[:],  # production (cohort,)
            float64[:],  # cohort_ages (cohort,)
            float64,  # recruitment_age (scalar for this cell)
            float64[:],  # d_tau (cohort,)
            float64[:],  # production_tendency (output)
            float64[:],  # recruitment_flux (output per cohort, for summing)
        ),
    ],
    "(c),(c),(),(c)->(c),(c)",
    nopython=True,
    fastmath=True,
    target="parallel",  # Parallelize over broadcast dimensions (y, x)
)
def production_dynamics_numba_parallel(
    production, cohort_ages, recruitment_age, d_tau, production_tendency, recruitment_flux
):
    """Numba kernel for production dynamics (parallel on y, x).

    Same logic as sequential version, but with target="parallel"
    to parallelize across spatial dimensions.
    """
    n_cohorts = len(production)
    prev_recruited = False
    prev_outflow = 0.0

    for c in range(n_cohorts):
        aging_rate = 1.0 / d_tau[c]
        outflow = production[c] * aging_rate

        if c == 0:
            influx = 0.0
        else:
            influx = prev_outflow if not prev_recruited else 0.0

        is_recruited = cohort_ages[c] >= recruitment_age
        production_tendency[c] = influx - outflow
        recruitment_flux[c] = outflow if is_recruited else 0.0

        prev_recruited = is_recruited
        prev_outflow = outflow


def compute_production_dynamics_numba(
    production: xr.DataArray,
    recruitment_age: xr.DataArray,
    cohort_ages: xr.DataArray,
    dt: float,
    parallel: bool = False,
) -> dict[str, xr.DataArray]:
    """Numba-accelerated production dynamics using apply_ufunc.

    Uses xr.apply_ufunc to ensure correct dimension ordering.
    """
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

    # Select kernel
    kernel = production_dynamics_numba_parallel if parallel else production_dynamics_numba

    # Use apply_ufunc for dimension safety
    # The kernel signature is: (c),(c),(),(c)->(c),(c)
    # production: (..., cohort)
    # cohort_ages: (cohort,)
    # recruitment_age: (...)
    # d_tau: (cohort,)
    production_tendency, recruitment_flux = xr.apply_ufunc(
        kernel,
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


# Warmup (JIT compilation for both versions)
_ = compute_production_dynamics_numba(production, recruitment_age, cohort_ages, dt, parallel=False)
_ = compute_production_dynamics_numba(production, recruitment_age, cohort_ages, dt, parallel=True)
_ = compute_production_dynamics_numba(production, recruitment_age, cohort_ages, dt, parallel=False)
_ = compute_production_dynamics_numba(production, recruitment_age, cohort_ages, dt, parallel=True)

# Benchmark SEQUENTIAL
times_numba = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    result_numba = compute_production_dynamics_numba(
        production, recruitment_age, cohort_ages, dt, parallel=False
    )
    times_numba.append(time.perf_counter() - t0)

mean_numba = np.mean(times_numba) * 1000
std_numba = np.std(times_numba) * 1000
print(f"Numba (sequential): {mean_numba:.3f} ± {std_numba:.3f} ms")

# Benchmark PARALLEL
times_parallel = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    result_parallel = compute_production_dynamics_numba(
        production, recruitment_age, cohort_ages, dt, parallel=True
    )
    times_parallel.append(time.perf_counter() - t0)

mean_parallel = np.mean(times_parallel) * 1000
std_parallel = np.std(times_parallel) * 1000
print(f"Numba (parallel):   {mean_parallel:.3f} ± {std_parallel:.3f} ms")

# %% [markdown]
"""
## Validation
"""

# %%
# Check numerical accuracy
prod_tend_ref = result_ref["production_tendency"].values
prod_tend_numba = result_numba["production_tendency"].values
bio_src_ref = result_ref["recruitment_source"].values
bio_src_numba = result_numba["recruitment_source"].values

diff_tend = np.abs(prod_tend_ref - prod_tend_numba)
diff_bio = np.abs(bio_src_ref - bio_src_numba)

print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)
print(f"Max diff (production_tendency): {diff_tend.max():.2e}")
print(f"Max diff (recruitment_source):  {diff_bio.max():.2e}")

if diff_tend.max() < 1e-10 and diff_bio.max() < 1e-10:
    print("✅ Numerical accuracy: PASSED")
else:
    print("⚠️ Numerical accuracy: DIFFERENCES DETECTED")

# %% [markdown]
"""
## Results Summary
"""

# %%
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"Grid: {NY} × {NX}, Cohorts: {N_COHORTS}")
print()
print(f"{'Implementation':<25} {'Time (ms)':<18} {'Speedup':<10}")
print("-" * 55)
print(f"{'Reference (xarray)':<25} {mean_ref:.3f} ± {std_ref:.3f}")
print(
    f"{'Numba (sequential)':<25} {mean_numba:.3f} ± {std_numba:.3f}       {mean_ref / mean_numba:.1f}x"
)
print(
    f"{'Numba (parallel)':<25} {mean_parallel:.3f} ± {std_parallel:.3f}       {mean_ref / mean_parallel:.1f}x"
)

speedup_seq = mean_ref / mean_numba
speedup_par = mean_ref / mean_parallel
parallel_gain = mean_numba / mean_parallel

print(f"\n🚀 Numba sequential: {speedup_seq:.1f}x faster than xarray")
print(f"🚀 Numba parallel:   {speedup_par:.1f}x faster than xarray")
print(f"⚡ Parallel vs sequential: {parallel_gain:.2f}x")
