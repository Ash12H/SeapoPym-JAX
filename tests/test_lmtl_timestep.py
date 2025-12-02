import numpy as np
import xarray as xr

from seapopym.lmtl.core import compute_production_dynamics


def test_production_dynamics_timestep_independence():
    """
    Verify that the production dynamics (aging) speed is independent of the timestep.

    The physical time it takes for a cohort to empty should depend on the cohort's age span,
    not on the simulation timestep 'dt'.

    We compare two cases:
    1. dt = cohort_duration (1 day)
    2. dt = small fraction of cohort_duration (1 hour)

    In a correct implementation, the flux rate in case 2 should be scaled so that
    it takes roughly the same physical time to move the mass.

    Current behavior (Bug):
    Case 2 moves mass 24x faster because rate is 1/dt.
    """
    # Setup
    day_seconds = 86400.0
    cohorts = np.array([0, 1, 2]) * day_seconds
    cohort_ages = xr.DataArray(cohorts, coords={"cohort": cohorts}, dims="cohort", name="cohort")

    # Initial state: Production only in cohort 0
    production = xr.DataArray([100.0, 0.0, 0.0], coords={"cohort": cohorts}, dims="cohort")

    # No recruitment (very high recruitment age)
    recruitment_age = xr.DataArray(100 * day_seconds)

    # Case 1: dt = 1 day
    dt_1 = day_seconds
    dynamics_1 = compute_production_dynamics(production, recruitment_age, cohort_ages, dt_1)
    tendency_1 = dynamics_1["production_tendency"]
    # Update for 1 step
    prod_1_next = production + tendency_1 * dt_1

    # Case 2: dt = 1 hour
    dt_2 = 3600.0
    dynamics_2 = compute_production_dynamics(production, recruitment_age, cohort_ages, dt_2)
    tendency_2 = dynamics_2["production_tendency"]
    # Update for 1 step
    prod_2_next = production + tendency_2 * dt_2

    # Analysis
    # In Case 1 (dt=1 day), we expect full transfer from C0 to C1 (since duration is 1 day).
    # P[0] becomes 0, P[1] becomes 100.

    # In Case 2 (dt=1 hour), we expect partial transfer.
    # Fraction transferred should be approx dt / duration = 1/24.
    # P[1] should be approx 100/24 ~= 4.16

    # Currently, the bug causes P[1] to be 100 in Case 2 as well.

    # We assert the *correct* behavior to fail if the bug is present.
    # Or we can assert the *buggy* behavior to confirm we reproduced it?
    # The user asked to "verify these behaviors".
    # Let's assert the CORRECT behavior, so the test fails, indicating the bug exists.

    # Check Case 1 (Reference)
    assert np.isclose(prod_1_next.isel(cohort=1), 100.0), "Case 1: Should transfer fully in 1 day"

    # Check Case 2 (Target behavior)
    transferred_amount = prod_2_next.isel(cohort=1).item()
    expected_amount = 100.0 * (dt_2 / day_seconds)

    # Allow some numerical tolerance, but it should be far from 100
    assert (
        transferred_amount < 50.0
    ), f"Case 2: Transfer was too fast! Got {transferred_amount}, expected approx {expected_amount}"
    assert np.isclose(
        transferred_amount, expected_amount, rtol=0.1
    ), f"Case 2: Transfer rate incorrect. Got {transferred_amount}, expected {expected_amount}"
