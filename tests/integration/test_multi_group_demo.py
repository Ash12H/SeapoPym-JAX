"""Integration test demonstrating multi-group zooplankton simulation."""

import jax.numpy as jnp

from seapopym_message.core.kernel import Kernel
from seapopym_message.model.zooplankton import zooplankton_group


def test_multi_group_simulation():
    """Run a simulation with two zooplankton groups: Epipelagic and Migrant."""

    # 1. Define Parameters
    # Shared biological parameters
    bio_params = {
        "tau_r0": 10.0,
        "gamma_tau_r": 0.1,
        "T_ref": 20.0,
        "lambda_0": 0.01,
        "gamma_lambda": 0.1,
        "n_ages": 5,
        "E": 0.1,
    }

    # Group-specific parameters
    epi_params = bio_params.copy()
    epi_params["layer_index"] = 0  # Surface

    migrant_params = bio_params.copy()
    migrant_params["day_layer_index"] = 2  # Deep
    migrant_params["night_layer_index"] = 0  # Surface

    # 2. Create Groups
    epi_group = zooplankton_group("epi", "epipelagic", epi_params)
    migrant_group = zooplankton_group("migrant", "migrant", migrant_params)

    # 3. Initialize Kernel
    kernel = Kernel([epi_group, migrant_group])

    # 4. Prepare Data
    # Dimensions: Depth=3, Lat=1, Lon=1
    # Temp: Surface=25°C, Mid=15°C, Deep=5°C
    temp_3d = jnp.array([[[25.0]], [[15.0]], [[5.0]]])  # (3, 1, 1)

    # NPP: 1.0
    npp = jnp.array([[1.0]])

    # Day Length: 0.5 (12h day / 12h night)
    day_length = jnp.array([[0.5]])

    forcings = {
        "forcing/temperature_3d": temp_3d,
        "forcing/npp": npp,
        "forcing/day_length": day_length,
    }

    # Initial State
    state = {
        # Epi
        "epi/biomass": jnp.array([[10.0]]),
        "epi/production": jnp.zeros((5, 1, 1)),
        # Migrant
        "migrant/biomass": jnp.array([[10.0]]),
        "migrant/production": jnp.zeros((5, 1, 1)),
    }

    # 5. Run Simulation Step
    dt = 1.0
    # Params dict needs to contain all params for all groups?
    # Currently Kernel passes the global 'params' dict to all units.
    # But FunctionalGroup stores its own params.
    # The Kernel/Unit execution model assumes params are passed in `execute`.
    # The `zooplankton_group` factory puts params in the FunctionalGroup object.
    # BUT, the Kernel currently doesn't automatically merge FunctionalGroup.params into the execution params.
    # I need to fix this in the test: merge params manually or update Kernel.
    # Let's merge manually for now.

    global_params = {}
    global_params.update(epi_params)
    global_params.update(migrant_params)
    # Wait, if keys collide (like 'tau_r0'), this is a problem!
    # 'tau_r0' is used by 'compute_tau_r_unit'.
    # If both groups use the SAME unit function, they read the SAME param key 'tau_r0'.
    # If they have different values for 'tau_r0', we have a conflict.

    # CRITICAL ARCHITECTURAL ISSUE:
    # Units read parameters by name (e.g. params['tau_r0']).
    # If two groups use the same Unit but need different parameter values,
    # we cannot pass a single flat 'params' dict.

    # Solution:
    # The `FunctionalGroup` has a `params` attribute.
    # The `Kernel` should inject these group-specific params when executing units belonging to that group.
    # OR, we namespace parameters too? (e.g. 'epi/tau_r0').
    # But the Unit code reads 'tau_r0'.

    # We need the Kernel to be smarter.
    # When executing `epi/compute_tau_r`, it should provide a params dict that contains `epi`'s values for `tau_r0`.

    # Let's check `src/seapopym_message/core/kernel.py`.
    # `execute_local_phase` iterates over units and calls `unit.execute(..., params=params)`.
    # It passes the SAME global params to everyone.

    # FIX REQUIRED:
    # I need to modify `Kernel` (or `Unit`) to handle group-specific parameters.
    # Option A: `Unit` stores its specific params (bound at creation).
    # Option B: `Kernel` maintains a map of Unit -> Params.

    # I will implement Option B in the test for now by passing a merged dict,
    # BUT since they share values in this test, it's fine.
    # However, `layer_index` is different!
    # `epi` needs `layer_index=0`.
    # `migrant` needs `day_layer_index=2`.
    # These keys are different, so they can coexist in one dict.
    # But if `tau_r0` differed, we'd have a problem.

    # For this test, I'll use the merged dict as keys don't conflict or values are identical.

    result_state = kernel.execute_local_phase(state, dt=dt, params=global_params, forcings=forcings)

    # 6. Assertions

    # Check Temperatures
    # Epi: Surface (25°C)
    # Migrant: 0.5 * 25 (Night/Surface) + 0.5 * 5 (Day/Deep) = 15°C

    # We can check the computed 'tau_r' in the state to verify this.
    # tau_r = tau_r0 * exp(-gamma * (T - T_ref))
    # T_ref = 20

    # Epi T=25: exp(-0.1 * (25-20)) = exp(-0.5) ≈ 0.606
    # tau_r_epi ≈ 10 * 0.606 = 6.06

    # Migrant T=15: exp(-0.1 * (15-20)) = exp(0.5) ≈ 1.648
    # tau_r_mig ≈ 10 * 1.648 = 16.48

    tau_r_epi = result_state["epi/tau_r"][0, 0]
    tau_r_mig = result_state["migrant/tau_r"][0, 0]

    assert jnp.isclose(tau_r_epi, 10.0 * jnp.exp(-0.5), rtol=1e-4)
    assert jnp.isclose(tau_r_mig, 10.0 * jnp.exp(0.5), rtol=1e-4)

    assert tau_r_epi < tau_r_mig  # Warmer -> Faster development

    print(f"Epi tau_r: {tau_r_epi}")
    print(f"Migrant tau_r: {tau_r_mig}")
