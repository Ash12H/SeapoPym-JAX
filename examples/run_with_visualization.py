"""Simulation example with visualization.

This example runs a simulation and creates various plots to visualize results.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import ray

from seapopym_message import setup_and_run
from seapopym_message.core.kernel import Kernel
from seapopym_message.kernels.biology import (
    compute_growth,
    compute_mortality,
    compute_recruitment_2d,
)
from seapopym_message.utils.grid import GridInfo
from seapopym_message.visualization import (
    plot_convergence,
    plot_diagnostics_timeseries,
    plot_spatial_field,
)


def main():
    """Run simulation with visualization."""
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=4)

    print("=== Simulation with Visualization ===\n")

    # Grid
    grid = GridInfo(
        lat_min=-10.0,
        lat_max=10.0,
        lon_min=140.0,
        lon_max=180.0,
        nlat=40,
        nlon=80,
    )
    print(f"Grid: {grid}\n")

    # Kernel
    kernel = Kernel([compute_recruitment_2d, compute_mortality, compute_growth])

    # Parameters
    params = {"R": 10.0, "lambda": 0.1}
    expected_equilibrium = params["R"] / params["lambda"]
    print(f"Parameters: R={params['R']}, lambda={params['lambda']}")
    print(f"Expected equilibrium: {expected_equilibrium}\n")

    # Initial condition
    def initial_state(lat_start, lat_end, lon_start, lon_end):
        nlat = lat_end - lat_start
        nlon = lon_end - lon_start
        return {"biomass": jnp.zeros((nlat, nlon))}

    # Run simulation
    print("Running simulation...")
    diagnostics, final_state = setup_and_run(
        grid=grid,
        kernel=kernel,
        params=params,
        initial_state_fn=initial_state,
        dt=0.1,
        t_max=50.0,
        num_workers_lat=2,
        num_workers_lon=2,
    )
    print(f"Simulation complete! {len(diagnostics)} timesteps.\n")

    # Create visualizations
    print("Creating visualizations...\n")

    # 1. Time series plot
    print("1. Time series of biomass evolution")
    fig1 = plot_diagnostics_timeseries(diagnostics, variable="biomass")
    fig1.savefig("biomass_timeseries.png", dpi=150)
    print("   Saved: biomass_timeseries.png")

    # 2. Convergence plot
    print("2. Convergence to equilibrium")
    fig2 = plot_convergence(diagnostics, expected_value=expected_equilibrium, variable="biomass")
    fig2.savefig("convergence.png", dpi=150)
    print("   Saved: convergence.png")

    # 3. Final spatial field
    print("3. Final biomass spatial distribution")
    fig3 = plot_spatial_field(
        final_state["biomass"],
        grid,
        title="Final Biomass Distribution",
        cmap="YlOrRd",
    )
    fig3.savefig("final_biomass_spatial.png", dpi=150)
    print("   Saved: final_biomass_spatial.png")

    # 4. Show results
    print("\nFinal statistics:")
    print(f"  Mean biomass: {diagnostics[-1]['biomass_global_mean']:.2f}")
    print(f"  Min biomass: {diagnostics[-1]['biomass_global_min']:.2f}")
    print(f"  Max biomass: {diagnostics[-1]['biomass_global_max']:.2f}")
    print(f"  Expected: {expected_equilibrium:.2f}")

    error = (
        abs(diagnostics[-1]["biomass_global_mean"] - expected_equilibrium) / expected_equilibrium
    )
    print(f"  Relative error: {error*100:.2f}%")

    plt.show()

    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main()
