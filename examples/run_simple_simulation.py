"""Simple end-to-end simulation example.

This example demonstrates how to set up and run a complete distributed
simulation with biology kernels (recruitment, mortality, growth).
"""

import jax.numpy as jnp
import ray

from seapopym_message import setup_and_run
from seapopym_message.core.kernel import Kernel
from seapopym_message.kernels.biology import (
    compute_growth,
    compute_mortality,
    compute_recruitment_2d,
)
from seapopym_message.utils.grid import GridInfo


def main():
    """Run a simple biology simulation."""
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=4)

    print("=== Simple Biology Simulation ===\n")

    # Define grid (Pacific Ocean subset)
    grid = GridInfo(
        lat_min=-10.0,  # 10°S
        lat_max=10.0,  # 10°N
        lon_min=140.0,  # 140°E
        lon_max=180.0,  # 180°E
        nlat=20,
        nlon=40,
    )
    print(f"Grid: {grid}\n")

    # Define kernel (biology only: recruitment + mortality + growth)
    kernel = Kernel([compute_recruitment_2d, compute_mortality, compute_growth])
    print(f"Kernel: {len(kernel.local_units)} local units\n")

    # Model parameters (simple logistic growth)
    params = {
        "R": 10.0,  # Recruitment rate
        "lambda": 0.1,  # Mortality rate
    }
    # Expected equilibrium: B_eq = R/lambda = 100.0
    print(f"Parameters: R={params['R']}, lambda={params['lambda']}")
    print(f"Expected equilibrium: B_eq = R/lambda = {params['R']/params['lambda']}\n")

    # Initial condition: start from zero biomass
    def initial_state(lat_start, lat_end, lon_start, lon_end):
        nlat = lat_end - lat_start
        nlon = lon_end - lon_start
        return {"biomass": jnp.zeros((nlat, nlon))}

    # Simulation parameters
    dt = 0.1  # 0.1 time units per step
    t_max = 50.0  # Run for 50 time units
    num_workers_lat = 2
    num_workers_lon = 2
    total_workers = num_workers_lat * num_workers_lon

    print(f"Simulation: dt={dt}, t_max={t_max}")
    print(f"Workers: {num_workers_lat}x{num_workers_lon} = {total_workers} total\n")

    # Run simulation
    print("Running simulation...")
    diagnostics, final_state = setup_and_run(
        grid=grid,
        kernel=kernel,
        params=params,
        initial_state_fn=initial_state,
        dt=dt,
        t_max=t_max,
        num_workers_lat=num_workers_lat,
        num_workers_lon=num_workers_lon,
        periodic_lon=False,
    )

    print(f"Simulation complete! {len(diagnostics)} timesteps executed.\n")

    # Analyze results
    print("=== Results ===")
    print(f"Initial biomass (global mean): {diagnostics[0]['biomass_global_mean']:.4f}")
    print(f"Final biomass (global mean): {diagnostics[-1]['biomass_global_mean']:.4f}")
    print(f"Final biomass (global min): {diagnostics[-1]['biomass_global_min']:.4f}")
    print(f"Final biomass (global max): {diagnostics[-1]['biomass_global_max']:.4f}")

    # Check convergence to equilibrium
    expected_eq = params["R"] / params["lambda"]
    final_biomass = diagnostics[-1]["biomass_global_mean"]
    error = abs(final_biomass - expected_eq) / expected_eq * 100

    print("\nConvergence check:")
    print(f"  Expected equilibrium: {expected_eq:.2f}")
    print(f"  Actual final biomass: {final_biomass:.2f}")
    print(f"  Relative error: {error:.2f}%")

    if error < 1.0:
        print("  ✓ Successfully converged to equilibrium!")
    else:
        print("  ⚠ Not fully converged yet (may need longer simulation)")

    # Print final state statistics
    print(f"\nFinal state shape: {final_state['biomass'].shape}")
    print(f"Final state mean: {float(jnp.mean(final_state['biomass'])):.4f}")
    print(f"Final state std: {float(jnp.std(final_state['biomass'])):.4f}")

    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main()
