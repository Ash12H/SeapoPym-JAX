"""Diffusion simulation example with global kernel and halo exchange.

This example demonstrates:
- Global kernel (diffusion) requiring halo exchange between workers
- Initial condition with spatial heterogeneity (Gaussian blob)
- Visualization of diffusion spreading over time
"""

import jax.numpy as jnp
import ray

from seapopym_message import setup_and_run
from seapopym_message.core.kernel import Kernel
from seapopym_message.kernels.transport import compute_diffusion_2d
from seapopym_message.utils.grid import GridInfo


def gaussian_blob(lat_start, lat_end, lon_start, lon_end, grid: GridInfo):
    """Create initial condition: Gaussian blob centered in domain.

    Args:
        lat_start, lat_end: Latitude indices for this patch.
        lon_start, lon_end: Longitude indices for this patch.
        grid: Global grid information.

    Returns:
        State dictionary with "biomass" field.
    """
    # Get global coordinates
    lat_coords_global = grid.lat_coords
    lon_coords_global = grid.lon_coords

    # Extract local coordinates
    lat_local = lat_coords_global[lat_start:lat_end]
    lon_local = lon_coords_global[lon_start:lon_end]

    # Create meshgrid
    LON, LAT = jnp.meshgrid(lon_local, lat_local)

    # Gaussian blob centered at grid center
    lat_center = (grid.lat_min + grid.lat_max) / 2.0
    lon_center = (grid.lon_min + grid.lon_max) / 2.0
    sigma_lat = (grid.lat_max - grid.lat_min) / 8.0
    sigma_lon = (grid.lon_max - grid.lon_min) / 8.0

    biomass = 100.0 * jnp.exp(
        -((LAT - lat_center) ** 2) / (2 * sigma_lat**2)
        - ((LON - lon_center) ** 2) / (2 * sigma_lon**2)
    )

    return {"biomass": biomass}


def main():
    """Run diffusion simulation."""
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=4)

    print("=== Diffusion Simulation with Halo Exchange ===\n")

    # Define grid
    grid = GridInfo(
        lat_min=-10.0,
        lat_max=10.0,
        lon_min=-10.0,
        lon_max=10.0,
        nlat=40,
        nlon=40,
    )
    print(f"Grid: {grid}\n")

    # Define kernel (diffusion only - global kernel)
    kernel = Kernel([compute_diffusion_2d])
    print(f"Kernel: {len(kernel.global_units)} global units (requires halo exchange)\n")

    # Diffusion parameters
    params = {
        "D": 0.5,  # Diffusion coefficient
        "dx": grid.dx,  # Grid spacing in meters
    }
    print(f"Parameters: D={params['D']}, dx={params['dx']/1000:.1f} km\n")

    # Initial condition: Gaussian blob
    def initial_state(lat_start, lat_end, lon_start, lon_end):
        return gaussian_blob(lat_start, lat_end, lon_start, lon_end, grid)

    # Simulation parameters
    dt = 0.01  # Small timestep for stability
    t_max = 1.0  # Run for 1.0 time units
    num_workers_lat = 2
    num_workers_lon = 2
    total_workers = num_workers_lat * num_workers_lon

    print(f"Simulation: dt={dt}, t_max={t_max}")
    print(f"Workers: {num_workers_lat}x{num_workers_lon} = {total_workers} total")
    print("Note: Workers will exchange halo data for diffusion\n")

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
    initial_mean = diagnostics[0]["biomass_global_mean"]
    initial_max = diagnostics[0]["biomass_global_max"]
    final_mean = diagnostics[-1]["biomass_global_mean"]
    final_max = diagnostics[-1]["biomass_global_max"]

    print("Initial state:")
    print(f"  Mean biomass: {initial_mean:.4f}")
    print(f"  Max biomass: {initial_max:.4f}")

    print("\nFinal state:")
    print(f"  Mean biomass: {final_mean:.4f}")
    print(f"  Max biomass: {final_max:.4f}")

    # Check conservation (diffusion should conserve mass)
    mass_change = abs(final_mean - initial_mean) / initial_mean * 100
    print("\nMass conservation:")
    print(f"  Change in mean: {mass_change:.4f}%")
    if mass_change < 1.0:
        print("  ✓ Mass well conserved!")
    else:
        print("  ⚠ Significant mass change (numerical diffusion)")

    # Check spreading
    spreading_factor = initial_max / final_max
    print("\nSpreading:")
    print(f"  Peak reduction: {spreading_factor:.2f}x")
    print("  Blob has spread out due to diffusion")

    print(f"\nFinal state shape: {final_state['biomass'].shape}")
    print(f"Final state std: {float(jnp.std(final_state['biomass'])):.4f}")

    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main()
