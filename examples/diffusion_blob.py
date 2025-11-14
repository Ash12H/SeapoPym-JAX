"""Diffusion of a biomass blob - Visualization example.

This example demonstrates diffusion by placing a concentrated biomass blob
at the center of a grid and observing how it spreads over time.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import ray

from seapopym_message import (
    create_distributed_simulation,
    get_global_state,
    initialize_workers,
)
from seapopym_message.core.kernel import Kernel
from seapopym_message.distributed.scheduler import EventScheduler
from seapopym_message.kernels.transport import compute_diffusion_2d
from seapopym_message.utils.grid import GridInfo


def main():
    """Run diffusion simulation and visualize spreading blob."""
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=4)

    print("=== Diffusion Blob Simulation ===\n")

    # Define grid (square domain)
    grid = GridInfo(
        lat_min=-5.0,
        lat_max=5.0,
        lon_min=-5.0,
        lon_max=5.0,
        nlat=50,
        nlon=50,
    )
    print(f"Grid: {grid}")
    print(f"Grid resolution: {grid.nlat}x{grid.nlon} = {grid.nlat * grid.nlon} cells")
    print(f"Grid spacing: dx={grid.dx / 1000:.2f} km, dy={grid.dy / 1000:.2f} km\n")

    # Diffusion kernel
    kernel = Kernel([compute_diffusion_2d])

    # Diffusion parameters
    # Use larger D to see visible diffusion over 24h
    # Ocean mesoscale eddy diffusion: D ~ 10^4 - 10^5 m^2/s
    D = 50000.0  # Diffusion coefficient (m^2/s)

    # CFL stability condition: dt <= dx^2 / (4*D)
    dt_max = (grid.dx**2 / (4 * D)) * 100
    dt = dt_max / (6 * 100)  # Use half of max for safety

    # CFL number check
    cfl = D * dt / grid.dx**2
    print("Diffusion parameters:")
    print(f"  D = {D} m^2/s")
    print(f"  dx = {grid.dx / 1000:.2f} km")
    print(f"  dt_max = {dt_max:.1f} s")
    print(f"  dt (used) = {dt:.1f} s")
    print(f"  CFL number = {cfl:.3f} (should be <= 0.25 for stability)\n")

    params = {"D": D, "dx": grid.dx}

    # Create distributed simulation (2x2 workers)
    print("\nCreating distributed simulation (2x2 workers)...")
    workers, patches = create_distributed_simulation(
        grid=grid,
        kernel=kernel,
        params=params,
        num_workers_lat=2,
        num_workers_lon=2,
        periodic_lon=False,
    )
    print(f"Created {len(workers)} workers\n")

    # Initial condition: Concentrated blob at center
    def blob_initial_state(lat_start, lat_end, lon_start, lon_end):
        """Create initial blob at grid center."""
        # Get global grid coordinates for this patch
        lat_coords = grid.lat_coords[lat_start:lat_end]
        lon_coords = grid.lon_coords[lon_start:lon_end]

        # Create meshgrid
        LON, LAT = jnp.meshgrid(lon_coords, lat_coords)

        # Gaussian blob centered at (0, 0) with small sigma (concentrated)
        lat_center = 0.0
        lon_center = 0.0
        sigma = 0.5  # Very narrow blob (0.5 degrees)

        biomass = 1000.0 * jnp.exp(
            -((LAT - lat_center) ** 2 + (LON - lon_center) ** 2) / (2 * sigma**2)
        )

        return {"biomass": biomass}

    # Initialize workers
    print("Initializing workers with concentrated blob at center...")
    initialize_workers(workers, patches, blob_initial_state)
    initial_state = get_global_state(workers, patches)
    print(
        f"Initial state: mean={float(jnp.mean(initial_state['biomass'])):.2f}, "
        f"max={float(jnp.max(initial_state['biomass'])):.2f}\n"
    )

    # Create figure with subplots for different times
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Times to snapshot (in seconds)
    # With dt ~ 258s, we want: 0, 1hr, 3hr, 6hr, 12hr, 24hr
    # times_to_plot = [0.0, 3600.0, 10800.0, 21600.0, 43200.0, 86400.0]  # In seconds
    times_to_plot = [
        0.0,
        dt * 1 * 100,
        dt * 2 * 100,
        dt * 3 * 100,
        dt * 5 * 100,
        dt * 6 * 100,
    ]  # In seconds
    times_labels = [f"{i} s" for i in times_to_plot]

    print("Running simulation and collecting snapshots...")
    print(f"Total simulation time: {max(times_to_plot) / 3600:.1f} hours")
    print(f"Number of steps: {int(max(times_to_plot) / dt)}\n")

    # Create scheduler
    scheduler = EventScheduler(workers=workers, dt=dt, t_max=max(times_to_plot))

    # Get global coordinates for plotting
    LON, LAT = jnp.meshgrid(grid.lon_coords, grid.lat_coords)

    # Collect snapshots
    snapshots = [initial_state]  # Start with t=0
    snapshot_times = [0.0]

    eps = dt / 10  # Tolerance for time comparison

    while scheduler.get_current_time() < max(times_to_plot) - eps:
        scheduler.step()
        current_time = scheduler.get_current_time()

        # Check if we need a snapshot
        for target_time in times_to_plot[1:]:
            if abs(current_time - target_time) < eps and target_time not in snapshot_times:
                print(f"  Snapshot at t={current_time:.1f}")
                state = get_global_state(workers, patches)
                snapshots.append(state)
                snapshot_times.append(target_time)
                break

    # Plot all snapshots
    vmax = float(jnp.max(initial_state["biomass"]))
    for idx in range(len(snapshots)):
        snapshot = snapshots[idx]
        label = times_labels[idx] if idx < len(times_labels) else f"t={snapshot_times[idx]:.0f}s"

        im = axes[idx].pcolormesh(
            LON, LAT, snapshot["biomass"], cmap="hot", shading="auto", vmin=0, vmax=vmax
        )
        axes[idx].set_title(f"t = {label}", fontsize=14, fontweight="bold")
        axes[idx].set_xlabel("Longitude (°)")
        axes[idx].set_ylabel("Latitude (°)")
        axes[idx].set_aspect("equal")
        axes[idx].grid(True, alpha=0.3)

        # Add text with statistics
        mean_val = float(jnp.mean(snapshot["biomass"]))
        max_val = float(jnp.max(snapshot["biomass"]))
        axes[idx].text(
            0.02,
            0.98,
            f"max={max_val:.1f}\nmean={mean_val:.1f}",
            transform=axes[idx].transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
        )

    # Add colorbar
    fig.colorbar(im, ax=axes, label="Biomass", fraction=0.046, pad=0.04)
    fig.suptitle("Diffusion of Biomass Blob Over Time", fontsize=18, fontweight="bold", y=0.995)
    plt.tight_layout()

    # Save figure (without showing)
    plt.savefig("diffusion_blob_evolution.png", dpi=150, bbox_inches="tight")
    print("\nSaved figure: diffusion_blob_evolution.png")
    plt.close(fig)

    # Final statistics
    final_state = snapshots[-1]
    final_time_hours = snapshot_times[-1] / 3600
    print(f"\nFinal biomass statistics (t={final_time_hours:.1f}h):")
    print(f"  Mean: {float(jnp.mean(final_state['biomass'])):.2f}")
    print(f"  Max: {float(jnp.max(final_state['biomass'])):.2f}")
    print(f"  Std: {float(jnp.std(final_state['biomass'])):.2f}")

    initial_mass = float(jnp.sum(initial_state["biomass"]))
    final_mass = float(jnp.sum(final_state["biomass"]))
    mass_change = abs(final_mass - initial_mass) / initial_mass * 100

    print("\nMass conservation:")
    print(f"  Initial total mass: {initial_mass:.2f}")
    print(f"  Final total mass: {final_mass:.2f}")
    print(f"  Relative change: {mass_change:.4f}%")

    if mass_change < 1.0:
        print("  ✓ Mass is well conserved!")
    else:
        print("  ⚠ Significant mass loss/gain detected")

    # Spreading analysis
    initial_max = float(jnp.max(initial_state["biomass"]))
    final_max = float(jnp.max(final_state["biomass"]))
    print("\nSpreading analysis:")
    print(f"  Initial peak: {initial_max:.2f}")
    print(f"  Final peak: {final_max:.2f}")
    print(f"  Peak reduction: {initial_max / final_max:.2f}x")
    print("  → Blob has spread out due to diffusion")

    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main()
