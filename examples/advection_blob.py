"""Advection of a biomass blob - Visualization example.

This example demonstrates advection by placing a concentrated biomass blob
and observing how it moves with ocean currents.

Two scenarios:
1. Uniform eastward flow (translation)
2. Rotational flow (circular motion)
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import ray
import xarray as xr

from seapopym_message import (
    create_distributed_simulation,
    get_global_state,
    initialize_workers,
)
from seapopym_message.core.kernel import Kernel
from seapopym_message.distributed.scheduler import EventScheduler
from seapopym_message.forcing import ForcingConfig, ForcingManager
from seapopym_message.kernels.transport import check_cfl_condition, compute_advection_2d
from seapopym_message.utils.grid import GridInfo


def create_uniform_currents(grid: GridInfo, u_vel: float, v_vel: float) -> xr.Dataset:
    """Create uniform velocity field for simple translation.

    Args:
        grid: Grid information.
        u_vel: Uniform zonal velocity (m/s).
        v_vel: Uniform meridional velocity (m/s).

    Returns:
        xarray Dataset with u and v velocity fields.
    """
    times = np.array([0.0, 1e9])  # Very large time range
    u_data = np.ones((len(times), grid.nlat, grid.nlon)) * u_vel
    v_data = np.ones((len(times), grid.nlat, grid.nlon)) * v_vel

    return xr.Dataset(
        {
            "u": (["time", "lat", "lon"], u_data),
            "v": (["time", "lat", "lon"], v_data),
        },
        coords={
            "time": times,
            "lat": grid.lat_coords,
            "lon": grid.lon_coords,
        },
    )


def create_rotational_currents(grid: GridInfo, omega: float) -> xr.Dataset:
    """Create rotational velocity field for circular motion.

    Args:
        grid: Grid information.
        omega: Angular velocity (rad/s).

    Returns:
        xarray Dataset with u and v velocity fields.
    """
    times = np.array([0.0, 1e9])  # Very large time range

    # Create meshgrid
    LON, LAT = np.meshgrid(grid.lon_coords, grid.lat_coords)

    # Rotational velocity: v = omega × r
    # u = -omega * y, v = omega * x (in Cartesian)
    # For small angles in degrees, approximate as:
    # Center at (0, 0)
    u_rot = -omega * LAT * 111000.0  # ~111 km/degree latitude
    v_rot = omega * LON * 111000.0 * np.cos(np.deg2rad(LAT))  # Adjusted for latitude

    # Repeat for time dimension
    u_data = np.stack([u_rot, u_rot])
    v_data = np.stack([v_rot, v_rot])

    return xr.Dataset(
        {
            "u": (["time", "lat", "lon"], u_data),
            "v": (["time", "lat", "lon"], v_data),
        },
        coords={
            "time": times,
            "lat": grid.lat_coords,
            "lon": grid.lon_coords,
        },
    )


def run_advection_simulation(scenario: str = "uniform"):
    """Run advection simulation with specified scenario.

    Args:
        scenario: "uniform" for translation or "rotation" for circular motion.
    """
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=4, runtime_env=None)

    print(f"\n=== Advection Blob Simulation ({scenario}) ===\n")

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

    # Create velocity fields based on scenario
    if scenario == "uniform":
        # Uniform eastward flow at 0.5 m/s
        u_vel = 1.0  # m/s eastward
        v_vel = 0.0  # m/s
        currents_ds = create_uniform_currents(grid, u_vel, v_vel)
        print(f"Velocity field: Uniform eastward flow (u={u_vel} m/s, v={v_vel} m/s)")
    else:
        # Rotational flow
        omega = 1e-5  # rad/s (weak rotation)
        currents_ds = create_rotational_currents(grid, omega)
        print(f"Velocity field: Rotational flow (omega={omega} rad/s)")

    # Create ForcingManager
    forcing_manager = ForcingManager(
        {
            "u": ForcingConfig(source=currents_ds, dims=["time", "lat", "lon"], units="m/s"),
            "v": ForcingConfig(source=currents_ds, dims=["time", "lat", "lon"], units="m/s"),
        }
    )

    # Get initial forcings to check CFL
    forcings_t0 = forcing_manager.prepare_timestep(time=0.0)
    u_array = forcings_t0["u"]
    v_array = forcings_t0["v"]

    # CFL stability condition for advection: CFL = max(|u|, |v|) * dt / min(dx, dy) <= 1
    max_u = float(np.max(np.abs(u_array)))
    max_v = float(np.max(np.abs(v_array)))

    # Use hardcoded timestep for consistent visualization
    dt_max = (60 * 60 * 24) * 30  # 30 days max
    dt = (60 * 60 * 24) / 24  # 1 hour timestep

    # CFL check
    cfl_result = check_cfl_condition(jnp.array(u_array), jnp.array(v_array), dt, grid.dx, grid.dy)
    print("\nAdvection parameters:")
    print(f"  max|u| = {max_u:.3f} m/s")
    print(f"  max|v| = {max_v:.3f} m/s")
    print(f"  dx = {grid.dx / 1000:.2f} km")
    print(f"  dt_max = {dt_max:.1f} s")
    print(f"  dt (used) = {dt:.1f} s")
    print(f"  CFL number = {cfl_result['cfl']:.3f} (should be <= 1.0 for stability)")
    if not cfl_result["stable"]:
        print("  ⚠ WARNING: CFL condition violated!")
    else:
        print("  ✓ CFL condition satisfied")

    # Advection kernel
    kernel = Kernel([compute_advection_2d])
    params = {"dx": grid.dx, "dy": grid.dy}

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

    # Initial condition: Concentrated blob
    def blob_initial_state(lat_start, lat_end, lon_start, lon_end):
        """Create initial blob."""
        lat_coords = grid.lat_coords[lat_start:lat_end]
        lon_coords = grid.lon_coords[lon_start:lon_end]
        LON, LAT = jnp.meshgrid(lon_coords, lat_coords)

        if scenario == "uniform":
            # Start on the left side for eastward advection
            lat_center = 0.0
            lon_center = -3.0
        else:
            # Start off-center for rotation
            lat_center = 0.0
            lon_center = 2.0

        sigma = 0.5  # Narrow blob

        biomass = 1000.0 * jnp.exp(
            -((LAT - lat_center) ** 2 + (LON - lon_center) ** 2) / (2 * sigma**2)
        )

        return {"biomass": biomass}

    # Initialize workers
    print("Initializing workers with concentrated blob...")
    initialize_workers(workers, patches, blob_initial_state)
    initial_state = get_global_state(workers, patches)
    print(
        f"Initial state: mean={float(jnp.mean(initial_state['biomass'])):.2f}, "
        f"max={float(jnp.max(initial_state['biomass'])):.2f}\n"
    )

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    times_to_plot = [
        0.0,
        dt * 2,
        dt * 4,
        dt * 8,
        dt * 16,
        dt_max,
    ]

    times_labels = [f"{t / 3600:.1f}h" if t > 0 else "0" for t in times_to_plot]

    print("Running simulation and collecting snapshots...")
    print(f"Total simulation time: {max(times_to_plot) / 3600:.1f} hours")
    print(f"Number of steps: {int(max(times_to_plot) / dt)}\n")

    # Create scheduler with forcing
    scheduler = EventScheduler(
        workers=workers, dt=dt, t_max=max(times_to_plot), forcing_manager=forcing_manager
    )

    # Get global coordinates for plotting
    LON, LAT = jnp.meshgrid(grid.lon_coords, grid.lat_coords)

    # Collect snapshots
    snapshots = [initial_state]
    snapshot_times = [0.0]
    eps = dt / 10

    while scheduler.get_current_time() < max(times_to_plot) - eps:
        scheduler.step()
        current_time = scheduler.get_current_time()

        for target_time in times_to_plot[1:]:
            if abs(current_time - target_time) < eps and target_time not in snapshot_times:
                print(f"  Snapshot at t={current_time / 3600:.1f}h")
                state = get_global_state(workers, patches)
                snapshots.append(state)
                snapshot_times.append(target_time)
                break

    # Plot all snapshots
    vmax = float(jnp.max(initial_state["biomass"]))
    for idx in range(len(snapshots)):
        snapshot = snapshots[idx]
        label = (
            times_labels[idx] if idx < len(times_labels) else f"t={snapshot_times[idx] / 3600:.1f}h"
        )

        im = axes[idx].pcolormesh(
            LON, LAT, snapshot["biomass"], cmap="viridis", shading="auto", vmin=0, vmax=vmax
        )
        axes[idx].set_title(f"t = {label}", fontsize=14, fontweight="bold")
        axes[idx].set_xlabel("Longitude (°)")
        axes[idx].set_ylabel("Latitude (°)")
        axes[idx].set_aspect("equal")
        axes[idx].grid(True, alpha=0.3)

        # Add velocity field quiver plot (subsample for clarity)
        if idx == 0:
            step = 5
            u_plot = u_array[::step, ::step]
            v_plot = v_array[::step, ::step]
            lon_plot = grid.lon_coords[::step]
            lat_plot = grid.lat_coords[::step]
            LON_plot, LAT_plot = np.meshgrid(lon_plot, lat_plot)
            axes[idx].quiver(
                LON_plot,
                LAT_plot,
                u_plot,
                v_plot,
                alpha=0.6,
                scale=5,
                width=0.004,
                color="white",
            )

        # Statistics
        mean_val = float(jnp.mean(snapshot["biomass"]))
        max_val = float(jnp.max(snapshot["biomass"]))
        axes[idx].text(
            0.02,
            0.98,
            f"max={max_val:.1f}\nmean={mean_val:.1f}",
            transform=axes[idx].transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    # Colorbar
    fig.colorbar(im, ax=axes, label="Biomass", fraction=0.046, pad=0.04)
    title = (
        "Advection: Uniform Translation" if scenario == "uniform" else "Advection: Rotational Flow"
    )
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.995)
    plt.tight_layout()

    # Save figure
    filename = f"advection_blob_{scenario}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure: {filename}")
    plt.close(fig)

    # Statistics
    final_state = snapshots[-1]
    final_time_hours = snapshot_times[-1] / 3600
    print(f"\nFinal biomass statistics (t={final_time_hours:.1f}h):")
    print(f"  Mean: {float(jnp.mean(final_state['biomass'])):.2f}")
    print(f"  Max: {float(jnp.max(final_state['biomass'])):.2f}")

    initial_mass = float(jnp.sum(initial_state["biomass"]))
    final_mass = float(jnp.sum(final_state["biomass"]))
    mass_change = abs(final_mass - initial_mass) / initial_mass * 100

    print("\nMass conservation:")
    print(f"  Initial total mass: {initial_mass:.2f}")
    print(f"  Final total mass: {final_mass:.2f}")
    print(f"  Relative change: {mass_change:.4f}%")

    if mass_change < 5.0:
        print("  ✓ Mass is reasonably conserved (upwind has some numerical diffusion)")
    else:
        print("  ⚠ Significant mass loss/gain detected")

    # Shutdown Ray
    ray.shutdown()


def main():
    """Run both scenarios."""
    # Scenario 1: Uniform eastward flow
    run_advection_simulation(scenario="uniform")

    print("\n" + "=" * 60 + "\n")

    # Scenario 2: Rotational flow
    run_advection_simulation(scenario="rotation")


if __name__ == "__main__":
    main()
