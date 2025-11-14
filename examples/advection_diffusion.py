"""Combined advection-diffusion example.

This example demonstrates the combination of advection and diffusion in a single
simulation, showing how a biomass blob is both transported by currents (advection)
and spread by turbulent mixing (diffusion).

This uses operator splitting:
1. Advection step (directional transport)
2. Diffusion step (isotropic spreading)
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
from seapopym_message.kernels.transport import (
    check_cfl_condition,
    compute_advection_2d,
    compute_diffusion_2d,
)
from seapopym_message.utils.grid import GridInfo


def create_uniform_currents(grid: GridInfo, u_vel: float, v_vel: float) -> xr.Dataset:
    """Create uniform velocity field.

    Args:
        grid: Grid information.
        u_vel: Uniform zonal velocity (m/s).
        v_vel: Uniform meridional velocity (m/s).

    Returns:
        xarray Dataset with u and v velocity fields.
    """
    times = np.array([0.0, 1e9])
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


def main():
    """Run combined advection-diffusion simulation."""
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=4, runtime_env=None)

    print("\n=== Combined Advection-Diffusion Simulation ===\n")

    # Define grid
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

    # Create velocity field (uniform eastward flow)
    u_vel = 0.5  # m/s
    v_vel = 0.0
    currents_ds = create_uniform_currents(grid, u_vel, v_vel)
    print(f"Velocity field: Uniform flow (u={u_vel} m/s, v={v_vel} m/s)")

    # Create ForcingManager
    forcing_manager = ForcingManager(
        {
            "u": ForcingConfig(source=currents_ds, dims=["time", "lat", "lon"], units="m/s"),
            "v": ForcingConfig(source=currents_ds, dims=["time", "lat", "lon"], units="m/s"),
        }
    )

    # Diffusion coefficient (moderate mixing)
    D = 10000.0  # m²/s
    print(f"Diffusion coefficient: D = {D} m²/s")

    # Check CFL conditions
    forcings_t0 = forcing_manager.prepare_timestep(time=0.0)
    u_array = forcings_t0["u"]
    v_array = forcings_t0["v"]

    # Advection CFL
    max_velocity = max(float(np.max(np.abs(u_array))), float(np.max(np.abs(v_array))))
    dt_adv_max = min(grid.dx, grid.dy) / max_velocity if max_velocity > 0 else 1e6

    # Diffusion CFL
    dt_diff_max = (grid.dx**2 / (4 * D)) * 0.25  # Conservative for 2D

    # Use the more restrictive condition
    dt = min(dt_adv_max, dt_diff_max) * 0.5
    print("\nCFL analysis:")
    print(f"  Advection dt_max: {dt_adv_max:.1f} s")
    print(f"  Diffusion dt_max: {dt_diff_max:.1f} s")
    print(f"  Using dt = {dt:.1f} s")

    # Verify CFL
    cfl_adv = check_cfl_condition(jnp.array(u_array), jnp.array(v_array), dt, grid.dx, grid.dy)
    cfl_diff = D * dt / grid.dx**2
    print(f"  Advection CFL: {cfl_adv['cfl']:.3f} (should be ≤ 1.0)")
    print(f"  Diffusion CFL: {cfl_diff:.3f} (should be ≤ 0.25)")

    if cfl_adv["stable"] and cfl_diff <= 0.25:
        print("  ✓ Both CFL conditions satisfied\n")
    else:
        print("  ⚠ WARNING: CFL condition violated!\n")

    # Create kernel with BOTH processes (operator splitting)
    # Order: advection first, then diffusion
    kernel = Kernel([compute_advection_2d, compute_diffusion_2d])
    params = {"dx": grid.dx, "dy": grid.dy, "D": D}

    # Create distributed simulation
    print("Creating distributed simulation (2x2 workers)...")
    workers, patches = create_distributed_simulation(
        grid=grid,
        kernel=kernel,
        params=params,
        num_workers_lat=2,
        num_workers_lon=2,
        periodic_lon=False,
    )
    print(f"Created {len(workers)} workers\n")

    # Initial condition: Blob on the left side
    def blob_initial_state(lat_start, lat_end, lon_start, lon_end):
        """Create initial blob."""
        lat_coords = grid.lat_coords[lat_start:lat_end]
        lon_coords = grid.lon_coords[lon_start:lon_end]
        LON, LAT = jnp.meshgrid(lon_coords, lat_coords)

        lat_center = 0.0
        lon_center = -3.0
        sigma = 0.5

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

    # Create figure with 3 scenarios for comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    times_to_plot = [0.0, 900.0, 1800.0, 2700.0, 3600.0, 5400.0]  # 0, 15min, 30min, 45min, 1h, 1.5h
    times_labels = [f"{t / 60:.0f}min" if t < 3600 else f"{t / 3600:.1f}h" for t in times_to_plot]

    print("Running simulation and collecting snapshots...")
    print(f"Total simulation time: {max(times_to_plot) / 3600:.1f} hours")
    print(f"Number of steps: {int(max(times_to_plot) / dt)}\n")

    # Create scheduler with forcing
    scheduler = EventScheduler(
        workers=workers, dt=dt, t_max=max(times_to_plot), forcing_manager=forcing_manager
    )

    # Get global coordinates
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
                print(f"  Snapshot at t={current_time / 60:.0f}min")
                state = get_global_state(workers, patches)
                snapshots.append(state)
                snapshot_times.append(target_time)
                break

    # Plot all snapshots
    axes = axes.flatten()
    vmax = float(jnp.max(initial_state["biomass"]))

    for idx in range(len(snapshots)):
        snapshot = snapshots[idx]
        label = (
            times_labels[idx] if idx < len(times_labels) else f"t={snapshot_times[idx] / 60:.0f}min"
        )

        im = axes[idx].pcolormesh(
            LON, LAT, snapshot["biomass"], cmap="plasma", shading="auto", vmin=0, vmax=vmax
        )
        axes[idx].set_title(f"t = {label}", fontsize=14, fontweight="bold")
        axes[idx].set_xlabel("Longitude (°)")
        axes[idx].set_ylabel("Latitude (°)")
        axes[idx].set_aspect("equal")
        axes[idx].grid(True, alpha=0.3)

        # Add velocity arrow on first subplot
        if idx == 0:
            axes[idx].annotate(
                "",
                xy=(-2, 4),
                xytext=(-3, 4),
                arrowprops={"arrowstyle": "->", "lw": 2, "color": "white"},
            )
            axes[idx].text(
                -2.5, 4.5, f"u={u_vel} m/s", color="white", fontweight="bold", fontsize=10
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
    fig.suptitle(
        f"Combined Advection-Diffusion (u={u_vel} m/s, D={D} m²/s)",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    # Save figure
    filename = "advection_diffusion_combined.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure: {filename}")
    plt.close(fig)

    # Analysis
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
        print("  ✓ Mass is reasonably conserved")
    else:
        print("  ⚠ Significant mass loss/gain detected")

    # Compare center of mass displacement
    def compute_center_of_mass(state, LON, LAT):
        """Compute center of mass."""
        biomass = state["biomass"]
        total_mass = jnp.sum(biomass)
        lon_cm = jnp.sum(LON * biomass) / total_mass
        lat_cm = jnp.sum(LAT * biomass) / total_mass
        return float(lon_cm), float(lat_cm)

    lon_cm_initial, lat_cm_initial = compute_center_of_mass(initial_state, LON, LAT)
    lon_cm_final, lat_cm_final = compute_center_of_mass(final_state, LON, LAT)

    distance_km = (
        np.sqrt((lon_cm_final - lon_cm_initial) ** 2 + (lat_cm_final - lat_cm_initial) ** 2) * 111.0
    )

    print("\nCenter of mass displacement:")
    print(f"  Initial: ({lon_cm_initial:.2f}°, {lat_cm_initial:.2f}°)")
    print(f"  Final: ({lon_cm_final:.2f}°, {lat_cm_final:.2f}°)")
    print(f"  Distance: {distance_km:.1f} km")
    print(f"  Expected (advection only): {u_vel * max(times_to_plot) / 1000:.1f} km")
    print("  → Blob is both advected (eastward) and diffused (spread)")

    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main()
