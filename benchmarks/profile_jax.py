#!/usr/bin/env python3
"""JAX profiling utilities for detailed performance analysis.

This script provides tools to profile JAX code and identify bottlenecks:
1. JAX built-in profiler (generates traces for TensorBoard)
2. Python cProfile integration
3. Memory profiling

The JAX profiler generates trace files that can be visualized in TensorBoard
or Chrome's trace viewer (chrome://tracing).

Usage:
    # Profile transport step and save trace
    python benchmarks/profile_jax.py --function transport --output-dir ./profiles

    # Profile zooplankton with custom grid size
    python benchmarks/profile_jax.py --function zooplankton --grid-size 360x720

    # View results in TensorBoard
    tensorboard --logdir ./profiles

Requirements:
    - tensorboard: pip install tensorboard
    - For GPU profiling: CUDA toolkit with nvprof
"""

import argparse
import contextlib
from pathlib import Path

import jax
import jax.numpy as jnp
from benchmark_utils import print_device_info

from seapopym_message.kernels.zooplankton import (
    age_production,
    compute_recruitment,
    update_biomass,
)
from seapopym_message.transport.advection import advection_upwind_flux
from seapopym_message.transport.boundary import BoundaryConditions, BoundaryType
from seapopym_message.transport.diffusion import diffusion_explicit_spherical
from seapopym_message.transport.grid import SphericalGrid
from seapopym_message.utils.grid import SphericalGridInfo


@contextlib.contextmanager
def jax_profiler(output_dir: Path, create_perfetto_trace: bool = True):
    """Context manager for JAX profiling.

    Args:
        output_dir: Directory to save profiler output
        create_perfetto_trace: Create perfetto trace for Chrome viewer

    Example:
        >>> with jax_profiler(Path("./profiles")):
        ...     result = my_jax_function(data)
        ...     result.block_until_ready()
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Start profiling
        jax.profiler.start_trace(str(output_dir), create_perfetto_trace=create_perfetto_trace)
        print(f"\nJAX Profiler started. Output directory: {output_dir}")
        print("Executing profiled code...")
        yield
    finally:
        # Stop profiling
        jax.profiler.stop_trace()
        print(f"\nProfiling complete. Results saved to: {output_dir}")
        print("\nTo view results:")
        print(f"  TensorBoard: tensorboard --logdir {output_dir}")
        if create_perfetto_trace:
            print("  Chrome: Open chrome://tracing and load the .json.gz file")


def profile_zooplankton(nlat: int, nlon: int, output_dir: Path) -> None:
    """Profile zooplankton functions.

    Args:
        nlat: Number of latitude cells
        nlon: Number of longitude cells
        output_dir: Directory to save profiler output
    """
    print(f"\n{'='*70}")
    print(f"Profiling Zooplankton Functions (grid: {nlat}x{nlon})")
    print(f"{'='*70}")

    # Setup
    n_ages = 11
    params = {"n_ages": n_ages, "E": 0.1668}
    forcings = {
        "npp": jnp.ones((nlat, nlon)) * 5.0,
        "tau_r": jnp.ones((nlat, nlon)) * 3.45,
        "mortality": jnp.ones((nlat, nlon)) * 0.01,
    }

    production = jnp.zeros((n_ages, nlat, nlon))
    biomass = jnp.ones((nlat, nlon)) * 100.0
    recruitment = jnp.ones((nlat, nlon)) * 5.0
    dt = 1.0

    # Warm-up (JIT compilation)
    print("\nWarm-up phase (JIT compilation)...")
    _ = age_production.func(production, dt, params, forcings)
    _ = compute_recruitment.func(production, dt, params, forcings)
    _ = update_biomass.func(biomass, recruitment, dt, {}, forcings)
    print("Warm-up complete.")

    # Profile execution
    with jax_profiler(output_dir / "zooplankton"):
        # Run multiple iterations for better profiling data
        for _ in range(10):
            production_new = age_production.func(production, dt, params, forcings)
            recruitment_new = compute_recruitment.func(production, dt, params, forcings)
            biomass_new = update_biomass.func(biomass, recruitment_new, dt, {}, forcings)

            # Ensure completion
            production_new.block_until_ready()
            recruitment_new.block_until_ready()
            biomass_new.block_until_ready()

            production = production_new
            biomass = biomass_new


def profile_transport(nlat: int, nlon: int, output_dir: Path) -> None:
    """Profile transport functions.

    Args:
        nlat: Number of latitude cells
        nlon: Number of longitude cells
        output_dir: Directory to save profiler output
    """
    print(f"\n{'='*70}")
    print(f"Profiling Transport Functions (grid: {nlat}x{nlon})")
    print(f"{'='*70}")

    # Setup grid
    grid_info = SphericalGridInfo(
        lat_min=-60.0,
        lat_max=60.0,
        lon_min=0.0,
        lon_max=360.0,
        nlat=nlat,
        nlon=nlon,
    )
    grid = SphericalGrid(grid_info=grid_info, R=6371e3)

    boundary = BoundaryConditions(
        north=BoundaryType.CLOSED,
        south=BoundaryType.CLOSED,
        east=BoundaryType.PERIODIC,
        west=BoundaryType.PERIODIC,
    )

    biomass = jnp.ones((nlat, nlon)) * 100.0
    u = jnp.ones((nlat, nlon)) * 0.1
    v = jnp.zeros((nlat, nlon))
    D = 1000.0
    dt = 3600.0

    # Warm-up
    print("\nWarm-up phase (JIT compilation)...")
    _ = advection_upwind_flux(biomass, u, v, dt, grid, boundary)
    _ = diffusion_explicit_spherical(biomass, D, dt, grid, boundary)
    print("Warm-up complete.")

    # Profile execution
    with jax_profiler(output_dir / "transport"):
        # Run multiple iterations
        for _ in range(10):
            biomass_adv = advection_upwind_flux(biomass, u, v, dt, grid, boundary)
            biomass_adv.block_until_ready()

            biomass_final = diffusion_explicit_spherical(biomass_adv, D, dt, grid, boundary)
            biomass_final.block_until_ready()

            biomass = biomass_final


def profile_full_step(nlat: int, nlon: int, output_dir: Path) -> None:
    """Profile a full simulation step (biology + transport).

    Args:
        nlat: Number of latitude cells
        nlon: Number of longitude cells
        output_dir: Directory to save profiler output
    """
    print(f"\n{'='*70}")
    print(f"Profiling Full Simulation Step (grid: {nlat}x{nlon})")
    print(f"{'='*70}")

    # Setup zooplankton
    n_ages = 11
    zp_params = {"n_ages": n_ages, "E": 0.1668}
    forcings = {
        "npp": jnp.ones((nlat, nlon)) * 5.0,
        "tau_r": jnp.ones((nlat, nlon)) * 3.45,
        "mortality": jnp.ones((nlat, nlon)) * 0.01,
    }

    production = jnp.zeros((n_ages, nlat, nlon))
    biomass = jnp.ones((nlat, nlon)) * 100.0

    # Setup transport
    grid_info = SphericalGridInfo(
        lat_min=-60.0, lat_max=60.0, lon_min=0.0, lon_max=360.0, nlat=nlat, nlon=nlon
    )
    grid = SphericalGrid(grid_info=grid_info, R=6371e3)
    boundary = BoundaryConditions(
        north=BoundaryType.CLOSED,
        south=BoundaryType.CLOSED,
        east=BoundaryType.PERIODIC,
        west=BoundaryType.PERIODIC,
    )

    u = jnp.ones((nlat, nlon)) * 0.1
    v = jnp.zeros((nlat, nlon))
    D = 1000.0
    dt = 1.0
    dt_transport = 3600.0

    # Warm-up
    print("\nWarm-up phase...")
    _ = compute_recruitment.func(production, dt, zp_params, forcings)
    _ = age_production.func(production, dt, zp_params, forcings)
    _ = update_biomass.func(biomass, jnp.ones((nlat, nlon)) * 5.0, dt, {}, forcings)
    _ = advection_upwind_flux(biomass, u, v, dt_transport, grid, boundary)
    _ = diffusion_explicit_spherical(biomass, D, dt_transport, grid, boundary)
    print("Warm-up complete.")

    # Profile full simulation step
    with jax_profiler(output_dir / "full_step"):
        for _ in range(5):
            # Biology phase
            recruitment = compute_recruitment.func(production, dt, zp_params, forcings)
            recruitment.block_until_ready()

            production = age_production.func(production, dt, zp_params, forcings)
            production.block_until_ready()

            biomass = update_biomass.func(biomass, recruitment, dt, {}, forcings)
            biomass.block_until_ready()

            # Transport phase
            biomass = advection_upwind_flux(biomass, u, v, dt_transport, grid, boundary)
            biomass.block_until_ready()

            biomass = diffusion_explicit_spherical(biomass, D, dt_transport, grid, boundary)
            biomass.block_until_ready()


def main():
    """Main profiler runner."""
    parser = argparse.ArgumentParser(description="Profile seapopym-message with JAX profiler")
    parser.add_argument(
        "--function",
        type=str,
        choices=["zooplankton", "transport", "full"],
        default="full",
        help="Which component to profile (default: full)",
    )
    parser.add_argument(
        "--grid-size",
        type=str,
        default="120x360",
        help="Grid size in format NxM (default: 120x360)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./profiles",
        help="Output directory for profiler results (default: ./profiles)",
    )

    args = parser.parse_args()

    # Parse grid size
    nlat, nlon = map(int, args.grid_size.split("x"))
    output_dir = Path(args.output_dir)

    # Print device info
    print_device_info()

    # Check JAX profiler availability
    print("\nJAX Profiler Status:")
    print(f"  JAX version: {jax.__version__}")
    print(f"  Profiler available: {hasattr(jax, 'profiler')}")

    # Run profiling
    if args.function == "zooplankton":
        profile_zooplankton(nlat, nlon, output_dir)
    elif args.function == "transport":
        profile_transport(nlat, nlon, output_dir)
    elif args.function == "full":
        profile_full_step(nlat, nlon, output_dir)

    print("\n" + "=" * 70)
    print("Profiling complete!")
    print("=" * 70)
    print("\nView results with:")
    print(f"  tensorboard --logdir {output_dir}")
    print("\nOr open the .json.gz trace file in Chrome at chrome://tracing")


if __name__ == "__main__":
    main()
