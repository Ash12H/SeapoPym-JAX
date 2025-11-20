#!/usr/bin/env python3
"""Run comprehensive benchmarks for seapopym-message.

This script benchmarks all core functions with detailed timing analysis.
Results are printed to stdout and optionally saved to JSON.

Usage:
    python benchmarks/run_benchmarks.py
    python benchmarks/run_benchmarks.py --save-json results.json
    python benchmarks/run_benchmarks.py --grid-size 360x720
"""

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
from benchmark_utils import benchmark_jax_function, print_device_info

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


def parse_grid_size(grid_str: str) -> tuple[int, int]:
    """Parse grid size string like '120x360' into (nlat, nlon)."""
    parts = grid_str.split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid grid size format: {grid_str}. Use format: NxM (e.g., 120x360)")
    return int(parts[0]), int(parts[1])


def benchmark_zooplankton(nlat: int = 120, nlon: int = 360, n_iterations: int = 20) -> dict:
    """Benchmark zooplankton functions.

    Args:
        nlat: Number of latitude cells
        nlon: Number of longitude cells
        n_iterations: Number of benchmark iterations

    Returns:
        Dictionary with benchmark results for each function
    """
    print(f"\n{'='*70}")
    print(f"Benchmarking Zooplankton Functions (grid: {nlat}x{nlon})")
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

    results = {}

    # Benchmark age_production
    print("\n1. age_production:")
    result = benchmark_jax_function(
        age_production.func,
        production,
        dt,
        params,
        forcings,
        n_iterations=n_iterations,
        problem_size=nlat * nlon * n_ages,
    )
    print(result)
    results["age_production"] = result.to_dict()

    # Benchmark compute_recruitment
    print("\n2. compute_recruitment:")
    result = benchmark_jax_function(
        compute_recruitment.func,
        production,
        dt,
        params,
        forcings,
        n_iterations=n_iterations,
        problem_size=nlat * nlon * n_ages,
    )
    print(result)
    results["compute_recruitment"] = result.to_dict()

    # Benchmark update_biomass
    print("\n3. update_biomass:")
    result = benchmark_jax_function(
        update_biomass.func,
        biomass,
        recruitment,
        dt,
        {},
        forcings,
        n_iterations=n_iterations,
        problem_size=nlat * nlon,
    )
    print(result)
    results["update_biomass"] = result.to_dict()

    return results


def benchmark_transport(nlat: int = 120, nlon: int = 360, n_iterations: int = 20) -> dict:
    """Benchmark transport functions.

    Args:
        nlat: Number of latitude cells
        nlon: Number of longitude cells
        n_iterations: Number of benchmark iterations

    Returns:
        Dictionary with benchmark results for each function
    """
    print(f"\n{'='*70}")
    print(f"Benchmarking Transport Functions (grid: {nlat}x{nlon})")
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
    u = jnp.ones((nlat, nlon)) * 0.1  # 0.1 m/s eastward
    v = jnp.zeros((nlat, nlon))
    D = 1000.0  # 1000 m²/s
    dt = 3600.0  # 1 hour

    results = {}

    # Benchmark advection
    print("\n1. advection_upwind_flux:")
    result = benchmark_jax_function(
        advection_upwind_flux,
        biomass,
        u,
        v,
        dt,
        grid,
        boundary,
        n_iterations=n_iterations,
        problem_size=nlat * nlon,
    )
    print(result)
    results["advection"] = result.to_dict()

    # Benchmark diffusion
    print("\n2. diffusion_explicit_spherical:")
    result = benchmark_jax_function(
        diffusion_explicit_spherical,
        biomass,
        D,
        dt,
        grid,
        boundary,
        n_iterations=n_iterations,
        problem_size=nlat * nlon,
    )
    print(result)
    results["diffusion"] = result.to_dict()

    # Benchmark combined transport
    print("\n3. Combined transport (advection + diffusion):")

    def full_transport(biomass, u, v, D, dt, grid, boundary):
        biomass_adv = advection_upwind_flux(biomass, u, v, dt, grid, boundary)
        biomass_final = diffusion_explicit_spherical(biomass_adv, D, dt, grid, boundary)
        return biomass_final

    result = benchmark_jax_function(
        full_transport,
        biomass,
        u,
        v,
        D,
        dt,
        grid,
        boundary,
        n_iterations=n_iterations,
        problem_size=nlat * nlon,
    )
    print(result)
    results["full_transport"] = result.to_dict()

    return results


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description="Run seapopym-message benchmarks")
    parser.add_argument(
        "--grid-size",
        type=str,
        default="120x360",
        help="Grid size in format NxM (e.g., 120x360, 360x720)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of benchmark iterations (default: 20)",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--zooplankton-only",
        action="store_true",
        help="Only benchmark zooplankton functions",
    )
    parser.add_argument(
        "--transport-only",
        action="store_true",
        help="Only benchmark transport functions",
    )

    args = parser.parse_args()

    # Parse grid size
    nlat, nlon = parse_grid_size(args.grid_size)

    # Print device info
    print_device_info()

    # Run benchmarks
    all_results = {
        "grid_size": {"nlat": nlat, "nlon": nlon},
        "iterations": args.iterations,
    }

    if not args.transport_only:
        all_results["zooplankton"] = benchmark_zooplankton(nlat, nlon, args.iterations)

    if not args.zooplankton_only:
        all_results["transport"] = benchmark_transport(nlat, nlon, args.iterations)

    # Save to JSON if requested
    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n{'='*70}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*70}")

    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
