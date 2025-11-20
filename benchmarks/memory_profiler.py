#!/usr/bin/env python3
"""Memory profiling tools for JAX functions.

This script measures memory consumption (RAM and VRAM) of JAX computations:
- Peak memory usage
- Memory allocated by JAX arrays
- System memory vs JAX memory
- Memory scaling with grid size

Usage:
    python benchmarks/memory_profiler.py
    python benchmarks/memory_profiler.py --grid-size 360x720
    python benchmarks/memory_profiler.py --save-json memory_results.json
"""

import argparse
import gc
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import psutil

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


def get_process_memory() -> dict:
    """Get current process memory usage.

    Returns:
        Dictionary with memory info in MB
    """
    process = psutil.Process()
    mem_info = process.memory_info()

    return {
        "rss_mb": mem_info.rss / 1024**2,  # Resident Set Size
        "vms_mb": mem_info.vms / 1024**2,  # Virtual Memory Size
    }


def get_jax_array_memory(array: jnp.ndarray) -> float:
    """Estimate memory used by a JAX array.

    Args:
        array: JAX array

    Returns:
        Memory in MB
    """
    # JAX arrays have dtype and shape
    bytes_per_element = array.dtype.itemsize
    total_bytes = np.prod(array.shape) * bytes_per_element
    return total_bytes / 1024**2


def profile_function_memory(func, *args, **kwargs) -> dict:
    """Profile memory usage of a function.

    Args:
        func: Function to profile
        *args, **kwargs: Arguments to pass to function

    Returns:
        Dictionary with memory metrics
    """
    # Force garbage collection
    gc.collect()

    # Get baseline memory
    mem_before = get_process_memory()

    # Run function
    result = func(*args, **kwargs)
    if hasattr(result, "block_until_ready"):
        result.block_until_ready()

    # Get memory after
    mem_after = get_process_memory()

    # Calculate increase
    memory_delta = {
        "rss_delta_mb": mem_after["rss_mb"] - mem_before["rss_mb"],
        "vms_delta_mb": mem_after["vms_mb"] - mem_before["vms_mb"],
        "rss_before_mb": mem_before["rss_mb"],
        "rss_after_mb": mem_after["rss_mb"],
    }

    # Estimate result memory
    if hasattr(result, "shape") and hasattr(result, "dtype"):
        memory_delta["result_array_mb"] = get_jax_array_memory(result)

    return memory_delta


def profile_zooplankton_memory(nlat: int, nlon: int) -> dict:
    """Profile memory usage of zooplankton functions.

    Args:
        nlat: Number of latitude cells
        nlon: Number of longitude cells

    Returns:
        Dictionary with memory profiles
    """
    print(f"\n{'='*70}")
    print(f"Memory Profiling: Zooplankton (grid: {nlat}x{nlon})")
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

    # Calculate theoretical array sizes
    production_size_mb = get_jax_array_memory(production)
    biomass_size_mb = get_jax_array_memory(biomass)

    results["theoretical"] = {
        "production_mb": production_size_mb,
        "biomass_mb": biomass_size_mb,
        "total_working_set_mb": production_size_mb + biomass_size_mb * 2,
    }

    print("\nTheoretical array sizes:")
    print(f"  production (11 ages): {production_size_mb:.2f} MB")
    print(f"  biomass: {biomass_size_mb:.2f} MB")

    # Profile age_production
    print("\n1. age_production:")
    mem = profile_function_memory(age_production.func, production, dt, params, forcings)
    print(f"   Memory delta: {mem['rss_delta_mb']:+.2f} MB")
    print(f"   Result size: {mem.get('result_array_mb', 0):.2f} MB")
    results["age_production"] = mem

    # Profile compute_recruitment
    print("\n2. compute_recruitment:")
    mem = profile_function_memory(compute_recruitment.func, production, dt, params, forcings)
    print(f"   Memory delta: {mem['rss_delta_mb']:+.2f} MB")
    print(f"   Result size: {mem.get('result_array_mb', 0):.2f} MB")
    results["compute_recruitment"] = mem

    # Profile update_biomass
    print("\n3. update_biomass:")
    mem = profile_function_memory(update_biomass.func, biomass, recruitment, dt, {}, forcings)
    print(f"   Memory delta: {mem['rss_delta_mb']:+.2f} MB")
    print(f"   Result size: {mem.get('result_array_mb', 0):.2f} MB")
    results["update_biomass"] = mem

    return results


def profile_transport_memory(nlat: int, nlon: int) -> dict:
    """Profile memory usage of transport functions.

    Args:
        nlat: Number of latitude cells
        nlon: Number of longitude cells

    Returns:
        Dictionary with memory profiles
    """
    print(f"\n{'='*70}")
    print(f"Memory Profiling: Transport (grid: {nlat}x{nlon})")
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

    results = {}

    # Theoretical sizes
    biomass_size_mb = get_jax_array_memory(biomass)
    results["theoretical"] = {
        "biomass_mb": biomass_size_mb,
        "velocity_mb": get_jax_array_memory(u),
        "total_working_set_mb": biomass_size_mb + get_jax_array_memory(u) * 2,
    }

    print("\nTheoretical array sizes:")
    print(f"  biomass: {biomass_size_mb:.2f} MB")
    print(f"  velocity (u/v): {get_jax_array_memory(u):.2f} MB each")

    # Profile advection
    print("\n1. advection_upwind_flux:")
    mem = profile_function_memory(advection_upwind_flux, biomass, u, v, dt, grid, boundary)
    print(f"   Memory delta: {mem['rss_delta_mb']:+.2f} MB")
    print(f"   Result size: {mem.get('result_array_mb', 0):.2f} MB")
    results["advection"] = mem

    # Profile diffusion
    print("\n2. diffusion_explicit_spherical:")
    mem = profile_function_memory(diffusion_explicit_spherical, biomass, D, dt, grid, boundary)
    print(f"   Memory delta: {mem['rss_delta_mb']:+.2f} MB")
    print(f"   Result size: {mem.get('result_array_mb', 0):.2f} MB")
    results["diffusion"] = mem

    return results


def analyze_memory_scaling(grid_sizes: list[tuple[int, int]]) -> dict:
    """Analyze how memory scales with grid size.

    Args:
        grid_sizes: List of (nlat, nlon) tuples

    Returns:
        Dictionary with scaling analysis
    """
    print(f"\n{'='*70}")
    print("Memory Scaling Analysis")
    print(f"{'='*70}")

    results = {
        "grid_sizes": [],
        "total_cells": [],
        "age_production_mem_mb": [],
        "advection_mem_mb": [],
        "theoretical_biomass_mb": [],
    }

    for nlat, nlon in grid_sizes:
        total_cells = nlat * nlon
        print(f"\nGrid: {nlat}x{nlon} ({total_cells:,} cells)")

        results["grid_sizes"].append(f"{nlat}x{nlon}")
        results["total_cells"].append(total_cells)

        # Theoretical biomass size
        biomass = jnp.ones((nlat, nlon))
        theoretical_mb = get_jax_array_memory(biomass)
        results["theoretical_biomass_mb"].append(theoretical_mb)

        # Quick profile of one function from each category
        n_ages = 11
        production = jnp.zeros((n_ages, nlat, nlon))
        params = {"n_ages": n_ages, "E": 0.1668}
        forcings = {
            "npp": jnp.ones((nlat, nlon)) * 5.0,
            "tau_r": jnp.ones((nlat, nlon)) * 3.45,
        }

        mem_zp = profile_function_memory(age_production.func, production, 1.0, params, forcings)
        results["age_production_mem_mb"].append(mem_zp["rss_delta_mb"])

        # Transport
        grid_info = SphericalGridInfo(
            lat_min=-60.0, lat_max=60.0, lon_min=0.0, lon_max=360.0, nlat=nlat, nlon=nlon
        )
        grid = SphericalGrid(grid_info=grid_info)
        boundary = BoundaryConditions(
            BoundaryType.CLOSED, BoundaryType.CLOSED, BoundaryType.PERIODIC, BoundaryType.PERIODIC
        )
        u = jnp.ones((nlat, nlon)) * 0.1
        v = jnp.zeros((nlat, nlon))

        mem_adv = profile_function_memory(
            advection_upwind_flux, biomass, u, v, 3600.0, grid, boundary
        )
        results["advection_mem_mb"].append(mem_adv["rss_delta_mb"])

        print(f"  age_production: {mem_zp['rss_delta_mb']:+.2f} MB")
        print(f"  advection: {mem_adv['rss_delta_mb']:+.2f} MB")

    return results


def main():
    """Main memory profiler."""
    parser = argparse.ArgumentParser(description="Profile memory usage")
    parser.add_argument(
        "--grid-size",
        type=str,
        default="120x360",
        help="Grid size (format: NxM)",
    )
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="Run memory scaling analysis",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    # Parse grid size
    nlat, nlon = map(int, args.grid_size.split("x"))

    # Print device info
    print(f"JAX backend: {jax.devices()[0].platform}")
    print(f"System memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")

    all_results = {
        "grid_size": {"nlat": nlat, "nlon": nlon},
        "device": str(jax.devices()[0]),
    }

    if args.scaling:
        # Scaling analysis
        grid_sizes = [(30, 60), (60, 120), (120, 360), (180, 540)]
        all_results["scaling"] = analyze_memory_scaling(grid_sizes)
    else:
        # Single grid profiling
        all_results["zooplankton"] = profile_zooplankton_memory(nlat, nlon)
        all_results["transport"] = profile_transport_memory(nlat, nlon)

    # Save results
    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
