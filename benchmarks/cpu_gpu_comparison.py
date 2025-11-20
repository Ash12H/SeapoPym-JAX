#!/usr/bin/env python3
"""CPU vs GPU performance comparison for JAX functions.

This script benchmarks functions on both CPU and GPU (if available)
to determine the speedup from using GPU acceleration.

Usage:
    python benchmarks/cpu_gpu_comparison.py
    python benchmarks/cpu_gpu_comparison.py --grid-size 360x720
    python benchmarks/cpu_gpu_comparison.py --save-json gpu_comparison.json

Note: Requires JAX with GPU support. Install with:
    pip install "jax[cuda12]"
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from benchmark_utils import benchmark_jax_function

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


def get_available_devices() -> dict:
    """Get list of available devices.

    Returns:
        Dictionary with device information
    """
    devices = jax.devices()
    cpu_devices = [d for d in devices if d.platform == "cpu"]
    gpu_devices = [d for d in devices if d.platform == "gpu"]

    return {
        "cpu_count": len(cpu_devices),
        "gpu_count": len(gpu_devices),
        "cpu_devices": [str(d) for d in cpu_devices],
        "gpu_devices": [str(d) for d in gpu_devices],
        "has_gpu": len(gpu_devices) > 0,
    }


def benchmark_on_device(func, device, *args, n_iterations=20, **kwargs):
    """Benchmark a function on a specific device.

    Args:
        func: Function to benchmark
        device: JAX device to use
        *args, **kwargs: Function arguments
        n_iterations: Number of iterations

    Returns:
        BenchmarkResult
    """
    # Transfer data to device
    with jax.default_device(device):
        # Move arrays to device
        device_args = jax.tree.map(lambda x: jnp.array(x) if hasattr(x, "__array__") else x, args)
        device_kwargs = jax.tree.map(
            lambda x: jnp.array(x) if hasattr(x, "__array__") else x, kwargs
        )

        # Benchmark
        result = benchmark_jax_function(
            func, *device_args, n_iterations=n_iterations, **device_kwargs
        )

    return result


def compare_zooplankton(nlat: int, nlon: int, n_iterations: int = 20) -> dict:
    """Compare zooplankton functions on CPU vs GPU.

    Args:
        nlat: Number of latitude cells
        nlon: Number of longitude cells
        n_iterations: Number of benchmark iterations

    Returns:
        Dictionary with comparison results
    """
    print(f"\n{'='*70}")
    print(f"CPU vs GPU: Zooplankton (grid: {nlat}x{nlon})")
    print(f"{'='*70}")

    devices_info = get_available_devices()

    if not devices_info["has_gpu"]:
        print("\nWARNING: No GPU detected. Only CPU benchmarks will run.")
        print("Install JAX with GPU support: pip install 'jax[cuda12]'")
        return {"error": "No GPU available"}

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

    cpu_device = jax.devices("cpu")[0]
    gpu_device = jax.devices("gpu")[0]

    results = {}

    # Benchmark age_production
    print("\n1. age_production:")
    print("   CPU...", end=" ", flush=True)
    cpu_result = benchmark_on_device(
        age_production.func,
        cpu_device,
        production,
        1.0,
        params,
        forcings,
        n_iterations=n_iterations,
    )
    print(f"{cpu_result.mean * 1000:.2f} ms")

    print("   GPU...", end=" ", flush=True)
    gpu_result = benchmark_on_device(
        age_production.func,
        gpu_device,
        production,
        1.0,
        params,
        forcings,
        n_iterations=n_iterations,
    )
    print(f"{gpu_result.mean * 1000:.2f} ms")

    speedup = cpu_result.mean / gpu_result.mean
    print(f"   Speedup: {speedup:.2f}x")

    results["age_production"] = {
        "cpu_ms": cpu_result.mean * 1000,
        "gpu_ms": gpu_result.mean * 1000,
        "speedup": speedup,
    }

    # Benchmark compute_recruitment
    print("\n2. compute_recruitment:")
    print("   CPU...", end=" ", flush=True)
    cpu_result = benchmark_on_device(
        compute_recruitment.func,
        cpu_device,
        production,
        1.0,
        params,
        forcings,
        n_iterations=n_iterations,
    )
    print(f"{cpu_result.mean * 1000:.2f} ms")

    print("   GPU...", end=" ", flush=True)
    gpu_result = benchmark_on_device(
        compute_recruitment.func,
        gpu_device,
        production,
        1.0,
        params,
        forcings,
        n_iterations=n_iterations,
    )
    print(f"{gpu_result.mean * 1000:.2f} ms")

    speedup = cpu_result.mean / gpu_result.mean
    print(f"   Speedup: {speedup:.2f}x")

    results["compute_recruitment"] = {
        "cpu_ms": cpu_result.mean * 1000,
        "gpu_ms": gpu_result.mean * 1000,
        "speedup": speedup,
    }

    # Benchmark update_biomass
    print("\n3. update_biomass:")
    print("   CPU...", end=" ", flush=True)
    cpu_result = benchmark_on_device(
        update_biomass.func,
        cpu_device,
        biomass,
        recruitment,
        1.0,
        {},
        forcings,
        n_iterations=n_iterations,
    )
    print(f"{cpu_result.mean * 1000:.2f} ms")

    print("   GPU...", end=" ", flush=True)
    gpu_result = benchmark_on_device(
        update_biomass.func,
        gpu_device,
        biomass,
        recruitment,
        1.0,
        {},
        forcings,
        n_iterations=n_iterations,
    )
    print(f"{gpu_result.mean * 1000:.2f} ms")

    speedup = cpu_result.mean / gpu_result.mean
    print(f"   Speedup: {speedup:.2f}x")

    results["update_biomass"] = {
        "cpu_ms": cpu_result.mean * 1000,
        "gpu_ms": gpu_result.mean * 1000,
        "speedup": speedup,
    }

    return results


def compare_transport(nlat: int, nlon: int, n_iterations: int = 20) -> dict:
    """Compare transport functions on CPU vs GPU.

    Args:
        nlat: Number of latitude cells
        nlon: Number of longitude cells
        n_iterations: Number of benchmark iterations

    Returns:
        Dictionary with comparison results
    """
    print(f"\n{'='*70}")
    print(f"CPU vs GPU: Transport (grid: {nlat}x{nlon})")
    print(f"{'='*70}")

    devices_info = get_available_devices()
    if not devices_info["has_gpu"]:
        return {"error": "No GPU available"}

    # Setup
    grid_info = SphericalGridInfo(
        lat_min=-60.0, lat_max=60.0, lon_min=0.0, lon_max=360.0, nlat=nlat, nlon=nlon
    )
    grid = SphericalGrid(grid_info=grid_info)
    boundary = BoundaryConditions(
        BoundaryType.CLOSED, BoundaryType.CLOSED, BoundaryType.PERIODIC, BoundaryType.PERIODIC
    )

    biomass = jnp.ones((nlat, nlon)) * 100.0
    u = jnp.ones((nlat, nlon)) * 0.1
    v = jnp.zeros((nlat, nlon))
    D = 1000.0
    dt = 3600.0

    cpu_device = jax.devices("cpu")[0]
    gpu_device = jax.devices("gpu")[0]

    results = {}

    # Benchmark advection
    print("\n1. advection_upwind_flux:")
    print("   CPU...", end=" ", flush=True)
    cpu_result = benchmark_on_device(
        advection_upwind_flux,
        cpu_device,
        biomass,
        u,
        v,
        dt,
        grid,
        boundary,
        n_iterations=n_iterations,
    )
    print(f"{cpu_result.mean * 1000:.2f} ms")

    print("   GPU...", end=" ", flush=True)
    gpu_result = benchmark_on_device(
        advection_upwind_flux,
        gpu_device,
        biomass,
        u,
        v,
        dt,
        grid,
        boundary,
        n_iterations=n_iterations,
    )
    print(f"{gpu_result.mean * 1000:.2f} ms")

    speedup = cpu_result.mean / gpu_result.mean
    print(f"   Speedup: {speedup:.2f}x")

    results["advection"] = {
        "cpu_ms": cpu_result.mean * 1000,
        "gpu_ms": gpu_result.mean * 1000,
        "speedup": speedup,
    }

    # Benchmark diffusion
    print("\n2. diffusion_explicit_spherical:")
    print("   CPU...", end=" ", flush=True)
    cpu_result = benchmark_on_device(
        diffusion_explicit_spherical,
        cpu_device,
        biomass,
        D,
        dt,
        grid,
        boundary,
        n_iterations=n_iterations,
    )
    print(f"{cpu_result.mean * 1000:.2f} ms")

    print("   GPU...", end=" ", flush=True)
    gpu_result = benchmark_on_device(
        diffusion_explicit_spherical,
        gpu_device,
        biomass,
        D,
        dt,
        grid,
        boundary,
        n_iterations=n_iterations,
    )
    print(f"{gpu_result.mean * 1000:.2f} ms")

    speedup = cpu_result.mean / gpu_result.mean
    print(f"   Speedup: {speedup:.2f}x")

    results["diffusion"] = {
        "cpu_ms": cpu_result.mean * 1000,
        "gpu_ms": gpu_result.mean * 1000,
        "speedup": speedup,
    }

    return results


def print_summary(zooplankton_results: dict, transport_results: dict):
    """Print summary of CPU vs GPU comparison.

    Args:
        zooplankton_results: Results from zooplankton comparison
        transport_results: Results from transport comparison
    """
    print(f"\n{'='*70}")
    print("SUMMARY: CPU vs GPU Speedup")
    print(f"{'='*70}\n")

    all_results = {**zooplankton_results, **transport_results}

    for func_name, data in all_results.items():
        if "speedup" in data:
            speedup = data["speedup"]
            interpretation = ""
            if speedup > 10:
                interpretation = "Excellent GPU acceleration!"
            elif speedup > 5:
                interpretation = "Good GPU speedup"
            elif speedup > 2:
                interpretation = "Moderate GPU benefit"
            elif speedup > 1:
                interpretation = "Marginal GPU benefit"
            else:
                interpretation = "GPU slower (overhead dominates)"

            print(f"{func_name}:")
            print(f"  CPU: {data['cpu_ms']:.2f} ms")
            print(f"  GPU: {data['gpu_ms']:.2f} ms")
            print(f"  Speedup: {speedup:.2f}x ({interpretation})")
            print()


def main():
    """Main CPU vs GPU comparison."""
    parser = argparse.ArgumentParser(description="Compare CPU vs GPU performance")
    parser.add_argument(
        "--grid-size",
        type=str,
        default="120x360",
        help="Grid size (format: NxM)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    # Parse grid size
    nlat, nlon = map(int, args.grid_size.split("x"))

    # Get device info
    devices_info = get_available_devices()
    print("Available devices:")
    print(f"  CPU: {devices_info['cpu_count']} device(s)")
    print(f"  GPU: {devices_info['gpu_count']} device(s)")

    if not devices_info["has_gpu"]:
        print("\nERROR: No GPU detected!")
        print("This script requires JAX with GPU support.")
        print("Install with: pip install 'jax[cuda12]'")
        return

    # Run comparisons
    all_results = {
        "grid_size": {"nlat": nlat, "nlon": nlon},
        "devices": devices_info,
    }

    zooplankton_results = compare_zooplankton(nlat, nlon, args.iterations)
    all_results["zooplankton"] = zooplankton_results

    transport_results = compare_transport(nlat, nlon, args.iterations)
    all_results["transport"] = transport_results

    # Print summary
    print_summary(zooplankton_results, transport_results)

    # Save results
    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
