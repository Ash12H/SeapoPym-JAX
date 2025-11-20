#!/usr/bin/env python3
"""Ray overhead analysis for distributed simulation.

This script measures the overhead introduced by Ray:
- Worker communication (put/get objects)
- Task scheduling and dispatch
- Serialization/deserialization
- Comparison: local execution vs Ray distributed

Usage:
    python benchmarks/ray_overhead_analysis.py
    python benchmarks/ray_overhead_analysis.py --grid-size 120x360
    python benchmarks/ray_overhead_analysis.py --num-workers 4
    python benchmarks/ray_overhead_analysis.py --save-json ray_overhead.json
"""

import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import ray
from benchmark_utils import benchmark_jax_function

from seapopym_message.transport.worker import TransportWorker


def measure_ray_put_get(data_size_mb: float, n_iterations: int = 50) -> dict:
    """Measure Ray put/get overhead.

    Args:
        data_size_mb: Size of data to transfer (MB)
        n_iterations: Number of iterations

    Returns:
        Dictionary with timing results
    """
    # Create data
    n_elements = int(data_size_mb * 1024**2 / 8)  # 8 bytes per float64
    data = np.random.randn(n_elements)

    # Measure put
    put_times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        ref = ray.put(data)
        ray.get(ref)  # Ensure put completes
        put_times.append(time.perf_counter() - start)

    # Measure get
    ref = ray.put(data)
    get_times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = ray.get(ref)
        get_times.append(time.perf_counter() - start)

    return {
        "data_size_mb": data_size_mb,
        "put_mean_ms": float(np.mean(put_times) * 1000),
        "put_std_ms": float(np.std(put_times) * 1000),
        "get_mean_ms": float(np.mean(get_times) * 1000),
        "get_std_ms": float(np.std(get_times) * 1000),
        "total_roundtrip_ms": float((np.mean(put_times) + np.mean(get_times)) * 1000),
        "bandwidth_mb_s": data_size_mb / (np.mean(put_times) + np.mean(get_times)),
    }


def measure_remote_task_overhead(n_iterations: int = 100) -> dict:
    """Measure overhead of Ray remote task vs local execution.

    Args:
        n_iterations: Number of iterations

    Returns:
        Dictionary with timing results
    """

    @ray.remote
    def simple_task(x):
        return x + 1

    def local_task(x):
        return x + 1

    # Measure local execution
    x = np.random.randn(100)
    local_times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = local_task(x)
        local_times.append(time.perf_counter() - start)

    # Measure Ray remote execution
    ray_times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        ref = simple_task.remote(x)
        _ = ray.get(ref)
        ray_times.append(time.perf_counter() - start)

    overhead = np.mean(ray_times) - np.mean(local_times)

    return {
        "local_mean_us": float(np.mean(local_times) * 1e6),
        "ray_mean_us": float(np.mean(ray_times) * 1e6),
        "overhead_us": float(overhead * 1e6),
        "overhead_factor": float(np.mean(ray_times) / np.mean(local_times)),
    }


def compare_transport_local_vs_ray(nlat: int, nlon: int, n_iterations: int = 10) -> dict:
    """Compare TransportWorker (Ray) vs local transport execution.

    Args:
        nlat: Number of latitude cells
        nlon: Number of longitude cells
        n_iterations: Number of iterations

    Returns:
        Dictionary with comparison results
    """
    print(f"\n{'='*70}")
    print(f"TransportWorker: Local vs Ray (grid: {nlat}x{nlon})")
    print(f"{'='*70}")

    from seapopym_message.transport.advection import advection_upwind_flux
    from seapopym_message.transport.boundary import BoundaryConditions, BoundaryType
    from seapopym_message.transport.diffusion import diffusion_explicit_spherical
    from seapopym_message.transport.grid import SphericalGrid
    from seapopym_message.utils.grid import SphericalGridInfo

    # Setup grid
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

    # Local execution (no Ray)
    print("\n1. Local execution (no Ray):")

    def local_transport_step(biomass, u, v, D, dt):
        biomass_adv = advection_upwind_flux(biomass, u, v, dt, grid, boundary)
        biomass_final = diffusion_explicit_spherical(biomass_adv, D, dt, grid, boundary)
        return biomass_final

    local_result = benchmark_jax_function(
        local_transport_step, biomass, u, v, D, dt, n_iterations=n_iterations
    )
    print(f"   Mean time: {local_result.mean * 1000:.2f} ms")

    # Ray execution (TransportWorker)
    print("\n2. Ray execution (TransportWorker):")

    # Create TransportWorker
    worker = TransportWorker.remote(
        grid_type="spherical",
        lat_min=-60.0,
        lat_max=60.0,
        lon_min=0.0,
        lon_max=360.0,
        nlat=nlat,
        nlon=nlon,
        lat_bc="closed",
        lon_bc="periodic",
    )

    # Warm-up
    _ = ray.get(worker.transport_step.remote(biomass, u, v, D, dt))

    # Benchmark
    ray_times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result_ref = worker.transport_step.remote(biomass, u, v, D, dt)
        _ = ray.get(result_ref)
        ray_times.append(time.perf_counter() - start)

    ray_mean = float(np.mean(ray_times))
    print(f"   Mean time: {ray_mean * 1000:.2f} ms")

    # Calculate overhead
    overhead = ray_mean - local_result.mean
    overhead_pct = (overhead / local_result.mean) * 100

    print("\n3. Ray overhead:")
    print(f"   Absolute: {overhead * 1000:.2f} ms")
    print(f"   Relative: {overhead_pct:.1f}%")

    return {
        "local_ms": local_result.mean * 1000,
        "ray_ms": ray_mean * 1000,
        "overhead_ms": overhead * 1000,
        "overhead_pct": overhead_pct,
        "slowdown_factor": ray_mean / local_result.mean,
    }


def analyze_data_transfer_scaling(grid_sizes: list[tuple[int, int]]) -> dict:
    """Analyze how Ray overhead scales with data size.

    Args:
        grid_sizes: List of (nlat, nlon) tuples

    Returns:
        Dictionary with scaling results
    """
    print(f"\n{'='*70}")
    print("Ray Data Transfer Scaling")
    print(f"{'='*70}")

    results = {
        "grid_sizes": [],
        "data_size_mb": [],
        "roundtrip_ms": [],
        "bandwidth_mb_s": [],
    }

    for nlat, nlon in grid_sizes:
        # Calculate data size (biomass + 2 velocity fields)
        biomass = jnp.ones((nlat, nlon))
        data_size_mb = (biomass.size * biomass.dtype.itemsize * 3) / 1024**2

        print(f"\nGrid {nlat}x{nlon} ({data_size_mb:.2f} MB)")

        transfer_metrics = measure_ray_put_get(data_size_mb, n_iterations=30)

        results["grid_sizes"].append(f"{nlat}x{nlon}")
        results["data_size_mb"].append(data_size_mb)
        results["roundtrip_ms"].append(transfer_metrics["total_roundtrip_ms"])
        results["bandwidth_mb_s"].append(transfer_metrics["bandwidth_mb_s"])

        print(f"  Roundtrip: {transfer_metrics['total_roundtrip_ms']:.2f} ms")
        print(f"  Bandwidth: {transfer_metrics['bandwidth_mb_s']:.1f} MB/s")

    return results


def main():
    """Main Ray overhead analysis."""
    parser = argparse.ArgumentParser(description="Analyze Ray overhead")
    parser.add_argument(
        "--grid-size",
        type=str,
        default="120x360",
        help="Grid size (format: NxM)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of Ray workers",
    )
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="Run data transfer scaling analysis",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=args.num_workers)

    print(f"Ray initialized with {args.num_workers} CPU(s)")
    print(f"Ray version: {ray.__version__}")

    # Parse grid size
    nlat, nlon = map(int, args.grid_size.split("x"))

    all_results = {
        "grid_size": {"nlat": nlat, "nlon": nlon},
        "num_workers": args.num_workers,
    }

    # Basic overhead measurements
    print(f"\n{'='*70}")
    print("Basic Ray Overhead Measurements")
    print(f"{'='*70}")

    print("\n1. Remote task overhead:")
    task_overhead = measure_remote_task_overhead()
    print(f"   Local execution: {task_overhead['local_mean_us']:.2f} µs")
    print(f"   Ray execution: {task_overhead['ray_mean_us']:.2f} µs")
    print(
        f"   Overhead: {task_overhead['overhead_us']:.2f} µs ({task_overhead['overhead_factor']:.2f}x)"
    )
    all_results["task_overhead"] = task_overhead

    print("\n2. Data transfer overhead:")
    biomass = jnp.ones((nlat, nlon))
    data_size_mb = (biomass.size * biomass.dtype.itemsize) / 1024**2
    transfer_metrics = measure_ray_put_get(data_size_mb)
    print(f"   Data size: {data_size_mb:.2f} MB")
    print(f"   Put: {transfer_metrics['put_mean_ms']:.2f} ms")
    print(f"   Get: {transfer_metrics['get_mean_ms']:.2f} ms")
    print(f"   Roundtrip: {transfer_metrics['total_roundtrip_ms']:.2f} ms")
    print(f"   Bandwidth: {transfer_metrics['bandwidth_mb_s']:.1f} MB/s")
    all_results["data_transfer"] = transfer_metrics

    # TransportWorker comparison
    transport_comparison = compare_transport_local_vs_ray(nlat, nlon, n_iterations=10)
    all_results["transport_comparison"] = transport_comparison

    # Scaling analysis
    if args.scaling:
        grid_sizes = [(30, 60), (60, 120), (120, 360), (180, 540)]
        scaling_results = analyze_data_transfer_scaling(grid_sizes)
        all_results["scaling"] = scaling_results

    # Save results
    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    # Cleanup
    ray.shutdown()


if __name__ == "__main__":
    main()
