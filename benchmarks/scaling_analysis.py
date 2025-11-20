#!/usr/bin/env python3
"""Scaling analysis for seapopym-message functions.

This script analyzes how performance scales with problem size (grid resolution).
It runs benchmarks at multiple grid sizes and analyzes the scaling behavior.

Outputs:
- Console report with scaling metrics
- Optional CSV file with raw data
- Optional plots (requires matplotlib)

Usage:
    python benchmarks/scaling_analysis.py
    python benchmarks/scaling_analysis.py --save-csv scaling_results.csv
    python benchmarks/scaling_analysis.py --plot --save-plot scaling.png
"""

import argparse
import csv
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from benchmark_utils import benchmark_jax_function, print_device_info

from seapopym_message.kernels.zooplankton import age_production
from seapopym_message.transport.advection import advection_upwind_flux
from seapopym_message.transport.boundary import BoundaryConditions, BoundaryType
from seapopym_message.transport.diffusion import diffusion_explicit_spherical
from seapopym_message.transport.grid import SphericalGrid
from seapopym_message.utils.grid import SphericalGridInfo


def analyze_scaling(grid_sizes: list[tuple[int, int]], n_iterations: int = 10) -> dict:
    """Analyze performance scaling across different grid sizes.

    Args:
        grid_sizes: List of (nlat, nlon) tuples to test
        n_iterations: Number of iterations per benchmark

    Returns:
        Dictionary with scaling results for each function
    """
    results = {
        "grid_sizes": [],
        "total_cells": [],
        "age_production": [],
        "advection": [],
        "diffusion": [],
    }

    for nlat, nlon in grid_sizes:
        print(f"\n{'='*70}")
        print(f"Testing grid size: {nlat}x{nlon} ({nlat * nlon:,} cells)")
        print(f"{'='*70}")

        total_cells = nlat * nlon
        results["grid_sizes"].append(f"{nlat}x{nlon}")
        results["total_cells"].append(total_cells)

        # Test age_production (zooplankton)
        print("\n1. Testing age_production...")
        n_ages = 11
        params = {"n_ages": n_ages, "E": 0.1668}
        forcings = {
            "npp": jnp.ones((nlat, nlon)) * 5.0,
            "tau_r": jnp.ones((nlat, nlon)) * 3.45,
            "mortality": jnp.ones((nlat, nlon)) * 0.01,
        }
        production = jnp.zeros((n_ages, nlat, nlon))

        result = benchmark_jax_function(
            age_production.func, production, 1.0, params, forcings, n_iterations=n_iterations
        )
        results["age_production"].append(result.mean)
        print(f"   Mean time: {result.mean * 1000:.3f} ms")

        # Test advection
        print("\n2. Testing advection_upwind_flux...")
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

        biomass = jnp.ones((nlat, nlon)) * 100.0
        u = jnp.ones((nlat, nlon)) * 0.1
        v = jnp.zeros((nlat, nlon))
        dt = 3600.0

        result = benchmark_jax_function(
            advection_upwind_flux, biomass, u, v, dt, grid, boundary, n_iterations=n_iterations
        )
        results["advection"].append(result.mean)
        print(f"   Mean time: {result.mean * 1000:.3f} ms")

        # Test diffusion
        print("\n3. Testing diffusion_explicit_spherical...")
        D = 1000.0

        result = benchmark_jax_function(
            diffusion_explicit_spherical, biomass, D, dt, grid, boundary, n_iterations=n_iterations
        )
        results["diffusion"].append(result.mean)
        print(f"   Mean time: {result.mean * 1000:.3f} ms")

    return results


def compute_scaling_metrics(results: dict) -> dict:
    """Compute scaling efficiency metrics.

    Analyzes how well performance scales with problem size.
    Ideal scaling: time ∝ n_cells (linear)

    Args:
        results: Results from analyze_scaling()

    Returns:
        Dictionary with scaling analysis for each function
    """
    n_cells = np.array(results["total_cells"])
    metrics = {}

    for func_name in ["age_production", "advection", "diffusion"]:
        times = np.array(results[func_name])

        # Normalize by smallest problem
        normalized_times = times / times[0]
        normalized_cells = n_cells / n_cells[0]

        # Compute scaling exponent by fitting: time ∝ n_cells^α
        # log(time) = α × log(n_cells) + const
        log_cells = np.log(normalized_cells)
        log_times = np.log(normalized_times)
        α = np.polyfit(log_cells, log_times, 1)[0]

        # Compute efficiency: ideal is 1.0 (linear scaling)
        # α = 1.0: perfect linear scaling
        # α < 1.0: super-linear (better than expected, e.g., cache effects)
        # α > 1.0: sub-linear (worse than expected, e.g., memory bandwidth)
        efficiency = 1.0 / α if α > 0 else float("inf")

        # Throughput (cells/second)
        throughput = n_cells / times

        metrics[func_name] = {
            "scaling_exponent": α,
            "efficiency": efficiency,
            "throughput": throughput.tolist(),
            "times_ms": (times * 1000).tolist(),
        }

    return metrics


def print_scaling_report(results: dict, metrics: dict) -> None:
    """Print formatted scaling analysis report."""
    print(f"\n{'='*70}")
    print("SCALING ANALYSIS REPORT")
    print(f"{'='*70}\n")

    for func_name in ["age_production", "advection", "diffusion"]:
        func_metrics = metrics[func_name]
        α = func_metrics["scaling_exponent"]
        efficiency = func_metrics["efficiency"]

        print(f"{func_name}:")
        print(f"  Scaling exponent (α): {α:.3f}")
        print(f"  Interpretation: time ∝ n_cells^{α:.3f}")

        if α < 0.95:
            interpretation = "Super-linear (excellent!)"
        elif α < 1.05:
            interpretation = "Linear (ideal)"
        elif α < 1.15:
            interpretation = "Slightly sub-linear (good)"
        elif α < 1.3:
            interpretation = "Sub-linear (acceptable)"
        else:
            interpretation = "Poor scaling (investigate!)"

        print(f"  Scaling quality: {interpretation}")
        print(f"  Efficiency: {efficiency:.2%}\n")

        # Print table
        print(f"  {'Grid Size':<12} {'Cells':>12} {'Time (ms)':>12} {'Throughput':>15}")
        print(f"  {'-'*55}")
        for i, grid_size in enumerate(results["grid_sizes"]):
            cells = results["total_cells"][i]
            time_ms = func_metrics["times_ms"][i]
            throughput = func_metrics["throughput"][i]
            print(f"  {grid_size:<12} {cells:12,} {time_ms:12.3f} {throughput:15.2e}")
        print()


def save_csv(results: dict, metrics: dict, output_path: Path) -> None:
    """Save scaling results to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(
            [
                "grid_size",
                "nlat",
                "nlon",
                "total_cells",
                "age_production_ms",
                "advection_ms",
                "diffusion_ms",
                "age_production_throughput",
                "advection_throughput",
                "diffusion_throughput",
            ]
        )

        # Data rows
        for i, grid_size in enumerate(results["grid_sizes"]):
            nlat, nlon = map(int, grid_size.split("x"))
            cells = results["total_cells"][i]

            row = [
                grid_size,
                nlat,
                nlon,
                cells,
                metrics["age_production"]["times_ms"][i],
                metrics["advection"]["times_ms"][i],
                metrics["diffusion"]["times_ms"][i],
                metrics["age_production"]["throughput"][i],
                metrics["advection"]["throughput"][i],
                metrics["diffusion"]["throughput"][i],
            ]
            writer.writerow(row)

    print(f"\nResults saved to: {output_path}")


def plot_scaling(results: dict, metrics: dict, output_path: Path | None = None) -> None:
    """Generate scaling plots.

    Args:
        results: Scaling results
        metrics: Scaling metrics
        output_path: Optional path to save plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nWarning: matplotlib not installed. Skipping plots.")
        print("Install with: pip install matplotlib")
        return

    n_cells = np.array(results["total_cells"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Execution time vs grid size
    for func_name, label in [
        ("age_production", "age_production"),
        ("advection", "advection"),
        ("diffusion", "diffusion"),
    ]:
        times_ms = metrics[func_name]["times_ms"]
        ax1.loglog(n_cells, times_ms, "o-", label=label, linewidth=2, markersize=8)

    ax1.set_xlabel("Number of grid cells", fontsize=12)
    ax1.set_ylabel("Execution time (ms)", fontsize=12)
    ax1.set_title("Scaling Analysis: Execution Time", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which="both")

    # Plot 2: Throughput vs grid size
    for func_name, label in [
        ("age_production", "age_production"),
        ("advection", "advection"),
        ("diffusion", "diffusion"),
    ]:
        throughput = metrics[func_name]["throughput"]
        ax2.semilogx(
            n_cells, np.array(throughput) / 1e6, "o-", label=label, linewidth=2, markersize=8
        )

    ax2.set_xlabel("Number of grid cells", fontsize=12)
    ax2.set_ylabel("Throughput (Mcells/s)", fontsize=12)
    ax2.set_title("Scaling Analysis: Throughput", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()


def main():
    """Main scaling analysis runner."""
    parser = argparse.ArgumentParser(description="Analyze performance scaling")
    parser.add_argument(
        "--grid-sizes",
        type=str,
        default="30x60,60x120,120x360,180x540",
        help="Comma-separated list of grid sizes (format: NxM)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations per grid size",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        help="Save results to CSV file",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate scaling plots",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        help="Save plot to file (implies --plot)",
    )

    args = parser.parse_args()

    # Parse grid sizes
    grid_sizes = []
    for size_str in args.grid_sizes.split(","):
        nlat, nlon = map(int, size_str.strip().split("x"))
        grid_sizes.append((nlat, nlon))

    # Print device info
    print_device_info()

    # Run scaling analysis
    print(f"\nRunning scaling analysis with {len(grid_sizes)} grid sizes...")
    results = analyze_scaling(grid_sizes, n_iterations=args.iterations)

    # Compute metrics
    metrics = compute_scaling_metrics(results)

    # Print report
    print_scaling_report(results, metrics)

    # Save CSV if requested
    if args.save_csv:
        save_csv(results, metrics, Path(args.save_csv))

    # Generate plots if requested
    if args.plot or args.save_plot:
        output_path = Path(args.save_plot) if args.save_plot else None
        plot_scaling(results, metrics, output_path)


if __name__ == "__main__":
    main()
