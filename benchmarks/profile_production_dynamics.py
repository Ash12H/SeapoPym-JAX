"""Line-by-line profiling of compute_production_dynamics using line_profiler.

This script uses the line_profiler package to identify exactly which lines
consume the most time in the compute_production_dynamics function.

Install line_profiler:
    uv pip install line_profiler
"""

import numpy as np
import xarray as xr
from line_profiler import LineProfiler

from seapopym.lmtl.core import compute_production_dynamics


def generate_test_data(ny: int = 500, nx: int = 500, n_cohorts: int = 12):
    """Generate synthetic test data."""
    lats = np.linspace(-20, 20, ny)
    lons = np.linspace(0, 40, nx)

    # Cohort ages in seconds
    cohort_ages_vals = np.arange(0, n_cohorts) * 86400.0
    cohort_ages = xr.DataArray(
        cohort_ages_vals,
        dims=["cohort"],
        coords={"cohort": cohort_ages_vals},
    )

    # Production field
    production_vals = np.random.rand(ny, nx, n_cohorts) * 10.0 + 1.0
    production = xr.DataArray(
        production_vals,
        dims=["y", "x", "cohort"],
        coords={
            "y": lats,
            "x": lons,
            "cohort": cohort_ages_vals,
        },
    )

    # Recruitment age
    recruitment_age_vals = np.random.rand(ny, nx) * 5 * 86400.0 + 3 * 86400.0
    recruitment_age = xr.DataArray(
        recruitment_age_vals,
        dims=["y", "x"],
        coords={"y": lats, "x": lons},
    )

    dt = 3600.0

    return production, recruitment_age, cohort_ages, dt


def main():
    """Profile compute_production_dynamics line by line."""
    print("=" * 80)
    print("LINE-BY-LINE PROFILING: compute_production_dynamics")
    print("=" * 80)
    print()

    # Generate test data
    print("Generating test data (500×500 grid, 12 cohorts)...")
    production, recruitment_age, cohort_ages, dt = generate_test_data(ny=500, nx=500, n_cohorts=12)
    print("✅ Test data generated")
    print()

    # Create profiler
    profiler = LineProfiler()

    # Add the function to profile
    profiler.add_function(compute_production_dynamics)

    # Run with profiling
    print("Running profiled execution (10 iterations)...")
    wrapped = profiler(compute_production_dynamics)

    # Execute multiple times for better statistics
    for i in range(10):
        result = wrapped(
            production=production,
            recruitment_age=recruitment_age,
            cohort_ages=cohort_ages,
            dt=dt,
        )
        if i == 0:
            print(f"  Iteration {i+1}/10 completed (verifying output)")
            print(f"    - production_tendency shape: {result['production_tendency'].shape}")
            print(f"    - recruitment_source shape: {result['recruitment_source'].shape}")
        else:
            print(f"  Iteration {i+1}/10 completed")

    print()
    print("=" * 80)
    print("PROFILING RESULTS")
    print("=" * 80)
    print()

    # Print the statistics
    profiler.print_stats(output_unit=1e-3)  # Show times in milliseconds

    print()
    print("=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print()
    print("Column descriptions:")
    print("  Line # : Line number in the source file")
    print("  Hits   : Number of times this line was executed")
    print("  Time   : Total time spent on this line (ms)")
    print("  Per Hit: Average time per execution (ms)")
    print("  % Time : Percentage of total function time")
    print()
    print("Look for lines with:")
    print("  - High % Time : These are the bottlenecks")
    print("  - High Per Hit: These operations are inherently slow")
    print()


if __name__ == "__main__":
    main()
