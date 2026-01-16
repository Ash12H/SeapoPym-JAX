"""Inspect chunking behavior of compute_production_dynamics.

This script verifies that the function:
1. Preserves input chunking structure
2. Does not create unwanted rechunking
3. Does not trigger transpositions
4. Maintains reasonable Dask graph size
"""

import dask.array as da
import numpy as np
import xarray as xr

from seapopym.lmtl.core import compute_production_dynamics


def visualize_chunks(data: xr.DataArray, name: str) -> None:
    """Pretty print chunking information."""
    if not hasattr(data.data, "chunks"):
        print(f"  {name}: Not a Dask array (numpy array)")
        return

    chunks = data.data.chunks
    print(f"  {name}:")
    print(f"    Shape: {data.shape}")
    print(f"    Chunks: {chunks}")
    print(f"    Num chunks per dim: {[len(c) for c in chunks]}")
    print(f"    Total chunks: {np.prod([len(c) for c in chunks])}")
    print(f"    Chunk sizes: {[c[0] if len(set(c)) == 1 else 'variable' for c in chunks]}")


def inspect_graph_size(data: xr.DataArray, name: str) -> int:
    """Inspect Dask graph size."""
    if not hasattr(data.data, "__dask_graph__"):
        return 0

    graph = data.data.__dask_graph__()
    num_tasks = len(graph)
    print(f"  {name}: {num_tasks} tasks in graph")
    return num_tasks


def test_chunking_preservation():
    """Test that chunking is preserved through the function."""
    print("=" * 80)
    print("TEST 1: CHUNKING PRESERVATION")
    print("=" * 80)
    print()

    # Setup
    ny, nx, n_cohorts = 500, 500, 12

    # Test different chunking strategies
    chunking_strategies = [
        {
            "name": "Chunk cohorts only (for strong scaling)",
            "chunks": {"y": -1, "x": -1, "cohort": 1},
        },
        {
            "name": "Chunk spatial",
            "chunks": {"y": 100, "x": 100, "cohort": -1},
        },
        {
            "name": "Chunk all dimensions",
            "chunks": {"y": 100, "x": 100, "cohort": 1},
        },
    ]

    for strategy in chunking_strategies:
        print(f"\n{'=' * 80}")
        print(f"Strategy: {strategy['name']}")
        print(f"Chunks: {strategy['chunks']}")
        print(f"{'=' * 80}")
        print()

        chunks = strategy["chunks"]

        # Create chunked inputs
        lats = np.linspace(-20, 20, ny)
        lons = np.linspace(0, 40, nx)
        cohort_ages_vals = np.arange(0, n_cohorts) * 86400.0

        # Production (3D) - chunked
        production_np = np.random.rand(ny, nx, n_cohorts) * 10.0 + 1.0
        production_dask = da.from_array(
            production_np,
            chunks=(
                chunks.get("y", -1) if chunks.get("y", -1) > 0 else ny,
                chunks.get("x", -1) if chunks.get("x", -1) > 0 else nx,
                chunks.get("cohort", -1) if chunks.get("cohort", -1) > 0 else n_cohorts,
            ),
        )
        production = xr.DataArray(
            production_dask,
            dims=["y", "x", "cohort"],
            coords={
                "y": lats,
                "x": lons,
                "cohort": cohort_ages_vals,
            },
        )

        # Recruitment age (2D) - chunked
        recruitment_age_np = np.random.rand(ny, nx) * 5 * 86400.0 + 3 * 86400.0
        recruitment_age_dask = da.from_array(
            recruitment_age_np,
            chunks=(
                chunks.get("y", -1) if chunks.get("y", -1) > 0 else ny,
                chunks.get("x", -1) if chunks.get("x", -1) > 0 else nx,
            ),
        )
        recruitment_age = xr.DataArray(
            recruitment_age_dask,
            dims=["y", "x"],
            coords={"y": lats, "x": lons},
        )

        # Cohort ages (1D) - not chunked (coordinate)
        cohort_ages = xr.DataArray(
            cohort_ages_vals,
            dims=["cohort"],
            coords={"cohort": cohort_ages_vals},
        )

        dt = 3600.0

        print("INPUT CHUNKING:")
        visualize_chunks(production, "production")
        visualize_chunks(recruitment_age, "recruitment_age")
        print("  cohort_ages: 1D coordinate (not chunked)")
        print()

        # Execute function
        print("Executing compute_production_dynamics...")
        result = compute_production_dynamics(
            production=production,
            recruitment_age=recruitment_age,
            cohort_ages=cohort_ages,
            dt=dt,
        )
        print("✅ Execution complete")
        print()

        # Inspect outputs
        print("OUTPUT CHUNKING:")
        visualize_chunks(result["production_tendency"], "production_tendency")
        visualize_chunks(result["recruitment_source"], "recruitment_source")
        print()

        # Check for rechunking
        print("CHUNKING ANALYSIS:")
        prod_input_chunks = production.data.chunks
        prod_output_chunks = result["production_tendency"].data.chunks

        if prod_input_chunks == prod_output_chunks:
            print("  ✅ production_tendency preserves input chunking exactly")
        else:
            print("  ⚠️  production_tendency has different chunking:")
            print(f"     Input:  {prod_input_chunks}")
            print(f"     Output: {prod_output_chunks}")

        # Check recruitment_source dimensions (should be 2D)
        if result["recruitment_source"].ndim == 2:
            print("  ✅ recruitment_source is 2D (cohort dimension reduced)")
        else:
            print(
                f"  ❌ recruitment_source has {result['recruitment_source'].ndim} dimensions (expected 2)"
            )

        print()

        # Graph size analysis
        print("DASK GRAPH SIZE:")
        tasks_input = inspect_graph_size(production, "production (input)")
        tasks_output = inspect_graph_size(
            result["production_tendency"], "production_tendency (output)"
        )
        _ = inspect_graph_size(result["recruitment_source"], "recruitment_source (output)")

        if tasks_output > 0:
            ratio = tasks_output / tasks_input if tasks_input > 0 else float("inf")
            print(f"  Graph expansion ratio: {ratio:.2f}×")
            if ratio > 10:
                print("  ⚠️  WARNING: Large graph expansion detected!")
            elif ratio > 5:
                print("  ⚠️  Moderate graph expansion")
            else:
                print("  ✅ Reasonable graph size")
        print()


def test_transpose_detection():
    """Test for unwanted transpose operations."""
    print("=" * 80)
    print("TEST 2: TRANSPOSE DETECTION")
    print("=" * 80)
    print()

    ny, nx, n_cohorts = 100, 100, 12

    # Create input with specific dimension order
    lats = np.linspace(-20, 20, ny)
    lons = np.linspace(0, 40, nx)
    cohort_ages_vals = np.arange(0, n_cohorts) * 86400.0

    production_np = np.random.rand(ny, nx, n_cohorts) * 10.0 + 1.0
    production = xr.DataArray(
        production_np,
        dims=["y", "x", "cohort"],  # Specific order
        coords={
            "y": lats,
            "x": lons,
            "cohort": cohort_ages_vals,
        },
    )

    recruitment_age = xr.DataArray(
        np.random.rand(ny, nx) * 5 * 86400.0 + 3 * 86400.0,
        dims=["y", "x"],
        coords={"y": lats, "x": lons},
    )

    cohort_ages = xr.DataArray(
        cohort_ages_vals,
        dims=["cohort"],
        coords={"cohort": cohort_ages_vals},
    )

    print("Input dimension order:")
    print(f"  production: {production.dims}")
    print(f"  recruitment_age: {recruitment_age.dims}")
    print()

    result = compute_production_dynamics(
        production=production,
        recruitment_age=recruitment_age,
        cohort_ages=cohort_ages,
        dt=3600.0,
    )

    print("Output dimension order:")
    print(f"  production_tendency: {result['production_tendency'].dims}")
    print(f"  recruitment_source: {result['recruitment_source'].dims}")
    print()

    # Check for transpose
    if production.dims == result["production_tendency"].dims:
        print("✅ No transpose detected - dimension order preserved")
    else:
        print("⚠️  Dimension order changed:")
        print(f"   Input:  {production.dims}")
        print(f"   Output: {result['production_tendency'].dims}")
        print("   This may indicate a transpose operation")
    print()


def test_shift_operation_behavior():
    """Test how .shift() affects chunking."""
    print("=" * 80)
    print("TEST 3: .shift() OPERATION BEHAVIOR")
    print("=" * 80)
    print()

    print("Testing .shift(cohort=1) with chunks={'cohort': 1}...")
    print()

    # Create a simple chunked array
    ny, nx, n_cohorts = 100, 100, 12

    data_np = np.arange(ny * nx * n_cohorts).reshape(ny, nx, n_cohorts)
    data_dask = da.from_array(data_np, chunks=(ny, nx, 1))  # Chunk each cohort

    data_xr = xr.DataArray(
        data_dask,
        dims=["y", "x", "cohort"],
        coords={
            "y": np.arange(ny),
            "x": np.arange(nx),
            "cohort": np.arange(n_cohorts),
        },
    )

    print("Before .shift():")
    visualize_chunks(data_xr, "data")
    tasks_before = len(data_xr.data.__dask_graph__())
    print(f"  Dask tasks: {tasks_before}")
    print()

    # Apply shift
    shifted = data_xr.shift(cohort=1, fill_value=0.0)

    print("After .shift(cohort=1):")
    visualize_chunks(shifted, "shifted")
    tasks_after = len(shifted.data.__dask_graph__())
    print(f"  Dask tasks: {tasks_after}")
    print(
        f"  Task increase: {tasks_after - tasks_before} (+{(tasks_after/tasks_before - 1)*100:.1f}%)"
    )
    print()

    # Check if chunks are preserved
    if data_xr.data.chunks == shifted.data.chunks:
        print("✅ .shift() preserves chunking structure")
    else:
        print("⚠️  .shift() changed chunking structure")
        print(f"   Before: {data_xr.data.chunks}")
        print(f"   After:  {shifted.data.chunks}")
    print()

    # Check for rechunking in graph
    graph = shifted.data.__dask_graph__()
    graph_str = str(graph)
    if "rechunk" in graph_str.lower():
        print("⚠️  WARNING: 'rechunk' operation detected in graph")
    else:
        print("✅ No explicit rechunking detected")
    print()


def main():
    """Run all chunking inspection tests."""
    print()
    print("=" * 80)
    print("CHUNKING BEHAVIOR INSPECTION")
    print("compute_production_dynamics")
    print("=" * 80)
    print()

    test_chunking_preservation()
    test_transpose_detection()
    test_shift_operation_behavior()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Key findings:")
    print("1. Check if chunking is preserved through the function")
    print("2. Verify no unwanted transpositions occur")
    print("3. Analyze .shift() operation impact on chunks")
    print("4. Assess Dask graph complexity")
    print()


if __name__ == "__main__":
    main()
