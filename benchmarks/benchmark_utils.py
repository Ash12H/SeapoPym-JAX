"""Utilities for benchmarking JAX functions.

This module provides tools for accurate performance measurement of JAX code:
1. Proper handling of JAX asynchronous execution (.block_until_ready())
2. JIT compilation warm-up
3. Statistical analysis (mean, std, percentiles)
4. Comparison tools

Key considerations for JAX benchmarking:
- JAX uses asynchronous dispatch: results are not computed until needed
- Must call .block_until_ready() to ensure computation completes
- First call includes JIT compilation time, so warm-up is essential
- Performance can vary between CPU and GPU
"""

import time
from collections.abc import Callable
from typing import Any

import jax
import numpy as np


class BenchmarkResult:
    """Container for benchmark results with statistics.

    Attributes:
        times: Array of execution times [seconds]
        mean: Mean execution time [seconds]
        std: Standard deviation [seconds]
        min: Minimum execution time [seconds]
        max: Maximum execution time [seconds]
        median: Median execution time [seconds]
        p95: 95th percentile [seconds]
        p99: 99th percentile [seconds]
        throughput: Operations per second (if problem_size provided)
    """

    def __init__(self, times: np.ndarray, problem_size: int | None = None) -> None:
        """Initialize benchmark result.

        Args:
            times: Array of execution times [seconds]
            problem_size: Optional problem size for throughput calculation
        """
        self.times = times
        self.mean = float(np.mean(times))
        self.std = float(np.std(times))
        self.min = float(np.min(times))
        self.max = float(np.max(times))
        self.median = float(np.median(times))
        self.p95 = float(np.percentile(times, 95))
        self.p99 = float(np.percentile(times, 99))

        self.throughput: float | None
        if problem_size is not None:
            self.throughput = problem_size / self.mean
        else:
            self.throughput = None

    def __repr__(self) -> str:
        """String representation of benchmark results."""
        lines = [
            "BenchmarkResult:",
            f"  Mean:   {self.mean * 1000:.3f} ms ± {self.std * 1000:.3f} ms",
            f"  Median: {self.median * 1000:.3f} ms",
            f"  Min:    {self.min * 1000:.3f} ms",
            f"  Max:    {self.max * 1000:.3f} ms",
            f"  P95:    {self.p95 * 1000:.3f} ms",
            f"  P99:    {self.p99 * 1000:.3f} ms",
        ]
        if self.throughput is not None:
            lines.append(f"  Throughput: {self.throughput:.2e} ops/s")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "mean_ms": self.mean * 1000,
            "std_ms": self.std * 1000,
            "min_ms": self.min * 1000,
            "max_ms": self.max * 1000,
            "median_ms": self.median * 1000,
            "p95_ms": self.p95 * 1000,
            "p99_ms": self.p99 * 1000,
            "throughput": self.throughput,
        }


def benchmark_jax_function(
    func: Callable,
    *args: Any,
    n_warmup: int = 3,
    n_iterations: int = 10,
    problem_size: int | None = None,
    **kwargs: Any,
) -> BenchmarkResult:
    """Benchmark a JAX function with proper synchronization.

    This function:
    1. Runs warm-up iterations to trigger JIT compilation
    2. Measures execution time over multiple iterations
    3. Uses .block_until_ready() to ensure synchronization
    4. Computes statistics over measured times

    Args:
        func: Function to benchmark (should return JAX array)
        *args: Positional arguments to pass to func
        n_warmup: Number of warm-up iterations (default: 3)
        n_iterations: Number of timed iterations (default: 10)
        problem_size: Optional problem size (e.g., grid cells) for throughput
        **kwargs: Keyword arguments to pass to func

    Returns:
        BenchmarkResult with timing statistics

    Example:
        >>> import jax.numpy as jnp
        >>> def my_function(x):
        ...     return jnp.sum(x ** 2)
        >>> x = jnp.ones((1000, 1000))
        >>> result = benchmark_jax_function(my_function, x, n_iterations=20)
        >>> print(result)
    """
    # Warm-up phase (trigger JIT compilation)
    for _ in range(n_warmup):
        result = func(*args, **kwargs)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()

    # Timing phase
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        end = time.perf_counter()
        times.append(end - start)

    return BenchmarkResult(np.array(times), problem_size=problem_size)


def compare_implementations(
    baseline_func: Callable,
    optimized_func: Callable,
    *args: Any,
    n_iterations: int = 10,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compare two implementations and compute speedup.

    Args:
        baseline_func: Baseline implementation
        optimized_func: Optimized implementation to compare
        *args: Arguments to pass to both functions
        n_iterations: Number of iterations for benchmarking
        **kwargs: Keyword arguments to pass to both functions

    Returns:
        Dictionary with:
            - baseline: BenchmarkResult for baseline
            - optimized: BenchmarkResult for optimized version
            - speedup: Speedup factor (baseline_time / optimized_time)
            - speedup_pct: Speedup as percentage improvement

    Example:
        >>> def baseline(x):
        ...     return jnp.sum(x ** 2)
        >>> def optimized(x):
        ...     return jnp.dot(x.ravel(), x.ravel())
        >>> x = jnp.ones((1000, 1000))
        >>> comparison = compare_implementations(baseline, optimized, x)
        >>> print(f"Speedup: {comparison['speedup']:.2f}x")
    """
    baseline_result = benchmark_jax_function(
        baseline_func, *args, n_iterations=n_iterations, **kwargs
    )
    optimized_result = benchmark_jax_function(
        optimized_func, *args, n_iterations=n_iterations, **kwargs
    )

    speedup = baseline_result.mean / optimized_result.mean
    speedup_pct = (speedup - 1.0) * 100

    return {
        "baseline": baseline_result,
        "optimized": optimized_result,
        "speedup": speedup,
        "speedup_pct": speedup_pct,
    }


def get_device_info() -> dict[str, Any]:
    """Get information about JAX devices.

    Returns:
        Dictionary with device information:
            - devices: List of available devices
            - default_device: Default device
            - device_count: Number of devices
            - backend: JAX backend (cpu, gpu, tpu)
    """
    devices = jax.devices()
    default_device = jax.devices()[0]

    return {
        "devices": [str(d) for d in devices],
        "default_device": str(default_device),
        "device_count": len(devices),
        "backend": default_device.platform,
    }


def print_device_info() -> None:
    """Print JAX device information."""
    info = get_device_info()
    print("JAX Device Information:")
    print(f"  Backend: {info['backend']}")
    print(f"  Device count: {info['device_count']}")
    print(f"  Default device: {info['default_device']}")
    print(f"  Available devices: {', '.join(info['devices'])}")
