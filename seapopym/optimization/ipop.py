"""IPOP-CMA-ES: CMA-ES with increasing population restarts.

Implements the IPOP strategy (Auger & Hansen, 2005) where the population
size doubles at each restart. Collects distinct modes by filtering
on parameter-space distance.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from seapopym.optimization.evolutionary import EvolutionaryOptimizer
from seapopym.optimization.optimizer import OptimizeResult
from seapopym.types import Array, Params


@dataclass
class IPOPResult:
    """Result of an IPOP-CMA-ES run.

    Attributes:
        modes: Distinct modes found, sorted by loss (best first).
        all_results: All restart results (including duplicates).
        n_restarts: Number of restarts performed.
    """

    modes: list[OptimizeResult]
    all_results: list[OptimizeResult] = field(default_factory=list)
    n_restarts: int = 0


def _params_distance(a: Params, b: Params, bounds: dict[str, tuple[float, float]]) -> float:
    """Euclidean distance between two parameter dicts, normalized by bounds."""
    dist_sq = 0.0
    for key in a:
        if key in b:
            va = jnp.atleast_1d(a[key])
            vb = jnp.atleast_1d(b[key])
            if key in bounds:
                low, high = bounds[key]
                va = (va - low) / (high - low)
                vb = (vb - low) / (high - low)
            diff = va - vb
            dist_sq += float(jnp.sum(diff**2))
    return float(jnp.sqrt(dist_sq))


def _is_new_mode(
    candidate: OptimizeResult,
    existing_modes: list[OptimizeResult],
    distance_threshold: float,
    bounds: dict[str, tuple[float, float]],
) -> bool:
    """Check if candidate is far enough from all existing modes (in normalized space)."""
    return all(
        _params_distance(candidate.params, mode.params, bounds) >= distance_threshold for mode in existing_modes
    )


def run_ipop_cmaes(
    loss_fn: Callable[[Params], Array],
    initial_params: Params,
    bounds: dict[str, tuple[float, float]],
    n_restarts: int = 5,
    initial_popsize: int = 32,
    n_generations: int = 100,
    distance_threshold: float = 0.1,
    seed: int = 0,
    verbose: bool = False,
) -> IPOPResult:
    """Run IPOP-CMA-ES with increasing population restarts.

    At each restart the population doubles (IPOP strategy, Auger & Hansen 2005).
    Initial positions are sampled uniformly within bounds. Distinct modes
    are collected by filtering on Euclidean distance in parameter space.

    Args:
        loss_fn: Function mapping params -> scalar loss to minimize.
        initial_params: Starting position for the first restart.
        bounds: Parameter bounds as {name: (min, max)}. Required for
            random initialization of restarts.
        n_restarts: Number of restarts to perform.
        initial_popsize: Population size for the first restart (doubles each time).
        n_generations: Number of generations per restart.
        distance_threshold: Minimum Euclidean distance between modes.
        seed: Random seed for reproducibility.
        verbose: If True, print progress per restart.

    Returns:
        IPOPResult with distinct modes sorted by loss.

    Example:
        >>> result = run_ipop_cmaes(
        ...     loss_fn, initial_params={"x": jnp.array(0.0)},
        ...     bounds={"x": (-5.0, 5.0)}, n_restarts=3,
        ... )
        >>> result.modes[0].loss  # best mode
    """
    key = jax.random.key(seed)
    all_results: list[OptimizeResult] = []
    modes: list[OptimizeResult] = []

    for i in range(n_restarts):
        popsize = initial_popsize * (2**i)
        key, init_key = jax.random.split(key)

        # First restart uses initial_params, subsequent restarts sample from bounds
        if i == 0:
            start_params = initial_params
        else:
            start_params = {}
            keys = jax.random.split(init_key, len(bounds))
            for (name, (low, high)), subkey in zip(bounds.items(), keys, strict=True):
                start_params[name] = jax.random.uniform(subkey, minval=low, maxval=high)

        optimizer = EvolutionaryOptimizer(
            popsize=popsize,
            bounds=bounds,
            seed=seed + i,
        )
        if verbose:
            import time

            print(f"\n--- Restart {i + 1}/{n_restarts}: popsize={popsize}, {n_generations} generations ---")
            t0 = time.time()

        result = optimizer.run(loss_fn, start_params, n_generations=n_generations, verbose=verbose)
        all_results.append(result)

        if _is_new_mode(result, modes, distance_threshold, bounds):
            modes.append(result)

        if verbose:
            elapsed = time.time() - t0
            actual_gens = result.n_iterations
            print(
                f"  -> loss={result.loss:.6e}, {actual_gens} gens, modes={len(modes)}, "
                f"elapsed={elapsed:.1f}s ({elapsed / max(actual_gens, 1):.3f} s/gen)"
            )

    # Sort modes by loss (best first)
    modes.sort(key=lambda r: r.loss)

    return IPOPResult(modes=modes, all_results=all_results, n_restarts=n_restarts)
