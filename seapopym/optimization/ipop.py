"""IPOP: Increasing population restarts for evolutionary strategies.

Implements the IPOP strategy (Auger & Hansen, 2005) where the population
size doubles at each restart. Works with any evosax strategy supported
by EvolutionaryOptimizer (CMA-ES, SimpleGA, etc.). Collects distinct modes
by filtering on parameter-space distance.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp

from seapopym.optimization.evolutionary import EvolutionaryOptimizer
from seapopym.optimization.optimizer import OptimizeResult
from seapopym.types import Array, Params

logger = logging.getLogger(__name__)


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


def run_ipop(
    loss_fn: Callable[[Params], Array],
    initial_params: Params,
    bounds: dict[str, tuple[float, float]],
    strategy: Literal["cma_es", "simple_ga"] = "cma_es",
    n_restarts: int = 5,
    initial_popsize: int = 32,
    n_generations: int = 100,
    distance_threshold: float = 0.1,
    seed: int = 0,
    progress_bar: bool = False,
    **strategy_kwargs,
) -> IPOPResult:
    """Run IPOP with increasing population restarts for any evosax strategy.

    At each restart the population doubles (IPOP strategy, Auger & Hansen 2005).
    Initial positions are sampled uniformly within bounds. Distinct modes
    are collected by filtering on Euclidean distance in parameter space.

    Args:
        loss_fn: Function mapping params -> scalar loss to minimize.
        initial_params: Starting position for the first restart.
        bounds: Parameter bounds as {name: (min, max)}. Required for
            random initialization of restarts.
        strategy: Evosax strategy name ("cma_es", "simple_ga").
        n_restarts: Number of restarts to perform.
        initial_popsize: Population size for the first restart (doubles each time).
        n_generations: Number of generations per restart.
        distance_threshold: Minimum Euclidean distance between modes.
        seed: Random seed for reproducibility.
        progress_bar: If True, display inline progress indicator.
        **strategy_kwargs: Extra keyword arguments passed to EvolutionaryOptimizer
            (e.g. crossover_rate, mutation_std for SimpleGA).

    Returns:
        IPOPResult with distinct modes sorted by loss.

    Example:
        >>> result = run_ipop(
        ...     loss_fn, initial_params={"x": jnp.array(0.0)},
        ...     bounds={"x": (-5.0, 5.0)}, strategy="cma_es", n_restarts=3,
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
            strategy=strategy,
            popsize=popsize,
            bounds=bounds,
            seed=seed + i,
            **strategy_kwargs,
        )
        logger.info(
            "Restart %d/%d (%s): popsize=%d, %d generations",
            i + 1, n_restarts, strategy, popsize, n_generations,
        )
        t0 = time.time()

        result = optimizer.run(loss_fn, start_params, n_generations=n_generations, progress_bar=progress_bar)
        all_results.append(result)

        if _is_new_mode(result, modes, distance_threshold, bounds):
            modes.append(result)

        elapsed = time.time() - t0
        actual_gens = result.n_iterations
        logger.info(
            "  -> loss=%.6e, %d gens, modes=%d, elapsed=%.1fs (%.3f s/gen)",
            result.loss, actual_gens, len(modes), elapsed, elapsed / max(actual_gens, 1),
        )

    # Sort modes by loss (best first)
    modes.sort(key=lambda r: r.loss)

    return IPOPResult(modes=modes, all_results=all_results, n_restarts=n_restarts)


def run_ipop_cmaes(
    loss_fn: Callable[[Params], Array],
    initial_params: Params,
    bounds: dict[str, tuple[float, float]],
    n_restarts: int = 5,
    initial_popsize: int = 32,
    n_generations: int = 100,
    distance_threshold: float = 0.1,
    seed: int = 0,
    progress_bar: bool = False,
) -> IPOPResult:
    """Run IPOP-CMA-ES. Convenience alias for ``run_ipop(strategy="cma_es", ...)``."""
    return run_ipop(
        loss_fn=loss_fn,
        initial_params=initial_params,
        bounds=bounds,
        strategy="cma_es",
        n_restarts=n_restarts,
        initial_popsize=initial_popsize,
        n_generations=n_generations,
        distance_threshold=distance_threshold,
        seed=seed,
        progress_bar=progress_bar,
    )
