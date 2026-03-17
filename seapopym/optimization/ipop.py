"""IPOP-CMA-ES: Increasing population restarts for CMA-ES.

Implements the IPOP strategy (Auger & Hansen, 2005) where the population
size doubles at each restart. Uses CMAESOptimizer internally. Collects
distinct modes by filtering on parameter-space distance.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from seapopym.optimization._common import (
    build_loss_fn,
    setup_objectives,
)
from seapopym.optimization.cmaes import CMAESOptimizer
from seapopym.optimization.gradient_optimizer import OptimizeResult
from seapopym.optimization.objective import Objective
from seapopym.types import Array, Params

if TYPE_CHECKING:
    from seapopym.compiler.model import CompiledModel

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
    """Euclidean distance between two parameter dicts, normalized by bounds.

    Only keys present in *bounds* contribute to the distance so that
    unbounded keys (whose raw scale may be arbitrary) do not dominate.
    """
    dist_sq = 0.0
    for key in a:
        if key not in b or key not in bounds:
            continue
        va = jnp.atleast_1d(a[key])
        vb = jnp.atleast_1d(b[key])
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
    return all(_params_distance(candidate.params, mode.params, bounds) >= distance_threshold for mode in existing_modes)


class IPOPCMAESOptimizer:
    """IPOP-CMA-ES optimizer with increasing population restarts.

    At each restart the population doubles (IPOP strategy, Auger & Hansen 2005).
    Uses :class:`CMAESOptimizer` internally. Collects distinct modes by
    filtering on Euclidean distance in parameter space.

    Args:
        objectives: List of ``(Objective, metric, weight)`` tuples.
        bounds: Parameter bounds as ``{name: (min, max)}``.
        n_restarts: Number of restarts to perform.
        initial_popsize: Population size for the first restart (doubles each time).
        n_generations: Number of generations per restart.
        distance_threshold: Minimum Euclidean distance between modes.
        seed: Random seed for reproducibility.
        chunk_size: Optional chunk size for time-stepping.

    Example::

        optimizer = IPOPCMAESOptimizer(
            objectives=[(Objective(observations=obs, transform=fn), "nrmse", 1.0)],
            bounds=BOUNDS,
            n_restarts=5,
            initial_popsize=8,
            n_generations=100,
        )
        result = optimizer.run(model, progress_bar=True)
    """

    def __init__(
        self,
        objectives: list[tuple[Objective, str | Callable, float]],
        bounds: dict[str, tuple[float, float]],
        n_restarts: int = 5,
        initial_popsize: int = 32,
        n_generations: int = 100,
        distance_threshold: float = 0.1,
        seed: int = 0,
        export_variables: list[str] | None = None,
        chunk_size: int | None = None,
    ) -> None:
        self.objectives = objectives
        self.bounds = bounds
        self.export_variables = export_variables
        self.chunk_size = chunk_size
        self.n_restarts = n_restarts
        self.initial_popsize = initial_popsize
        self.n_generations = n_generations
        self.distance_threshold = distance_threshold
        self.seed = seed

    def run(
        self,
        model: CompiledModel,
        progress_bar: bool = False,
    ) -> IPOPResult:
        """Run IPOP-CMA-ES optimization.

        Args:
            model: Compiled model to calibrate.
            progress_bar: If True, display inline progress indicator.

        Returns:
            IPOPResult with distinct modes sorted by loss.
        """
        prepared = setup_objectives(self.objectives, model.coords)
        loss_fn = build_loss_fn(model, prepared, self.export_variables, self.chunk_size)
        initial_params = {k: model.parameters[k] for k in self.bounds}

        return self._run_loss_fn(loss_fn, initial_params, progress_bar)

    def _run_loss_fn(
        self,
        loss_fn: Callable[[Params], Array],
        initial_params: Params,
        progress_bar: bool = False,
    ) -> IPOPResult:
        """Run IPOP-CMA-ES on a raw loss function."""
        key = jax.random.key(self.seed)
        all_results: list[OptimizeResult] = []
        modes: list[OptimizeResult] = []

        for i in range(self.n_restarts):
            popsize = self.initial_popsize * (2**i)
            key, init_key = jax.random.split(key)

            if i == 0:
                start_params = initial_params
            else:
                start_params = {}
                keys = jax.random.split(init_key, len(self.bounds))
                for (name, (low, high)), subkey in zip(self.bounds.items(), keys, strict=True):
                    shape = jnp.shape(initial_params[name])
                    start_params[name] = jax.random.uniform(subkey, shape=shape, minval=low, maxval=high)

            optimizer = CMAESOptimizer(
                objectives=self.objectives,
                bounds=self.bounds,
                popsize=popsize,
                seed=self.seed + i,
            )

            logger.info(
                "Restart %d/%d (CMA-ES): popsize=%d, %d generations",
                i + 1,
                self.n_restarts,
                popsize,
                self.n_generations,
            )
            t0 = time.time()

            result = optimizer._run_loss_fn(
                loss_fn,
                start_params,
                n_generations=self.n_generations,
                progress_bar=progress_bar,
            )
            all_results.append(result)

            if _is_new_mode(result, modes, self.distance_threshold, self.bounds):
                modes.append(result)

            elapsed = time.time() - t0
            actual_gens = result.n_iterations
            logger.info(
                "  -> loss=%.6e, %d gens, modes=%d, elapsed=%.1fs (%.3f s/gen)",
                result.loss,
                actual_gens,
                len(modes),
                elapsed,
                elapsed / max(actual_gens, 1),
            )

        modes.sort(key=lambda r: r.loss)

        return IPOPResult(modes=modes, all_results=all_results, n_restarts=self.n_restarts)
