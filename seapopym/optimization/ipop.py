"""IPOP-CMA-ES: Increasing population restarts for CMA-ES.

Implements the IPOP strategy (Auger & Hansen, 2005) where the population
size doubles at each restart. Uses CMAESOptimizer internally. Collects
distinct modes by filtering on parameter-space distance.

Example::

    optimizer = IPOPCMAESOptimizer(
        bounds=BOUNDS,
        initial_params=initial_params,
        n_restarts=5,
        initial_popsize=8,
    )
    result = optimizer.run(loss_fn)
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
    OptimizeResult,
    build_loss_fn,
    params_distance,
    setup_objectives,
)
from seapopym.optimization.cmaes import CMAESOptimizer
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


def _is_new_mode(
    candidate: OptimizeResult,
    existing_modes: list[OptimizeResult],
    distance_threshold: float,
    bounds: dict[str, tuple[float, float]],
) -> bool:
    """Check if candidate is far enough from all existing modes (in normalized space)."""
    return all(params_distance(candidate.params, mode.params, bounds) >= distance_threshold for mode in existing_modes)


class IPOPCMAESOptimizer:
    """IPOP-CMA-ES optimizer with increasing population restarts.

    At each restart the population doubles (IPOP strategy, Auger & Hansen 2005).
    Uses :class:`CMAESOptimizer` internally. Collects distinct modes by
    filtering on Euclidean distance in parameter space.

    Args:
        bounds: Parameter bounds as ``{name: (min, max)}``.
        initial_params: Starting point as ``{name: value}``.
        n_restarts: Number of restarts to perform.
        initial_popsize: Population size for the first restart (doubles each time).
        n_generations: Number of generations per restart.
        distance_threshold: Minimum Euclidean distance between modes.
        tol_fun: Absolute improvement threshold for early stopping.
        patience: Generations without improvement before stopping.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        bounds: dict[str, tuple[float, float]],
        initial_params: Params,
        n_restarts: int = 5,
        initial_popsize: int = 32,
        n_generations: int = 100,
        distance_threshold: float = 0.1,
        tol_fun: float = 1e-9,
        patience: int = 50,
        seed: int = 0,
    ) -> None:
        self.bounds = bounds
        self._initial_params = initial_params
        self.n_restarts = n_restarts
        self.initial_popsize = initial_popsize
        self.n_generations = n_generations
        self.distance_threshold = distance_threshold
        self.tol_fun = tol_fun
        self.patience = patience
        self.seed = seed

    def run(
        self,
        eval_fn: Callable[[Params], Array],
    ) -> IPOPResult:
        """Run IPOP-CMA-ES optimization.

        Args:
            eval_fn: Loss function mapping ``Params -> scalar``.

        Returns:
            IPOPResult with distinct modes sorted by loss.
        """
        key = jax.random.key(self.seed)
        all_results: list[OptimizeResult] = []
        modes: list[OptimizeResult] = []

        for i in range(self.n_restarts):
            popsize = self.initial_popsize * (2**i)
            key, init_key = jax.random.split(key)

            if i == 0:
                start_params = self._initial_params
            else:
                start_params = {}
                keys = jax.random.split(init_key, len(self.bounds))
                for (name, (low, high)), subkey in zip(self.bounds.items(), keys, strict=True):
                    shape = jnp.shape(self._initial_params[name])
                    start_params[name] = jax.random.uniform(subkey, shape=shape, minval=low, maxval=high)

            optimizer = CMAESOptimizer(
                bounds=self.bounds,
                initial_params=start_params,
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

            result = optimizer.run(
                eval_fn,
                max_gen=self.n_generations,
                patience=self.patience,
                tol_fun=self.tol_fun,
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

    # ------------------------------------------------------------------
    # High-level model-aware entry point
    # ------------------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model: CompiledModel,
        objectives: list[tuple[Objective, str | Callable, float]],
        bounds: dict[str, tuple[float, float]],
        n_restarts: int = 5,
        initial_popsize: int = 32,
        n_generations: int = 100,
        distance_threshold: float = 0.1,
        tol_fun: float = 1e-9,
        patience: int = 50,
        seed: int = 0,
        export_variables: list[str] | None = None,
        chunk_size: int | None = None,
    ) -> tuple[IPOPCMAESOptimizer, Callable[[Params], Array]]:
        """Create optimizer and loss function from a compiled model.

        Returns:
            Tuple of ``(optimizer, loss_fn)`` ready for ``run()``.

        Example::

            optimizer, loss_fn = IPOPCMAESOptimizer.from_model(
                model, objectives, bounds, n_restarts=5,
            )
            result = optimizer.run(loss_fn)
        """
        prepared = setup_objectives(objectives, model.coords)
        loss_fn = build_loss_fn(model, prepared, export_variables, chunk_size)
        initial_params = {k: model.parameters[k] for k in bounds}

        optimizer = cls(
            bounds=bounds,
            initial_params=initial_params,
            n_restarts=n_restarts,
            initial_popsize=initial_popsize,
            n_generations=n_generations,
            distance_threshold=distance_threshold,
            tol_fun=tol_fun,
            patience=patience,
            seed=seed,
        )
        return optimizer, loss_fn
