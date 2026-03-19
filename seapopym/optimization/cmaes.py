"""CMA-ES optimizer wrapping evosax CMA_ES.

Single-run CMA-ES with typed arguments matching the underlying library.
Handles Objective setup, loss building, normalization, and optimization.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from evosax.algorithms import CMA_ES

from seapopym.optimization._common import (
    HallOfFame,
    build_bounds_arrays,
    build_loss_fn,
    denormalize,
    flatten_params,
    normalize,
    run_evolution_strategy,
    setup_objectives,
    unflatten_params,
)
from seapopym.optimization.gradient_optimizer import OptimizeResult
from seapopym.optimization.objective import Objective
from seapopym.types import Array, Params

if TYPE_CHECKING:
    from seapopym.compiler.model import CompiledModel


class CMAESOptimizer:
    """CMA-ES optimizer for model calibration.

    Wraps evosax ``CMA_ES`` with a high-level interface that handles
    objective setup, loss building, and parameter normalization.

    Args:
        objectives: List of ``(Objective, metric, weight)`` tuples.
        bounds: Parameter bounds as ``{name: (min, max)}``.
        popsize: Population size (rounded up to even if odd).
        seed: Random seed for reproducibility.
        chunk_size: Optional chunk size for time-stepping.

    Example::

        optimizer = CMAESOptimizer(
            objectives=[(Objective(observations=obs, transform=fn), "nrmse", 1.0)],
            bounds={"x": (0.0, 10.0)},
            popsize=32,
        )
        result = optimizer.run(model, n_generations=100)
    """

    def __init__(
        self,
        objectives: list[tuple[Objective, str | Callable, float]],
        bounds: dict[str, tuple[float, float]],
        popsize: int = 32,
        seed: int = 0,
        export_variables: list[str] | None = None,
        chunk_size: int | None = None,
        hall_of_fame_size: int | None = None,
        distance_threshold: float = 0.1,
    ) -> None:
        if popsize % 2 != 0:
            popsize += 1

        self.objectives = objectives
        self.bounds = bounds
        self.export_variables = export_variables
        self.chunk_size = chunk_size
        self.popsize = popsize
        self.seed = seed
        self.hall_of_fame_size = hall_of_fame_size
        self.distance_threshold = distance_threshold

    def run(
        self,
        model: CompiledModel,
        n_generations: int = 100,
        tol_fun: float = 1e-9,
        patience: int = 50,
        progress_bar: bool = False,
    ) -> OptimizeResult:
        """Run CMA-ES optimization.

        Args:
            model: Compiled model to calibrate.
            n_generations: Maximum number of generations.
            tol_fun: Absolute improvement threshold for early stopping.
            patience: Generations without improvement before stopping.
            progress_bar: If True, display inline progress indicator.

        Returns:
            OptimizeResult with optimized parameters and diagnostics.
        """
        prepared = setup_objectives(self.objectives, model.coords)
        loss_fn = build_loss_fn(model, prepared, self.export_variables, self.chunk_size)
        initial_params = {k: model.parameters[k] for k in self.bounds}

        return self._run_loss_fn(loss_fn, initial_params, n_generations, tol_fun, patience, progress_bar)

    def _run_loss_fn(
        self,
        loss_fn: Callable[[Params], Array],
        initial_params: Params,
        n_generations: int = 100,
        tol_fun: float = 1e-9,
        patience: int = 50,
        progress_bar: bool = False,
    ) -> OptimizeResult:
        """Run CMA-ES on a raw loss function (used by IPOP internally)."""
        keys, x0 = flatten_params(initial_params)
        shapes = {k: jnp.atleast_1d(initial_params[k]).shape for k in keys}
        lower, upper = build_bounds_arrays(keys, initial_params, self.bounds)

        x0_norm = normalize(x0, lower, upper)

        def eval_one(flat_norm: Array) -> Array:
            flat_orig = denormalize(flat_norm, lower, upper)
            params = unflatten_params(keys, flat_orig, shapes, initial_params)
            return jnp.squeeze(loss_fn(params))

        eval_population = jax.jit(jax.vmap(eval_one))

        strategy = CMA_ES(population_size=self.popsize, solution=x0_norm)
        es_params = strategy.default_params

        key = jax.random.key(self.seed)
        key, init_key = jax.random.split(key)
        state = strategy.init(init_key, x0_norm, es_params)

        norm_lower = jnp.zeros_like(x0_norm)
        norm_upper = jnp.ones_like(x0_norm)

        # Hall-of-fame setup
        hof = None
        denorm_fn = None
        if self.hall_of_fame_size is not None:
            hof = HallOfFame(self.hall_of_fame_size, self.distance_threshold, self.bounds)
            denorm_fn = lambda flat_norm: unflatten_params(keys, denormalize(flat_norm, lower, upper), shapes, initial_params)

        best_flat_norm, loss_history, converged = run_evolution_strategy(
            strategy=strategy,
            es_params=es_params,
            state=state,
            eval_population=eval_population,
            n_generations=n_generations,
            tol_fun=tol_fun,
            patience=patience,
            key=key,
            norm_lower=norm_lower,
            norm_upper=norm_upper,
            x0_norm=x0_norm,
            progress_bar=progress_bar,
            hall_of_fame=hof,
            denorm_fn=denorm_fn,
        )

        best_flat_orig = denormalize(best_flat_norm, lower, upper)
        best_params = unflatten_params(keys, best_flat_orig, shapes, initial_params)
        best_loss = min(loss_history) if loss_history else float("inf")

        hof_results = None
        if hof is not None:
            hof_results = [OptimizeResult(params=p, loss=l) for p, l in hof.members]

        return OptimizeResult(
            params=best_params,
            loss=best_loss,
            loss_history=loss_history,
            n_iterations=len(loss_history),
            converged=converged,
            message=f"Converged after {len(loss_history)} generations"
            if converged
            else f"Reached max iterations ({n_generations})",
            hall_of_fame=hof_results,
        )
