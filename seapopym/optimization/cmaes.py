"""CMA-ES optimizer wrapping evosax CMA_ES.

Provides a step-based API for fine-grained control, and a convenience
``run()`` method for standard optimization with patience-based stopping.

Example (step-based)::

    optimizer = CMAESOptimizer(
        bounds={"x": (0.0, 10.0), "y": (-1.0, 1.0)},
        initial_params={"x": 5.0, "y": 0.0},
        popsize=32,
    )
    for i in range(100):
        gen = optimizer.step(loss_fn)
        print(f"gen {gen.gen}: best={gen.best_loss:.4f}")
        if converged:
            break

Example (convenience)::

    result = optimizer.run(loss_fn, max_gen=100, patience=20)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from evosax.algorithms import CMA_ES

from seapopym.optimization._common import (
    GenerationResult,
    OptimizeResult,
    build_bounds_arrays,
    build_loss_fn,
    denormalize,
    flatten_params,
    normalize,
    setup_objectives,
    unflatten_params,
)
from seapopym.optimization.objective import Objective
from seapopym.types import Array, Params

if TYPE_CHECKING:
    from seapopym.compiler.model import CompiledModel

logger = logging.getLogger(__name__)


class CMAESOptimizer:
    """CMA-ES optimizer with step-based API.

    The optimizer is initialized with bounds, initial parameters, and
    hyperparameters. Call :meth:`step` repeatedly to advance one generation
    at a time, or use :meth:`run` for a standard loop with patience.

    Args:
        bounds: Parameter bounds as ``{name: (min, max)}``.
        initial_params: Starting point as ``{name: value}``.
        popsize: Population size (rounded up to even if odd).
        seed: Random seed for reproducibility.
        nan_penalty: Fitness value assigned to non-finite individuals.
    """

    def __init__(
        self,
        bounds: dict[str, tuple[float, float]],
        initial_params: Params,
        popsize: int = 32,
        seed: int = 0,
        nan_penalty: float = 1e6,
    ) -> None:
        if popsize % 2 != 0:
            popsize += 1

        self.bounds = bounds
        self.popsize = popsize
        self.nan_penalty = nan_penalty

        # Flatten and normalize
        self._keys, x0 = flatten_params(initial_params)
        self._shapes = {k: jnp.atleast_1d(initial_params[k]).shape for k in self._keys}
        self._initial_params = initial_params
        self._lower, self._upper = build_bounds_arrays(self._keys, initial_params, bounds)

        x0_norm = normalize(x0, self._lower, self._upper)

        # Initialize evosax strategy
        self._strategy = CMA_ES(population_size=popsize, solution=x0_norm)
        self._es_params = self._strategy.default_params

        key = jax.random.key(seed)
        key, init_key = jax.random.split(key)
        self._state = self._strategy.init(init_key, x0_norm, self._es_params)
        self._key = key

        self._norm_lower = jnp.zeros_like(x0_norm)
        self._norm_upper = jnp.ones_like(x0_norm)

        # Generation counter
        self._gen = 0

        # Build vmapped evaluator (set on first step call)
        self._eval_population: Callable | None = None
        self._current_eval_fn: Callable | None = None

    def _denorm_params(self, flat_norm: Array) -> Params:
        """Convert normalized flat vector back to parameter dict."""
        flat_orig = denormalize(flat_norm, self._lower, self._upper)
        return unflatten_params(self._keys, flat_orig, self._shapes, self._initial_params)

    def _build_eval(self, eval_fn: Callable[[Params], Array]) -> Callable:
        """Build vmapped population evaluator with NaN penalty."""
        penalty = jnp.float32(self.nan_penalty)

        def eval_one(flat_norm: Array) -> Array:
            params = self._denorm_params(flat_norm)
            loss = jnp.squeeze(eval_fn(params))
            return jnp.where(jnp.isfinite(loss), loss, penalty)

        return jax.jit(jax.vmap(eval_one))

    def step(self, eval_fn: Callable[[Params], Array]) -> GenerationResult:
        """Advance one generation.

        Args:
            eval_fn: Loss function mapping ``Params -> scalar``.

        Returns:
            GenerationResult with population data for this generation.
        """
        # Rebuild evaluator if eval_fn changed (e.g., switching train/test)
        if eval_fn is not self._current_eval_fn:
            self._eval_population = self._build_eval(eval_fn)
            self._current_eval_fn = eval_fn

        eval_pop = self._eval_population
        if eval_pop is None:  # pragma: no cover
            msg = "eval_population not initialized"
            raise RuntimeError(msg)

        self._key, ask_key, tell_key = jax.random.split(self._key, 3)

        # Ask
        population, self._state = self._strategy.ask(ask_key, self._state, self._es_params)
        population = jnp.clip(population, self._norm_lower, self._norm_upper)

        # Evaluate
        fitness = eval_pop(population)
        fitness_np = np.array(fitness)

        # Tell
        self._state, _ = self._strategy.tell(tell_key, population, fitness, self._state, self._es_params)

        # Build result
        valid_mask = np.isfinite(fitness_np)
        best_idx = int(np.argmin(fitness_np))
        best_loss = float(fitness_np[best_idx])
        mean_loss = float(np.mean(fitness_np[valid_mask])) if valid_mask.any() else float("inf")
        n_valid = int(valid_mask.sum())

        best_params = self._denorm_params(population[best_idx])

        # Denormalize all individuals
        pop_params = [self._denorm_params(population[i]) for i in range(self.popsize)]

        gen_result = GenerationResult(
            gen=self._gen,
            best_loss=best_loss,
            mean_loss=mean_loss,
            n_valid=n_valid,
            best_params=best_params,
            population_params=pop_params,
            population_fitness=fitness_np,
        )

        self._gen += 1
        return gen_result

    def run(
        self,
        eval_fn: Callable[[Params], Array],
        max_gen: int = 200,
        patience: int = 50,
        tol_fun: float = 1e-9,
    ) -> OptimizeResult:
        """Convenience: run CMA-ES with patience-based early stopping.

        Args:
            eval_fn: Loss function mapping ``Params -> scalar``.
            max_gen: Maximum number of generations.
            patience: Stop after this many generations without improvement.
            tol_fun: Minimum improvement to reset patience counter.

        Returns:
            OptimizeResult with best parameters and diagnostics.
        """
        best_loss = float("inf")
        best_params = self._initial_params
        loss_history: list[float] = []
        stalls = 0
        converged = False

        for _ in range(max_gen):
            gen = self.step(eval_fn)
            loss_history.append(gen.best_loss)

            if gen.best_loss < best_loss - tol_fun:
                best_loss = gen.best_loss
                best_params = gen.best_params
                stalls = 0
            else:
                stalls += 1

            if stalls >= patience:
                converged = True
                break

        return OptimizeResult(
            params=best_params,
            loss=best_loss,
            loss_history=loss_history,
            n_iterations=len(loss_history),
            converged=converged,
            message=f"Converged after {len(loss_history)} generations"
            if converged
            else f"Reached max iterations ({max_gen})",
        )

    # ------------------------------------------------------------------
    # High-level model-aware entry point (backward compatible)
    # ------------------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model: CompiledModel,
        objectives: list[tuple[Objective, str | Callable, float]],
        bounds: dict[str, tuple[float, float]],
        popsize: int = 32,
        seed: int = 0,
        nan_penalty: float = 1e6,
        export_variables: list[str] | None = None,
        chunk_size: int | None = None,
    ) -> tuple[CMAESOptimizer, Callable[[Params], Array]]:
        """Create optimizer and loss function from a compiled model.

        Returns:
            Tuple of ``(optimizer, loss_fn)`` ready for ``step()`` or ``run()``.

        Example::

            optimizer, loss_fn = CMAESOptimizer.from_model(
                model, objectives, bounds, popsize=64,
            )
            result = optimizer.run(loss_fn, max_gen=200)
        """
        prepared = setup_objectives(objectives, model.coords)
        loss_fn = build_loss_fn(model, prepared, export_variables, chunk_size)
        initial_params = {k: model.parameters[k] for k in bounds}

        optimizer = cls(
            bounds=bounds,
            initial_params=initial_params,
            popsize=popsize,
            seed=seed,
            nan_penalty=nan_penalty,
        )
        return optimizer, loss_fn
