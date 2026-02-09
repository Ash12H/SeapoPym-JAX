"""Evolutionary optimization using CMA-ES via evosax.

Provides a wrapper around evosax strategies with the same API as Optimizer,
enabling easy comparison between gradient-based and evolutionary methods.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
from evosax.algorithms import CMA_ES

from seapopym.optimization.optimizer import OptimizeResult

from seapopym.types import Array, Params


class EvolutionaryOptimizer:
    """CMA-ES optimizer with same API as Optimizer.

    Uses evosax's CMA-ES implementation for derivative-free optimization.
    Automatically handles conversion between parameter dicts and flat arrays.

    Example:
        >>> optimizer = EvolutionaryOptimizer(popsize=32, bounds={"x": (0, 10)})
        >>> result = optimizer.run(loss_fn, {"x": jnp.array(5.0)}, n_generations=100)
    """

    STRATEGIES = {
        "cma_es": CMA_ES,
    }

    def __init__(
        self,
        strategy: Literal["cma_es"] = "cma_es",
        popsize: int = 32,
        bounds: dict[str, tuple[float, float]] | None = None,
        seed: int = 0,
    ) -> None:
        """Initialize the evolutionary optimizer.

        Args:
            strategy: Evolution strategy to use.
            popsize: Population size. Will be rounded up to even number if odd.
            bounds: Parameter bounds as {param_name: (min, max)}.
            seed: Random seed for reproducibility.
        """
        if strategy not in self.STRATEGIES:
            msg = f"Unknown strategy '{strategy}'. Available: {list(self.STRATEGIES.keys())}"
            raise ValueError(msg)

        # Ensure popsize is even (required by CMA-ES)
        if popsize % 2 != 0:
            popsize += 1

        self.strategy_name = strategy
        self.popsize = popsize
        self.bounds = bounds or {}
        self.seed = seed

        self._strategy_cls = self.STRATEGIES[strategy]

    def _flatten(self, params: Params) -> tuple[list[str], Array, Array | None, Array | None]:
        """Flatten parameter dict to array.

        Args:
            params: Parameter dict.

        Returns:
            Tuple of (keys, flat_array, lower_bounds, upper_bounds).
        """
        keys = sorted(params.keys())
        values = [jnp.atleast_1d(params[k]).flatten() for k in keys]
        flat = jnp.concatenate(values) if values else jnp.array([])

        # Build bounds arrays
        if self.bounds:
            lowers = []
            uppers = []
            for k in keys:
                size = jnp.atleast_1d(params[k]).size
                if k in self.bounds:
                    low, high = self.bounds[k]
                    lowers.extend([low] * size)
                    uppers.extend([high] * size)
                else:
                    lowers.extend([-jnp.inf] * size)
                    uppers.extend([jnp.inf] * size)
            return keys, flat, jnp.array(lowers), jnp.array(uppers)

        return keys, flat, None, None

    def _unflatten(
        self, keys: list[str], flat: Array, shapes: dict[str, tuple], original_params: Params | None = None
    ) -> Params:
        """Unflatten array back to parameter dict.

        Args:
            keys: Parameter names in order.
            flat: Flat array of values.
            shapes: Original shapes of each parameter.
            original_params: Original params to check if values were scalars.

        Returns:
            Parameter dict.
        """
        params = {}
        idx = 0
        for k in keys:
            shape = shapes[k]
            size = int(jnp.prod(jnp.array(shape))) if shape else 1
            values = flat[idx : idx + size]

            # Check if original was a scalar (0-dim array)
            if original_params is not None and jnp.ndim(original_params[k]) == 0:
                params[k] = values[0]
            elif shape:
                params[k] = values.reshape(shape)
            else:
                params[k] = values[0]
            idx += size
        return params

    def _apply_bounds(self, population: Array, lower: Array | None, upper: Array | None) -> Array:
        """Clip population to bounds.

        Args:
            population: Population array of shape (popsize, n_dims).
            lower: Lower bounds array.
            upper: Upper bounds array.

        Returns:
            Clipped population.
        """
        if lower is None or upper is None:
            return population
        return jnp.clip(population, lower, upper)

    def run(
        self,
        loss_fn: Callable[[Params], Array],
        initial_params: Params,
        n_generations: int = 100,
        verbose: bool = False,
    ) -> OptimizeResult:
        """Run the evolutionary optimization.

        Args:
            loss_fn: Function mapping params -> scalar loss.
            initial_params: Starting parameter values (used as initial mean).
            n_generations: Number of generations to run.
            verbose: If True, print progress every 10 generations.

        Returns:
            OptimizeResult with optimized parameters and diagnostics.
        """
        # Flatten initial params
        keys, x0, lower, upper = self._flatten(initial_params)
        shapes = {k: jnp.atleast_1d(initial_params[k]).shape for k in keys}

        # Initialize strategy
        strategy = self._strategy_cls(population_size=self.popsize, solution=x0)
        es_params = strategy.default_params

        key = jax.random.key(self.seed)
        key, init_key = jax.random.split(key)
        state = strategy.init(init_key, x0, es_params)

        # Create vectorized loss function
        def eval_one(flat_params: Array) -> Array:
            params = self._unflatten(keys, flat_params, shapes, initial_params)
            loss = loss_fn(params)
            # Ensure scalar output (squeeze any extra dimensions)
            return jnp.squeeze(loss)

        eval_population = jax.vmap(eval_one)

        # Optimization loop
        loss_history: list[float] = []
        best_loss = float("inf")
        best_params = initial_params

        for gen in range(n_generations):
            key, ask_key, tell_key = jax.random.split(key, 3)

            # Ask for new population
            population, state = strategy.ask(ask_key, state, es_params)

            # Apply bounds
            population = self._apply_bounds(population, lower, upper)

            # Evaluate fitness (lower is better)
            fitness = eval_population(population)

            # Tell results (returns state, metrics)
            state, _metrics = strategy.tell(tell_key, population, fitness, state, es_params)

            # Track best
            min_fitness = float(jnp.min(fitness))
            loss_history.append(min_fitness)

            if min_fitness < best_loss:
                best_loss = min_fitness
                best_idx = jnp.argmin(fitness)
                best_flat = population[best_idx]
                best_params = self._unflatten(keys, best_flat, shapes, initial_params)

            # Verbose output
            if verbose and gen % 10 == 0:
                print(f"Generation {gen}: best_loss = {best_loss:.6e}")

        return OptimizeResult(
            params=best_params,
            loss=best_loss,
            loss_history=loss_history,
            n_iterations=n_generations,
            converged=False,  # CMA-ES doesn't have built-in convergence check
            message=f"Completed {n_generations} generations",
        )
