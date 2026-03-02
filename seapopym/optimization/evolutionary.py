"""Evolutionary optimization using evosax strategies (CMA-ES, GA, etc.).

Provides a wrapper around evosax strategies with the same API as GradientOptimizer,
enabling easy comparison between gradient-based and evolutionary methods.

Population evaluation is JIT-compiled and vmapped for GPU acceleration.
Parameters are normalized to [0,1] using bounds for proper scaling.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
import optax
from evosax.algorithms import CMA_ES, SimpleGA

from seapopym.optimization.optimizer import OptimizeResult
from seapopym.types import Array, Params

logger = logging.getLogger(__name__)


class EvolutionaryOptimizer:
    """Evolutionary optimizer wrapping evosax strategies (CMA-ES, SimpleGA).

    Uses evosax strategies for derivative-free optimization.
    Automatically handles conversion between parameter dicts and flat arrays.

    When bounds are provided, the optimization is performed in normalized
    [0,1] space so that the strategy can handle parameters spanning many
    orders of magnitude.

    Example:
        >>> optimizer = EvolutionaryOptimizer(popsize=32, bounds={"x": (0, 10)})
        >>> result = optimizer.run(loss_fn, {"x": jnp.array(5.0)}, n_generations=100)
        >>> optimizer = EvolutionaryOptimizer(
        ...     strategy="simple_ga", popsize=64, bounds={"x": (0, 10)},
        ...     crossover_rate=0.5, mutation_std=0.1,
        ... )
    """

    STRATEGIES = {
        "cma_es": CMA_ES,
        "simple_ga": SimpleGA,
    }

    def __init__(
        self,
        strategy: Literal["cma_es", "simple_ga"] = "cma_es",
        popsize: int = 32,
        bounds: dict[str, tuple[float, float]] | None = None,
        seed: int = 0,
        crossover_rate: float = 0.5,
        mutation_std: float = 0.1,
    ) -> None:
        if strategy not in self.STRATEGIES:
            msg = f"Unknown strategy '{strategy}'. Available: {list(self.STRATEGIES.keys())}"
            raise ValueError(msg)

        # Ensure popsize is even (required by CMA-ES only)
        if strategy == "cma_es" and popsize % 2 != 0:
            popsize += 1

        self.strategy_name = strategy
        self.popsize = popsize
        self.bounds = bounds or {}
        self.seed = seed
        self.crossover_rate = crossover_rate
        self.mutation_std = mutation_std

        self._strategy_cls = self.STRATEGIES[strategy]

    def _flatten(self, params: Params) -> tuple[list[str], Array]:
        """Flatten parameter dict to a 1D array."""
        keys = sorted(params.keys())
        values = [jnp.atleast_1d(params[k]).flatten() for k in keys]
        flat = jnp.concatenate(values) if values else jnp.array([])
        return keys, flat

    def _build_bounds_arrays(self, keys: list[str], params: Params) -> tuple[Array, Array]:
        """Build lower/upper bound arrays matching the flattened parameter vector."""
        lowers = []
        uppers = []
        for k in keys:
            size = jnp.atleast_1d(params[k]).size
            if k in self.bounds:
                low, high = self.bounds[k]
                lowers.extend([low] * size)
                uppers.extend([high] * size)
            else:
                lowers.extend([0.0] * size)
                uppers.extend([1.0] * size)
        return jnp.array(lowers), jnp.array(uppers)

    def _normalize(self, flat: Array, lower: Array, upper: Array) -> Array:
        """Transform from original space to [0,1]."""
        return (flat - lower) / (upper - lower)

    def _denormalize(self, flat_norm: Array, lower: Array, upper: Array) -> Array:
        """Transform from [0,1] back to original space."""
        return flat_norm * (upper - lower) + lower

    def _unflatten(
        self, keys: list[str], flat: Array, shapes: dict[str, tuple], original_params: Params | None = None
    ) -> Params:
        params = {}
        idx = 0
        for k in keys:
            shape = shapes[k]
            size = math.prod(shape) if shape else 1
            values = flat[idx : idx + size]

            if original_params is not None and jnp.ndim(original_params[k]) == 0:
                params[k] = values[0]
            elif shape:
                params[k] = values.reshape(shape)
            else:
                params[k] = values[0]
            idx += size
        return params

    def _build_strategy(self, solution: Array):
        """Construct the evosax strategy with appropriate parameters."""
        if self.strategy_name == "simple_ga":
            return self._strategy_cls(
                population_size=self.popsize,
                solution=solution,
                std_schedule=optax.constant_schedule(self.mutation_std),
            )
        return self._strategy_cls(population_size=self.popsize, solution=solution)

    def _build_params(self, strategy):
        """Build strategy params, applying GA-specific overrides."""
        params = strategy.default_params
        if self.strategy_name == "simple_ga":
            params = params.replace(crossover_rate=self.crossover_rate)
        return params

    def _init_state(self, strategy, key, x0_norm, es_params, eval_population):
        """Initialize strategy state, handling API differences.

        CMA_ES.init(key, mean, params) — distribution-based, takes a mean.
        SimpleGA.init(key, population, fitness, params) — population-based,
        takes an initial population and its fitness.
        """
        if self.strategy_name == "simple_ga":
            # Generate initial population uniformly in [0,1]
            pop = jax.random.uniform(key, shape=(self.popsize, x0_norm.shape[0]))
            fitness = eval_population(pop)
            return strategy.init(key, pop, fitness, es_params)
        return strategy.init(key, x0_norm, es_params)

    def run(
        self,
        loss_fn: Callable[[Params], Array],
        initial_params: Params,
        n_generations: int = 100,
        tol_fun: float = 1e-9,
        patience: int = 50,
        progress_bar: bool = False,
    ) -> OptimizeResult:
        """Run the evolutionary optimization with early stopping.

        The optimization runs in normalized [0,1] space. Samples are
        clipped to [0,1] then denormalized before evaluating the loss.

        Stops early when the best loss has not improved by more than
        ``tol_fun`` (relative) over the last ``patience`` generations,
        following the TolFun criterion from Hansen (2023).

        Args:
            loss_fn: Function mapping params -> scalar loss.
            initial_params: Starting parameter values (used as initial mean
                for CMA-ES; ignored by SimpleGA which starts from a random
                population within bounds).
            n_generations: Maximum number of generations to run.
            tol_fun: Relative improvement threshold for early stopping.
            patience: Number of generations without improvement before stopping.
            progress_bar: If True, display inline progress indicator.

        Returns:
            OptimizeResult with optimized parameters and diagnostics.
        """
        # Flatten initial params and build bounds
        keys, x0 = self._flatten(initial_params)
        shapes = {k: jnp.atleast_1d(initial_params[k]).shape for k in keys}
        lower, upper = self._build_bounds_arrays(keys, initial_params)

        # Normalize initial point to [0,1]
        x0_norm = self._normalize(x0, lower, upper)

        # JIT-compiled vectorized loss: normalized [0,1] -> loss
        def eval_one(flat_norm: Array) -> Array:
            flat_orig = self._denormalize(flat_norm, lower, upper)
            params = self._unflatten(keys, flat_orig, shapes, initial_params)
            return jnp.squeeze(loss_fn(params))

        eval_population = jax.jit(jax.vmap(eval_one))

        # Initialize strategy in normalized space
        strategy = self._build_strategy(x0_norm)
        es_params = self._build_params(strategy)

        key = jax.random.key(self.seed)
        key, init_key = jax.random.split(key)
        state = self._init_state(strategy, init_key, x0_norm, es_params, eval_population)

        # Normalized bounds for clipping
        norm_lower = jnp.zeros_like(x0_norm)
        norm_upper = jnp.ones_like(x0_norm)

        # Optimization loop
        loss_history: list[float] = []
        best_loss = float("inf")
        best_flat_norm = x0_norm
        stall_count = 0
        converged = False

        for gen in range(n_generations):
            key, ask_key, tell_key = jax.random.split(key, 3)

            # Ask for new population (in [0,1] space)
            population, state = strategy.ask(ask_key, state, es_params)

            # Clip to [0,1]
            population = jnp.clip(population, norm_lower, norm_upper)

            # Evaluate fitness — JIT + vmap on GPU
            fitness = eval_population(population)

            # Tell results
            state, _metrics = strategy.tell(tell_key, population, fitness, state, es_params)

            # Track best
            min_fitness = float(jnp.min(fitness))
            loss_history.append(min_fitness)

            if min_fitness < best_loss * (1 - tol_fun):
                best_loss = min_fitness
                best_idx = jnp.argmin(fitness)
                best_flat_norm = population[best_idx]
                stall_count = 0
            else:
                stall_count += 1

            # Early stopping
            if stall_count >= patience:
                converged = True
                logger.info(
                    "Converged at generation %d: no improvement over %d generations (best_loss=%.6e)",
                    gen, patience, best_loss,
                )
                break

            # Logging
            if gen % 50 == 0:
                logger.info(
                    "gen %d/%d: best_loss=%.6e, stall=%d/%d",
                    gen, n_generations, best_loss, stall_count, patience,
                )

            # Progress bar
            if progress_bar:
                print_rate = max(1, n_generations // 20)
                if gen % print_rate == 0 or gen == n_generations - 1:
                    print(f"\r  [{gen+1}/{n_generations}] loss={best_loss:.4e}", end="", flush=True)
                if gen == n_generations - 1:
                    print()  # newline at end

        if progress_bar:
            print()  # newline after progress bar

        # Denormalize best result
        best_flat_orig = self._denormalize(best_flat_norm, lower, upper)
        best_params = self._unflatten(keys, best_flat_orig, shapes, initial_params)

        return OptimizeResult(
            params=best_params,
            loss=best_loss,
            loss_history=loss_history,
            n_iterations=len(loss_history),
            converged=converged,
            message=f"Converged after {len(loss_history)} generations"
            if converged
            else f"Reached max iterations ({n_generations})",
        )
