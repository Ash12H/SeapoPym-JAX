"""Hybrid optimization combining CMA-ES exploration with gradient refinement.

Uses CMA-ES for global exploration, then refines the top K candidates
using gradient-based optimization for faster local convergence.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import jax
import jax.numpy as jnp

from seapopym.optimization.evolutionary import EvolutionaryOptimizer
from seapopym.optimization.optimizer import GradientOptimizer, OptimizeResult
from seapopym.types import Array, Params

logger = logging.getLogger(__name__)


class HybridOptimizer:
    """CMA-ES exploration followed by gradient refinement on top K members.

    This hybrid approach combines the global exploration capabilities of
    CMA-ES with the fast local convergence of gradient-based methods.

    Example:
        >>> optimizer = HybridOptimizer(popsize=32, top_k=5, bounds={"x": (0, 10)})
        >>> result = optimizer.run(loss_fn, {"x": jnp.array(5.0)}, n_generations=50)
    """

    def __init__(
        self,
        popsize: int = 32,
        top_k: int = 5,
        bounds: dict[str, tuple[float, float]] | None = None,
        gradient_steps: int = 50,
        gradient_lr: float = 0.1,
        parallel_gradients: int | None = None,
        seed: int = 0,
    ) -> None:
        """Initialize the hybrid optimizer.

        Args:
            popsize: Population size for CMA-ES phase.
            top_k: Number of top candidates to refine with gradient descent.
            bounds: Parameter bounds as {param_name: (min, max)}.
            gradient_steps: Number of gradient descent steps per candidate.
            gradient_lr: Learning rate for gradient descent.
            parallel_gradients: Max number of gradient refinements to run in parallel.
                If None, runs all top_k in parallel. Set lower to reduce memory usage.
            seed: Random seed for reproducibility.
        """
        self.popsize = popsize
        self.top_k = top_k
        self.bounds = bounds or {}
        self.gradient_steps = gradient_steps
        self.gradient_lr = gradient_lr
        self.parallel_gradients = parallel_gradients or top_k
        self.seed = seed

    def run(
        self,
        loss_fn: Callable[[Params], Array],
        initial_params: Params,
        n_generations: int = 50,
        progress_bar: bool = False,
    ) -> OptimizeResult:
        """Run hybrid optimization.

        Phase 1: CMA-ES exploration for n_generations.
        Phase 2: Gradient refinement on top K candidates.

        Args:
            loss_fn: Function mapping params -> scalar loss.
            initial_params: Starting parameter values.
            n_generations: Number of CMA-ES generations.
            progress_bar: If True, display inline progress indicator.

        Returns:
            OptimizeResult with the best optimized parameters.
        """
        logger.info("=== Phase 1: CMA-ES Exploration ===")

        # Phase 1: CMA-ES
        evo_opt = EvolutionaryOptimizer(
            strategy="cma_es",
            popsize=self.popsize,
            bounds=self.bounds,
            seed=self.seed,
        )

        # Run CMA-ES and collect top K candidates
        top_candidates = self._run_cma_and_get_top_k(evo_opt, loss_fn, initial_params, n_generations, progress_bar)

        logger.info("=== Phase 2: Gradient Refinement (top %d) ===", self.top_k)

        # Phase 2: Gradient refinement
        grad_opt = GradientOptimizer(
            algorithm="adam",
            learning_rate=self.gradient_lr,
            bounds=self.bounds,
            scaling="bounds" if self.bounds else "none",
        )

        # Refine candidates (in batches if memory constrained)
        refined_results = self._refine_candidates(grad_opt, loss_fn, top_candidates)

        # Find best result
        best_result = min(refined_results, key=lambda r: r.loss)

        # Combine loss histories
        # CMA-ES history + best gradient history
        combined_history = (
            top_candidates[0][1]  # CMA-ES loss history from first candidate
            + best_result.loss_history
        )

        return OptimizeResult(
            params=best_result.params,
            loss=best_result.loss,
            loss_history=combined_history,
            n_iterations=n_generations + self.gradient_steps,
            converged=best_result.converged,
            message=f"CMA-ES ({n_generations} gen) + Gradient ({self.gradient_steps} steps)",
        )

    def _run_cma_and_get_top_k(
        self,
        evo_opt: EvolutionaryOptimizer,
        loss_fn: Callable[[Params], Array],
        initial_params: Params,
        n_generations: int,
        progress_bar: bool,
    ) -> list[tuple[Params, list[float], float]]:
        """Run CMA-ES and return top K candidates with their loss histories.

        Returns:
            List of (params, loss_history, final_loss) tuples for top K.
        """
        # We need to track the population during CMA-ES
        # For simplicity, we'll run CMA-ES and then evaluate the final population

        keys = sorted(initial_params.keys())
        shapes = {k: jnp.atleast_1d(initial_params[k]).shape for k in keys}

        # Flatten initial params
        flat_keys, x0 = evo_opt._flatten(initial_params)
        lower, upper = evo_opt._build_bounds_arrays(flat_keys, initial_params)

        # Initialize strategy
        from evosax.algorithms import CMA_ES

        strategy = CMA_ES(population_size=evo_opt.popsize, solution=x0)
        es_params = strategy.default_params

        key = jax.random.key(evo_opt.seed)
        key, init_key = jax.random.split(key)
        state = strategy.init(init_key, x0, es_params)

        # Create vectorized loss function
        def eval_one(flat_params: Array) -> Array:
            params = evo_opt._unflatten(flat_keys, flat_params, shapes, initial_params)
            loss = loss_fn(params)
            return jnp.squeeze(loss)

        eval_population = jax.vmap(eval_one)

        # Run CMA-ES
        loss_history: list[float] = []
        final_population = None
        final_fitness = None

        for gen in range(n_generations):
            key, ask_key, tell_key = jax.random.split(key, 3)

            population, state = strategy.ask(ask_key, state, es_params)
            population = jnp.clip(population, lower, upper)
            fitness = eval_population(population)
            state, _metrics = strategy.tell(tell_key, population, fitness, state, es_params)

            min_fitness = float(jnp.min(fitness))
            loss_history.append(min_fitness)

            if gen % 10 == 0:
                logger.info("Generation %d: best_loss = %.6e", gen, min_fitness)

            if progress_bar:
                print_rate = max(1, n_generations // 20)
                if gen % print_rate == 0 or gen == n_generations - 1:
                    print(f"\r  [{gen + 1}/{n_generations}] loss={min_fitness:.4e}", end="", flush=True)
                if gen == n_generations - 1:
                    print()

            final_population = population
            final_fitness = fitness

        # Get top K from final population
        if final_fitness is None or final_population is None:
            # Should never happen if n_generations > 0
            raise RuntimeError("CMA-ES did not run any generations")

        top_indices = jnp.argsort(final_fitness)[: self.top_k]

        top_candidates = []
        for idx in top_indices:
            flat_params = final_population[int(idx)]
            params = evo_opt._unflatten(flat_keys, flat_params, shapes, initial_params)
            candidate_loss = float(final_fitness[int(idx)])
            top_candidates.append((params, loss_history.copy(), candidate_loss))

        logger.info("Top %d losses: %s", self.top_k, [c[2] for c in top_candidates])

        return top_candidates

    def _refine_candidates(
        self,
        grad_opt: GradientOptimizer,
        loss_fn: Callable[[Params], Array],
        candidates: list[tuple[Params, list[float], float]],
    ) -> list[OptimizeResult]:
        """Refine candidates using gradient descent.

        Args:
            grad_opt: Gradient optimizer to use.
            loss_fn: Loss function.
            candidates: List of (params, loss_history, loss) tuples.

        Returns:
            List of OptimizeResult for each refined candidate.
        """
        results = []

        # Process in batches based on parallel_gradients
        for batch_start in range(0, len(candidates), self.parallel_gradients):
            batch_end = min(batch_start + self.parallel_gradients, len(candidates))
            batch = candidates[batch_start:batch_end]

            logger.info("Refining candidates %d-%d...", batch_start + 1, batch_end)

            for i, (params, _, _) in enumerate(batch):
                result = grad_opt.run(
                    loss_fn=loss_fn,
                    initial_params=params,
                    n_steps=self.gradient_steps,
                )
                results.append(result)

                logger.info("Candidate %d: loss %.6e", batch_start + i + 1, result.loss)

        return results
