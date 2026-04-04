"""Multi-start gradient optimizer.

Launches N parallel Adam runs from diverse initial points sampled via
prior distributions. Combines gradient precision with global exploration.

Example::

    optimizer, loss_fn = MultiStartGradientOptimizer.from_model(
        model, objectives, bounds, n_starts=8,
    )
    result = optimizer.run(loss_fn, max_steps=200)
    print(result.best.loss)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import optax

from seapopym.optimization._common import OptimizeResult
from seapopym.optimization.prior import PriorSet, Uniform
from seapopym.types import Array, Params

if TYPE_CHECKING:
    from seapopym.compiler.model import CompiledModel
    from seapopym.optimization.objective import Objective

logger = logging.getLogger(__name__)


@dataclass
class MultiStartResult:
    """Result of a multi-start gradient optimization.

    Attributes:
        all_results: Individual ``OptimizeResult`` for each start.
    """

    all_results: list[OptimizeResult] = field(default_factory=list)

    @property
    def best(self) -> OptimizeResult:
        """Return the result with the lowest loss."""
        return min(self.all_results, key=lambda r: r.loss)


class MultiStartGradientOptimizer:
    """Multi-start gradient optimizer with model-agnostic API.

    Samples ``n_starts`` initial parameter sets from prior distributions
    (default: ``Uniform`` from bounds), then runs Adam in parallel via
    ``jax.vmap(jax.value_and_grad(loss_fn))``.

    Args:
        bounds: Parameter bounds as ``{name: (min, max)}``.  Required.
        n_starts: Number of parallel gradient runs.
        priors: Prior distributions for initialization sampling.
            ``None`` defaults to ``Uniform`` from bounds.
        algorithm: Optimization algorithm name.
        learning_rate: Learning rate (step size).
        scaling: Parameter scaling mode.
        seed: Random seed for initialization sampling.
    """

    ALGORITHMS = {
        "adam": optax.adam,
        "sgd": optax.sgd,
        "rmsprop": optax.rmsprop,
        "adagrad": optax.adagrad,
    }

    def __init__(
        self,
        bounds: dict[str, tuple[float, float]],
        n_starts: int = 8,
        priors: PriorSet | None = None,
        algorithm: Literal["adam", "sgd", "rmsprop", "adagrad"] = "adam",
        learning_rate: float = 0.01,
        scaling: Literal["none", "bounds"] = "bounds",
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Available: {list(self.ALGORITHMS.keys())}")

        self.bounds = bounds
        self.n_starts = n_starts
        self.priors = priors
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.scaling = scaling
        self.seed = seed

        optimizer_fn = self.ALGORITHMS[algorithm]
        self._optimizer = optimizer_fn(learning_rate, **kwargs)

        # Set by from_model() or manually before run()
        self._param_shapes: dict[str, tuple] | None = None

    def run(
        self,
        eval_fn: Callable[[Params], Array],
        max_steps: int = 200,
        tolerance: float = 1e-6,
        patience: int = 50,
        param_shapes: dict[str, tuple] | None = None,
    ) -> MultiStartResult:
        """Run multi-start gradient optimization.

        Args:
            eval_fn: Loss function mapping ``Params -> scalar``.
            max_steps: Maximum optimization steps per start.
            tolerance: Per-start convergence tolerance.
            patience: Global early stopping patience.
            param_shapes: Shape of each parameter ``{name: shape}``.
                Required if not set via :meth:`from_model`.

        Returns:
            MultiStartResult with all individual results.
        """
        shapes = param_shapes or self._param_shapes
        if shapes is None:
            msg = "param_shapes must be provided either via from_model() or as argument to run()"
            raise ValueError(msg)

        # Build PriorSet for initialization
        priors = self.priors
        if priors is None:
            priors = PriorSet({name: Uniform(low, high) for name, (low, high) in self.bounds.items()})

        # Sample N initial parameter sets
        key = jax.random.key(self.seed)
        free_param_names = sorted(self.bounds.keys())
        batched_shapes = {k: (self.n_starts, *shapes[k]) for k in free_param_names}
        batched_init = priors.sample(key, shapes=batched_shapes)

        return self._run_batched(eval_fn, batched_init, free_param_names, max_steps, tolerance, patience)

    def _run_batched(
        self,
        eval_fn: Callable[[Params], Array],
        batched_init: Params,
        free_param_names: list[str],
        max_steps: int,
        tolerance: float,
        patience: int,
    ) -> MultiStartResult:
        """Run N parallel gradient descents."""
        n_starts = self.n_starts

        # Normalize if scaling=bounds
        batched_params = self._normalize_batched(batched_init, free_param_names)

        # Initialize optax state (batched)
        opt_state = self._optimizer.init(batched_params)

        # Build vmapped value_and_grad
        def scaled_loss(params_norm: Params) -> Array:
            params_orig = self._denormalize(params_norm, free_param_names)
            return eval_fn(params_orig)

        batched_vg = jax.jit(jax.vmap(jax.value_and_grad(scaled_loss)))

        # Storage for per-start history
        loss_histories: list[list[float]] = [[] for _ in range(n_starts)]
        prev_losses = [float("inf")] * n_starts
        converged_flags = [False] * n_starts
        converged_iters = [max_steps] * n_starts

        # Global patience tracking
        global_best_loss = float("inf")
        stall_count = 0

        for i in range(max_steps):
            losses, grads = batched_vg(batched_params)
            loss_vals = [float(losses[j]) for j in range(n_starts)]

            for j in range(n_starts):
                loss_histories[j].append(loss_vals[j])
                if not converged_flags[j] and abs(prev_losses[j] - loss_vals[j]) < tolerance:
                    converged_flags[j] = True
                    converged_iters[j] = i + 1

            prev_losses = loss_vals

            # Global patience: track best loss across all starts
            current_best = min(loss_vals)
            if current_best < global_best_loss - tolerance:
                global_best_loss = current_best
                stall_count = 0
            else:
                stall_count += 1

            # Optax update (batched)
            updates, opt_state = self._optimizer.update(grads, opt_state, batched_params)
            batched_params: Params = optax.apply_updates(batched_params, updates)  # type: ignore[assignment]
            batched_params = self._clip_batched(batched_params, free_param_names)

            if i % 10 == 0:
                logger.info(
                    "Step %d/%d: best_loss=%.6e, stall=%d/%d", i, max_steps, current_best, stall_count, patience
                )

            # Stop early if global patience exhausted or ALL per-start converged
            if stall_count >= patience:
                for j in range(n_starts):
                    if not converged_flags[j]:
                        converged_flags[j] = True
                        converged_iters[j] = i + 1
                logger.info("Global patience exhausted at step %d (best_loss=%.6e)", i, global_best_loss)
                break

            if all(converged_flags):
                break

        # Build individual results
        all_results = []
        for j in range(n_starts):
            params_j = {k: batched_params[k][j] for k in free_param_names}
            params_orig = self._denormalize(params_j, free_param_names)
            final_loss = float(eval_fn(params_orig))
            all_results.append(
                OptimizeResult(
                    params=params_orig,
                    loss=final_loss,
                    loss_history=loss_histories[j],
                    n_iterations=converged_iters[j],
                    converged=converged_flags[j],
                    message=f"Start {j}: " + ("Converged" if converged_flags[j] else f"Reached max ({max_steps})"),
                )
            )

        return MultiStartResult(all_results=all_results)

    # ------------------------------------------------------------------
    # High-level model-aware entry point
    # ------------------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model: CompiledModel,
        objectives: list[tuple[Objective, str | Callable, float]],
        bounds: dict[str, tuple[float, float]],
        n_starts: int = 8,
        priors: PriorSet | None = None,
        algorithm: Literal["adam", "sgd", "rmsprop", "adagrad"] = "adam",
        learning_rate: float = 0.01,
        scaling: Literal["none", "bounds"] = "bounds",
        seed: int = 0,
        export_variables: list[str] | None = None,
        chunk_size: int | None = None,
        checkpoint: bool = True,
        **kwargs: Any,
    ) -> tuple[MultiStartGradientOptimizer, Callable[[Params], Array]]:
        """Create optimizer and loss function from a compiled model.

        Returns:
            Tuple of ``(optimizer, loss_fn)`` ready for ``run()``.

        Example::

            optimizer, loss_fn = MultiStartGradientOptimizer.from_model(
                model, objectives, bounds, n_starts=8,
            )
            result = optimizer.run(loss_fn, max_steps=200)
        """
        from seapopym.optimization._common import build_loss_fn, setup_objectives

        prepared = setup_objectives(objectives, model.coords)
        loss_fn = build_loss_fn(model, prepared, export_variables, chunk_size, checkpoint)

        optimizer = cls(
            bounds=bounds,
            n_starts=n_starts,
            priors=priors,
            algorithm=algorithm,
            learning_rate=learning_rate,
            scaling=scaling,
            seed=seed,
            **kwargs,
        )
        # Store param shapes for run() to use when sampling priors
        optimizer._param_shapes = {k: jnp.shape(model.parameters[k]) for k in sorted(bounds.keys())}

        return optimizer, loss_fn

    # ------------------------------------------------------------------
    # Scaling helpers (batched)
    # ------------------------------------------------------------------

    def _normalize_batched(self, batched_params: Params, keys: list[str]) -> Params:
        if self.scaling == "none":
            return batched_params
        result = {}
        for k in keys:
            if k in self.bounds:
                low, high = self.bounds[k]
                result[k] = (batched_params[k] - low) / (high - low)
            else:
                result[k] = batched_params[k]
        return result

    def _denormalize(self, params: Params, keys: list[str]) -> Params:
        if self.scaling == "none":
            return params
        result = {}
        for k in keys:
            if k in self.bounds:
                low, high = self.bounds[k]
                result[k] = params[k] * (high - low) + low
            else:
                result[k] = params[k]
        return result

    def _clip_batched(self, batched_params: Params, keys: list[str]) -> Params:
        if self.scaling == "bounds":
            return {k: jnp.clip(batched_params[k], 0.0, 1.0) for k in keys}
        if self.scaling == "none" and self.bounds:
            result = {}
            for k in keys:
                if k in self.bounds:
                    low, high = self.bounds[k]
                    result[k] = jnp.clip(batched_params[k], low, high)
                else:
                    result[k] = batched_params[k]
            return result
        return batched_params
