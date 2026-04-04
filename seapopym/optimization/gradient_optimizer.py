"""Gradient-based optimizer wrapping Optax algorithms.

Provides a step-based API for fine-grained control, and a convenience
``run()`` method for standard optimization with tolerance-based stopping.

Example (step-based)::

    optimizer = GradientOptimizer(
        bounds={"rate": (0.0, 1.0)},
        initial_params={"rate": 0.5},
        algorithm="adam",
        learning_rate=0.01,
        scaling="bounds",
    )
    for i in range(100):
        result = optimizer.step(loss_fn)
        print(f"step {result.step}: loss={result.loss:.4f}")

Example (convenience)::

    result = optimizer.run(loss_fn, max_steps=300, tolerance=1e-8)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import optax

from seapopym.optimization._common import GradientStepResult, OptimizeResult
from seapopym.types import Array, Params

if TYPE_CHECKING:
    from seapopym.compiler.model import CompiledModel
    from seapopym.optimization.objective import Objective

logger = logging.getLogger(__name__)


class GradientOptimizer:
    """Gradient-based optimizer with step-based API.

    The optimizer is initialized with bounds, initial parameters, and
    hyperparameters. Call :meth:`step` repeatedly to advance one gradient
    step at a time, or use :meth:`run` for a standard loop with tolerance.

    Args:
        bounds: Parameter bounds as ``{name: (min, max)}``.
        initial_params: Starting point as ``{name: value}``.
        algorithm: Optimization algorithm name.
        learning_rate: Learning rate (step size).
        scaling: Parameter scaling mode:
            - ``"none"``: No scaling (default)
            - ``"bounds"``: Normalize to [0,1] using bounds (requires bounds)
            - ``"log"``: Log-space for positive parameters
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
        initial_params: Params,
        algorithm: Literal["adam", "sgd", "rmsprop", "adagrad"] = "adam",
        learning_rate: float = 0.01,
        scaling: Literal["none", "bounds", "log"] = "none",
        **kwargs: Any,
    ) -> None:
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Available: {list(self.ALGORITHMS.keys())}")

        if scaling == "bounds" and not bounds:
            raise ValueError("scaling='bounds' requires bounds to be provided")

        self.bounds = bounds
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.scaling = scaling

        # Create optax optimizer eagerly
        optimizer_fn = self.ALGORITHMS[algorithm]
        self._optimizer = optimizer_fn(learning_rate, **kwargs)

        # Normalize initial params and init optimizer state
        self._params_norm = self._normalize(initial_params)
        self._opt_state = self._optimizer.init(self._params_norm)

        # Step counter
        self._step_count = 0

        # Lazy-build value_and_grad_fn (depends on eval_fn)
        self._current_eval_fn: Callable | None = None
        self._value_and_grad_fn: Callable | None = None

    def _build_value_and_grad(self, eval_fn: Callable[[Params], Array]) -> Callable:
        """Build value_and_grad function wrapping denormalization + eval."""

        def scaled_loss(params_norm: Params) -> Array:
            params_orig = self._denormalize(params_norm)
            return eval_fn(params_orig)

        return jax.value_and_grad(scaled_loss)

    def step(self, eval_fn: Callable[[Params], Array]) -> GradientStepResult:
        """Advance one gradient step.

        Args:
            eval_fn: Loss function mapping ``Params -> scalar``.

        Returns:
            GradientStepResult with loss, gradient norm, and current params.
        """
        # Rebuild value_and_grad if eval_fn changed
        if eval_fn is not self._current_eval_fn:
            self._value_and_grad_fn = self._build_value_and_grad(eval_fn)
            self._current_eval_fn = eval_fn

        vg_fn = self._value_and_grad_fn
        assert vg_fn is not None

        # Forward + backward
        loss, grads = vg_fn(self._params_norm)
        loss_val = float(loss)

        # Compute gradient norm
        grad_leaves = jax.tree.leaves(grads)
        grad_norm = float(jnp.sqrt(sum(jnp.sum(g**2) for g in grad_leaves)))

        # Update
        updates, self._opt_state = self._optimizer.update(grads, self._opt_state, self._params_norm)
        self._params_norm = optax.apply_updates(self._params_norm, updates)
        self._params_norm = self._apply_bounds_normalized(self._params_norm)  # type: ignore[arg-type]

        # Denormalize for result
        current_params = self._denormalize(self._params_norm)

        result = GradientStepResult(
            step=self._step_count,
            loss=loss_val,
            grad_norm=grad_norm,
            params=current_params,
        )

        self._step_count += 1
        return result

    def run(
        self,
        eval_fn: Callable[[Params], Array],
        max_steps: int = 100,
        tolerance: float = 1e-6,
    ) -> OptimizeResult:
        """Convenience: run gradient descent with tolerance-based stopping.

        Args:
            eval_fn: Loss function mapping ``Params -> scalar``.
            max_steps: Maximum number of optimization steps.
            tolerance: Stop if ``|prev_loss - loss| < tolerance``.

        Returns:
            OptimizeResult with optimized parameters and diagnostics.
        """
        loss_history: list[float] = []
        prev_loss = float("inf")
        converged = False

        for _ in range(max_steps):
            result = self.step(eval_fn)
            loss_history.append(result.loss)

            if abs(prev_loss - result.loss) < tolerance:
                converged = True
                break

            prev_loss = result.loss

        final_params = self._denormalize(self._params_norm)  # type: ignore[reportArgumentType]
        final_loss = float(eval_fn(final_params))

        return OptimizeResult(
            params=final_params,
            loss=final_loss,
            loss_history=loss_history,
            n_iterations=len(loss_history),
            converged=converged,
            message="Converged" if converged else f"Reached max iterations ({max_steps})",
        )

    # ------------------------------------------------------------------
    # High-level model-aware entry point
    # ------------------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model: CompiledModel,
        objectives: list[tuple[Objective, str | Callable, float]],
        bounds: dict[str, tuple[float, float]],
        algorithm: Literal["adam", "sgd", "rmsprop", "adagrad"] = "adam",
        learning_rate: float = 0.01,
        scaling: Literal["none", "bounds", "log"] = "none",
        export_variables: list[str] | None = None,
        chunk_size: int | None = None,
        checkpoint: bool = True,
        **kwargs: Any,
    ) -> tuple[GradientOptimizer, Callable[[Params], Array]]:
        """Create optimizer and loss function from a compiled model.

        Returns:
            Tuple of ``(optimizer, loss_fn)`` ready for ``step()`` or ``run()``.

        Example::

            optimizer, loss_fn = GradientOptimizer.from_model(
                model, objectives, bounds,
                algorithm="adam", learning_rate=0.01,
            )
            result = optimizer.run(loss_fn, max_steps=300)
        """
        from seapopym.optimization._common import build_loss_fn, setup_objectives

        prepared = setup_objectives(objectives, model.coords)
        loss_fn = build_loss_fn(model, prepared, export_variables, chunk_size, checkpoint)
        initial_params = {k: model.parameters[k] for k in bounds}

        optimizer = cls(
            bounds=bounds,
            initial_params=initial_params,
            algorithm=algorithm,
            learning_rate=learning_rate,
            scaling=scaling,
            **kwargs,
        )
        return optimizer, loss_fn

    # ------------------------------------------------------------------
    # Internal scaling helpers
    # ------------------------------------------------------------------

    def _normalize(self, params: Params) -> Params:
        if self.scaling == "none":
            return params

        new_params = {}
        for name, value in params.items():
            if self.scaling == "bounds" and name in self.bounds:
                low, high = self.bounds[name]
                new_params[name] = (value - low) / (high - low)
            elif self.scaling == "log":
                new_params[name] = jnp.log(value)
            else:
                new_params[name] = value
        return new_params

    def _denormalize(self, params: Params) -> Params:
        if self.scaling == "none":
            return params

        new_params = {}
        for name, value in params.items():
            if self.scaling == "bounds" and name in self.bounds:
                low, high = self.bounds[name]
                new_params[name] = value * (high - low) + low
            elif self.scaling == "log":
                new_params[name] = jnp.exp(value)
            else:
                new_params[name] = value
        return new_params

    def _apply_bounds_normalized(self, params_norm: Params) -> Params:
        if self.scaling == "bounds":
            new_params = {}
            for name, value in params_norm.items():
                if name in self.bounds:
                    new_params[name] = jnp.clip(value, 0.0, 1.0)
                else:
                    new_params[name] = value
            return new_params
        elif self.scaling == "none" and self.bounds:
            new_params = {}
            for name, value in params_norm.items():
                if name in self.bounds:
                    low, high = self.bounds[name]
                    new_params[name] = jnp.clip(value, low, high)
                else:
                    new_params[name] = value
            return new_params
        else:
            return params_norm
