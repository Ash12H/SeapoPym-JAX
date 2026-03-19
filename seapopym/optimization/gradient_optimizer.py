"""Gradient-based optimizer wrapping Optax algorithms.

Provides a high-level interface for gradient-based parameter optimization
with optional parameter bounds and automatic parameter scaling.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import optax

from seapopym.types import Array, Params

if TYPE_CHECKING:
    from seapopym.compiler.model import CompiledModel
    from seapopym.optimization.objective import Objective

logger = logging.getLogger(__name__)


@dataclass
class OptimizeResult:
    """Result of an optimization run.

    Attributes:
        params: Optimized parameter values.
        loss: Final loss value.
        loss_history: Loss value at each iteration.
        n_iterations: Number of iterations performed.
        converged: Whether optimization converged (loss change < tolerance).
        message: Human-readable status message.
    """

    params: Params
    loss: float
    loss_history: list[float] = field(default_factory=list)
    n_iterations: int = 0
    converged: bool = False
    message: str = ""
    hall_of_fame: list[OptimizeResult] | None = None


class GradientOptimizer:
    """Gradient-based optimizer for model calibration.

    Wraps Optax algorithms (adam, sgd, rmsprop, adagrad) with a high-level
    interface that handles objective setup, loss building, and parameter
    normalization.

    Args:
        objectives: List of ``(Objective, metric, weight)`` tuples.
        bounds: Optional parameter bounds as ``{name: (min, max)}``.
        algorithm: Optimization algorithm name.
        learning_rate: Learning rate (step size).
        scaling: Parameter scaling mode:
            - ``"none"``: No scaling (default)
            - ``"bounds"``: Normalize to [0,1] using bounds (requires bounds)
            - ``"log"``: Log-space for positive parameters
        chunk_size: Optional chunk size for time-stepping.
        checkpoint: If ``True`` (default), enable gradient checkpointing
            to reduce memory usage during backpropagation.

    Example::

        optimizer = GradientOptimizer(
            objectives=[(Objective(observations=obs, transform=fn), "nrmse", 1.0)],
            bounds={"rate": (0.0, 1.0)},
            algorithm="adam",
            learning_rate=0.01,
            scaling="bounds",
        )
        result = optimizer.run(model, n_steps=100)
    """

    ALGORITHMS = {
        "adam": optax.adam,
        "sgd": optax.sgd,
        "rmsprop": optax.rmsprop,
        "adagrad": optax.adagrad,
    }

    def __init__(
        self,
        objectives: list[tuple[Objective, str | Callable, float]],
        bounds: dict[str, tuple[float, float]] | None = None,
        algorithm: Literal["adam", "sgd", "rmsprop", "adagrad"] = "adam",
        learning_rate: float = 0.01,
        scaling: Literal["none", "bounds", "log"] = "none",
        export_variables: list[str] | None = None,
        chunk_size: int | None = None,
        checkpoint: bool = True,
        **kwargs: Any,
    ) -> None:
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Available: {list(self.ALGORITHMS.keys())}")

        if scaling == "bounds" and not bounds:
            raise ValueError("scaling='bounds' requires bounds to be provided")

        self.objectives = objectives
        self.bounds = bounds or {}
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.scaling = scaling
        self.export_variables = export_variables
        self.chunk_size = chunk_size
        self.checkpoint = checkpoint

        optimizer_fn = self.ALGORITHMS[algorithm]
        self._optimizer = optimizer_fn(learning_rate, **kwargs)
        self._opt_state: optax.OptState | None = None

    def run(
        self,
        model: CompiledModel,
        n_steps: int = 100,
        tolerance: float = 1e-6,
        progress_bar: bool = False,
    ) -> OptimizeResult:
        """Run gradient-based optimization.

        Args:
            model: Compiled model to calibrate.
            n_steps: Maximum number of optimization steps.
            tolerance: Convergence tolerance (stop if loss change < tolerance).
            progress_bar: If True, display inline progress indicator.

        Returns:
            OptimizeResult with optimized parameters and diagnostics.
        """
        from seapopym.optimization._common import build_loss_fn, setup_objectives

        prepared = setup_objectives(self.objectives, model.coords)
        loss_fn = build_loss_fn(model, prepared, self.export_variables, self.chunk_size, self.checkpoint)
        initial_params = {k: model.parameters[k] for k in self.bounds} if self.bounds else dict(model.parameters)

        return self._run_loss_fn(loss_fn, initial_params, n_steps, tolerance, progress_bar)

    def _run_loss_fn(
        self,
        loss_fn: Callable[[Params], Array],
        initial_params: Params,
        n_steps: int = 100,
        tolerance: float = 1e-6,
        progress_bar: bool = False,
    ) -> OptimizeResult:
        """Run optimization on a raw loss function."""
        params_norm = self._normalize(initial_params)
        self._opt_state = self._optimizer.init(params_norm)

        def scaled_loss_fn(params_norm: Params) -> Array:
            params_orig = self._denormalize(params_norm)
            return loss_fn(params_orig)

        value_and_grad_fn = jax.value_and_grad(scaled_loss_fn)

        loss_history: list[float] = []
        prev_loss = float("inf")
        converged = False

        for i in range(n_steps):
            loss, grads = value_and_grad_fn(params_norm)
            loss_val = float(loss)
            loss_history.append(loss_val)

            if abs(prev_loss - loss_val) < tolerance:
                converged = True
                logger.info("Converged at iteration %d with loss %.6e", i, loss_val)
                break

            updates, self._opt_state = self._optimizer.update(grads, self._opt_state, params_norm)
            params_norm = optax.apply_updates(params_norm, updates)
            params_norm = self._apply_bounds_normalized(params_norm)  # type: ignore[arg-type]

            if i % 10 == 0:
                logger.info("Iteration %d/%d: loss = %.6e", i, n_steps, loss_val)

            if progress_bar:
                print_rate = max(1, n_steps // 20)
                if i % print_rate == 0 or i == n_steps - 1:
                    print(f"\r  [{i + 1}/{n_steps}] loss={loss_val:.4e}", end="", flush=True)

            prev_loss = loss_val

        if progress_bar:
            print()

        final_params = self._denormalize(params_norm)
        final_loss = float(loss_fn(final_params))

        return OptimizeResult(
            params=final_params,
            loss=final_loss,
            loss_history=loss_history,
            n_iterations=len(loss_history),
            converged=converged,
            message="Converged" if converged else f"Reached max iterations ({n_steps})",
        )

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
