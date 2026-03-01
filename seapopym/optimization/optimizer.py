"""Optimizer wrapper for gradient-based parameter optimization.

Provides a unified interface for different optimization algorithms
from Optax (and optionally JAXopt for L-BFGS).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import optax

from seapopym.types import Array, Params

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


class Optimizer:
    """Unified optimizer interface wrapping Optax algorithms.

    Supports gradient-based optimization with optional parameter bounds
    and automatic parameter scaling for better conditioning.

    Example:
        >>> optimizer = Optimizer(algorithm="adam", learning_rate=0.01)
        >>> optimizer = Optimizer(
        ...     algorithm="adam",
        ...     learning_rate=0.01,
        ...     bounds={"rate": (0.0, 1.0), "scale": (0.1, 10.0)},
        ...     scaling="bounds",  # Normalize parameters to [0, 1]
        ... )
    """

    ALGORITHMS = {
        "adam": optax.adam,
        "sgd": optax.sgd,
        "rmsprop": optax.rmsprop,
        "adagrad": optax.adagrad,
    }

    def __init__(
        self,
        algorithm: Literal["adam", "sgd", "rmsprop", "adagrad"] = "adam",
        learning_rate: float = 0.01,
        bounds: dict[str, tuple[float, float]] | None = None,
        scaling: Literal["none", "bounds", "log"] = "none",
        **kwargs: Any,
    ) -> None:
        """Initialize the optimizer.

        Args:
            algorithm: Optimization algorithm name.
            learning_rate: Learning rate (step size).
            bounds: Optional parameter bounds as {param_name: (min, max)}.
            scaling: Parameter scaling mode:
                - "none": No scaling (default, backward compatible)
                - "bounds": Normalize to [0, 1] using bounds (requires bounds)
                - "log": Use log-space for positive parameters
            **kwargs: Additional arguments passed to the Optax optimizer.
        """
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Available: {list(self.ALGORITHMS.keys())}")

        if scaling == "bounds" and not bounds:
            raise ValueError("scaling='bounds' requires bounds to be provided")

        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.bounds = bounds or {}
        self.scaling = scaling

        # Create Optax optimizer
        optimizer_fn = self.ALGORITHMS[algorithm]
        self._optimizer = optimizer_fn(learning_rate, **kwargs)
        self._opt_state: optax.OptState | None = None

    def init(self, params: Params) -> None:
        """Initialize optimizer state for given parameters.

        Args:
            params: Initial parameter values.
        """
        self._opt_state = self._optimizer.init(params)

    def step(self, params: Params, grads: Params) -> Params:
        """Perform one optimization step.

        Args:
            params: Current parameter values.
            grads: Gradients of loss with respect to parameters.

        Returns:
            Updated parameter values.
        """
        if self._opt_state is None:
            self.init(params)

        # Compute updates
        updates, self._opt_state = self._optimizer.update(grads, self._opt_state, params)

        # Apply updates
        new_params = optax.apply_updates(params, updates)

        # Apply bounds (projection)
        # Note: optax returns generic Params type, but we use dict[str, Array]
        new_params = self._apply_bounds(new_params)  # type: ignore[arg-type]

        return new_params

    def _apply_bounds(self, params: Params) -> Params:
        """Project parameters to satisfy bounds.

        Args:
            params: Parameter values to project.

        Returns:
            Projected parameter values within bounds.
        """
        if not self.bounds:
            return params

        new_params = {}
        for name, value in params.items():
            if name in self.bounds:
                low, high = self.bounds[name]
                new_params[name] = jnp.clip(value, low, high)
            else:
                new_params[name] = value

        return new_params

    def _normalize(self, params: Params) -> Params:
        """Transform parameters to normalized space.

        Args:
            params: Parameter values in original space.

        Returns:
            Parameter values in normalized space.
        """
        if self.scaling == "none":
            return params

        new_params = {}
        for name, value in params.items():
            if self.scaling == "bounds" and name in self.bounds:
                low, high = self.bounds[name]
                # Normalize to [0, 1]
                new_params[name] = (value - low) / (high - low)
            elif self.scaling == "log":
                # Log transform for positive parameters
                new_params[name] = jnp.log(value)
            else:
                new_params[name] = value

        return new_params

    def _denormalize(self, params: Params) -> Params:
        """Transform parameters back to original space.

        Args:
            params: Parameter values in normalized space.

        Returns:
            Parameter values in original space.
        """
        if self.scaling == "none":
            return params

        new_params = {}
        for name, value in params.items():
            if self.scaling == "bounds" and name in self.bounds:
                low, high = self.bounds[name]
                # Denormalize from [0, 1]
                new_params[name] = value * (high - low) + low
            elif self.scaling == "log":
                # Exp transform back
                new_params[name] = jnp.exp(value)
            else:
                new_params[name] = value

        return new_params

    def _step_normalized(self, params_norm: Params, grads: Params) -> Params:
        """Perform one optimization step in normalized space.

        Args:
            params_norm: Current parameter values in normalized space.
            grads: Gradients in normalized space.

        Returns:
            Updated parameter values in normalized space.
        """
        if self._opt_state is None:
            self.init(params_norm)

        # Compute updates
        updates, self._opt_state = self._optimizer.update(grads, self._opt_state, params_norm)

        # Apply updates
        new_params = optax.apply_updates(params_norm, updates)

        # Apply bounds in normalized space
        new_params = self._apply_bounds_normalized(new_params)  # type: ignore[arg-type]

        return new_params

    def _apply_bounds_normalized(self, params_norm: Params) -> Params:
        """Apply bounds in normalized space.

        Args:
            params_norm: Parameter values in normalized space.

        Returns:
            Bounded parameter values in normalized space.
        """
        if self.scaling == "bounds":
            # In normalized space, bounds are [0, 1]
            new_params = {}
            for name, value in params_norm.items():
                if name in self.bounds:
                    new_params[name] = jnp.clip(value, 0.0, 1.0)
                else:
                    new_params[name] = value
            return new_params
        elif self.scaling == "none" and self.bounds:
            # Apply original bounds
            return self._apply_bounds(params_norm)
        else:
            # No bounds to apply (log scaling has no natural bounds)
            return params_norm

    def run(
        self,
        loss_fn: Callable[[Params], Array],
        initial_params: Params,
        n_steps: int = 100,
        tolerance: float = 1e-6,
        callback: Callable[[int, Params, float], None] | None = None,
        progress_bar: bool = False,
    ) -> OptimizeResult:
        """Run the optimization loop.

        Args:
            loss_fn: Function mapping params -> scalar loss.
            initial_params: Starting parameter values.
            n_steps: Maximum number of optimization steps.
            tolerance: Convergence tolerance (stop if loss change < tolerance).
            callback: Optional function called at each step with (iteration, params, loss).
                Note: callback receives denormalized (original space) params.
            progress_bar: If True, display inline progress indicator.

        Returns:
            OptimizeResult with optimized parameters and diagnostics.
        """
        # Normalize initial params
        params_norm = self._normalize(initial_params)
        self.init(params_norm)

        # Wrap loss_fn to denormalize before evaluation
        def scaled_loss_fn(params_norm: Params) -> Array:
            params_orig = self._denormalize(params_norm)
            return loss_fn(params_orig)

        # Create gradient function
        value_and_grad_fn = jax.value_and_grad(scaled_loss_fn)

        loss_history: list[float] = []
        prev_loss = float("inf")
        converged = False

        for i in range(n_steps):
            # Compute loss and gradients (in normalized space)
            loss, grads = value_and_grad_fn(params_norm)
            loss_val = float(loss)
            loss_history.append(loss_val)

            # Check convergence
            if abs(prev_loss - loss_val) < tolerance:
                converged = True
                logger.info("Converged at iteration %d with loss %.6e", i, loss_val)
                break

            # Update parameters (in normalized space)
            params_norm = self._step_normalized(params_norm, grads)

            # Callback with denormalized params
            if callback is not None:
                callback(i, self._denormalize(params_norm), loss_val)

            # Logging
            if i % 10 == 0:
                logger.info("Iteration %d/%d: loss = %.6e", i, n_steps, loss_val)

            # Progress bar
            if progress_bar:
                print_rate = max(1, n_steps // 20)
                if i % print_rate == 0 or i == n_steps - 1:
                    print(f"\r  [{i+1}/{n_steps}] loss={loss_val:.4e}", end="", flush=True)

            prev_loss = loss_val

        if progress_bar:
            print()  # newline after progress bar

        # Denormalize final params
        final_params = self._denormalize(params_norm)

        # Final evaluation
        final_loss = float(loss_fn(final_params))

        return OptimizeResult(
            params=final_params,
            loss=final_loss,
            loss_history=loss_history,
            n_iterations=len(loss_history),
            converged=converged,
            message="Converged" if converged else f"Reached max iterations ({n_steps})",
        )
