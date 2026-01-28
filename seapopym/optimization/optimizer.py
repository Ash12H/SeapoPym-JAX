"""Optimizer wrapper for gradient-based parameter optimization.

Provides a unified interface for different optimization algorithms
from Optax (and optionally JAXopt for L-BFGS).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import optax

# Type aliases
Array = jnp.ndarray
Params = dict[str, Array]


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

    Supports gradient-based optimization with optional parameter bounds.
    Bounds are enforced by projecting parameters after each update.

    Example:
        >>> optimizer = Optimizer(algorithm="adam", learning_rate=0.01)
        >>> optimizer = Optimizer(
        ...     algorithm="adam",
        ...     learning_rate=0.01,
        ...     bounds={"rate": (0.0, 1.0), "scale": (0.1, 10.0)}
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
        **kwargs: Any,
    ) -> None:
        """Initialize the optimizer.

        Args:
            algorithm: Optimization algorithm name.
            learning_rate: Learning rate (step size).
            bounds: Optional parameter bounds as {param_name: (min, max)}.
            **kwargs: Additional arguments passed to the Optax optimizer.
        """
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Available: {list(self.ALGORITHMS.keys())}")

        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.bounds = bounds or {}

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

    def run(
        self,
        loss_fn: Callable[[Params], Array],
        initial_params: Params,
        n_steps: int = 100,
        tolerance: float = 1e-6,
        callback: Callable[[int, Params, float], None] | None = None,
        verbose: bool = False,
    ) -> OptimizeResult:
        """Run the optimization loop.

        Args:
            loss_fn: Function mapping params -> scalar loss.
            initial_params: Starting parameter values.
            n_steps: Maximum number of optimization steps.
            tolerance: Convergence tolerance (stop if loss change < tolerance).
            callback: Optional function called at each step with (iteration, params, loss).
            verbose: If True, print progress every 10 iterations.

        Returns:
            OptimizeResult with optimized parameters and diagnostics.
        """
        # Initialize
        params = initial_params
        self.init(params)

        # Create gradient function
        value_and_grad_fn = jax.value_and_grad(loss_fn)

        loss_history: list[float] = []
        prev_loss = float("inf")
        converged = False

        for i in range(n_steps):
            # Compute loss and gradients
            loss, grads = value_and_grad_fn(params)
            loss_val = float(loss)
            loss_history.append(loss_val)

            # Check convergence
            if abs(prev_loss - loss_val) < tolerance:
                converged = True
                if verbose:
                    print(f"Converged at iteration {i} with loss {loss_val:.6e}")
                break

            # Update parameters
            params = self.step(params, grads)

            # Callback
            if callback is not None:
                callback(i, params, loss_val)

            # Verbose output
            if verbose and i % 10 == 0:
                print(f"Iteration {i}: loss = {loss_val:.6e}")

            prev_loss = loss_val

        # Final evaluation
        final_loss = float(loss_fn(params))

        return OptimizeResult(
            params=params,
            loss=final_loss,
            loss_history=loss_history,
            n_iterations=len(loss_history),
            converged=converged,
            message="Converged" if converged else f"Reached max iterations ({n_steps})",
        )
