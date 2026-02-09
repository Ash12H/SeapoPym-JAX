"""Gradient-based optimization runner.

Provides a runner that can compute gradients of the loss with respect
to model parameters, enabling gradient-based optimization.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp

from seapopym.engine.step import build_step_fn
from seapopym.optimization.loss import mse, nrmse, rmse
from seapopym.optimization.optimizer import Optimizer, OptimizeResult

if TYPE_CHECKING:
    from seapopym.compiler import CompiledModel

# Type aliases
Array = jnp.ndarray
Params = dict[str, Array]


@dataclass
class SparseObservations:
    """Sparse observations with indices.

    For efficient handling of observations that are sparse in space and/or time.

    Attributes:
        variable: Name of the variable being observed (e.g., "biomass").
        times: Time indices of observations.
        y: Y (latitude) indices of observations.
        x: X (longitude) indices of observations.
        values: Observed values at each (time, y, x) location.
    """

    variable: str
    times: Array
    y: Array
    x: Array
    values: Array

    def __post_init__(self) -> None:
        """Validate that all arrays have the same length."""
        n = len(self.values)
        if len(self.times) != n or len(self.y) != n or len(self.x) != n:
            raise ValueError(
                f"All observation arrays must have the same length. Got times={len(self.times)}, y={len(self.y)}, x={len(self.x)}, values={n}"
            )


class GradientRunner:
    """Runner for gradient-based parameter optimization.

    This runner wraps a compiled model and provides methods to:
    - Run the model with arbitrary parameter values
    - Compute loss between model outputs and observations
    - Compute gradients of the loss with respect to parameters
    - Run optimization loops

    Example:
        >>> runner = GradientRunner(compiled_model)
        >>> observations = SparseObservations(
        ...     variable="biomass",
        ...     times=jnp.array([10, 20, 30]),
        ...     y=jnp.array([5, 10, 15]),
        ...     x=jnp.array([5, 10, 15]),
        ...     values=jnp.array([1.0, 1.5, 2.0]),
        ... )
        >>> loss_fn = runner.make_loss_fn(observations)
        >>> grads = runner.compute_gradient(model.parameters, loss_fn)
    """

    def __init__(self, model: CompiledModel) -> None:
        """Initialize the gradient runner.

        Args:
            model: Compiled model to optimize.
        """
        self.model = model
        self._step_fn = build_step_fn(model, params_as_argument=True)

    def run_with_params(
        self,
        params: Params,
        initial_state: dict[str, Array] | None = None,
        forcings: dict[str, Array] | None = None,
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        """Run the model with specified parameters.

        Args:
            params: Parameter values to use for this run.
            initial_state: Initial state. If None, uses model's initial state.
            forcings: Forcing data. If None, uses model's forcings.

        Returns:
            Tuple of (final_state, outputs) where outputs contains all
            timesteps stacked along axis 0.
        """
        import jax.lax as lax

        # Use defaults if not provided
        if initial_state is None:
            initial_state = self.model.state
        if forcings is None:
            forcings = self.model.forcings.get_all()

        # Initial carry includes both state and params
        init_carry = (initial_state, params)

        # Run scan
        (final_state, _), outputs = lax.scan(
            self._step_fn,
            init_carry,
            forcings,
        )

        return final_state, outputs

    def make_loss_fn(
        self,
        observations: SparseObservations | list[SparseObservations],
        loss_type: Literal["mse", "rmse", "nrmse"] = "mse",
        nrmse_mode: Literal["std", "mean", "minmax"] = "std",
        weights: dict[str, float] | None = None,
    ) -> Callable[[Params], Array]:
        """Create a differentiable loss function.

        Args:
            observations: Sparse observations to compare against. Can be a single
                SparseObservations or a list for multiple variables.
            loss_type: Type of loss function ("mse", "rmse", or "nrmse").
            nrmse_mode: Normalization mode for NRMSE (if loss_type="nrmse").
            weights: Optional weights for each variable (by name).

        Returns:
            Function mapping params -> scalar loss, suitable for jax.grad().
        """
        # Normalize to list
        obs_list = [observations] if isinstance(observations, SparseObservations) else observations

        # Default weights
        if weights is None:
            weights = {obs.variable: 1.0 for obs in obs_list}

        # Select loss function
        if loss_type == "mse":
            loss_func = mse
        elif loss_type == "rmse":
            loss_func = rmse
        elif loss_type == "nrmse":

            def _nrmse_wrapper(predictions: Array, observations: Array, mask: Array | None = None) -> Array:
                return nrmse(predictions, observations, mask, mode=nrmse_mode)

            loss_func = _nrmse_wrapper
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        def compute_loss(params: Params) -> Array:
            """Compute total loss for given parameters."""
            # Run model
            _, outputs = self.run_with_params(params)

            total_loss = jnp.array(0.0)

            for obs in obs_list:
                # Get predictions for this variable
                if obs.variable not in outputs:
                    raise KeyError(
                        f"Variable '{obs.variable}' not found in model outputs. Available: {list(outputs.keys())}"
                    )

                pred_full = outputs[obs.variable]  # Shape: (T, Y, X) or (T, Y, X, C)

                # Extract predictions at observation locations
                # Handle different dimensionalities
                if pred_full.ndim == 3:
                    # (T, Y, X)
                    pred_sparse = pred_full[obs.times, obs.y, obs.x]
                elif pred_full.ndim == 4:
                    # (T, Y, X, C) - sum over cohorts or take first
                    pred_sparse = pred_full[obs.times, obs.y, obs.x, 0]
                else:
                    # Fallback: try direct indexing
                    pred_sparse = pred_full[obs.times, obs.y, obs.x]

                # Compute loss for this variable
                var_loss = loss_func(pred_sparse, obs.values, mask=None)

                # Weight and accumulate
                weight = weights.get(obs.variable, 1.0)
                total_loss = total_loss + weight * var_loss

            return total_loss

        return compute_loss

    def compute_gradient(
        self,
        params: Params,
        loss_fn: Callable[[Params], Array],
    ) -> Params:
        """Compute gradient of loss with respect to parameters.

        Args:
            params: Current parameter values.
            loss_fn: Loss function (e.g., from make_loss_fn).

        Returns:
            Gradients with same structure as params.
        """
        grad_fn = jax.grad(loss_fn)
        return grad_fn(params)

    def compute_value_and_gradient(
        self,
        params: Params,
        loss_fn: Callable[[Params], Array],
    ) -> tuple[Array, Params]:
        """Compute both loss value and gradient.

        More efficient than calling loss_fn and compute_gradient separately.

        Args:
            params: Current parameter values.
            loss_fn: Loss function.

        Returns:
            Tuple of (loss_value, gradients).
        """
        value_and_grad_fn = jax.value_and_grad(loss_fn)
        return value_and_grad_fn(params)

    def optimize(
        self,
        observations: SparseObservations | list[SparseObservations],
        params_to_optimize: list[str],
        optimizer: Optimizer | None = None,
        n_steps: int = 100,
        loss_type: Literal["mse", "rmse", "nrmse"] = "mse",
        verbose: bool = False,
    ) -> OptimizeResult:
        """Run full optimization loop.

        Args:
            observations: Sparse observations to fit.
            params_to_optimize: List of parameter names to optimize.
            optimizer: Optimizer instance. If None, uses Adam with lr=0.01.
            n_steps: Maximum number of optimization steps.
            loss_type: Type of loss function.
            verbose: If True, print progress.

        Returns:
            OptimizeResult with optimized parameters.
        """
        # Default optimizer
        if optimizer is None:
            optimizer = Optimizer(algorithm="adam", learning_rate=0.01)

        # Extract initial values for parameters to optimize
        initial_params = {name: self.model.parameters[name] for name in params_to_optimize}

        # Create loss function that only takes the subset of params
        full_loss_fn = self.make_loss_fn(observations, loss_type=loss_type)

        def partial_loss_fn(subset_params: Params) -> Array:
            """Loss function over subset of parameters."""
            # Merge subset with fixed params
            full_params = dict(self.model.parameters)
            full_params.update(subset_params)
            return full_loss_fn(full_params)

        # Run optimization
        result = optimizer.run(
            loss_fn=partial_loss_fn,
            initial_params=initial_params,
            n_steps=n_steps,
            verbose=verbose,
        )

        return result

    def extract_trainable_params(self) -> Params:
        """Extract parameters marked as trainable in the model.

        Returns:
            Dict of trainable parameter names to values.
        """
        trainable_names = getattr(self.model, "trainable_params", [])
        return {name: self.model.parameters[name] for name in trainable_names if name in self.model.parameters}
