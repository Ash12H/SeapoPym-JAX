"""High-level Optimizer for parameter calibration.

Orchestrates the calibration pipeline: assembles a ``loss_fn`` from
Runner + PriorSet + Objectives, then dispatches to the appropriate
low-level optimizer (gradient or evolutionary).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from seapopym.engine.runner import Runner
from seapopym.optimization.gradient_optimizer import GradientOptimizer, OptimizeResult
from seapopym.optimization.loss import mse, nrmse, rmse
from seapopym.optimization.objective import Objective, PreparedObjective
from seapopym.optimization.prior import PriorSet
from seapopym.types import Array, Params

if TYPE_CHECKING:
    from seapopym.compiler.model import CompiledModel

# Metric name → callable(predictions, observations) → scalar
METRICS: dict[str, Callable[[Array, Array], Array]] = {
    "mse": mse,
    "rmse": rmse,
    "nrmse": lambda p, o: nrmse(p, o, mode="std"),
    "nrmse_std": lambda p, o: nrmse(p, o, mode="std"),
    "nrmse_mean": lambda p, o: nrmse(p, o, mode="mean"),
    "nrmse_minmax": lambda p, o: nrmse(p, o, mode="minmax"),
}

GRADIENT_STRATEGIES = {"adam", "sgd", "rmsprop", "adagrad"}
EVOLUTIONARY_STRATEGIES = {"cma_es", "simple_ga"}


def _resolve_metric(metric: str | Callable[[Array, Array], Array]) -> Callable[[Array, Array], Array]:
    """Resolve a metric name to a callable."""
    if callable(metric):
        return metric
    if metric in METRICS:
        return METRICS[metric]
    available = sorted(METRICS.keys())
    msg = f"Unknown metric '{metric}'. Available: {available}"
    raise ValueError(msg)


class Optimizer:
    """High-level optimizer for model calibration.

    Assembles a loss function from objectives, metrics, and weights,
    then dispatches to a low-level optimizer based on ``strategy``.

    Args:
        runner: Execution strategy (from :class:`Runner`).
        priors: Defines which parameters are free and their constraints.
        objectives: List of ``(Objective, metric, weight)`` tuples.
            *metric* can be a string name (``"rmse"``, ``"nrmse_std"``, ...)
            or a callable ``(predictions, observations) -> scalar``.
        strategy: Optimization algorithm.  Gradient-based: ``"adam"``,
            ``"sgd"``, ``"rmsprop"``, ``"adagrad"``.  Evolutionary:
            ``"cma_es"``, ``"simple_ga"``.
        strategy_kwargs: Extra keyword arguments forwarded to the
            low-level optimizer constructor (e.g. ``learning_rate``,
            ``popsize``).

    Example::

        optimizer = Optimizer(
            runner=Runner.optimization(),
            priors=PriorSet(priors={"growth_rate": Uniform(0.01, 1.0)}),
            objectives=[
                (Objective(observations=obs_xr, target="biomass"), "nrmse_std", 1.0),
            ],
            strategy="cma_es",
        )
        result = optimizer.run(model, n_generations=100)
    """

    def __init__(
        self,
        runner: Runner,
        priors: PriorSet,
        objectives: list[tuple[Objective, str | Callable, float]],
        strategy: str = "cma_es",
        **strategy_kwargs: Any,
    ) -> None:
        all_strategies = GRADIENT_STRATEGIES | EVOLUTIONARY_STRATEGIES
        if strategy not in all_strategies:
            msg = f"Unknown strategy '{strategy}'. Available: {sorted(all_strategies)}"
            raise ValueError(msg)

        self.runner = runner
        self.priors = priors
        self.objectives = objectives
        self.strategy = strategy
        self.strategy_kwargs = strategy_kwargs

    def run(self, model: CompiledModel, **run_kwargs: Any) -> OptimizeResult:
        """Run the optimization.

        Args:
            model: Compiled model to calibrate.
            run_kwargs: Forwarded to the low-level optimizer's ``run()``
                method (e.g. ``n_steps``, ``n_generations``,
                ``tolerance``).

        Returns:
            Optimization result with best parameters found.
        """
        # 1. Setup each objective
        prepared: list[tuple[PreparedObjective, Callable, float]] = []
        for obj, metric, weight in self.objectives:
            p = obj.setup(model.coords)
            prepared.append((p, _resolve_metric(metric), weight))

        # 2. Build composite loss function
        loss_fn = self._build_loss_fn(model, prepared)

        # 3. Extract initial values for free parameters
        initial_params = {k: model.parameters[k] for k in self.priors.priors}

        # 4. Dispatch to low-level optimizer
        return self._dispatch(loss_fn, initial_params, **run_kwargs)

    def _build_loss_fn(
        self,
        model: CompiledModel,
        prepared: list[tuple[PreparedObjective, Callable, float]],
    ) -> Callable[[Params], Array]:
        """Build the composite loss: sum(w_i * metric_i) + prior_penalty."""
        runner = self.runner
        priors = self.priors

        def loss_fn(free_params: Params) -> Array:
            outputs = runner(model, free_params)

            total = jnp.array(0.0)
            for p, metric_fn, weight in prepared:
                pred = p.extract_fn(outputs)
                total = total + weight * metric_fn(pred, p.obs_array)

            # Prior penalty: -log_prob penalizes unlikely params
            penalty = -priors.log_prob(free_params)
            total = total + penalty
            return total

        return loss_fn

    def _dispatch(
        self,
        loss_fn: Callable[[Params], Array],
        initial_params: Params,
        **run_kwargs: Any,
    ) -> OptimizeResult:
        """Route to the correct low-level optimizer."""
        bounds = self.priors.get_bounds()

        if self.strategy in GRADIENT_STRATEGIES:
            opt = GradientOptimizer(
                algorithm=self.strategy,  # type: ignore[arg-type]
                bounds=bounds,
                scaling="bounds",
                **self.strategy_kwargs,
            )
            return opt.run(loss_fn, initial_params, **run_kwargs)

        if self.strategy in EVOLUTIONARY_STRATEGIES:
            from seapopym.optimization.evolutionary import EvolutionaryOptimizer

            opt_evo = EvolutionaryOptimizer(
                strategy=self.strategy,  # type: ignore[arg-type]
                bounds=bounds,
                **self.strategy_kwargs,
            )
            return opt_evo.run(loss_fn, initial_params, **run_kwargs)

        msg = f"Unknown strategy '{self.strategy}'"
        raise ValueError(msg)
