"""Shared helpers for optimization classes.

Provides loss-building utilities, metric resolution, and parameter space
transformations used by CMAESOptimizer, GAOptimizer, IPOPCMAESOptimizer,
and GradientOptimizer.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from seapopym.engine.run import run
from seapopym.engine.step import build_step_fn
from seapopym.optimization.loss import mse, nrmse, rmse
from seapopym.optimization.objective import Objective, PreparedObjective
from seapopym.types import Array, Params

if TYPE_CHECKING:
    from seapopym.compiler.model import CompiledModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric resolution
# ---------------------------------------------------------------------------

METRICS: dict[str, Callable[[Array, Array], Array]] = {
    "mse": mse,
    "rmse": rmse,
    "nrmse": lambda p, o: nrmse(p, o, mode="std"),
    "nrmse_std": lambda p, o: nrmse(p, o, mode="std"),
    "nrmse_mean": lambda p, o: nrmse(p, o, mode="mean"),
    "nrmse_minmax": lambda p, o: nrmse(p, o, mode="minmax"),
}


def resolve_metric(metric: str | Callable[[Array, Array], Array]) -> Callable[[Array, Array], Array]:
    """Resolve a metric name to a callable."""
    if callable(metric):
        return metric
    if metric in METRICS:
        return METRICS[metric]
    available = sorted(METRICS.keys())
    msg = f"Unknown metric '{metric}'. Available: {available}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Objective helpers
# ---------------------------------------------------------------------------


def setup_objectives(
    objectives: list[tuple[Objective, str | Callable, float]],
    model_coords: dict[str, Array],
) -> list[tuple[PreparedObjective, Callable, float]]:
    """Prepare objectives for evaluation."""
    prepared: list[tuple[PreparedObjective, Callable, float]] = []
    for obj, metric, weight in objectives:
        p = obj.setup(model_coords)
        prepared.append((p, resolve_metric(metric), weight))
    return prepared


def build_loss_fn(
    model: CompiledModel,
    prepared_objectives: list[tuple[PreparedObjective, Callable, float]],
    export_variables: list[str] | None = None,
    chunk_size: int | None = None,
    checkpoint: bool = True,
) -> Callable[[Params], Array]:
    """Build composite loss: sum(w_i * metric_i).

    Args:
        model: Compiled model.
        prepared_objectives: List of (PreparedObjective, metric_fn, weight).
        export_variables: Variables to export from the simulation.
        chunk_size: Timesteps per chunk for ``run()``.
        checkpoint: If ``True`` (default), enable gradient checkpointing
            in ``run()`` to reduce memory usage during ``jax.grad``.
    """
    step_fn = build_step_fn(model, export_variables=export_variables)

    def loss_fn(free_params: Params) -> Array:
        merged = {**model.parameters, **free_params}
        state = dict(model.state)
        _, outputs = run(step_fn, model, state, merged, chunk_size=chunk_size, checkpoint=checkpoint)

        total = jnp.array(0.0)
        for p, metric_fn, weight in prepared_objectives:
            pred = p.extract_fn(outputs)
            total = total + weight * metric_fn(pred, p.obs_array)

        return total

    return loss_fn


# ---------------------------------------------------------------------------
# Parameter-space utilities
# ---------------------------------------------------------------------------


def flatten_params(params: Params) -> tuple[list[str], Array]:
    """Flatten parameter dict to sorted keys + 1D array."""
    keys = sorted(params.keys())
    values = [jnp.atleast_1d(params[k]).flatten() for k in keys]
    flat = jnp.concatenate(values) if values else jnp.array([])
    return keys, flat


def unflatten_params(
    keys: list[str],
    flat: Array,
    shapes: dict[str, tuple],
    original_params: Params | None = None,
) -> Params:
    """Reconstruct parameter dict from flat array."""
    params: Params = {}
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


def build_bounds_arrays(
    keys: list[str],
    params: Params,
    bounds: dict[str, tuple[float, float]],
) -> tuple[Array, Array]:
    """Build lower/upper bound arrays matching the flattened parameter vector."""
    lowers: list[float] = []
    uppers: list[float] = []
    for k in keys:
        size = jnp.atleast_1d(params[k]).size
        if k in bounds:
            low, high = bounds[k]
            if low >= high:
                msg = f"Invalid bounds for '{k}': low ({low}) >= high ({high})"
                raise ValueError(msg)
            lowers.extend([low] * size)
            uppers.extend([high] * size)
        else:
            lowers.extend([0.0] * size)
            uppers.extend([1.0] * size)
    return jnp.array(lowers), jnp.array(uppers)


def normalize(flat: Array, lower: Array, upper: Array) -> Array:
    """Transform from original space to [0,1]."""
    return (flat - lower) / (upper - lower)


def denormalize(flat_norm: Array, lower: Array, upper: Array) -> Array:
    """Transform from [0,1] back to original space."""
    return flat_norm * (upper - lower) + lower


# ---------------------------------------------------------------------------
# Hall-of-fame utilities
# ---------------------------------------------------------------------------


def params_distance(a: Params, b: Params, bounds: dict[str, tuple[float, float]]) -> float:
    """Euclidean distance between two parameter dicts, normalized by bounds.

    Only keys present in *bounds* contribute to the distance so that
    unbounded keys (whose raw scale may be arbitrary) do not dominate.
    """
    dist_sq = 0.0
    for key in a:
        if key not in b or key not in bounds:
            continue
        va = jnp.atleast_1d(a[key])
        vb = jnp.atleast_1d(b[key])
        low, high = bounds[key]
        va = (va - low) / (high - low)
        vb = (vb - low) / (high - low)
        diff = va - vb
        dist_sq += float(jnp.sum(diff**2))
    return float(jnp.sqrt(dist_sq))


class HallOfFame:
    """Maintains a diverse set of the best solutions found during optimization.

    At each update, candidates better than the worst member are considered.
    If a candidate is too close to an existing member (distance < threshold),
    the worse of the two is evicted. Otherwise the candidate is added and
    the worst member is evicted if the hall is full.

    Args:
        max_size: Maximum number of members.
        distance_threshold: Minimum normalized Euclidean distance between members.
        bounds: Parameter bounds for normalization ``{name: (min, max)}``.
    """

    def __init__(
        self,
        max_size: int,
        distance_threshold: float,
        bounds: dict[str, tuple[float, float]],
    ) -> None:
        self.max_size = max_size
        self.distance_threshold = distance_threshold
        self.bounds = bounds
        self.members: list[tuple[Params, float]] = []  # (params, loss)

    def update(self, params: Params, loss: float) -> None:
        """Try to insert a candidate into the hall-of-fame."""
        loss = float(loss)
        if not jnp.isfinite(loss):
            return

        # If hall not full, always consider; otherwise candidate must beat worst
        if len(self.members) >= self.max_size and loss >= self.members[-1][1]:
            return

        # Check distance against all current members
        for i, (m_params, m_loss) in enumerate(self.members):
            dist = params_distance(params, m_params, self.bounds)
            if dist < self.distance_threshold:
                # Too close — keep the better one
                if loss < m_loss:
                    self.members[i] = (params, loss)
                    self.members.sort(key=lambda x: x[1])
                return

        # New diverse member — add and evict worst if full
        self.members.append((params, loss))
        self.members.sort(key=lambda x: x[1])
        if len(self.members) > self.max_size:
            self.members.pop()

    def update_population(self, all_params: list[Params], all_losses: list[float]) -> None:
        """Update from a population of candidates (sorted best-first internally)."""
        # Sort by loss to process best candidates first
        paired = sorted(zip(all_losses, all_params, strict=True), key=lambda x: x[0])
        for loss, params in paired:
            self.update(params, loss)

    def to_results(self) -> list[dict]:
        """Export as list of dicts with params and loss."""
        return [{"params": p, "loss": loss} for p, loss in self.members]


# ---------------------------------------------------------------------------
# Shared evolution-strategy loop
# ---------------------------------------------------------------------------


def run_evolution_strategy(
    strategy,
    es_params,
    state,
    eval_population: Callable[[Array], Array],
    n_generations: int,
    tol_fun: float,
    patience: int,
    key: Array,
    norm_lower: Array,
    norm_upper: Array,
    x0_norm: Array,
    progress_bar: bool = False,
    hall_of_fame: HallOfFame | None = None,
    denorm_fn: Callable[[Array], Params] | None = None,
) -> tuple[Array, list[float], bool]:
    """Shared ask/tell loop for evolutionary strategies.

    Args:
        hall_of_fame: Optional HallOfFame to update each generation.
        denorm_fn: Function to convert flat normalized array to Params dict.
            Required when hall_of_fame is not None.

    Returns ``(best_flat_norm, loss_history, converged)``.
    """
    loss_history: list[float] = []
    best_loss = float("inf")
    best_flat_norm = x0_norm
    stall_count = 0
    converged = False

    for gen in range(n_generations):
        key, ask_key, tell_key = jax.random.split(key, 3)

        population, state = strategy.ask(ask_key, state, es_params)
        population = jnp.clip(population, norm_lower, norm_upper)
        fitness = eval_population(population)
        state, _metrics = strategy.tell(tell_key, population, fitness, state, es_params)

        min_fitness = float(jnp.min(fitness))
        loss_history.append(min_fitness)

        if min_fitness < best_loss - tol_fun:
            best_loss = min_fitness
            best_idx = jnp.argmin(fitness)
            best_flat_norm = population[best_idx]
            stall_count = 0
        else:
            stall_count += 1

        # Update hall-of-fame with this generation's population
        if hall_of_fame is not None and denorm_fn is not None:
            for i in range(population.shape[0]):
                f = float(fitness[i])
                if len(hall_of_fame.members) < hall_of_fame.max_size or f < hall_of_fame.members[-1][1]:
                    hall_of_fame.update(denorm_fn(population[i]), f)

        if stall_count >= patience:
            converged = True
            logger.info(
                "Converged at generation %d: no improvement over %d generations (best_loss=%.6e)",
                gen,
                patience,
                best_loss,
            )
            break

        if gen % 50 == 0:
            logger.info(
                "gen %d/%d: best_loss=%.6e, stall=%d/%d",
                gen,
                n_generations,
                best_loss,
                stall_count,
                patience,
            )

        if progress_bar:
            print_rate = max(1, n_generations // 20)
            if gen % print_rate == 0 or gen == n_generations - 1:
                print(f"\r  [{gen + 1}/{n_generations}] loss={best_loss:.4e}", end="", flush=True)

    if progress_bar:
        print()

    return best_flat_norm, loss_history, converged
