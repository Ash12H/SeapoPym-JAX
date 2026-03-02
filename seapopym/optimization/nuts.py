"""NUTS sampler integration via BlackJAX.

Provides a simple interface to run NUTS (No-U-Turn Sampler) with
automatic warmup (step size and mass matrix adaptation).

Accepts a log-posterior function from likelihood.py and returns
samples with diagnostics.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import blackjax
import blackjax.progress_bar
import jax
import jax.numpy as jnp

from seapopym.types import Array, Params

logger = logging.getLogger(__name__)


@dataclass
class NUTSResult:
    """Result of a NUTS sampling run.

    Attributes:
        samples: Dict of parameter samples, {name: array of shape (n_samples,)}.
        log_posterior_values: Log-posterior at each sample, shape (n_samples,).
        divergences: Boolean array, True where a divergence occurred, shape (n_samples,).
        acceptance_rate: Mean acceptance probability across all samples.
        n_warmup: Number of warmup steps used.
        n_samples: Number of samples collected.
    """

    samples: Params
    log_posterior_values: Array
    divergences: Array
    acceptance_rate: float
    n_warmup: int
    n_samples: int
    kernel_params: dict | None = None


def run_nuts(
    log_posterior_fn: Callable[[Params], Array],
    initial_params: Params,
    n_warmup: int = 1000,
    n_samples: int = 2000,
    seed: int = 0,
    target_acceptance_rate: float = 0.8,
    is_mass_matrix_diagonal: bool = True,
    progress_bar: bool = False,
    kernel_params: dict | None = None,
) -> NUTSResult:
    """Run NUTS sampling with automatic warmup.

    Uses BlackJAX's window adaptation to tune step size and mass matrix,
    then collects samples via jax.lax.scan for efficiency.

    Args:
        log_posterior_fn: Function mapping Params -> scalar log-posterior.
            Must be compatible with jax.grad().
        initial_params: Starting position for the sampler (dict of arrays).
        n_warmup: Number of warmup steps for adaptation.
        n_samples: Number of samples to collect after warmup.
        seed: Random seed for reproducibility.
        target_acceptance_rate: Target acceptance rate for step size adaptation.
        is_mass_matrix_diagonal: If True, adapt a diagonal mass matrix.
            If False, adapt a dense mass matrix (more expensive but captures correlations).
        progress_bar: If True, display progress bars for warmup and sampling phases.
        kernel_params: Pre-computed kernel parameters (step_size + inverse_mass_matrix)
            from a previous warmup. If provided, warmup is skipped entirely.

    Returns:
        NUTSResult with samples, log-posterior values, and diagnostics.

    Example:
        >>> from seapopym.optimization.prior import Normal, PriorSet
        >>> prior_set = PriorSet({"x": Normal(0.0, 1.0)})
        >>> log_post = lambda p: -(p["x"] ** 2) + prior_set.log_prob(p)
        >>> result = run_nuts(log_post, {"x": jnp.array(0.0)}, n_warmup=200, n_samples=500)
        >>> result.samples["x"].shape
        (500,)
    """
    key = jax.random.key(seed)
    key, warmup_key, sample_key = jax.random.split(key, 3)

    if kernel_params is None:
        # Warmup: adapt step size and mass matrix
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            log_posterior_fn,
            is_mass_matrix_diagonal=is_mass_matrix_diagonal,
            target_acceptance_rate=target_acceptance_rate,
            progress_bar=progress_bar,
        )
        (state, kernel_params), _ = warmup.run(warmup_key, initial_params, num_steps=n_warmup)  # type: ignore[call-arg]
        logger.info("Warmup: %d steps, step_size=%.4g", n_warmup, float(kernel_params["step_size"]))
    else:
        # Skip warmup, init state from position
        logger.info("Using provided kernel_params (step_size=%.4g), skipping warmup", float(kernel_params["step_size"]))
        sampler = blackjax.nuts(log_posterior_fn, **kernel_params)
        state = sampler.init(initial_params)
        n_warmup = 0

    # Build sampling kernel
    kernel = blackjax.nuts(log_posterior_fn, **kernel_params).step

    # Sample via scan (with optional progress bar)
    # BlackJAX progress_bar expects xs=(arange, ...) so that iter_num is an int.
    # Without progress bar, we pass sample_keys directly.
    sample_keys = jax.random.split(sample_key, n_samples)

    if progress_bar:
        def one_step(state, x):  # type: ignore[no-untyped-def]
            _iter_num, rng_key = x
            state, info = kernel(rng_key, state)
            return state, (state.position, state.logdensity, info.is_divergent, info.acceptance_rate)  # type: ignore[attr-defined]

        scan_fn = blackjax.progress_bar.gen_scan_fn(n_samples, progress_bar=True)
        xs = (jnp.arange(n_samples), sample_keys)
    else:
        def one_step(state, rng_key):  # type: ignore[no-untyped-def]
            state, info = kernel(rng_key, state)
            return state, (state.position, state.logdensity, info.is_divergent, info.acceptance_rate)  # type: ignore[attr-defined]

        scan_fn = jax.lax.scan
        xs = sample_keys

    _, (positions, log_densities, divergences, acceptance_rates) = scan_fn(
        one_step, state, xs
    )

    # Compute mean acceptance rate
    mean_acceptance = float(jnp.mean(acceptance_rates))

    logger.info("Sampling: %d samples, accept=%.2f%%", n_samples, mean_acceptance * 100)

    return NUTSResult(
        samples=positions,
        log_posterior_values=log_densities,
        divergences=divergences,
        acceptance_rate=mean_acceptance,
        n_warmup=n_warmup,
        n_samples=n_samples,
        kernel_params=kernel_params,
    )
