"""High-level Sampler for Bayesian calibration.

Orchestrates the sampling pipeline: builds a ``log_posterior`` from
Runner + PriorSet + Objectives + Likelihood, then runs NUTS.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from seapopym.optimization.likelihood import (
    GaussianLikelihood,
    reparameterize_log_posterior,
)
from seapopym.optimization.nuts import NUTSResult, run_nuts
from seapopym.optimization.objective import Objective, PreparedObjective
from seapopym.optimization.prior import HalfNormal, PriorSet
from seapopym.optimization.runner import CalibrationRunner
from seapopym.types import Array, Params

if TYPE_CHECKING:
    from seapopym.compiler.model import CompiledModel


class Sampler:
    """High-level Bayesian sampler for model calibration.

    Builds a log-posterior from objectives and likelihood, then runs
    NUTS via BlackJAX.

    Args:
        runner: Execution strategy (from :class:`CalibrationRunner`).
        priors: Defines free parameters and their prior distributions.
        objectives: List of :class:`Objective` instances.
        likelihood: Likelihood model (default: Gaussian with free sigma).
        sigma_prior: Prior on observation noise sigma (when sigma is
            free).  Ignored if ``likelihood.sigma`` is fixed.
        reparameterize: If ``True`` (default), sample in unit space
            ``[0, 1]^d`` for better conditioning.  Samples are
            converted back to physical space in the returned result.

    Example::

        sampler = Sampler(
            runner=CalibrationRunner.standard(),
            priors=PriorSet(priors={"growth_rate": Uniform(0.01, 1.0)}),
            objectives=[Objective(observations=obs_xr, target="biomass")],
            likelihood=GaussianLikelihood(sigma=0.1),
        )
        result = sampler.run(model, n_samples=1000)
    """

    def __init__(
        self,
        runner: CalibrationRunner,
        priors: PriorSet,
        objectives: list[Objective],
        likelihood: GaussianLikelihood | None = None,
        sigma_prior: HalfNormal | None = None,
        *,
        reparameterize: bool = True,
    ) -> None:
        self.runner = runner
        self.priors = priors
        self.objectives = objectives
        self.likelihood = likelihood or GaussianLikelihood()
        self.sigma_prior = sigma_prior or HalfNormal(scale=1.0)
        self.reparameterize = reparameterize

    def run(self, model: CompiledModel, **run_kwargs: Any) -> NUTSResult:
        """Run NUTS sampling.

        Args:
            model: Compiled model to calibrate.
            run_kwargs: Forwarded to :func:`run_nuts` (e.g.
                ``n_warmup``, ``n_samples``, ``seed``).

        Returns:
            Sampling result with posterior samples.  If
            ``reparameterize=True``, samples are in physical space.
        """
        # 1. Setup each objective
        prepared: list[PreparedObjective] = [
            obj.setup(model.coords) for obj in self.objectives
        ]

        # 2. Build log-posterior
        log_posterior_fn = self._build_log_posterior(model, prepared)

        # 3. Initial params for free parameters
        initial_params = {k: model.parameters[k] for k in self.priors.priors}

        # Add sigma to initial params if free
        if self.likelihood.sigma is None:
            initial_params["sigma"] = jnp.array(1.0)

        # 4. Reparameterize to unit space if requested
        if self.reparameterize:
            log_posterior_unit = reparameterize_log_posterior(
                log_posterior_fn, self.priors
            )
            initial_unit = self.priors.to_unit(initial_params)
            # sigma is not part of priors, pass through as-is
            if self.likelihood.sigma is None:
                initial_unit["sigma"] = initial_params["sigma"]

            result = run_nuts(log_posterior_unit, initial_unit, **run_kwargs)

            # Convert samples back to physical space
            result.samples = self.priors.from_unit(result.samples)
            return result

        # 5. Run in physical space
        return run_nuts(log_posterior_fn, initial_params, **run_kwargs)

    def _build_log_posterior(
        self,
        model: CompiledModel,
        prepared: list[PreparedObjective],
    ) -> Callable[[Params], Array]:
        """Build log-posterior: Σ(log_lik_i) + log_prior."""
        runner = self.runner
        priors = self.priors
        likelihood = self.likelihood
        sigma_prior = self.sigma_prior

        def log_posterior(params: Params) -> Array:
            outputs = runner(model, params)

            # Resolve sigma
            sigma = params["sigma"] if likelihood.sigma is None else jnp.array(likelihood.sigma)

            # Sum log-likelihoods over all objectives
            total_ll = jnp.array(0.0)
            for p in prepared:
                pred = p.extract_fn(outputs)
                total_ll = total_ll + likelihood.log_likelihood(
                    pred, p.obs_array, sigma=sigma
                )

            # Log-prior on model parameters
            lp = priors.log_prob(params)

            # Log-prior on sigma (if free)
            if likelihood.sigma is None and sigma_prior is not None:
                lp = lp + sigma_prior.log_prob(params["sigma"])

            return total_ll + lp

        return log_posterior
