"""Deprecated Runner shim — delegates to :func:`run` and :func:`simulate`.

This module preserves backward compatibility for code that uses the
``Runner`` class (e.g. optimizers). New code should use ``run()`` and
``simulate()`` directly from ``seapopym.engine``.

.. deprecated::
    ``Runner`` and ``RunnerConfig`` will be removed in a future version.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import jax

from seapopym.types import Outputs, Params, State

from .exceptions import EngineError
from .run import run, simulate
from .step import build_step_fn

if TYPE_CHECKING:
    from seapopym.compiler import CompiledModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunnerConfig:
    """Configuration for Runner execution strategy.

    .. deprecated:: Use ``run()`` and ``simulate()`` directly.
    """

    vmap_params: bool = False
    chunk_size: int | None = None
    output_mode: Literal["disk", "memory", "raw"] = "raw"

    def __post_init__(self) -> None:
        if self.vmap_params and self.output_mode != "raw":
            msg = "vmap_params=True requires output_mode='raw'"
            raise EngineError(msg)


class Runner:
    """Deprecated runner — delegates to ``run()`` and ``simulate()``.

    Preserved for backward compatibility with optimizers. New code should
    use :func:`seapopym.engine.run` and :func:`seapopym.engine.simulate`.

    .. deprecated::
        Will be removed when the optimizer workflow is completed.
    """

    def __init__(self, config: RunnerConfig) -> None:
        warnings.warn(
            "Runner is deprecated. Use run() and simulate() from seapopym.engine instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.config = config

    # --- Presets ---

    @classmethod
    def simulation(
        cls,
        chunk_size: int | None = None,
        output: Literal["memory", "disk"] = "memory",
    ) -> Runner:
        """Create a runner configured for production simulation."""
        return cls(
            RunnerConfig(
                vmap_params=False,
                chunk_size=chunk_size,
                output_mode=output,
            )
        )

    @classmethod
    def optimization(cls, vmap: bool = False, chunk_size: int | None = None) -> Runner:
        """Create a runner configured for optimization/calibration."""
        return cls(
            RunnerConfig(
                vmap_params=vmap,
                chunk_size=chunk_size,
                output_mode="raw",
            )
        )

    # --- Simulation interface ---

    def run(
        self,
        model: CompiledModel,
        output_path: str | Path | None = None,
        export_variables: list[str] | None = None,
    ) -> tuple[State, Any | None]:
        """Execute the full simulation (delegates to simulate())."""
        return simulate(
            model,
            chunk_size=self.config.chunk_size,
            output_path=output_path,
            export_variables=export_variables,
        )

    # --- Optimization interface ---

    def __call__(
        self,
        model: CompiledModel,
        free_params: Params,
        export_variables: list[str] | None = None,
    ) -> Outputs:
        """Run the model with merged parameters (optimization interface)."""
        step_fn = build_step_fn(model, export_variables=export_variables)

        def eval_one(single_free: Params) -> Outputs:
            merged = {**model.parameters, **single_free}
            state = dict(model.state)
            _, outputs = run(step_fn, model, state, merged, chunk_size=self.config.chunk_size)
            return outputs

        if self.config.vmap_params:
            return jax.vmap(eval_one)(free_params)
        return eval_one(free_params)
