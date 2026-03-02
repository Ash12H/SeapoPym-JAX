"""Calibration runner — executes the model for optimization or sampling.

The CalibrationRunner merges fixed parameters (from CompiledModel) with
free parameters (proposed by an Optimizer or Sampler), then runs the model.
It encapsulates the execution strategy (standard single-run or vmapped
over a population of parameter sets).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax

from seapopym.types import Outputs, Params

if TYPE_CHECKING:
    from seapopym.compiler.model import CompiledModel


@dataclass(frozen=True)
class CalibrationRunner:
    """Executes the model for calibration purposes.

    Merges fixed parameters (from ``CompiledModel.parameters``) with
    free parameters (proposed by Optimizer/Sampler) and runs the model.

    Use the factory methods to create instances::

        runner = CalibrationRunner.standard()   # single run
        runner = CalibrationRunner.vmapped()     # vmap over population

    Then call as a function::

        outputs = runner(model, free_params)

    Attributes:
        use_vmap: Whether to vmap over a population of parameter sets.
    """

    use_vmap: bool = False

    @classmethod
    def standard(cls) -> CalibrationRunner:
        """Single-run runner (carry + scan)."""
        return cls(use_vmap=False)

    @classmethod
    def vmapped(cls) -> CalibrationRunner:
        """Population runner (carry + scan + vmap over params)."""
        return cls(use_vmap=True)

    def __call__(self, model: CompiledModel, free_params: Params) -> Outputs:
        """Run the model with merged parameters.

        Args:
            model: Compiled model containing fixed parameters, state,
                and forcings.
            free_params: Free parameters proposed by the optimizer or
                sampler.  Keys that match ``model.parameters`` override
                the fixed values.

                For *vmapped* mode, each value must have a leading batch
                dimension (e.g. shape ``(pop_size,)`` for scalars).

        Returns:
            Model outputs (dict of arrays).  For vmapped mode, each
            value has an extra leading batch dimension.
        """
        if self.use_vmap:
            return self._run_vmap(model, free_params)
        return self._run_single(model, free_params)

    def _run_single(self, model: CompiledModel, free_params: Params) -> Outputs:
        """Merge fixed+free and run once."""
        merged = {**model.parameters, **free_params}
        _, outputs = model.run_with_params(merged)
        return outputs

    def _run_vmap(self, model: CompiledModel, free_params: Params) -> Outputs:
        """Vmap over a population of free parameter sets."""

        def run_one(single_free: Params) -> Outputs:
            merged = {**model.parameters, **single_free}
            _, outputs = model.run_with_params(merged)
            return outputs

        return jax.vmap(run_one)(free_params)
