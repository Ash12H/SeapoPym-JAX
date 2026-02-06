"""Sobol sensitivity analysis for SeapoPym models.

Orchestrates the three phases of Sobol analysis:
1. Sampling (CPU): Generate Saltelli parameter samples via SALib
2. Evaluation (GPU): Run batched model evaluations via SobolRunner
3. Analysis (CPU): Compute Sobol indices via SALib

Example:
    >>> from seapopym.sensitivity import SobolAnalyzer
    >>> analyzer = SobolAnalyzer(compiled_model)
    >>> result = analyzer.analyze(
    ...     param_bounds={"growth_rate": (0.001, 0.1), "mortality": (0.01, 0.5)},
    ...     extraction_points=[(90, 180), (45, 90)],
    ...     n_samples=1024,
    ... )
    >>> print(result.S1)  # First-order Sobol indices
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from seapopym.sensitivity.checkpoint import SobolCheckpoint
from seapopym.sensitivity.qoi import available_qoi, compute_qoi
from seapopym.sensitivity.runner import SobolRunner

if TYPE_CHECKING:
    from seapopym.compiler import CompiledModel

logger = logging.getLogger(__name__)

Array = Any  # jax.Array | np.ndarray

_DEFAULT_QOI = ["mean", "var", "argmax", "median"]


@dataclass
class SobolResult:
    """Result of a Sobol sensitivity analysis.

    Attributes:
        S1: First-order indices. DataFrame with MultiIndex (qoi, point), columns = param names.
        ST: Total-order indices. Same structure as S1.
        S2: Second-order indices (if computed). None otherwise.
        S1_conf: Confidence intervals for S1.
        ST_conf: Confidence intervals for ST.
        n_samples: Base sample size N used.
        problem: SALib problem dict (for reproducibility).
    """

    S1: pd.DataFrame
    ST: pd.DataFrame
    S2: pd.DataFrame | None
    S1_conf: pd.DataFrame
    ST_conf: pd.DataFrame
    n_samples: int
    problem: dict = field(repr=False)


class SobolAnalyzer:
    """Orchestrates Sobol sensitivity analysis on a compiled model.

    Args:
        model: Compiled SeapoPym model (JAX backend required).
    """

    def __init__(self, model: CompiledModel) -> None:
        if model.backend != "jax":
            raise ValueError("SobolAnalyzer requires a model compiled with backend='jax'.")
        self.model = model

    def analyze(
        self,
        param_bounds: dict[str, tuple[float, float]],
        extraction_points: list[tuple[int, int]],
        output_variable: str = "biomass",
        n_samples: int = 1024,
        calc_second_order: bool = False,
        qoi: list[str] | None = None,
        batch_size: int = 256,
        chunk_size: int = 365,
        checkpoint_path: str | Path | None = None,
    ) -> SobolResult:
        """Run Sobol sensitivity analysis.

        Args:
            param_bounds: Dict mapping parameter names to (lower, upper) bounds.
            extraction_points: List of (y, x) grid indices for point extraction.
            output_variable: Name of the model output variable to analyze.
            n_samples: Base sample size N (must be a power of 2).
            calc_second_order: Whether to compute second-order indices.
            qoi: List of QoI names. Defaults to ["mean", "var", "argmax", "median"].
            batch_size: Number of parameter sets to evaluate simultaneously.
            chunk_size: Number of timesteps per temporal chunk.
            checkpoint_path: Path for Parquet checkpoint. Enables pause/resume.

        Returns:
            SobolResult with Sobol indices per QoI and per extraction point.
        """
        from SALib.analyze import sobol as sobol_analyze
        from SALib.sample import sobol as sobol_sample

        qoi_names = qoi if qoi is not None else _DEFAULT_QOI
        self._validate_inputs(param_bounds, qoi_names, n_samples)

        # Build SALib problem
        problem = {
            "num_vars": len(param_bounds),
            "names": list(param_bounds.keys()),
            "bounds": [list(b) for b in param_bounds.values()],
        }

        # Phase 1: Generate Saltelli samples (CPU)
        logger.info(f"Generating Saltelli samples: N={n_samples}, D={problem['num_vars']}")
        param_samples = sobol_sample.sample(problem, n_samples, calc_second_order=calc_second_order)
        n_total = param_samples.shape[0]
        logger.info(f"Total model evaluations: {n_total}")

        # Phase 2: Evaluate model in batches (GPU)
        runner = SobolRunner(self.model, extraction_points, output_variable, chunk_size)

        # Setup checkpoint if requested
        checkpoint = None
        start_sample = 0
        if checkpoint_path is not None:
            checkpoint = SobolCheckpoint(Path(checkpoint_path), problem, qoi_names, extraction_points)
            start_sample, _ = checkpoint.load()
            if start_sample > 0:
                logger.info(f"Resuming from sample {start_sample}/{n_total}")

        # Collect all QoI results
        all_qoi: dict[str, list[np.ndarray]] = {name: [] for name in qoi_names}

        # If resuming, load already computed QoI
        if start_sample > 0 and checkpoint is not None:
            _, existing_df = checkpoint.load()
            for qoi_name in qoi_names:
                all_qoi[qoi_name].append(checkpoint.extract_qoi_array(existing_df, qoi_name))

        # Process remaining batches
        for batch_start in range(start_sample, n_total, batch_size):
            batch_end = min(batch_start + batch_size, n_total)
            actual_batch_size = batch_end - batch_start

            logger.info(f"Evaluating batch {batch_start}-{batch_end} / {n_total}")

            # Extract parameter values for this batch
            batch_params_np = param_samples[batch_start:batch_end]

            # Pad if last batch is smaller (to avoid JIT recompilation)
            needs_padding = actual_batch_size < batch_size
            if needs_padding:
                pad_size = batch_size - actual_batch_size
                batch_params_np = np.concatenate(
                    [batch_params_np, np.zeros((pad_size, problem["num_vars"]))],
                    axis=0,
                )

            # Convert to JAX param dict (varied params only)
            varied_params = self._array_to_param_dict(batch_params_np, problem["names"])

            # Merge with model defaults: broadcast all params to batch, then override varied ones
            params_batch = {
                k: jnp.broadcast_to(v, (batch_size,) + v.shape) for k, v in self.model.parameters.items()
            }
            params_batch.update(varied_params)

            # Run batched simulation
            time_series = runner.run_batch(params_batch, batch_size)

            # Remove padding
            if needs_padding:
                time_series = time_series[:actual_batch_size]

            # Compute QoI
            qoi_values = compute_qoi(time_series, qoi_names)

            # Convert to numpy and collect
            qoi_np = {name: np.asarray(val) for name, val in qoi_values.items()}
            for name in qoi_names:
                all_qoi[name].append(qoi_np[name])

            # Save checkpoint
            if checkpoint is not None:
                checkpoint.save_batch(
                    sample_indices=np.arange(batch_start, batch_end),
                    params_values=param_samples[batch_start:batch_end],
                    qoi_values=qoi_np,
                )

        # Phase 3: Compute Sobol indices (CPU)
        logger.info("Computing Sobol indices...")
        return self._compute_indices(
            problem=problem,
            all_qoi=all_qoi,
            qoi_names=qoi_names,
            extraction_points=extraction_points,
            n_samples=n_samples,
            calc_second_order=calc_second_order,
            sobol_analyze=sobol_analyze,
        )

    def _validate_inputs(
        self,
        param_bounds: dict[str, tuple[float, float]],
        qoi_names: list[str],
        n_samples: int,
    ) -> None:
        """Validate analysis inputs."""
        # Check N is power of 2
        if n_samples & (n_samples - 1) != 0:
            raise ValueError(f"n_samples must be a power of 2, got {n_samples}")

        # Check params exist in model
        for name in param_bounds:
            if name not in self.model.parameters:
                raise ValueError(
                    f"Parameter '{name}' not found in model. Available: {list(self.model.parameters.keys())}"
                )

        # Check QoI names
        valid_qoi = available_qoi()
        for name in qoi_names:
            if name not in valid_qoi:
                raise ValueError(f"Unknown QoI '{name}'. Available: {valid_qoi}")

    @staticmethod
    def _array_to_param_dict(params_array: np.ndarray, param_names: list[str]) -> dict[str, Array]:
        """Convert parameter array to dict of JAX arrays.

        Args:
            params_array: Shape (batch_size, n_params).
            param_names: Parameter names matching columns.

        Returns:
            Dict mapping param name to JAX array of shape (batch_size,).
        """
        return {name: jnp.array(params_array[:, i]) for i, name in enumerate(param_names)}

    @staticmethod
    def _compute_indices(
        problem: dict,
        all_qoi: dict[str, list[np.ndarray]],
        qoi_names: list[str],
        extraction_points: list[tuple[int, int]],
        n_samples: int,
        calc_second_order: bool,
        sobol_analyze: Any,
    ) -> SobolResult:
        """Compute Sobol indices from collected QoI values.

        Args:
            problem: SALib problem dict.
            all_qoi: Dict mapping QoI name to list of arrays.
            qoi_names: Names of QoI computed.
            extraction_points: List of (y, x) points.
            n_samples: Base sample size.
            calc_second_order: Whether second-order was requested.
            sobol_analyze: SALib sobol analyze module.

        Returns:
            SobolResult with Sobol indices.
        """
        param_names = problem["names"]

        # Infer n_points from QoI data (0D models produce 1 point even with empty extraction_points)
        first_qoi = all_qoi[qoi_names[0]]
        n_points = np.concatenate(first_qoi, axis=0).shape[1] if first_qoi else len(extraction_points)

        # Build result DataFrames
        index_tuples = [(qoi_name, i) for qoi_name in qoi_names for i in range(n_points)]
        multi_index = pd.MultiIndex.from_tuples(index_tuples, names=["qoi", "point"])

        s1_data = np.zeros((len(index_tuples), len(param_names)))
        st_data = np.zeros_like(s1_data)
        s1_conf_data = np.zeros_like(s1_data)
        st_conf_data = np.zeros_like(s1_data)

        row = 0
        for qoi_name in qoi_names:
            # Concatenate all batches: (n_total, n_points)
            qoi_array = np.concatenate(all_qoi[qoi_name], axis=0)

            for point_idx in range(n_points):
                # SALib expects 1D array of model outputs
                y = qoi_array[:, point_idx]

                si = sobol_analyze.analyze(problem, y, calc_second_order=calc_second_order)

                s1_data[row, :] = si["S1"]
                st_data[row, :] = si["ST"]
                s1_conf_data[row, :] = si["S1_conf"]
                st_conf_data[row, :] = si["ST_conf"]

                row += 1

        s1_df = pd.DataFrame(s1_data, index=multi_index, columns=param_names)
        st_df = pd.DataFrame(st_data, index=multi_index, columns=param_names)
        s1_conf_df = pd.DataFrame(s1_conf_data, index=multi_index, columns=param_names)
        st_conf_df = pd.DataFrame(st_conf_data, index=multi_index, columns=param_names)

        return SobolResult(
            S1=s1_df,
            ST=st_df,
            S2=None,  # TODO: extract S2 if calc_second_order
            S1_conf=s1_conf_df,
            ST_conf=st_conf_df,
            n_samples=n_samples,
            problem=problem,
        )
