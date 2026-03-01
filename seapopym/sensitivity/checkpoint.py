"""Incremental checkpoint for Sobol analysis via Parquet.

Saves QoI results after each parameter batch, enabling pause/resume
for long-running sensitivity analyses.

Parquet schema:
    - sample_index (int): Global index in the Saltelli sample matrix
    - param_<name> (float): Parameter value for each analyzed parameter
    - qoi_<name>_point_<i> (float): QoI value at extraction point i
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SobolCheckpoint:
    """Manages Parquet persistence for incremental Sobol analysis.

    Each batch of evaluated parameter sets is appended to a Parquet file.
    Metadata (problem definition, QoI names) is stored in a companion JSON file
    to ensure consistency when resuming.

    Args:
        path: Path to the Parquet file (created if needed).
        problem: SALib problem dict (param names, bounds, etc.).
        qoi_names: List of QoI names being computed.
        extraction_points: List of (y, x) extraction point indices.
    """

    def __init__(
        self,
        path: Path,
        problem: dict,
        qoi_names: list[str],
        extraction_points: list[tuple[int, int]],
    ) -> None:
        self.path = Path(path)
        self.meta_path = self.path.with_suffix(".meta.json")
        self.problem = problem
        self.qoi_names = qoi_names
        self.extraction_points = extraction_points
        self._columns = self._build_columns()

    def _build_columns(self) -> list[str]:
        """Build the column names for the DataFrame."""
        cols = ["sample_index"]
        # Parameter columns
        for name in self.problem["names"]:
            cols.append(f"param_{name}")
        # QoI columns: one per (qoi, point) pair
        for qoi_name in self.qoi_names:
            for i in range(len(self.extraction_points)):
                cols.append(f"qoi_{qoi_name}_point_{i}")
        return cols

    def save_batch(
        self,
        sample_indices: np.ndarray,
        params_values: np.ndarray,
        qoi_values: dict[str, np.ndarray],
    ) -> None:
        """Append a batch of results to the Parquet file.

        Args:
            sample_indices: Global sample indices, shape (batch_size,).
            params_values: Parameter values, shape (batch_size, n_params).
            qoi_values: Dict mapping QoI name to array of shape (batch_size, n_points).
        """
        batch_size = len(sample_indices)
        data = np.empty((batch_size, len(self._columns)), dtype=np.float64)

        # Sample index
        data[:, 0] = sample_indices

        # Parameters
        n_params = len(self.problem["names"])
        data[:, 1 : 1 + n_params] = params_values

        # QoI values
        col_offset = 1 + n_params
        for qoi_name in self.qoi_names:
            arr = np.asarray(qoi_values[qoi_name])  # (batch_size, n_points)
            n_points = arr.shape[1]
            data[:, col_offset : col_offset + n_points] = arr
            col_offset += n_points

        df = pd.DataFrame(data, columns=self._columns)

        if self.path.exists():
            existing = pd.read_parquet(self.path)
            df = pd.concat([existing, df], ignore_index=True)

        df.to_parquet(self.path, index=False)
        self._save_metadata()

        logger.debug(f"Checkpoint saved: {batch_size} samples appended ({len(df)} total)")

    def load(self) -> tuple[int, pd.DataFrame]:
        """Load existing checkpoint.

        Returns:
            Tuple of (n_completed_samples, dataframe).
            If no checkpoint exists, returns (0, empty DataFrame).
        """
        if not self.path.exists():
            return 0, pd.DataFrame(columns=self._columns)

        self._validate_metadata()

        df = pd.read_parquet(self.path)
        n_done = len(df)
        logger.info(f"Checkpoint loaded: {n_done} samples completed")
        return n_done, df

    def extract_qoi_array(self, df: pd.DataFrame, qoi_name: str) -> np.ndarray:
        """Extract QoI values from checkpoint DataFrame.

        Args:
            df: Checkpoint DataFrame.
            qoi_name: Name of the QoI to extract.

        Returns:
            Array of shape (n_samples, n_points).
        """
        cols = [c for c in df.columns if c.startswith(f"qoi_{qoi_name}_point_")]
        return df[cols].values

    def _save_metadata(self) -> None:
        """Save metadata to companion JSON file."""
        meta = {
            "problem": self.problem,
            "qoi_names": self.qoi_names,
            "extraction_points": self.extraction_points,
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _validate_metadata(self) -> None:
        """Validate that existing metadata matches current configuration."""
        if not self.meta_path.exists():
            return

        with open(self.meta_path, encoding="utf-8") as f:
            existing = json.load(f)

        if existing["qoi_names"] != self.qoi_names:
            raise ValueError(
                f"QoI mismatch: checkpoint has {existing['qoi_names']}, but current analysis uses {self.qoi_names}"
            )
        if existing["problem"]["names"] != self.problem["names"]:
            raise ValueError(
                f"Parameter mismatch: checkpoint has {existing['problem']['names']}, "
                f"but current analysis uses {self.problem['names']}"
            )
