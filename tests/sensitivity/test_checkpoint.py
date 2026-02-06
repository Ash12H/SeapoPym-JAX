"""Tests for SobolCheckpoint (Parquet persistence)."""

import numpy as np
import pytest

from seapopym.sensitivity.checkpoint import SobolCheckpoint


@pytest.fixture
def problem():
    return {
        "num_vars": 2,
        "names": ["alpha", "beta"],
        "bounds": [[0.0, 1.0], [0.0, 10.0]],
    }


@pytest.fixture
def qoi_names():
    return ["mean", "var"]


@pytest.fixture
def extraction_points():
    return [(10, 20), (30, 40)]


class TestSobolCheckpoint:
    """Tests for save/load round-trip."""

    def test_save_and_load(self, tmp_path, problem, qoi_names, extraction_points):
        path = tmp_path / "test.parquet"
        ckpt = SobolCheckpoint(path, problem, qoi_names, extraction_points)

        # Save a batch
        indices = np.array([0, 1, 2, 3])
        params = np.random.rand(4, 2)
        qoi_values = {
            "mean": np.random.rand(4, 2),
            "var": np.random.rand(4, 2),
        }
        ckpt.save_batch(indices, params, qoi_values)

        # Load back
        n_done, df = ckpt.load()
        assert n_done == 4
        assert len(df) == 4
        assert "sample_index" in df.columns
        assert "param_alpha" in df.columns
        assert "param_beta" in df.columns
        assert "qoi_mean_point_0" in df.columns
        assert "qoi_var_point_1" in df.columns

    def test_append_multiple_batches(self, tmp_path, problem, qoi_names, extraction_points):
        path = tmp_path / "test.parquet"
        ckpt = SobolCheckpoint(path, problem, qoi_names, extraction_points)

        # Save two batches
        for batch_idx in range(2):
            offset = batch_idx * 4
            ckpt.save_batch(
                sample_indices=np.arange(offset, offset + 4),
                params_values=np.random.rand(4, 2),
                qoi_values={"mean": np.random.rand(4, 2), "var": np.random.rand(4, 2)},
            )

        n_done, df = ckpt.load()
        assert n_done == 8
        assert len(df) == 8

    def test_empty_load(self, tmp_path, problem, qoi_names, extraction_points):
        path = tmp_path / "nonexistent.parquet"
        ckpt = SobolCheckpoint(path, problem, qoi_names, extraction_points)
        n_done, df = ckpt.load()
        assert n_done == 0
        assert len(df) == 0

    def test_extract_qoi_array(self, tmp_path, problem, qoi_names, extraction_points):
        path = tmp_path / "test.parquet"
        ckpt = SobolCheckpoint(path, problem, qoi_names, extraction_points)

        expected_mean = np.array([[1.0, 2.0], [3.0, 4.0]])
        ckpt.save_batch(
            sample_indices=np.array([0, 1]),
            params_values=np.zeros((2, 2)),
            qoi_values={"mean": expected_mean, "var": np.zeros((2, 2))},
        )

        _, df = ckpt.load()
        result = ckpt.extract_qoi_array(df, "mean")
        np.testing.assert_array_almost_equal(result, expected_mean)

    def test_metadata_validation_mismatch(self, tmp_path, problem, qoi_names, extraction_points):
        """Should raise if resuming with different QoI names."""
        path = tmp_path / "test.parquet"
        ckpt = SobolCheckpoint(path, problem, qoi_names, extraction_points)
        ckpt.save_batch(
            sample_indices=np.array([0]),
            params_values=np.zeros((1, 2)),
            qoi_values={"mean": np.zeros((1, 2)), "var": np.zeros((1, 2))},
        )

        # Try to load with different QoI
        ckpt2 = SobolCheckpoint(path, problem, ["mean", "std"], extraction_points)
        with pytest.raises(ValueError, match="QoI mismatch"):
            ckpt2.load()

    def test_metadata_validation_param_mismatch(self, tmp_path, qoi_names, extraction_points):
        """Should raise if resuming with different parameters."""
        problem1 = {"num_vars": 2, "names": ["alpha", "beta"], "bounds": [[0, 1], [0, 1]]}
        problem2 = {"num_vars": 2, "names": ["alpha", "gamma"], "bounds": [[0, 1], [0, 1]]}

        path = tmp_path / "test.parquet"
        ckpt1 = SobolCheckpoint(path, problem1, qoi_names, extraction_points)
        ckpt1.save_batch(
            sample_indices=np.array([0]),
            params_values=np.zeros((1, 2)),
            qoi_values={"mean": np.zeros((1, 2)), "var": np.zeros((1, 2))},
        )

        ckpt2 = SobolCheckpoint(path, problem2, qoi_names, extraction_points)
        with pytest.raises(ValueError, match="Parameter mismatch"):
            ckpt2.load()
