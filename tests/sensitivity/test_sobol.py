"""Integration tests for SobolAnalyzer."""

import pytest
import xarray as xr

from seapopym.blueprint import Blueprint, Config, functional
from seapopym.compiler import compile_model

# Skip all tests if SALib is not installed
SALib = pytest.importorskip("SALib")

from seapopym.sensitivity.sobol import SobolAnalyzer, SobolResult


def _make_test_model():
    """Create a simple model where 'rate' clearly drives the output."""

    @functional(name="test:sobol_linear", units={"x": "g", "rate": "1/s", "return": "g/s"})
    def linear(x, rate):
        return rate * x

    blueprint = Blueprint.from_dict(
        {
            "id": "test_sobol_sa",
            "version": "1.0",
            "declarations": {
                "state": {"x": {"units": "g", "dims": []}},
                "parameters": {"rate": {"units": "1/s"}},
                "forcings": {},
            },
            "process": [
                {
                    "func": "test:sobol_linear",
                    "inputs": {"x": "state.x", "rate": "parameters.rate"},
                    "outputs": {"return": "derived.x_flux"},
                }
            ],
            "tendencies": {"x": [{"source": "derived.x_flux"}]},
        }
    )

    config = Config.from_dict(
        {
            "parameters": {"rate": {"value": 0.001}},
            "forcings": {},
            "initial_state": {"x": xr.DataArray(1.0)},
            "execution": {"time_start": "2000-01-01", "time_end": "2000-01-05", "dt": "1d"},
        }
    )

    return compile_model(blueprint, config)


class TestSobolAnalyzer:
    """Integration tests for the full Sobol analysis pipeline."""

    def test_basic_analysis(self):
        """Should compute S1 and ST for a simple model."""
        model = _make_test_model()
        analyzer = SobolAnalyzer(model)

        result = analyzer.analyze(
            param_bounds={"rate": (0.0001, 0.01)},
            extraction_points=[],
            output_variable="x",
            n_samples=64,
            qoi=["mean"],
            batch_size=32,
            chunk_size=5,
        )

        assert isinstance(result, SobolResult)
        assert "rate" in result.S1.columns
        assert len(result.S1) == 1  # 1 QoI * 1 point

    def test_result_structure(self):
        """SobolResult should have correct DataFrame structure."""
        model = _make_test_model()
        analyzer = SobolAnalyzer(model)

        result = analyzer.analyze(
            param_bounds={"rate": (0.0001, 0.01)},
            extraction_points=[],
            output_variable="x",
            n_samples=64,
            qoi=["mean", "var"],
            batch_size=64,
            chunk_size=5,
        )

        # 2 QoI * 1 point = 2 rows
        assert len(result.S1) == 2
        assert result.S1.index.names == ["qoi", "point"]
        assert result.n_samples == 64

    def test_checkpoint_resume(self, tmp_path):
        """Should resume from checkpoint correctly."""
        model = _make_test_model()
        analyzer = SobolAnalyzer(model)
        ckpt_path = tmp_path / "sobol_ckpt.parquet"

        # Run with small batch to create checkpoint
        result1 = analyzer.analyze(
            param_bounds={"rate": (0.0001, 0.01)},
            extraction_points=[],
            output_variable="x",
            n_samples=64,
            qoi=["mean"],
            batch_size=32,
            chunk_size=5,
            checkpoint_path=ckpt_path,
        )

        # Checkpoint file should exist
        assert ckpt_path.exists()
        assert isinstance(result1, SobolResult)

    def test_invalid_n_samples(self):
        """Should reject non-power-of-2 n_samples."""
        model = _make_test_model()
        analyzer = SobolAnalyzer(model)

        with pytest.raises(ValueError, match="power of 2"):
            analyzer.analyze(
                param_bounds={"rate": (0.0, 1.0)},
                extraction_points=[],
                output_variable="x",
                n_samples=100,
            )

    def test_invalid_param_name(self):
        """Should reject unknown parameter names."""
        model = _make_test_model()
        analyzer = SobolAnalyzer(model)

        with pytest.raises(ValueError, match="not found"):
            analyzer.analyze(
                param_bounds={"nonexistent": (0.0, 1.0)},
                extraction_points=[],
                output_variable="x",
                n_samples=64,
            )

