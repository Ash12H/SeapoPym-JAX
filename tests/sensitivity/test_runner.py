"""Tests for SobolRunner (batched model evaluation)."""

import jax.numpy as jnp
import xarray as xr

from seapopym.blueprint import Blueprint, Config, functional
from seapopym.compiler import compile_model
from seapopym.sensitivity.runner import SobolRunner


def _make_minimal_model(backend="jax"):
    """Create a minimal compiled model for testing."""

    @functional(name="test:sobol_growth", backend="jax", units={"x": "g", "rate": "1/s", "return": "g/s"})
    def growth(x, rate):
        return rate * x

    blueprint = Blueprint.from_dict(
        {
            "id": "test_sobol",
            "version": "1.0",
            "declarations": {
                "state": {"x": {"units": "g", "dims": []}},
                "parameters": {"rate": {"units": "1/s"}},
                "forcings": {},
            },
            "process": [
                {
                    "func": "test:sobol_growth",
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
            "execution": {"time_start": "2000-01-01", "time_end": "2000-01-10", "dt": "1d"},
        }
    )

    return compile_model(blueprint, config, backend=backend)


class TestSobolRunner:
    """Tests for SobolRunner on a minimal 0D model."""

    def test_run_batch_shapes(self):
        """Output should have correct shape (batch, T, n_points)."""
        model = _make_minimal_model()
        runner = SobolRunner(model, extraction_points=[], output_variable="x", chunk_size=5)

        # 0D model: extraction_points is empty, but _extract_points handles 0D
        batch_size = 4
        params_batch = {"rate": jnp.array([0.001, 0.002, 0.003, 0.004])}

        result = runner.run_batch(params_batch, batch_size)

        # 0D model: n_points=1 (broadcast), T=10 timesteps (2000-01-01 to 2000-01-10)
        assert result.shape[0] == batch_size
        assert result.shape[1] == model.n_timesteps
        assert result.shape[2] == 1  # 0D → 1 point

    def test_different_params_give_different_outputs(self):
        """Different parameter values should produce different time series."""
        model = _make_minimal_model()
        runner = SobolRunner(model, extraction_points=[], output_variable="x", chunk_size=10)

        params_batch = {"rate": jnp.array([0.0001, 0.01])}
        result = runner.run_batch(params_batch, 2)

        # Higher rate → higher final values
        final_low = result[0, -1, 0]
        final_high = result[1, -1, 0]
        assert final_high > final_low

    def test_chunking_consistency(self):
        """Results should be identical regardless of chunk_size."""
        model = _make_minimal_model()
        params_batch = {"rate": jnp.array([0.001, 0.002])}

        runner_full = SobolRunner(model, extraction_points=[], output_variable="x", chunk_size=10)
        result_full = runner_full.run_batch(params_batch, 2)

        runner_chunked = SobolRunner(model, extraction_points=[], output_variable="x", chunk_size=3)
        result_chunked = runner_chunked.run_batch(params_batch, 2)

        assert jnp.allclose(result_full, result_chunked, atol=1e-5)

    def test_padding_removed(self):
        """Padded batch should give same results as non-padded for valid samples."""
        model = _make_minimal_model()
        runner = SobolRunner(model, extraction_points=[], output_variable="x", chunk_size=10)

        # Run with 2 real samples
        params_real = {"rate": jnp.array([0.001, 0.005])}
        result_real = runner.run_batch(params_real, 2)

        # Run with 4 (2 real + 2 padded zeros)
        params_padded = {"rate": jnp.array([0.001, 0.005, 0.0, 0.0])}
        result_padded = runner.run_batch(params_padded, 4)

        # First 2 should match
        assert jnp.allclose(result_real, result_padded[:2], atol=1e-5)
