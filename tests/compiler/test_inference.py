"""Tests for inference module."""

import numpy as np
import pytest
import xarray as xr

from seapopym.blueprint import Config, ExecutionParams
from seapopym.compiler.exceptions import GridAlignmentError
from seapopym.compiler.inference import infer_shapes
from seapopym.compiler.time_grid import TimeGrid


def _make_config(**kwargs):
    """Helper to build a Config with default execution block."""
    kwargs.setdefault(
        "execution",
        ExecutionParams(time_start="2000-01-01", time_end="2000-01-02", dt="1d"),
    )
    return Config(**kwargs)


class TestInferShapes:
    """Tests for infer_shapes function."""

    def test_from_xarray_forcing(self):
        """Test inference from xarray DataArray in config."""
        config = _make_config(
            forcings={"temperature": xr.DataArray(np.random.rand(30, 10, 20), dims=["T", "Y", "X"])},
        )
        result = infer_shapes(config)
        assert result == {"T": 30, "Y": 10, "X": 20}

    def test_from_multiple_forcings(self):
        """Test inference from multiple forcings."""
        config = _make_config(
            forcings={
                "temperature": xr.DataArray(np.random.rand(30, 10, 20), dims=["T", "Y", "X"]),
                "mask": xr.DataArray(np.ones((10, 20)), dims=["Y", "X"]),
            },
        )
        result = infer_shapes(config)
        assert result == {"T": 30, "Y": 10, "X": 20}

    def test_grid_alignment_error(self):
        """Test that misaligned dimensions raise error."""
        config = _make_config(
            forcings={
                "temperature": xr.DataArray(np.random.rand(30, 10, 20), dims=["T", "Y", "X"]),
                "salinity": xr.DataArray(np.random.rand(30, 15, 20), dims=["T", "Y", "X"]),
            },
        )

        with pytest.raises(GridAlignmentError) as exc_info:
            infer_shapes(config)
        assert exc_info.value.dimension == "Y"

    def test_from_initial_state(self):
        """Test inference from initial state."""
        config = _make_config(
            initial_state={"biomass": xr.DataArray(np.random.rand(10, 20), dims=["Y", "X"])},
        )
        result = infer_shapes(config)
        assert result == {"Y": 10, "X": 20}

    def test_from_parameters(self):
        """Test inference from parameters with dimensions."""
        config = _make_config(
            parameters={"mortality": xr.DataArray([0.1, 0.2, 0.3], dims=["C"])},
        )
        result = infer_shapes(config)
        assert result == {"C": 3}

    def test_empty_config(self):
        """Test with empty config."""
        config = _make_config()
        result = infer_shapes(config)
        assert result == {}

    def test_combined_forcings_and_state(self):
        """Test combining forcings and initial state."""
        config = _make_config(
            forcings={"temperature": xr.DataArray(np.random.rand(30, 10, 20), dims=["T", "Y", "X"])},
            initial_state={"biomass": xr.DataArray(np.random.rand(10, 20, 5), dims=["Y", "X", "C"])},
        )
        result = infer_shapes(config)
        assert result == {"T": 30, "Y": 10, "X": 20, "C": 5}

    def test_with_time_grid(self):
        """Test that time_grid overrides T dimension from data."""
        config = _make_config(
            forcings={"temperature": xr.DataArray(np.random.rand(30, 10, 20), dims=["T", "Y", "X"])},
        )
        time_grid = TimeGrid.from_config("2000-01-01", "2000-04-10", "1d")

        result = infer_shapes(config, time_grid=time_grid)
        assert result["T"] == time_grid.n_timesteps
        assert result["Y"] == 10
        assert result["X"] == 20

    def test_scalar_parameter(self):
        """Test inference from scalar parameter (no dims)."""
        config = _make_config(
            parameters={"rate": xr.DataArray(0.1)},
        )
        result = infer_shapes(config)
        assert result == {}
