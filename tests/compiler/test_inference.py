"""Tests for inference module."""

import numpy as np
import pytest
import xarray as xr

from seapopym.blueprint import Config
from seapopym.compiler.exceptions import GridAlignmentError
from seapopym.compiler.inference import infer_shapes, infer_shapes_from_array


class TestInferShapesFromArray:
    """Tests for infer_shapes_from_array function."""

    def test_basic_inference(self):
        """Test basic shape inference from array."""
        arr = np.random.rand(10, 20, 30)
        dims = ["T", "Y", "X"]
        result = infer_shapes_from_array(arr, dims)
        assert result == {"T": 10, "Y": 20, "X": 30}

    def test_no_dims(self):
        """Test with None dims."""
        arr = np.random.rand(10, 20)
        result = infer_shapes_from_array(arr, None)
        assert result == {}

    def test_mismatched_dims(self):
        """Test with mismatched number of dims."""
        arr = np.random.rand(10, 20)
        dims = ["T", "Y", "X"]  # 3 dims but array has 2
        result = infer_shapes_from_array(arr, dims)
        assert result == {}

    def test_scalar_like(self):
        """Test with 0-d array."""
        arr = np.array(5.0)
        dims: list[str] = []
        result = infer_shapes_from_array(arr, dims)
        assert result == {}


class TestInferShapes:
    """Tests for infer_shapes function."""

    def test_from_xarray_forcing(self):
        """Test inference from xarray DataArray in config."""
        da = xr.DataArray(
            np.random.rand(30, 10, 20),
            dims=["T", "Y", "X"],
        )
        config = Config.from_dict(
            {
                "forcings": {"temperature": da},
                "execution": {
                    "dt": "1d",
                    "time_start": "2000-01-01",
                    "time_end": "2000-01-02",
                },
            }
        )
        result = infer_shapes(config)
        assert result == {"T": 30, "Y": 10, "X": 20}

    def test_from_multiple_forcings(self):
        """Test inference from multiple forcings."""
        temp = xr.DataArray(np.random.rand(30, 10, 20), dims=["T", "Y", "X"])
        mask = xr.DataArray(np.ones((10, 20)), dims=["Y", "X"])

        config = Config.from_dict(
            {
                "forcings": {
                    "temperature": temp,
                    "mask": mask,
                },
                "execution": {
                    "dt": "1d",
                    "time_start": "2000-01-01",
                    "time_end": "2000-01-02",
                },
            }
        )
        result = infer_shapes(config)
        assert result == {"T": 30, "Y": 10, "X": 20}

    def test_grid_alignment_error(self):
        """Test that misaligned dimensions raise error."""
        temp = xr.DataArray(np.random.rand(30, 10, 20), dims=["T", "Y", "X"])
        salt = xr.DataArray(np.random.rand(30, 15, 20), dims=["T", "Y", "X"])  # Y=15 vs Y=10

        config = Config.from_dict(
            {
                "forcings": {
                    "temperature": temp,
                    "salinity": salt,
                },
                "execution": {
                    "dt": "1d",
                    "time_start": "2000-01-01",
                    "time_end": "2000-01-02",
                },
            }
        )

        with pytest.raises(GridAlignmentError) as exc_info:
            infer_shapes(config)

        assert exc_info.value.dimension == "Y"

    def test_from_initial_state(self):
        """Test inference from initial state."""
        biomass = xr.DataArray(np.random.rand(10, 20), dims=["Y", "X"])

        config = Config.from_dict(
            {
                "initial_state": {"biomass": biomass},
                "execution": {
                    "dt": "1d",
                    "time_start": "2000-01-01",
                    "time_end": "2000-01-02",
                },
            }
        )
        result = infer_shapes(config)
        assert result == {"Y": 10, "X": 20}

    def test_from_numpy_with_blueprint_dims(self):
        """Test inference from numpy array with blueprint dims."""
        arr = np.random.rand(30, 10, 20)

        config = Config.from_dict(
            {
                "forcings": {"temperature": arr},
                "execution": {
                    "dt": "1d",
                    "time_start": "2000-01-01",
                    "time_end": "2000-01-02",
                },
            }
        )

        blueprint_dims: dict[str, list[str] | None] = {"forcings.temperature": ["T", "Y", "X"]}
        result = infer_shapes(config, blueprint_dims)
        assert result == {"T": 30, "Y": 10, "X": 20}

    def test_nested_initial_state(self):
        """Test inference from nested initial state."""
        tuna_biomass = xr.DataArray(np.random.rand(10, 20, 5), dims=["Y", "X", "C"])

        config = Config.from_dict(
            {
                "initial_state": {
                    "tuna": {"biomass": tuna_biomass},
                },
                "execution": {
                    "dt": "1d",
                    "time_start": "2000-01-01",
                    "time_end": "2000-01-02",
                },
            }
        )
        result = infer_shapes(config)
        assert result == {"Y": 10, "X": 20, "C": 5}

    def test_empty_config(self):
        """Test with empty config."""
        config = Config.from_dict(
            {
                "execution": {
                    "dt": "1d",
                    "time_start": "2000-01-01",
                    "time_end": "2000-01-02",
                }
            }
        )
        result = infer_shapes(config)
        assert result == {}

    def test_combined_forcings_and_state(self):
        """Test combining forcings and initial state."""
        temp = xr.DataArray(np.random.rand(30, 10, 20), dims=["T", "Y", "X"])
        biomass = xr.DataArray(np.random.rand(10, 20, 5), dims=["Y", "X", "C"])

        config = Config.from_dict(
            {
                "forcings": {"temperature": temp},
                "initial_state": {"biomass": biomass},
                "execution": {
                    "dt": "1d",
                    "time_start": "2000-01-01",
                    "time_end": "2000-01-02",
                },
            }
        )
        result = infer_shapes(config)
        assert result == {"T": 30, "Y": 10, "X": 20, "C": 5}
