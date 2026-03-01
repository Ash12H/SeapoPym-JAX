"""Tests for inference module."""

import numpy as np
import pytest
import xarray as xr

from seapopym.blueprint import Config
from seapopym.compiler.exceptions import GridAlignmentError, ShapeInferenceError
from seapopym.compiler.inference import infer_shapes, infer_shapes_from_array, infer_shapes_from_file
from seapopym.compiler.time_grid import TimeGrid


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


class TestInferShapesFromFile:
    """Tests for infer_shapes_from_file function."""

    def test_from_netcdf(self, tmp_path):
        """Test reading shapes from a NetCDF file."""
        ds = xr.Dataset({"temp": (["T", "Y", "X"], np.zeros((30, 10, 20)))})
        path = tmp_path / "data.nc"
        ds.to_netcdf(path)

        result = infer_shapes_from_file(path)
        assert result == {"T": 30, "Y": 10, "X": 20}

    def test_from_zarr(self, tmp_path):
        """Test reading shapes from a Zarr store."""
        ds = xr.Dataset({"temp": (["T", "Y", "X"], np.zeros((30, 10, 20)))})
        path = tmp_path / "data.zarr"
        ds.to_zarr(path)

        result = infer_shapes_from_file(path)
        assert result == {"T": 30, "Y": 10, "X": 20}

    def test_invalid_file_raises(self, tmp_path):
        """Test that a non-existent file raises ShapeInferenceError."""
        path = tmp_path / "nonexistent.nc"
        with pytest.raises(ShapeInferenceError) as exc_info:
            infer_shapes_from_file(path)
        assert exc_info.value.path == str(path)


class TestInferShapesExtended:
    """Extended tests for infer_shapes covering file paths, Datasets, and time_grid."""

    def _make_config(self, **kwargs):
        """Helper to build a Config with default execution block."""
        kwargs.setdefault("execution", {"dt": "1d", "time_start": "2000-01-01", "time_end": "2000-01-02"})
        return Config.from_dict(kwargs)

    def test_from_file_forcing(self, tmp_path):
        """Test inference from a file path in forcings."""
        ds = xr.Dataset({"temperature": (["T", "Y", "X"], np.zeros((30, 10, 20)))})
        path = tmp_path / "temp.nc"
        ds.to_netcdf(path)

        config = self._make_config(forcings={"temperature": str(path)})
        result = infer_shapes(config)
        assert result == {"T": 30, "Y": 10, "X": 20}

    def test_from_dataset_forcing(self):
        """Test inference from xr.Dataset forcing."""
        ds = xr.Dataset({"temp": (["T", "Y", "X"], np.zeros((30, 10, 20)))})
        config = self._make_config(forcings={"temperature": ds})
        result = infer_shapes(config)
        assert result == {"T": 30, "Y": 10, "X": 20}

    def test_numpy_forcing_without_blueprint_dims(self):
        """Test that numpy forcing without blueprint_dims is silently ignored."""
        arr = np.random.rand(30, 10, 20)
        config = self._make_config(forcings={"temperature": arr})
        # No blueprint_dims → numpy array shapes cannot be inferred
        result = infer_shapes(config)
        assert result == {}

    def test_from_file_initial_state(self, tmp_path):
        """Test inference from a file path in initial_state."""
        ds = xr.Dataset({"biomass": (["Y", "X"], np.zeros((10, 20)))})
        path = tmp_path / "state.nc"
        ds.to_netcdf(path)

        config = self._make_config(initial_state={"biomass": str(path)})
        result = infer_shapes(config)
        assert result == {"Y": 10, "X": 20}

    def test_numpy_initial_state_with_blueprint_dims(self):
        """Test inference from numpy initial_state with blueprint dims."""
        arr = np.random.rand(10, 20)
        config = self._make_config(initial_state={"biomass": arr})
        # blueprint maps "state.biomass" (initial_state.biomass → state.biomass)
        blueprint_dims: dict[str, list[str] | None] = {"state.biomass": ["Y", "X"]}
        result = infer_shapes(config, blueprint_dims)
        assert result == {"Y": 10, "X": 20}

    def test_numpy_initial_state_without_blueprint_dims(self):
        """Test that numpy initial_state without blueprint_dims is ignored."""
        arr = np.random.rand(10, 20)
        config = self._make_config(initial_state={"biomass": arr})
        result = infer_shapes(config)
        assert result == {}

    def test_with_time_grid(self):
        """Test that time_grid overrides T dimension from data."""
        temp = xr.DataArray(np.random.rand(30, 10, 20), dims=["T", "Y", "X"])
        config = self._make_config(forcings={"temperature": temp})
        time_grid = TimeGrid.from_config("2000-01-01", "2000-04-10", "1d")

        result = infer_shapes(config, time_grid=time_grid)
        # T comes from time_grid (100 days), not from data (30)
        assert result["T"] == time_grid.n_timesteps
        assert result["Y"] == 10
        assert result["X"] == 20
