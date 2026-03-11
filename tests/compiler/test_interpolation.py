"""Tests for temporal interpolation of forcings."""

import numpy as np
import pandas as pd
import xarray as xr

from seapopym.blueprint import Blueprint, Config
from seapopym.compiler.compiler import _extract_blueprint_dims, _prepare_forcings
from seapopym.compiler.time_grid import TimeGrid


class TestForcingInterpolation:
    """Tests for the forcing interpolation logic in Compiler."""

    def _make_blueprint(self, forcings_decl):
        """Create a minimal blueprint with given forcing declarations."""
        return Blueprint.from_dict(
            {
                "id": "test",
                "version": "1.0",
                "declarations": {
                    "state": {},
                    "parameters": {},
                    "forcings": forcings_decl,
                },
                "process": [],
            }
        )

    def test_static_forcing_no_interpolation(self):
        """Test that static forcings (no time dim) are NOT interpolated."""
        ny, nx = 10, 5
        nt = 100

        blueprint = self._make_blueprint({"time_index": {"dims": ["T"]}, "bathy": {"dims": ["Y", "X"]}})

        bathy_data = np.random.rand(ny, nx)
        target_coords = pd.date_range("2000-01-01", periods=nt, freq="1D").to_numpy()

        config = Config(
            forcings={
                "bathy": xr.DataArray(bathy_data, dims=["Y", "X"]),
                "time_index": xr.DataArray(np.arange(nt, dtype=float), dims=["T"]),
            },
            execution={
                "forcing_interpolation": "linear",
                "time_start": "2000-01-01",
                "time_end": "2000-01-02",
            },
        )

        shapes = {"T": nt, "Y": ny, "X": nx}
        blueprint_dims = _extract_blueprint_dims(blueprint)

        forcings, _ = _prepare_forcings(
            config,
            {},
            shapes,
            time_grid=TimeGrid(
                start=np.datetime64("2000-01-01"),
                end=np.datetime64("2000-01-02"),
                dt_seconds=86400,
                n_timesteps=nt,
                coords=target_coords,
            ),
            blueprint_dims=blueprint_dims,
        )

        # Bathy is static — access via get_statics
        statics = forcings.get_statics()
        assert statics["bathy"].shape == (ny, nx)
        np.testing.assert_allclose(np.asarray(statics["bathy"]), bathy_data, rtol=1e-6)

    def test_temporal_interpolation_linear(self):
        """Test linear interpolation of undersampled time forcing."""
        blueprint = self._make_blueprint({"temp": {"dims": ["T", "Y", "X"]}})

        source_t = 5
        target_t = 9

        source_times = pd.date_range("2000-01-01", periods=source_t, freq="1D")
        data = np.linspace(0, 40, source_t).reshape(-1, 1, 1)
        target_coords = pd.date_range(source_times[0], source_times[-1], periods=target_t).to_numpy()

        config = Config(
            forcings={
                "temp": xr.DataArray(data, dims=["T", "Y", "X"], coords={"T": source_times}),
            },
            execution={
                "forcing_interpolation": "linear",
                "time_start": "2000-01-01",
                "time_end": "2000-01-05",
            },
        )

        shapes = {"T": target_t, "Y": 1, "X": 1}
        blueprint_dims = _extract_blueprint_dims(blueprint)
        forcings, _ = _prepare_forcings(
            config,
            {},
            shapes,
            time_grid=TimeGrid(
                start=source_times[0].to_datetime64(),
                end=source_times[-1].to_datetime64(),
                dt_seconds=86400,
                n_timesteps=target_t,
                coords=target_coords,
            ),
            blueprint_dims=blueprint_dims,
        )

        all_forcings = forcings.get_all_dynamic()
        res = np.asarray(all_forcings["temp"]).flatten()
        expected = np.linspace(0, 40, target_t)
        np.testing.assert_allclose(res, expected, atol=1e-5)

    def test_temporal_interpolation_nearest(self):
        """Test nearest neighbor interpolation."""
        blueprint = self._make_blueprint({"temp": {"dims": ["T", "Y", "X"]}})

        target_t = 4
        source_times = pd.date_range("2000-01-01", periods=2, freq="1D")
        data = np.array([10.0, 20.0]).reshape(-1, 1, 1)
        target_coords = pd.date_range(source_times[0], source_times[-1], periods=target_t).to_numpy()

        config = Config(
            forcings={
                "temp": xr.DataArray(data, dims=["T", "Y", "X"], coords={"T": source_times}),
            },
            execution={
                "forcing_interpolation": "nearest",
                "time_start": "2000-01-01",
                "time_end": "2000-01-02",
            },
        )

        shapes = {"T": target_t, "Y": 1, "X": 1}
        blueprint_dims = _extract_blueprint_dims(blueprint)
        forcings, _ = _prepare_forcings(
            config,
            {},
            shapes,
            time_grid=TimeGrid(
                start=source_times[0].to_datetime64(),
                end=source_times[-1].to_datetime64(),
                dt_seconds=86400,
                n_timesteps=target_t,
                coords=target_coords,
            ),
            blueprint_dims=blueprint_dims,
        )

        all_forcings = forcings.get_all_dynamic()
        res = np.asarray(all_forcings["temp"]).flatten()
        expected = [10.0, 10.0, 20.0, 20.0]
        np.testing.assert_array_equal(res, expected)

    def test_temporal_interpolation_ffill(self):
        """Test forward fill interpolation."""
        blueprint = self._make_blueprint({"temp": {"dims": ["T", "Y", "X"]}})

        target_t = 4
        source_times = pd.date_range("2000-01-01", periods=2, freq="1D")
        data = np.array([10.0, 20.0]).reshape(-1, 1, 1)
        target_coords = pd.date_range(source_times[0], source_times[-1], periods=target_t).to_numpy()

        config = Config(
            forcings={
                "temp": xr.DataArray(data, dims=["T", "Y", "X"], coords={"T": source_times}),
            },
            execution={
                "forcing_interpolation": "ffill",
                "time_start": "2000-01-01",
                "time_end": "2000-01-02",
            },
        )

        shapes = {"T": target_t, "Y": 1, "X": 1}
        blueprint_dims = _extract_blueprint_dims(blueprint)
        forcings, _ = _prepare_forcings(
            config,
            {},
            shapes,
            time_grid=TimeGrid(
                start=source_times[0].to_datetime64(),
                end=source_times[-1].to_datetime64(),
                dt_seconds=86400,
                n_timesteps=target_t,
                coords=target_coords,
            ),
            blueprint_dims=blueprint_dims,
        )

        all_forcings = forcings.get_all_dynamic()
        res = np.asarray(all_forcings["temp"]).flatten()
        expected = [10.0, 10.0, 10.0, 20.0]
        np.testing.assert_array_equal(res, expected)
