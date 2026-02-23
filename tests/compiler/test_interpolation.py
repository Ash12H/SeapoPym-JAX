"""Tests for temporal interpolation of forcings."""

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("jax")

from seapopym.blueprint import Blueprint, Config
from seapopym.compiler.compiler import Compiler, TimeGrid


class TestForcingInterpolation:
    """Tests for the forcing interpolation logic in Compiler."""

    @pytest.fixture
    def blueprint(self):
        """Create a minimal blueprint."""
        return Blueprint.from_dict(
            {
                "id": "test",
                "version": "1.0",
                "declarations": {
                    "state": {},
                    "parameters": {},
                    "forcings": {"temp": {"dims": ["T", "Y", "X"]}, "bathy": {"dims": ["Y", "X"]}},
                },
                "process": [],
            }
        )

    def test_static_forcing_no_interpolation(self, blueprint):
        """Test that static forcings (no time dim) are NOT interpolated."""
        # Config with static forcing (Y, X)
        ny, nx = 10, 5
        nt = 100

        bathy_data = np.random.rand(ny, nx)

        config = Config.from_dict(
            {
                "parameters": {},
                "forcings": {
                    "bathy": xr.DataArray(bathy_data, dims=["Y", "X"]),
                    "time_index": xr.DataArray(np.arange(nt), dims=["T"]),
                },
                "execution": {
                    "forcing_interpolation": "linear",  # Enable interpolation globally
                    "time_start": "2000-01-01",
                    "time_end": "2000-01-02",
                },
                "initial_state": {},
            }
        )

        compiler = Compiler()
        # We access internal method _prepare_forcings to test isolation
        # Need to mimic what compile() does before calling it
        shapes = {"T": nt, "Y": ny, "X": nx}
        dim_mapping = {}
        blueprint_dims = compiler._extract_blueprint_dims(blueprint)

        forcings, _ = compiler._prepare_forcings(
            config,
            dim_mapping,
            shapes,
            time_grid=TimeGrid(
                start=np.datetime64("2000-01-01"),
                end=np.datetime64("2000-01-02"),
                dt_seconds=86400,
                n_timesteps=nt,
                coords=np.array([]),
            ),
            blueprint_dims=blueprint_dims,
        )

        # Bathy should match input shape (ny, nx), NOT (nt, nx)
        assert forcings["bathy"].shape == (ny, nx)
        np.testing.assert_allclose(np.asarray(forcings["bathy"]), bathy_data, rtol=1e-6)

    def test_temporal_interpolation_linear(self, blueprint):
        """Test linear interpolation of undersampled time forcing."""
        # Source: 5 timesteps. Target: 9 timesteps.
        # 0, 1, 2, 3, 4 -> 0.0, 0.5, 1.0, 1.5, ...
        source_t = 5
        target_t = 9  # implies dt is halved approx

        # Simple ramp: 0, 10, 20, 30, 40
        data = np.linspace(0, 40, source_t).reshape(-1, 1, 1)  # (T, Y, X)

        config = Config.from_dict(
            {
                "parameters": {},
                "forcings": {
                    "temp": xr.DataArray(data, dims=["T", "Y", "X"]),
                    "time_index": xr.DataArray(np.arange(target_t), dims=["T"]),
                },
                "execution": {
                    "forcing_interpolation": "linear",
                    "time_start": "2000-01-01",
                    "time_end": "2000-01-02",
                },
                "initial_state": {},
            }
        )

        compiler = Compiler()
        shapes = {"T": target_t, "Y": 1, "X": 1}
        blueprint_dims = compiler._extract_blueprint_dims(blueprint)
        forcings, _ = compiler._prepare_forcings(
            config,
            {},
            shapes,
            time_grid=TimeGrid(
                start=np.datetime64("2000-01-01"),
                end=np.datetime64("2000-01-02"),
                dt_seconds=86400,
                n_timesteps=target_t,
                coords=np.array([]),
            ),
            blueprint_dims=blueprint_dims,
        )

        # Interpolation is deferred — use get_all() to materialize
        all_forcings = forcings.get_all()
        res = np.asarray(all_forcings["temp"]).flatten()
        expected = np.linspace(0, 40, target_t)

        # Should be close
        np.testing.assert_allclose(res, expected, atol=1e-5)

    def test_temporal_interpolation_nearest(self, blueprint):
        """Test nearest neighbor interpolation."""
        # source_t = 2 (implicit in data shape)
        target_t = 4

        # [10, 20] -> [10, 10, 20, 20] roughly
        data = np.array([10.0, 20.0]).reshape(-1, 1, 1)

        config = Config.from_dict(
            {
                "parameters": {},
                "forcings": {
                    "temp": xr.DataArray(data, dims=["T", "Y", "X"]),
                    "time_index": xr.DataArray(np.arange(target_t), dims=["T"]),
                },
                "execution": {
                    "forcing_interpolation": "nearest",
                    "time_start": "2000-01-01",
                    "time_end": "2000-01-02",
                },
                "initial_state": {},
            }
        )

        compiler = Compiler()
        shapes = {"T": target_t, "Y": 1, "X": 1}
        blueprint_dims = compiler._extract_blueprint_dims(blueprint)
        forcings, _ = compiler._prepare_forcings(
            config,
            {},
            shapes,
            time_grid=TimeGrid(
                start=np.datetime64("2000-01-01"),
                end=np.datetime64("2000-01-02"),
                dt_seconds=86400,
                n_timesteps=target_t,
                coords=np.array([]),
            ),
            blueprint_dims=blueprint_dims,
        )

        # Interpolation is deferred — use get_all() to materialize
        all_forcings = forcings.get_all()
        res = np.asarray(all_forcings["temp"]).flatten()
        # indices: 0->0, 1->0.33(0), 2->0.66(1), 3->1
        # With linspace(0, 1, 4): 0.0, 0.33, 0.66, 1.0
        # round(0)=0, round(0.33)=0, round(0.66)=1, round(1)=1
        expected = [10.0, 10.0, 20.0, 20.0]

        np.testing.assert_array_equal(res, expected)

    def test_temporal_interpolation_ffill(self, blueprint):
        """Test forward fill interpolation."""
        # source_t = 2 (implicit in data shape)
        target_t = 4

        data = np.array([10.0, 20.0]).reshape(-1, 1, 1)

        config = Config.from_dict(
            {
                "parameters": {},
                "forcings": {
                    "temp": xr.DataArray(data, dims=["T", "Y", "X"]),
                    "time_index": xr.DataArray(np.arange(target_t), dims=["T"]),
                },
                "execution": {
                    "forcing_interpolation": "ffill",
                    "time_start": "2000-01-01",
                    "time_end": "2000-01-02",
                },
                "initial_state": {},
            }
        )

        compiler = Compiler()
        shapes = {"T": target_t, "Y": 1, "X": 1}
        blueprint_dims = compiler._extract_blueprint_dims(blueprint)
        forcings, _ = compiler._prepare_forcings(
            config,
            {},
            shapes,
            time_grid=TimeGrid(
                start=np.datetime64("2000-01-01"),
                end=np.datetime64("2000-01-02"),
                dt_seconds=86400,
                n_timesteps=target_t,
                coords=np.array([]),
            ),
            blueprint_dims=blueprint_dims,
        )

        # Interpolation is deferred — use get_all() to materialize
        all_forcings = forcings.get_all()
        res = np.asarray(all_forcings["temp"]).flatten()
        # [0, 0.33, 0.66, 1.0] -> floor -> [0, 0, 0, 1]
        # Wait, current implementation uses floor(linspace(0, N-1, M))
        # linspace(0, 1, 4): 0, 0.33, 0.66, 1
        # floor: 0, 0, 0, 1
        # So we expect 10, 10, 10, 20
        expected = [10.0, 10.0, 10.0, 20.0]

        np.testing.assert_array_equal(res, expected)
