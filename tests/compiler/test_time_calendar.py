"""Tests for temporal calendar system (TimeGrid and ExecutionParams)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from seapopym.blueprint import Blueprint, Config, ConfigValidationError, ExecutionParams
from seapopym.compiler import compile_model
from seapopym.compiler.time_grid import TimeGrid


class TestTimeGrid:
    """Tests for TimeGrid.from_config()."""

    def test_basic_parsing(self):
        """Test basic date parsing and n_timesteps calculation."""
        grid = TimeGrid.from_config("2000-01-01", "2000-01-10", "1d")

        assert grid.start == np.datetime64("2000-01-01")
        assert grid.end == np.datetime64("2000-01-10")
        assert grid.dt_seconds == 86400.0  # 1 day
        assert grid.n_timesteps == 9  # [01-01, 01-10) with 1d step
        assert grid.coords.shape == (9,)
        assert grid.coords[0] == np.datetime64("2000-01-01")
        assert grid.coords[-1] == np.datetime64("2000-01-09")

    def test_fractional_timestep(self):
        """Test fractional timestep like 0.05d."""
        grid = TimeGrid.from_config("2000-01-01", "2000-01-02", "0.05d")

        assert grid.dt_seconds == pytest.approx(0.05 * 86400)
        assert grid.n_timesteps == 20  # 1 day / 0.05d = 20 steps

    def test_hourly_timestep(self):
        """Test hourly timestep."""
        grid = TimeGrid.from_config("2000-01-01T00:00:00", "2000-01-01T06:00:00", "1h")

        assert grid.dt_seconds == 3600.0
        assert grid.n_timesteps == 6

    def test_end_before_start_error(self):
        """Test error when time_end <= time_start."""
        with pytest.raises(ValueError, match="must be after"):
            TimeGrid.from_config("2000-01-10", "2000-01-01", "1d")

    def test_end_equals_start_error(self):
        """Test error when time_end == time_start."""
        with pytest.raises(ValueError, match="must be after"):
            TimeGrid.from_config("2000-01-01", "2000-01-01", "1d")

    def test_misaligned_range_error(self):
        """Test error when time range is not evenly divisible by dt."""
        # 10 days is not evenly divisible by 3 days (10/3 = 3.33...)
        with pytest.raises(ValueError, match="not evenly divisible"):
            TimeGrid.from_config("2000-01-01", "2000-01-11", "3d")

    def test_aligned_range_success(self):
        """Test that aligned range (evenly divisible) works."""
        # 9 days is evenly divisible by 3 days (9/3 = 3)
        grid = TimeGrid.from_config("2000-01-01", "2000-01-10", "3d")
        assert grid.n_timesteps == 3

    @pytest.mark.parametrize(
        "start, end, expected",
        [
            ("2000-02-01", "2000-03-01", 29),  # leap year
            ("2001-02-01", "2001-03-01", 28),  # non-leap year
            ("2000-01-01", "2000-02-01", 31),  # January
            ("2000-04-01", "2000-05-01", 30),  # April
        ],
        ids=["leap_feb", "non_leap_feb", "january", "april"],
    )
    def test_calendar_month_lengths(self, start, end, expected):
        """Test handling of variable month lengths and leap years."""
        grid = TimeGrid.from_config(start, end, "1d")
        assert grid.n_timesteps == expected

    def test_long_simulation(self):
        """Test long simulation (20 years)."""
        grid = TimeGrid.from_config("2000-01-01", "2020-01-01", "1d")

        # 20 years * 365 days + 5 leap days (2000, 2004, 2008, 2012, 2016)
        expected_days = 20 * 365 + 5
        assert grid.n_timesteps == expected_days
        assert grid.coords.shape == (expected_days,)

    def test_super_sampling(self):
        """Test super-sampling with fine timestep (reproduce bug scenario)."""
        # 20 years with dt=0.05d (20 steps per day)
        grid = TimeGrid.from_config("2000-01-01", "2020-01-01", "0.05d")

        expected_days = 20 * 365 + 5  # 7305 days
        expected_steps = expected_days * 20  # 146100 steps
        assert grid.n_timesteps == expected_steps

    def test_coords_dtype(self):
        """Test that coords are datetime64."""
        grid = TimeGrid.from_config("2000-01-01", "2000-01-10", "1d")
        assert grid.coords.dtype.kind == "M"  # datetime64


class TestExecutionParams:
    """Tests for ExecutionParams validation."""

    def test_valid_params(self):
        """Test valid execution parameters."""
        params = ExecutionParams(
            time_start="2000-01-01",
            time_end="2020-12-31",
            dt="1d",
            forcing_interpolation="linear",
        )

        assert params.time_start == "2000-01-01"
        assert params.time_end == "2020-12-31"
        assert params.dt == "1d"
        assert params.forcing_interpolation == "linear"

    def test_minimal_params(self):
        """Test minimal required parameters."""
        params = ExecutionParams(time_start="2000-01-01", time_end="2000-12-31")

        assert params.dt == "1d"  # default
        assert params.forcing_interpolation == "constant"  # default

    def test_missing_time_start_error(self):
        """Test error when time_start is missing."""
        with pytest.raises(ValueError, match="Field required"):
            ExecutionParams(time_end="2020-12-31")  # type: ignore

    def test_missing_time_end_error(self):
        """Test error when time_end is missing."""
        with pytest.raises(ValueError, match="Field required"):
            ExecutionParams(time_start="2000-01-01")  # type: ignore

    def test_end_before_start_error(self):
        """Test error when time_end < time_start."""
        with pytest.raises(ValueError, match="must be after"):
            ExecutionParams(time_start="2020-01-01", time_end="2000-01-01")

    def test_end_equals_start_error(self):
        """Test error when time_end == time_start."""
        with pytest.raises(ValueError, match="must be after"):
            ExecutionParams(time_start="2000-01-01", time_end="2000-01-01")

    def test_invalid_datetime_format_start(self):
        """Test error with invalid datetime format for time_start."""
        with pytest.raises(ValueError, match="Invalid datetime format"):
            ExecutionParams(time_start="not-a-date", time_end="2020-01-01")

    def test_invalid_datetime_format_end(self):
        """Test error with invalid datetime format for time_end."""
        with pytest.raises(ValueError, match="Invalid datetime format"):
            ExecutionParams(time_start="2000-01-01", time_end="not-a-date")

    def test_invalid_forcing_interpolation(self):
        """Test error with invalid forcing_interpolation method."""
        with pytest.raises(ValueError):
            ExecutionParams(
                time_start="2000-01-01",
                time_end="2020-01-01",
                forcing_interpolation="invalid",  # type: ignore
            )


class TestCompileTimeGrid:
    """Tests for integration of TimeGrid in Compiler.compile()."""

    @pytest.fixture
    def blueprint(self):
        """Minimal blueprint for testing."""
        return Blueprint.from_dict(
            {
                "id": "test",
                "version": "1.0",
                "declarations": {
                    "state": {},
                    "parameters": {},
                    "forcings": {"temp": {"dims": ["T", "Y", "X"]}},
                },
                "process": [],
            }
        )

    def test_timesteps_from_time_grid(self, blueprint):
        """Test that n_timesteps is calculated from time_grid, not forcings."""
        # Config: 10 days, daily timestep -> 10 steps
        # Forcing: Only 2 steps (should trigger interpolation if configured)
        start = "2000-01-01"
        end = "2000-01-11"  # 10 days
        dt = "1d"

        data = np.zeros((2, 1, 1))

        config = Config.from_dict(
            {
                "parameters": {},
                "forcings": {
                    "temp": xr.DataArray(
                        data, dims=["T", "Y", "X"], coords={"T": pd.to_datetime(["2000-01-01", "2000-01-11"])}
                    )
                },
                "execution": {"time_start": start, "time_end": end, "dt": dt, "forcing_interpolation": "linear"},
                "initial_state": {},
            }
        )

        model = compile_model(blueprint, config)

        assert model.time_grid is not None
        assert model.time_grid.n_timesteps == 10
        assert model.shapes["T"] == 10
        # Interpolation is deferred to runtime — use get_all() to verify
        all_forcings = model.forcings.get_all_dynamic()
        assert all_forcings["temp"].shape[0] == 10  # Interpolated at runtime

    def test_coords_generation(self, blueprint):
        """Test that T coords are generated from time_grid."""
        start = "2000-01-01"
        end = "2000-01-05"
        dt = "1d"

        config = Config.from_dict(
            {
                "parameters": {},
                "forcings": {
                    # Force strictly no coords to see if we generate them
                    "temp": xr.DataArray(np.zeros((4, 1, 1)), dims=["T", "Y", "X"])
                },
                "execution": {"time_start": start, "time_end": end, "dt": dt},
                "initial_state": {},
            }
        )

        model = compile_model(blueprint, config)

        assert "T" in model.coords
        assert len(model.coords["T"]) == 4
        assert model.coords["T"][0] == np.datetime64("2000-01-01")
        assert model.coords["T"][-1] == np.datetime64("2000-01-04")

    def test_forcing_out_of_range_error(self, blueprint):
        """Test error when forcing range does not cover simulation range."""
        # Simulation: 2000-01-01 to 2000-01-10
        # Forcing:    2000-01-05 to 2000-01-15 (starts too late)

        config = Config.from_dict(
            {
                "parameters": {},
                "forcings": {
                    "temp": xr.DataArray(
                        np.zeros((10, 1, 1)),
                        dims=["T", "Y", "X"],

                        coords={"T": pd.date_range("2000-01-05", periods=10, freq="1D")},
                    )
                },
                "execution": {"time_start": "2000-01-01", "time_end": "2000-01-10", "dt": "1d"},
                "initial_state": {},
            }
        )

        with pytest.raises(ConfigValidationError, match="does not cover simulation range"):
            compile_model(blueprint, config)

    def test_interpolation_triggered(self, blueprint):
        """Test that interpolation logic is triggered for mismatching shapes."""
        # 4 steps in grid vs 2 steps in forcing
        start = "2000-01-01"
        end = "2000-01-05"  # 4 days
        dt = "1d"

        # Linear ramp: 0 to 10
        data = np.array([0.0, 10.0]).reshape(-1, 1, 1)  # Shape (2,1,1)

        config = Config.from_dict(
            {
                "parameters": {},
                "forcings": {
                    "temp": xr.DataArray(
                        data,
                        dims=["T", "Y", "X"],
                        # Coords must cover range for validation to pass
                        coords={"T": pd.to_datetime(["2000-01-01", "2000-01-05"])},
                    )
                },
                "execution": {"time_start": start, "time_end": end, "dt": dt, "forcing_interpolation": "linear"},
                "initial_state": {},
            }
        )

        model = compile_model(blueprint, config)

        # Interpolation is deferred to runtime — use get_all() to verify
        all_forcings = model.forcings.get_all_dynamic()
        result = np.asarray(all_forcings["temp"]).flatten()
        # xarray interpolates in real time space:
        # source [0, 10] at [Jan 1, Jan 5], target [Jan 1, Jan 2, Jan 3, Jan 4]
        expected = np.array([0.0, 2.5, 5.0, 7.5])

        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_temporal_slicing(self, blueprint):
        """Test that forcings are sliced to simulation temporal range before interpolation."""
        forcing_start = "2001-01-01"
        forcing_end = "2021-01-01"
        forcing_dates = pd.date_range(forcing_start, forcing_end, freq="1D", inclusive="left")

        # Create a forcing with a clear pattern: value = day_of_year
        forcing_values = np.array([d.dayofyear for d in forcing_dates], dtype=float)

        config = Config.from_dict(
            {
                "parameters": {},
                "forcings": {
                    "temp": xr.DataArray(
                        forcing_values.reshape(-1, 1, 1),
                        dims=["T", "Y", "X"],
                        coords={"T": forcing_dates},
                    )
                },
                "execution": {
                    "time_start": "2001-01-01",
                    "time_end": "2002-01-01",
                    "dt": "1d",
                    "forcing_interpolation": "linear",
                },
                "initial_state": {},
            }
        )

        model = compile_model(blueprint, config)

        # Should have 365 timesteps (1 non-leap year)
        assert model.n_timesteps == 365

        # Interpolation is deferred — use get_all() to materialize
        all_forcings = model.forcings.get_all_dynamic()
        result = np.asarray(all_forcings["temp"]).flatten()

        # Values should be day_of_year from 1 to 365 (first year only)
        expected = np.arange(1, 366, dtype=float)

        np.testing.assert_allclose(result, expected, atol=1e-5)
