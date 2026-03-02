"""Tests for coords_to_indices coordinate conversion."""

import numpy as np
import pytest

from seapopym.compiler.coords import coords_to_indices


class TestCoordsToIndicesNominal:
    """Tests for nominal coords_to_indices behavior."""

    @pytest.fixture
    def grid(self):
        """Simple lat/lon grid."""
        return {
            "Y": np.array([40.0, 41.0, 42.0, 43.0, 44.0]),
            "X": np.array([-5.0, -4.0, -3.0, -2.0, -1.0]),
        }

    def test_exact_match(self, grid):
        """Exact coordinates should return correct indices."""
        (y_idx,) = coords_to_indices(grid, Y=np.array([42.0]))
        assert y_idx == 2

    def test_nearest_match(self, grid):
        """Non-exact coordinates should snap to nearest."""
        (y_idx,) = coords_to_indices(grid, Y=np.array([42.3]))
        assert y_idx == 2  # 42.3 is closest to 42.0

    def test_multiple_points(self, grid):
        """Multiple coordinate values should return multiple indices."""
        (y_idx,) = coords_to_indices(grid, Y=np.array([40.0, 43.0, 44.0]))
        np.testing.assert_array_equal(y_idx, [0, 3, 4])

    def test_multiple_dimensions(self, grid):
        """Multiple dimensions should return tuple of index arrays."""
        y_idx, x_idx = coords_to_indices(
            grid,
            Y=np.array([41.0, 43.0]),
            X=np.array([-3.0, -1.0]),
        )
        np.testing.assert_array_equal(y_idx, [1, 3])
        np.testing.assert_array_equal(x_idx, [2, 4])

    def test_single_scalar_value(self, grid):
        """Scalar coordinate value should work."""
        (x_idx,) = coords_to_indices(grid, X=np.array(-3.0))
        assert int(x_idx) == 2

    def test_preserves_dimension_order(self, grid):
        """Output order should match kwargs order."""
        x_idx, y_idx = coords_to_indices(
            grid,
            X=np.array([-5.0]),
            Y=np.array([44.0]),
        )
        assert x_idx == 0
        assert y_idx == 4


class TestCoordsToIndicesErrors:
    """Tests for coords_to_indices error handling."""

    @pytest.fixture
    def grid(self):
        """Simple grid."""
        return {
            "Y": np.array([40.0, 41.0, 42.0]),
            "X": np.array([-3.0, -2.0, -1.0]),
        }

    def test_unknown_dimension_raises(self, grid):
        """Unknown dimension name should raise ValueError."""
        with pytest.raises(ValueError, match="Dimension 'Z' not found"):
            coords_to_indices(grid, Z=np.array([0.0]))

    def test_error_shows_available_dims(self, grid):
        """Error message should list available dimensions."""
        with pytest.raises(ValueError, match="Available"):
            coords_to_indices(grid, T=np.array([0.0]))


class TestCoordsToIndicesSelKwargs:
    """Tests for custom sel_kwargs."""

    @pytest.fixture
    def grid(self):
        """Simple grid."""
        return {"Y": np.array([40.0, 41.0, 42.0, 43.0, 44.0])}

    def test_exact_match_with_method_none(self, grid):
        """method=None requires exact match."""
        (y_idx,) = coords_to_indices(grid, sel_kwargs={}, Y=np.array([42.0]))
        assert y_idx == 2

    def test_exact_match_fails_for_non_exact(self, grid):
        """method=None should raise for non-exact coordinates."""
        with pytest.raises(KeyError):
            coords_to_indices(grid, sel_kwargs={}, Y=np.array([42.3]))

    def test_nearest_with_tolerance(self, grid):
        """method=nearest with tolerance should reject far values."""
        with pytest.raises(KeyError):
            coords_to_indices(
                grid,
                sel_kwargs={"method": "nearest", "tolerance": 0.01},
                Y=np.array([42.5]),
            )

    def test_nearest_within_tolerance(self, grid):
        """method=nearest with tolerance should accept close values."""
        (y_idx,) = coords_to_indices(
            grid,
            sel_kwargs={"method": "nearest", "tolerance": 0.6},
            Y=np.array([42.5]),
        )
        assert y_idx in (2, 3)  # Either 42.0 or 43.0


class TestCoordsToIndicesDatetime:
    """Tests for datetime coordinate support."""

    def test_datetime_coords(self):
        """Datetime coordinates should work with nearest matching."""
        grid = {
            "T": np.array(["2000-01-01", "2000-01-11", "2000-01-21"], dtype="datetime64[ns]"),
        }
        (t_idx,) = coords_to_indices(
            grid,
            T=np.array(["2000-01-05", "2000-01-15"], dtype="datetime64[ns]"),
        )
        np.testing.assert_array_equal(t_idx, [0, 1])
