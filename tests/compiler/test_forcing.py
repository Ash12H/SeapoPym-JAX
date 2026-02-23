"""Tests for ForcingStore, compute_source_window, and interpolate_chunk."""

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("jax")

from seapopym.compiler.forcing import (
    ForcingStore,
    _interpolate_full,
    compute_source_window,
    interpolate_chunk,
)


class TestComputeSourceWindow:
    """Tests for compute_source_window()."""

    def test_full_range(self):
        """Full chunk covers all source indices."""
        src_start, src_end = compute_source_window(
            source_len=5, target_len=10, start=0, end=10, method="linear"
        )
        assert src_start == 0
        assert src_end == 5

    def test_first_half_linear(self):
        """First half of target should need first portion of source."""
        src_start, src_end = compute_source_window(
            source_len=10, target_len=20, start=0, end=10, method="linear"
        )
        assert src_start == 0
        # target_indices[9] = linspace(0, 9, 20)[9] = 9 * 9/19 ≈ 4.26
        # ceil(4.26) = 5
        assert src_end == 6  # indices 0..5 inclusive

    def test_second_half_linear(self):
        """Second half of target should need second portion of source."""
        src_start, src_end = compute_source_window(
            source_len=10, target_len=20, start=10, end=20, method="linear"
        )
        # target_indices[10] = linspace(0, 9, 20)[10] = 10 * 9/19 ≈ 4.73
        # floor(4.73) = 4
        assert src_start == 4
        assert src_end == 10

    def test_nearest(self):
        """Nearest uses round()."""
        src_start, src_end = compute_source_window(
            source_len=4, target_len=8, start=0, end=4, method="nearest"
        )
        # target_indices = linspace(0, 3, 8) = [0, 0.43, 0.86, 1.29, ...]
        # round([0, 0.43, 0.86, 1.29]) = [0, 0, 1, 1]
        assert src_start == 0
        assert src_end == 2  # 0 and 1

    def test_ffill(self):
        """Forward fill uses floor()."""
        src_start, src_end = compute_source_window(
            source_len=2, target_len=4, start=0, end=4, method="ffill"
        )
        # target_indices = linspace(0, 1, 4) = [0, 0.33, 0.66, 1]
        # floor = [0, 0, 0, 1]
        assert src_start == 0
        assert src_end == 2

    def test_single_source(self):
        """Edge case: only 1 source timestep."""
        src_start, src_end = compute_source_window(
            source_len=1, target_len=10, start=0, end=10, method="linear"
        )
        assert src_start == 0
        assert src_end == 1


class TestInterpolateChunk:
    """Tests for interpolate_chunk() and _interpolate_full()."""

    def test_linear_1d(self):
        """Linear interpolation of a 1D ramp."""
        source = np.array([0.0, 10.0, 20.0, 30.0, 40.0]).reshape(-1, 1)
        source_idx = np.arange(5, dtype=np.float64)
        target_idx = np.linspace(0, 4, 9)

        result = interpolate_chunk(source, source_idx, target_idx, "linear")
        expected = np.linspace(0, 40, 9).reshape(-1, 1)
        np.testing.assert_allclose(np.asarray(result), expected, atol=1e-5)

    def test_linear_2d(self):
        """Linear interpolation preserves spatial dims."""
        source = np.arange(6).reshape(2, 3).astype(float)  # (2, 3)
        source_idx = np.array([0.0, 1.0])
        target_idx = np.array([0.0, 0.5, 1.0])

        result = interpolate_chunk(source, source_idx, target_idx, "linear")
        assert result.shape == (3, 3)
        np.testing.assert_allclose(np.asarray(result[0]), source[0], atol=1e-5)
        np.testing.assert_allclose(np.asarray(result[2]), source[1], atol=1e-5)
        np.testing.assert_allclose(np.asarray(result[1]), (source[0] + source[1]) / 2, atol=1e-5)

    def test_nearest(self):
        """Nearest neighbor interpolation."""
        source = np.array([10.0, 20.0]).reshape(-1, 1)
        source_idx = np.array([0.0, 1.0])
        target_idx = np.linspace(0, 1, 4)  # [0, 0.33, 0.66, 1]

        result = interpolate_chunk(source, source_idx, target_idx, "nearest")
        expected = np.array([10.0, 10.0, 20.0, 20.0]).reshape(-1, 1)
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_ffill(self):
        """Forward fill interpolation."""
        source = np.array([10.0, 20.0]).reshape(-1, 1)
        source_idx = np.array([0.0, 1.0])
        target_idx = np.linspace(0, 1, 4)  # [0, 0.33, 0.66, 1]

        result = interpolate_chunk(source, source_idx, target_idx, "ffill")
        expected = np.array([10.0, 10.0, 10.0, 20.0]).reshape(-1, 1)
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_interpolate_full_linear(self):
        """Full interpolation from source_len to target_len."""
        source = np.linspace(0, 40, 5).reshape(-1, 1, 1)
        result = _interpolate_full(source, 5, 9, "linear")
        expected = np.linspace(0, 40, 9).reshape(-1, 1, 1)
        np.testing.assert_allclose(np.asarray(result), expected, atol=1e-5)

    def test_interpolate_full_nearest(self):
        """Full interpolation with nearest method."""
        source = np.array([10.0, 20.0]).reshape(-1, 1, 1)
        result = _interpolate_full(source, 2, 4, "nearest")
        expected = np.array([10.0, 10.0, 20.0, 20.0]).reshape(-1, 1, 1)
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_interpolate_full_ffill(self):
        """Full interpolation with ffill method."""
        source = np.array([10.0, 20.0]).reshape(-1, 1, 1)
        result = _interpolate_full(source, 2, 4, "ffill")
        expected = np.array([10.0, 10.0, 10.0, 20.0]).reshape(-1, 1, 1)
        np.testing.assert_array_equal(np.asarray(result), expected)


class TestForcingStore:
    """Tests for ForcingStore."""

    def test_get_chunk_static(self):
        """Static forcing is broadcast to chunk length."""
        mask = np.ones((5, 5))
        store = ForcingStore(
            _forcings={"mask": mask},
            n_timesteps=10,
        )

        chunk = store.get_chunk(0, 3)
        assert chunk["mask"].shape == (3, 5, 5)
        np.testing.assert_array_equal(np.asarray(chunk["mask"][0]), mask)

    def test_get_chunk_dynamic(self):
        """Dynamic forcing is sliced to chunk range."""
        temp = np.random.rand(10, 5, 5)
        store = ForcingStore(
            _forcings={"temperature": temp},
            n_timesteps=10,
            _dynamic_forcings={"temperature"},
        )

        chunk = store.get_chunk(2, 5)
        assert chunk["temperature"].shape == (3, 5, 5)
        np.testing.assert_allclose(np.asarray(chunk["temperature"]), temp[2:5], rtol=1e-6)

    def test_get_all(self):
        """get_all materializes full time range."""
        temp = np.random.rand(10, 5, 5)
        mask = np.ones((5, 5))
        store = ForcingStore(
            _forcings={"temperature": temp, "mask": mask},
            n_timesteps=10,
            _dynamic_forcings={"temperature"},
        )

        all_forcings = store.get_all()
        assert all_forcings["temperature"].shape == (10, 5, 5)
        assert all_forcings["mask"].shape == (10, 5, 5)

    def test_get_single(self):
        """get() returns raw forcing value."""
        mask = np.ones((5, 5))
        store = ForcingStore(
            _forcings={"mask": mask},
            n_timesteps=10,
        )

        result = store.get("mask")
        np.testing.assert_array_equal(np.asarray(result), mask)

    def test_get_default(self):
        """get() returns default when key not found."""
        store = ForcingStore(
            _forcings={},
            n_timesteps=10,
        )
        assert store.get("missing", 1.0) == 1.0

    def test_contains(self):
        """__contains__ works."""
        store = ForcingStore(
            _forcings={"mask": np.ones((5, 5))},
            n_timesteps=10,
        )
        assert "mask" in store
        assert "temperature" not in store

    def test_getitem(self):
        """__getitem__ works."""
        mask = np.ones((5, 5))
        store = ForcingStore(
            _forcings={"mask": mask},
            n_timesteps=10,
        )
        np.testing.assert_array_equal(np.asarray(store["mask"]), mask)

    def test_lazy_xarray_static(self):
        """Lazy xr.DataArray without T dim returns values directly."""
        da = xr.DataArray(np.ones((5, 5)), dims=["Y", "X"])
        store = ForcingStore(
            _forcings={"mask": da},
            n_timesteps=10,
        )

        chunk = store.get_chunk(0, 3)
        assert chunk["mask"].shape == (3, 5, 5)

    def test_lazy_xarray_aligned(self):
        """Lazy xr.DataArray with T dim matching n_timesteps is sliced."""
        data = np.random.rand(10, 5, 5).astype(np.float32)
        da = xr.DataArray(data, dims=["T", "Y", "X"])
        store = ForcingStore(
            _forcings={"temp": da},
            n_timesteps=10,
            _dynamic_forcings={"temp"},
        )

        chunk = store.get_chunk(2, 5)
        assert chunk["temp"].shape == (3, 5, 5)
        np.testing.assert_array_equal(np.asarray(chunk["temp"]), data[2:5])

    def test_lazy_xarray_interpolated(self):
        """Lazy xr.DataArray needing interpolation."""
        # 5 source timesteps → 10 target timesteps, linear
        data = np.linspace(0, 40, 5).reshape(-1, 1, 1).astype(np.float64)
        da = xr.DataArray(data, dims=["T", "Y", "X"])
        store = ForcingStore(
            _forcings={"temp": da},
            n_timesteps=10,
            interp_method="linear",
            _dynamic_forcings={"temp"},
        )

        all_data = store.get_chunk(0, 10)
        result = np.asarray(all_data["temp"]).flatten()
        expected = np.linspace(0, 40, 10)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_jax_arrays(self):
        """ForcingStore produces JAX arrays."""
        temp = np.random.rand(10, 5, 5)
        store = ForcingStore(
            _forcings={"temperature": temp},
            n_timesteps=10,
            _dynamic_forcings={"temperature"},
        )

        chunk = store.get_chunk(0, 3)
        assert hasattr(chunk["temperature"], "device")  # JAX array check
        assert chunk["temperature"].shape == (3, 5, 5)

    def test_chunk_consistency(self):
        """Chunks cover the full range without gaps."""
        temp = np.arange(20).reshape(20, 1).astype(float)
        store = ForcingStore(
            _forcings={"temp": temp},
            n_timesteps=20,
            _dynamic_forcings={"temp"},
        )

        # Get all data in chunks of 7
        chunks = []
        for start in range(0, 20, 7):
            end = min(start + 7, 20)
            chunk = store.get_chunk(start, end)
            chunks.append(np.asarray(chunk["temp"]))

        reconstructed = np.concatenate(chunks, axis=0)
        np.testing.assert_array_equal(reconstructed, temp)

    def test_xarray_not_materialized_at_compile(self):
        """xr.DataArray forcings are stored lazy in ForcingStore."""
        da = xr.DataArray(np.random.rand(10, 5, 5), dims=["T", "Y", "X"])
        store = ForcingStore(
            _forcings={"temp": da},
            n_timesteps=10,
            _dynamic_forcings={"temp"},
        )
        # The stored value is still an xr.DataArray, not materialized
        assert isinstance(store._forcings["temp"], xr.DataArray)

        # get_chunk materializes just the chunk
        chunk = store.get_chunk(0, 3)
        assert chunk["temp"].shape == (3, 5, 5)
