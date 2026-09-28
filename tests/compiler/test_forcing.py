"""Tests for ForcingStore."""

import numpy as np
import pytest
import xarray as xr

from seapopym.compiler.forcing import ForcingStore


class TestForcingStore:
    """Tests for ForcingStore."""

    def test_get_statics(self):
        """get_statics() returns JAX arrays for all static forcings."""
        mask = xr.DataArray(np.ones((5, 5)), dims=["Y", "X"])
        store = ForcingStore(
            _static={"mask": mask},
            n_timesteps=10,
        )

        statics = store.get_statics()
        assert statics["mask"].shape == (5, 5)
        np.testing.assert_array_equal(np.asarray(statics["mask"]), 1.0)

    def test_get_chunk_dynamic(self):
        """Dynamic forcing is sliced to chunk range."""
        data = np.random.rand(10, 5, 5)
        temp = xr.DataArray(data, dims=["T", "Y", "X"])
        store = ForcingStore(
            _dynamic={"temperature": temp},
            n_timesteps=10,
        )

        chunk = store.get_chunk(2, 5)
        assert chunk["temperature"].shape == (3, 5, 5)
        np.testing.assert_allclose(np.asarray(chunk["temperature"]), data[2:5], rtol=1e-6)

    def test_get_all(self):
        """get_all materializes full time range for dynamics."""
        data = np.random.rand(10, 5, 5)
        temp = xr.DataArray(data, dims=["T", "Y", "X"])
        store = ForcingStore(
            _dynamic={"temperature": temp},
            n_timesteps=10,
        )

        all_forcings = store.get_all_dynamic()
        assert all_forcings["temperature"].shape == (10, 5, 5)

    def test_get_single_static(self):
        """get() returns a static forcing as JAX array."""
        mask = xr.DataArray(np.ones((5, 5)), dims=["Y", "X"])
        store = ForcingStore(_static={"mask": mask})

        result = store.get("mask")
        np.testing.assert_array_equal(np.asarray(result), 1.0)

    def test_get_single_dynamic(self):
        """get() returns a dynamic forcing as JAX array."""
        data = np.random.rand(10, 5, 5)
        store = ForcingStore(
            _dynamic={"temp": xr.DataArray(data, dims=["T", "Y", "X"])},
        )

        result = store.get("temp")
        np.testing.assert_allclose(np.asarray(result), data, rtol=1e-6)

    def test_get_default(self):
        """get() returns default when key not found."""
        store = ForcingStore()
        assert store.get("missing", 1.0) == 1.0

    def test_contains(self):
        """__contains__ works for both static and dynamic."""
        store = ForcingStore(
            _static={"mask": xr.DataArray(np.ones((5, 5)), dims=["Y", "X"])},
            _dynamic={"temp": xr.DataArray(np.random.rand(10, 5, 5), dims=["T", "Y", "X"])},
        )
        assert "mask" in store
        assert "temp" in store
        assert "missing" not in store

    def test_getitem(self):
        """__getitem__ works."""
        mask = xr.DataArray(np.ones((5, 5)), dims=["Y", "X"])
        store = ForcingStore(_static={"mask": mask})
        np.testing.assert_array_equal(np.asarray(store["mask"]), 1.0)

    def test_jax_arrays(self):
        """ForcingStore produces JAX arrays."""
        data = np.random.rand(10, 5, 5)
        store = ForcingStore(
            _dynamic={"temperature": xr.DataArray(data, dims=["T", "Y", "X"])},
            n_timesteps=10,
        )

        chunk = store.get_chunk(0, 3)
        assert hasattr(chunk["temperature"], "device")
        assert chunk["temperature"].shape == (3, 5, 5)

    def test_chunk_consistency(self):
        """Chunks cover the full range without gaps."""
        data = np.arange(20).reshape(20, 1).astype(float)
        store = ForcingStore(
            _dynamic={"temp": xr.DataArray(data, dims=["T", "Y"])},
            n_timesteps=20,
        )

        chunks = []
        for start in range(0, 20, 7):
            end = min(start + 7, 20)
            chunk = store.get_chunk(start, end)
            chunks.append(np.asarray(chunk["temp"]))

        reconstructed = np.concatenate(chunks, axis=0)
        np.testing.assert_array_equal(reconstructed, data)

    def test_xarray_not_materialized_at_compile(self):
        """xr.DataArray forcings are stored lazy in ForcingStore."""
        da = xr.DataArray(np.random.rand(10, 5, 5), dims=["T", "Y", "X"])
        store = ForcingStore(
            _dynamic={"temp": da},
            n_timesteps=10,
        )
        assert isinstance(store._dynamic["temp"], xr.DataArray)

        chunk = store.get_chunk(0, 3)
        assert chunk["temp"].shape == (3, 5, 5)

    def test_from_config(self):
        """from_config() correctly separates static/dynamic."""
        forcings = {
            "temp": xr.DataArray(np.random.rand(10, 5, 5), dims=["T", "Y", "X"]),
            "mask": xr.DataArray(np.ones((5, 5)), dims=["Y", "X"]),
        }
        blueprint_dims = {
            "forcings.temp": ["T", "Y", "X"],
            "forcings.mask": ["Y", "X"],
        }

        store = ForcingStore.from_config(
            forcings=forcings,
            blueprint_dims=blueprint_dims,
            n_timesteps=10,
        )

        assert "temp" in store._dynamic
        assert "mask" in store._static
        assert "temp" not in store._static
        assert "mask" not in store._dynamic

    def test_chunk_consistency_with_interpolation_linear(self):
        """Chunked interpolation (linear) matches get_all."""
        import pandas as pd

        source_times = pd.date_range("2000-01-01", periods=5, freq="2D")
        target_times = pd.date_range("2000-01-01", periods=9, freq="1D")
        data = np.linspace(0, 40, 5).reshape(-1, 1).astype(np.float64)
        da = xr.DataArray(data, dims=["T", "Y"], coords={"T": source_times})

        store = ForcingStore(
            _dynamic={"temp": da},
            n_timesteps=9,
            interp_method="linear",
            _time_coords=target_times.values,
        )

        all_data = np.asarray(store.get_all_dynamic()["temp"])
        chunks = []
        for start in range(0, 9, 3):
            end = min(start + 3, 9)
            chunks.append(np.asarray(store.get_chunk(start, end)["temp"]))
        reconstructed = np.concatenate(chunks, axis=0)
        np.testing.assert_allclose(reconstructed, all_data, rtol=1e-10)

    def test_chunk_consistency_with_interpolation_nearest(self):
        """Chunked interpolation (nearest) matches get_all."""
        import pandas as pd

        source_times = pd.date_range("2000-01-01", periods=5, freq="2D")
        target_times = pd.date_range("2000-01-01", periods=9, freq="1D")
        data = np.linspace(0, 40, 5).reshape(-1, 1).astype(np.float64)
        da = xr.DataArray(data, dims=["T", "Y"], coords={"T": source_times})

        store = ForcingStore(
            _dynamic={"temp": da},
            n_timesteps=9,
            interp_method="nearest",
            _time_coords=target_times.values,
        )

        all_data = np.asarray(store.get_all_dynamic()["temp"])
        chunks = []
        for start in range(0, 9, 3):
            end = min(start + 3, 9)
            chunks.append(np.asarray(store.get_chunk(start, end)["temp"]))
        reconstructed = np.concatenate(chunks, axis=0)
        np.testing.assert_allclose(reconstructed, all_data, rtol=1e-10)

    def test_chunk_consistency_with_interpolation_ffill(self):
        """Chunked interpolation (ffill) matches get_all."""
        import pandas as pd

        source_times = pd.date_range("2000-01-01", periods=5, freq="2D")
        target_times = pd.date_range("2000-01-01", periods=9, freq="1D")
        data = np.linspace(0, 40, 5).reshape(-1, 1).astype(np.float64)
        da = xr.DataArray(data, dims=["T", "Y"], coords={"T": source_times})

        store = ForcingStore(
            _dynamic={"temp": da},
            n_timesteps=9,
            interp_method="ffill",
            _time_coords=target_times.values,
        )

        all_data = np.asarray(store.get_all_dynamic()["temp"])
        chunks = []
        for start in range(0, 9, 3):
            end = min(start + 3, 9)
            chunks.append(np.asarray(store.get_chunk(start, end)["temp"]))
        reconstructed = np.concatenate(chunks, axis=0)
        np.testing.assert_allclose(reconstructed, all_data, rtol=1e-10)

    def test_windowing_reduces_source_size(self):
        """Windowing slices source to a small window around target times."""
        import pandas as pd

        source_times = pd.date_range("2000-01-01", periods=100, freq="1D")
        data = np.arange(100).reshape(-1, 1).astype(np.float64)
        da = xr.DataArray(data, dims=["T", "Y"], coords={"T": source_times})

        store = ForcingStore(interp_method="linear")
        target = pd.date_range("2000-03-01", periods=5, freq="1D").values
        windowed = store._compute_source_window(da, target)
        assert windowed.sizes["T"] <= 7

    def test_windowing_full_range_passthrough(self):
        """When target covers all source, return the same object."""
        import pandas as pd

        source_times = pd.date_range("2000-01-01", periods=10, freq="1D")
        data = np.arange(10).reshape(-1, 1).astype(np.float64)
        da = xr.DataArray(data, dims=["T", "Y"], coords={"T": source_times})

        store = ForcingStore(interp_method="linear")
        target = pd.date_range("1999-12-01", periods=60, freq="1D").values
        windowed = store._compute_source_window(da, target)
        assert windowed is da

    def test_windowing_no_t_coord_passthrough(self):
        """DataArray without T dim is returned unchanged."""
        da = xr.DataArray(np.ones((5, 5)), dims=["Y", "X"])
        store = ForcingStore(interp_method="linear")
        windowed = store._compute_source_window(da, np.array([]))
        assert windowed is da

    def test_get_chunk_nan_raises(self):
        """get_chunk() raises ValueError when dynamic forcing contains NaN."""
        data = np.ones((10, 5, 5))
        data[3, 2, 2] = np.nan
        store = ForcingStore(
            _dynamic={"temp": xr.DataArray(data, dims=["T", "Y", "X"])},
            n_timesteps=10,
        )
        with pytest.raises(ValueError, match="NaN"):
            store.get_chunk(0, 10)

    def test_get_statics_nan_raises(self):
        """get_statics() raises ValueError when static forcing contains NaN."""
        mask = np.ones((5, 5))
        mask[2, 2] = np.nan
        store = ForcingStore(
            _static={"mask": xr.DataArray(mask, dims=["Y", "X"])},
        )
        with pytest.raises(ValueError, match="NaN"):
            store.get_statics()

    def test_get_chunk_int_no_nan_check(self):
        """Integer arrays skip NaN check (int has no NaN)."""
        data = np.ones((10, 5), dtype=np.int32)
        store = ForcingStore(
            _dynamic={"flags": xr.DataArray(data, dims=["T", "Y"])},
            n_timesteps=10,
        )
        chunk = store.get_chunk(0, 5)
        assert chunk["flags"].shape == (5, 5)

    def test_linear_interpolation_on_nan_free_frames_ignores_unused_nan(self):
        """A NaN in a frame no target reads does not raise, one in a read frame does."""
        import pandas as pd

        source_times = pd.date_range("2000-01-01", periods=6, freq="1D")
        data = np.ones((6, 3), dtype=np.float32)
        data[5, 1] = np.nan
        da = xr.DataArray(data, dims=["T", "Y"], coords={"T": source_times})
        targets = pd.date_range("2000-01-01", periods=10, freq="6h").values
        store = ForcingStore(_dynamic={"temp": da}, n_timesteps=10, interp_method="linear", _time_coords=targets)
        np.testing.assert_array_equal(np.asarray(store.get_chunk(0, 10)["temp"]), 1.0)

        data[2, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            store.get_chunk(0, 10)

    def test_interpolation_without_time_coords_error(self):
        """Interpolation with _time_coords=None raises ValueError."""
        import pandas as pd

        source_times = pd.date_range("2000-01-01", periods=5, freq="2D")
        data = np.linspace(0, 40, 5).reshape(-1, 1, 1).astype(np.float64)
        da = xr.DataArray(data, dims=["T", "Y", "X"], coords={"T": source_times})

        store = ForcingStore(
            _dynamic={"temp": da},
            n_timesteps=10,
            interp_method="linear",
            _time_coords=None,
        )
        with pytest.raises(ValueError, match="_time_coords is None"):
            store.get_chunk(0, 10)


class TestLinearInterpolationOnDevice:
    """Linear interpolation builds the steps on the device, as xarray would on the host."""

    @staticmethod
    def _store(data, source_times, target_times):
        import pandas as pd

        da = xr.DataArray(data, dims=["T", "Y", "X"], coords={"T": pd.DatetimeIndex(source_times)})
        return da, ForcingStore(
            _dynamic={"temp": da},
            n_timesteps=len(target_times),
            interp_method="linear",
            _time_coords=np.asarray(target_times),
        )

    def test_matches_xarray_between_on_and_beyond_frames(self):
        """Steps between frames, on frames and past the last frame follow xarray's interp."""
        import pandas as pd

        rng = np.random.default_rng(0)
        source = pd.date_range("2015-09-01", periods=4, freq="1D")
        # 40-minute steps from the first frame to beyond the last: extrapolation included.
        target = pd.date_range("2015-09-01", periods=36 * 5, freq="40min")
        data = rng.uniform(-2.0, 30.0, (4, 3, 5)).astype(np.float32)
        da, store = self._store(data, source, target)

        got = np.asarray(store.get_chunk(0, len(target))["temp"])
        expected = da.interp(T=target.values, method="linear", kwargs={"fill_value": "extrapolate"}).values
        assert got.shape == expected.shape
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-5)

    def test_targets_on_frames_are_the_frames(self):
        """A step at a source time gets that frame exactly."""
        import pandas as pd

        rng = np.random.default_rng(1)
        source = pd.date_range("2015-09-01", periods=3, freq="1D")
        target = pd.date_range("2015-09-01", periods=3 * 36, freq="40min")
        data = rng.uniform(0.0, 1.0, (3, 2, 2)).astype(np.float32)
        _, store = self._store(data, source, target)

        got = np.asarray(store.get_chunk(0, len(target))["temp"])
        np.testing.assert_array_equal(got[::36], data)

    def test_chunks_equal_one_call(self):
        """Chunked steps are the steps of a single call, to the bit."""
        import pandas as pd

        rng = np.random.default_rng(2)
        source = pd.date_range("2015-09-01", periods=5, freq="1D")
        target = pd.date_range("2015-09-01", periods=36 * 4, freq="40min")
        data = rng.uniform(0.0, 1.0, (5, 2, 3)).astype(np.float32)
        _, store = self._store(data, source, target)

        whole = np.asarray(store.get_all_dynamic()["temp"])
        parts = np.concatenate(
            [np.asarray(store.get_chunk(s, min(s + 36, len(target)))["temp"]) for s in range(0, len(target), 36)]
        )
        np.testing.assert_array_equal(parts, whole)

    def test_integer_forcing_becomes_float(self):
        """An integer forcing is interpolated in floating point, as xarray returns it."""
        import pandas as pd

        source = pd.date_range("2015-09-01", periods=2, freq="1D")
        target = pd.date_range("2015-09-01", periods=3, freq="12h")
        data = np.array([[[0]], [[2]]], dtype=np.int32)
        _, store = self._store(data, source, target)

        got = np.asarray(store.get_chunk(0, 3)["temp"])
        assert np.issubdtype(got.dtype, np.floating)
        np.testing.assert_allclose(got[:, 0, 0], [0.0, 1.0, 2.0])

    def test_result_is_a_device_array(self):
        """The chunk is a JAX array, built where it is used."""
        import pandas as pd

        source = pd.date_range("2015-09-01", periods=2, freq="1D")
        target = pd.date_range("2015-09-01", periods=4, freq="6h")
        _, store = self._store(np.zeros((2, 2, 2), dtype=np.float32), source, target)
        assert hasattr(store.get_chunk(0, 4)["temp"], "device")

    @pytest.mark.parametrize(
        "days", [["2015-09-01", "2015-09-02", "2015-09-02"], ["2015-09-01", "2015-09-03", "2015-09-02"]]
    )
    def test_repeated_or_unsorted_times_raise(self, days):
        """Linear brackets need strictly increasing source times: anything else is an error."""
        import pandas as pd

        target = pd.date_range("2015-09-01", periods=4, freq="12h")
        _, store = self._store(np.zeros((3, 1, 1), dtype=np.float32), pd.DatetimeIndex(days), target)
        with pytest.raises(ValueError, match="strictly increasing"):
            store.get_chunk(0, 4)

    def test_half_precision_is_interpolated_in_single(self):
        """A float16 forcing comes back in float32, as the host path cast xarray's float64."""
        import pandas as pd

        source = pd.date_range("2015-09-01", periods=2, freq="1D")
        target = pd.date_range("2015-09-01", periods=3, freq="12h")
        data = np.array([[[0.1]], [[0.3]]], dtype=np.float16)
        _, store = self._store(data, source, target)
        got = store.get_chunk(0, 3)["temp"]
        assert got.dtype == np.float32
        np.testing.assert_allclose(np.asarray(got)[:, 0, 0], [0.1, 0.2, 0.3], rtol=1e-3)

    def test_time_axis_keeps_its_position(self):
        """A forcing whose time axis is not first comes back as xarray interpolates it."""
        import pandas as pd

        rng = np.random.default_rng(3)
        source = pd.date_range("2015-09-01", periods=3, freq="1D")
        target = pd.date_range("2015-09-01", periods=9, freq="6h")
        da = xr.DataArray(rng.uniform(0, 1, (4, 3)).astype(np.float32), dims=["Y", "T"], coords={"T": source})
        store = ForcingStore(_dynamic={"temp": da}, n_timesteps=9, interp_method="linear", _time_coords=target.values)
        got = np.asarray(store.get_chunk(0, 9)["temp"])
        expected = da.interp(T=target.values, method="linear", kwargs={"fill_value": "extrapolate"}).values
        assert got.shape == expected.shape == (4, 9)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-7)
