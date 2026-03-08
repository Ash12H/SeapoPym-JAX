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

    @pytest.mark.xfail(reason="get_chunk() no longer includes statics — Runner must use get_statics() separately (workflow Runner)")
    def test_get_chunk_includes_statics(self):
        """get_chunk() returns both dynamic and static forcings (broadcast)."""
        store = ForcingStore(
            _static={"mask": xr.DataArray(np.ones((5, 5)), dims=["Y", "X"])},
            _dynamic={"temp": xr.DataArray(np.random.rand(10, 5, 5), dims=["T", "Y", "X"])},
            n_timesteps=10,
        )

        chunk = store.get_chunk(0, 3)
        assert "temp" in chunk
        assert "mask" in chunk
        assert chunk["mask"].shape == (3, 5, 5)

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
