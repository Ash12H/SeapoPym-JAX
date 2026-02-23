"""Tests for ForcingStore."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from seapopym.compiler.forcing import ForcingStore


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
        source_times = pd.date_range("2000-01-01", periods=5, freq="2D")
        data = np.linspace(0, 40, 5).reshape(-1, 1, 1).astype(np.float64)
        da = xr.DataArray(data, dims=["T", "Y", "X"], coords={"T": source_times})
        target_times = pd.date_range(source_times[0], source_times[-1], periods=10).to_numpy()

        store = ForcingStore(
            _forcings={"temp": da},
            n_timesteps=10,
            interp_method="linear",
            _dynamic_forcings={"temp"},
            _time_coords=target_times,
        )
        result = np.asarray(store.get_chunk(0, 10)["temp"]).flatten()
        np.testing.assert_allclose(result, np.linspace(0, 40, 10), atol=1e-5)

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
