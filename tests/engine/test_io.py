"""Tests for async I/O functionality."""

import numpy as np
import pytest

from seapopym.engine.exceptions import EngineIOError
from seapopym.engine.io import DiskWriter


class TestDiskWriter:
    """Tests for DiskWriter class."""

    def test_init_creates_directory(self, tmp_path):
        """Test that initialization creates output directory."""
        output_path = tmp_path / "output"
        assert not output_path.exists()

        writer = DiskWriter(output_path)
        writer.initialize({"Y": 10, "X": 20}, ["biomass"])

        assert output_path.exists()
        writer.close()

    def test_write_single_chunk(self, tmp_path):
        """Test writing a single chunk."""
        output_path = tmp_path / "output"
        writer = DiskWriter(output_path)
        writer.initialize({"Y": 5, "X": 5}, ["biomass"])

        # Write a chunk
        data = {"biomass": np.ones((10, 5, 5)) * 100.0}
        writer.append(data, chunk_index=0)
        writer.finalize()
        writer.close()

        # Verify data was written
        import zarr

        store = zarr.open(str(output_path), mode="r")
        assert "biomass" in store
        assert store["biomass"].shape[0] == 10  # type: ignore[union-attr]  # 10 timesteps

    def test_write_multiple_chunks(self, tmp_path):
        """Test writing multiple chunks."""
        output_path = tmp_path / "output"
        writer = DiskWriter(output_path)
        writer.initialize({"Y": 5, "X": 5}, ["biomass"])

        # Write multiple chunks
        for i in range(3):
            data = {"biomass": np.ones((5, 5, 5)) * (i + 1)}
            writer.append(data, chunk_index=i)

        writer.finalize()
        writer.close()

        # Verify all data was written
        import zarr

        store = zarr.open(str(output_path), mode="r")
        assert store["biomass"].shape[0] == 15  # type: ignore[union-attr]  # 3 chunks * 5 timesteps

    def test_write_multiple_variables(self, tmp_path):
        """Test writing multiple variables."""
        output_path = tmp_path / "output"
        writer = DiskWriter(output_path)
        writer.initialize({"Y": 5, "X": 5}, ["biomass", "temperature"])

        data = {
            "biomass": np.ones((5, 5, 5)) * 100.0,
            "temperature": np.ones((5, 5, 5)) * 20.0,
        }
        writer.append(data, chunk_index=0)
        writer.finalize()
        writer.close()

        import zarr

        store = zarr.open(str(output_path), mode="r")
        assert "biomass" in store
        assert "temperature" in store

    def test_context_manager(self, tmp_path):
        """Test using writer as context manager."""
        output_path = tmp_path / "output"

        with DiskWriter(output_path) as writer:
            writer.initialize({"Y": 5, "X": 5}, ["biomass"])
            data = {"biomass": np.ones((5, 5, 5)) * 100.0}
            writer.append(data, chunk_index=0)

        # Should have flushed and closed
        import zarr

        store = zarr.open(str(output_path), mode="r")
        assert store["biomass"].shape[0] == 5  # type: ignore[union-attr]

    def test_write_without_init_raises(self, tmp_path):
        """Test that writing without initialization raises error."""
        output_path = tmp_path / "output"
        writer = DiskWriter(output_path)

        with pytest.raises(EngineIOError):
            writer.append({"biomass": np.ones((5, 5, 5))}, chunk_index=0)

        writer.close()

    def test_jax_array_conversion(self, tmp_path):
        """Test that JAX arrays are converted to numpy."""
        pytest.importorskip("jax")
        import jax.numpy as jnp

        output_path = tmp_path / "output"
        with DiskWriter(output_path) as writer:
            writer.initialize({"Y": 5, "X": 5}, ["biomass"])

            # Use JAX array
            data = {"biomass": jnp.ones((5, 5, 5)) * 100.0}
            writer.append(data, chunk_index=0)

        # Should have been converted and written
        import zarr

        store = zarr.open(str(output_path), mode="r")
        assert store["biomass"].shape[0] == 5  # type: ignore[union-attr]


class TestDiskWriterConcurrency:
    """Tests for concurrent write behavior."""

    def test_parallel_writes(self, tmp_path):
        """Test that multiple writes can proceed in parallel."""
        output_path = tmp_path / "output"
        writer = DiskWriter(output_path, max_workers=2)
        writer.initialize({"Y": 10, "X": 10}, ["biomass"])

        # Submit multiple writes quickly
        for i in range(5):
            data = {"biomass": np.random.rand(10, 10, 10)}
            writer.append(data, chunk_index=i)

        # Flush should wait for all
        writer.finalize()
        writer.close()

        # Verify all data written
        import zarr

        store = zarr.open(str(output_path), mode="r")
        assert store["biomass"].shape[0] == 50  # type: ignore[union-attr]  # 5 chunks * 10 timesteps
