"""Tests for I/O functionality."""

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
        writer.append(data)
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
            writer.append(data)

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
        writer.append(data)
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
            writer.append(data)

        # Should have flushed and closed
        import zarr

        store = zarr.open(str(output_path), mode="r")
        assert store["biomass"].shape[0] == 5  # type: ignore[union-attr]

    def test_write_without_init_raises(self, tmp_path):
        """Test that writing without initialization raises error."""
        output_path = tmp_path / "output"
        writer = DiskWriter(output_path)

        with pytest.raises(EngineIOError):
            writer.append({"biomass": np.ones((5, 5, 5))})

        writer.close()

    def test_jax_array_conversion(self, tmp_path):
        """Test that JAX arrays are converted to numpy."""
        import jax.numpy as jnp

        output_path = tmp_path / "output"
        with DiskWriter(output_path) as writer:
            writer.initialize({"Y": 5, "X": 5}, ["biomass"])

            # Use JAX array
            data = {"biomass": jnp.ones((5, 5, 5)) * 100.0}
            writer.append(data)

        # Should have been converted and written
        import zarr

        store = zarr.open(str(output_path), mode="r")
        assert store["biomass"].shape[0] == 5  # type: ignore[union-attr]

    def test_var_dims_shapes(self, tmp_path):
        """Test that var_dims produces correct per-variable shapes."""
        output_path = tmp_path / "output"
        writer = DiskWriter(output_path)
        writer.initialize(
            {"C": 3, "Y": 4, "X": 5},
            ["biomass", "cohort_var"],
            var_dims={"biomass": ("Y", "X"), "cohort_var": ("C", "Y", "X")},
        )

        import zarr

        # biomass: (T, Y, X) → initial shape (0, 4, 5)
        assert writer.store["biomass"].shape == (0, 4, 5)
        assert writer.store["biomass"].attrs["_ARRAY_DIMENSIONS"] == ["T", "Y", "X"]

        # cohort_var: (T, C, Y, X) → initial shape (0, 3, 4, 5)
        assert writer.store["cohort_var"].shape == (0, 3, 4, 5)
        assert writer.store["cohort_var"].attrs["_ARRAY_DIMENSIONS"] == ["T", "C", "Y", "X"]

        writer.close()

    def test_coords_written_to_zarr(self, tmp_path):
        """Test that coordinate arrays are written to the zarr store."""
        output_path = tmp_path / "output"
        writer = DiskWriter(output_path)

        coords = {"Y": np.arange(5), "X": np.arange(3)}
        writer.initialize({"Y": 5, "X": 3}, ["biomass"], coords=coords)

        import zarr

        store = zarr.open(str(output_path), mode="r")
        np.testing.assert_array_equal(store["Y"][:], np.arange(5))
        np.testing.assert_array_equal(store["X"][:], np.arange(3))

        writer.close()

    def test_array_dimensions_attr(self, tmp_path):
        """Test that _ARRAY_DIMENSIONS attribute is set on variables."""
        output_path = tmp_path / "output"
        writer = DiskWriter(output_path)
        writer.initialize({"Y": 5, "X": 5}, ["biomass"])

        import zarr

        store = zarr.open(str(output_path), mode="r")
        assert store["biomass"].attrs["_ARRAY_DIMENSIONS"] == ["T", "Y", "X"]

        writer.close()


