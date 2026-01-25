"""Tests for backend validation utilities."""

import numpy as np
import pytest
import xarray as xr

from seapopym.backend.exceptions import BackendConfigurationError
from seapopym.backend.validation import (
    get_chunked_variables,
    has_chunked_arrays,
    validate_has_chunks,
    validate_no_chunks,
)


@pytest.fixture
def chunked_dataset():
    ds = xr.Dataset(
        {"var1": (("x", "y"), np.zeros((10, 10)))}, coords={"x": np.arange(10), "y": np.arange(10)}
    )
    return ds.chunk({"x": 5})


@pytest.fixture
def unchunked_dataset():
    return xr.Dataset(
        {"var1": (("x", "y"), np.zeros((10, 10)))}, coords={"x": np.arange(10), "y": np.arange(10)}
    )


def test_has_chunked_arrays(chunked_dataset, unchunked_dataset):
    assert has_chunked_arrays(chunked_dataset) is True
    assert has_chunked_arrays(unchunked_dataset) is False


def test_get_chunked_variables(chunked_dataset, unchunked_dataset):
    chunks = get_chunked_variables(chunked_dataset)
    assert "var1" in chunks
    # Check structure ((5, 5), (10,)) depending on xarray/dask version but definitely tuples
    assert isinstance(chunks["var1"], tuple)

    assert get_chunked_variables(unchunked_dataset) == {}


def test_validate_no_chunks_error(chunked_dataset):
    with pytest.raises(BackendConfigurationError, match="detected chunked Dask arrays"):
        validate_no_chunks(chunked_dataset, "TestBackend")


def test_validate_no_chunks_success(unchunked_dataset):
    # Should not raise
    validate_no_chunks(unchunked_dataset, "TestBackend")


def test_validate_has_chunks_warning(unchunked_dataset):
    with pytest.warns(UserWarning, match="selected but no chunked Dask arrays found"):
        validate_has_chunks(unchunked_dataset, "TestBackend")


def test_validate_has_chunks_success(chunked_dataset, caplog):
    # Should check logs for info
    import logging

    with caplog.at_level(logging.INFO):
        validate_has_chunks(chunked_dataset, "TestBackend")

    assert "TestBackend: Found 1 chunked variable" in caplog.text
