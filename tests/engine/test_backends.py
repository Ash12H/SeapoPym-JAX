"""Tests for backend implementations."""

import numpy as np
import pytest

from seapopym.engine.backends import JAXBackend, NumpyBackend, get_backend
from seapopym.engine.exceptions import BackendError


class TestNumpyBackend:
    """Tests for NumpyBackend."""

    def test_scan_basic(self):
        """Test basic scan operation."""
        backend = NumpyBackend()

        def step_fn(carry, x):
            new_carry = carry + x
            output = carry * 2
            return new_carry, output

        init = np.array(0.0)
        xs = np.array([1.0, 2.0, 3.0, 4.0])

        final, outputs = backend.scan(step_fn, init, xs)

        assert final == 10.0  # 0 + 1 + 2 + 3 + 4
        np.testing.assert_array_equal(outputs, [0.0, 2.0, 6.0, 12.0])

    def test_scan_with_dict_carry(self):
        """Test scan with dict as carry."""
        backend = NumpyBackend()

        def step_fn(carry, x):
            new_carry = {"value": carry["value"] + x}
            output = {"doubled": carry["value"] * 2}
            return new_carry, output

        init = {"value": np.array(0.0)}
        xs = np.array([1.0, 2.0, 3.0])

        final, outputs = backend.scan(step_fn, init, xs)

        assert final["value"] == 6.0
        np.testing.assert_array_equal(outputs["doubled"], [0.0, 2.0, 6.0])

    def test_scan_with_dict_xs(self):
        """Test scan with dict as xs."""
        backend = NumpyBackend()

        def step_fn(carry, x):
            new_carry = carry + x["a"] + x["b"]
            output = carry
            return new_carry, output

        init = np.array(0.0)
        xs = {
            "a": np.array([1.0, 2.0, 3.0]),
            "b": np.array([10.0, 20.0, 30.0]),
        }

        final, outputs = backend.scan(step_fn, init, xs)

        assert final == 66.0  # 0 + (1+10) + (2+20) + (3+30)

    def test_scan_with_length(self):
        """Test scan with explicit length."""
        backend = NumpyBackend()

        counter = {"n": 0}

        def step_fn(carry, x):
            counter["n"] += 1
            return carry + 1, carry

        init = np.array(0.0)

        final, outputs = backend.scan(step_fn, init, None, length=5)

        assert counter["n"] == 5
        assert final == 5.0

    def test_scan_empty(self):
        """Test scan with empty sequence."""
        backend = NumpyBackend()

        def step_fn(carry, x):
            return carry + x, carry

        init = np.array(10.0)
        xs = np.array([])

        final, outputs = backend.scan(step_fn, init, xs)

        assert final == 10.0
        assert len(outputs) == 0


class TestJAXBackend:
    """Tests for JAXBackend."""

    @pytest.fixture(autouse=True)
    def skip_if_no_jax(self):
        """Skip tests if JAX is not available."""
        pytest.importorskip("jax")

    def test_scan_basic(self):
        """Test basic scan operation."""
        import jax.numpy as jnp

        backend = JAXBackend()

        def step_fn(carry, x):
            new_carry = carry + x
            output = carry * 2
            return new_carry, output

        init = jnp.array(0.0)
        xs = jnp.array([1.0, 2.0, 3.0, 4.0])

        final, outputs = backend.scan(step_fn, init, xs)

        assert float(final) == 10.0
        np.testing.assert_array_almost_equal(outputs, [0.0, 2.0, 6.0, 12.0])

    def test_scan_with_dict_carry(self):
        """Test scan with dict as carry."""
        import jax.numpy as jnp

        backend = JAXBackend()

        def step_fn(carry, x):
            new_carry = {"value": carry["value"] + x}
            output = {"doubled": carry["value"] * 2}
            return new_carry, output

        init = {"value": jnp.array(0.0)}
        xs = jnp.array([1.0, 2.0, 3.0])

        final, outputs = backend.scan(step_fn, init, xs)

        assert float(final["value"]) == 6.0
        np.testing.assert_array_almost_equal(outputs["doubled"], [0.0, 2.0, 6.0])

    def test_scan_with_dict_xs(self):
        """Test scan with dict as xs (pytree)."""
        import jax.numpy as jnp

        backend = JAXBackend()

        def step_fn(carry, x):
            new_carry = carry + x["a"] + x["b"]
            output = carry
            return new_carry, output

        init = jnp.array(0.0)
        xs = {
            "a": jnp.array([1.0, 2.0, 3.0]),
            "b": jnp.array([10.0, 20.0, 30.0]),
        }

        final, outputs = backend.scan(step_fn, init, xs)

        assert float(final) == 66.0


class TestGetBackend:
    """Tests for get_backend function."""

    def test_get_numpy_backend(self):
        """Test getting numpy backend."""
        backend = get_backend("numpy")
        assert isinstance(backend, NumpyBackend)

    def test_get_jax_backend(self):
        """Test getting jax backend."""
        pytest.importorskip("jax")
        backend = get_backend("jax")
        assert isinstance(backend, JAXBackend)

    def test_unknown_backend(self):
        """Test error for unknown backend."""
        with pytest.raises(BackendError) as exc_info:
            get_backend("unknown")  # type: ignore
        assert "unknown" in str(exc_info.value)
