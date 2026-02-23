"""Tests for backend implementations."""

import numpy as np
import pytest

from seapopym.engine.backends import JAXBackend
from seapopym.engine.exceptions import BackendError


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
