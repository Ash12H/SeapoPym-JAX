"""Tests for automatic vmap vectorization."""

import jax.numpy as jnp
import numpy as np

from seapopym.engine.vectorize import (
    compute_broadcast_dims,
    compute_in_axes,
    remove_dim_from_inputs,
    wrap_with_vmap,
)


class TestComputeBroadcastDims:
    """Tests for compute_broadcast_dims function."""

    def test_single_core_dim(self):
        """Test with single core dimension."""
        input_dims = {"production": ("C", "Y", "X"), "rate": ()}
        core_dims = {"production": ["C"]}

        result = compute_broadcast_dims(input_dims, core_dims)

        assert result == ["Y", "X"]

    def test_multiple_core_dims(self):
        """Test with multiple core dimensions."""
        input_dims = {"data": ("C", "Z", "Y", "X"), "param": ("Z",)}
        core_dims = {"data": ["C", "Z"], "param": ["Z"]}

        result = compute_broadcast_dims(input_dims, core_dims)

        assert result == ["Y", "X"]

    def test_no_broadcast_dims(self):
        """Test when all dims are core dims."""
        input_dims = {"data": ("C",), "param": ()}
        core_dims = {"data": ["C"]}

        result = compute_broadcast_dims(input_dims, core_dims)

        assert result == []

    def test_canonical_order(self):
        """Test that broadcast dims are returned in canonical order."""
        # X comes before Y in input, but Y should come before X in output
        input_dims = {"data": ("X", "Y", "C")}
        core_dims = {"data": ["C"]}

        result = compute_broadcast_dims(input_dims, core_dims)

        # Canonical order: E, T, F, C, Z, Y, X
        assert result == ["Y", "X"]


    def test_non_canonical_dims_not_broadcast(self):
        """Test that non-canonical dims are never included in broadcast dims."""
        input_dims = {"nn_params": ("F", "P"), "production": ("F", "C", "Y", "X")}
        core_dims = {"nn_params": ["P"], "production": ["C"]}

        result = compute_broadcast_dims(input_dims, core_dims)

        # P is non-canonical, should not be broadcast even though it's not a core dim of production
        # Only canonical dims F, Y, X are broadcast candidates
        assert result == ["F", "Y", "X"]
        assert "P" not in result


class TestComputeInAxes:
    """Tests for compute_in_axes function."""

    def test_basic(self):
        """Test basic in_axes computation."""
        input_dims = {"production": ("C", "Y", "X"), "rate": ()}
        arg_order = ["production", "rate"]

        result = compute_in_axes(input_dims, "X", arg_order)

        assert result == (2, None)

    def test_dimension_not_present(self):
        """Test when dimension is not in some inputs."""
        input_dims = {"a": ("C", "Y", "X"), "b": ("Y", "X"), "c": ()}
        arg_order = ["a", "b", "c"]

        result = compute_in_axes(input_dims, "C", arg_order)

        assert result == (0, None, None)

    def test_all_have_dimension(self):
        """Test when all inputs have the dimension."""
        input_dims = {"a": ("Y", "X"), "b": ("Y", "X")}
        arg_order = ["a", "b"]

        result = compute_in_axes(input_dims, "Y", arg_order)

        assert result == (0, 0)


class TestRemoveDimFromInputs:
    """Tests for remove_dim_from_inputs function."""

    def test_basic(self):
        """Test basic dimension removal."""
        input_dims = {"a": ("C", "Y", "X"), "b": ("Y", "X")}

        result = remove_dim_from_inputs(input_dims, "X")

        assert result == {"a": ("C", "Y"), "b": ("Y",)}

    def test_dimension_not_present(self):
        """Test removal of dimension not in some inputs."""
        input_dims = {"a": ("C", "Y", "X"), "b": ("C",)}

        result = remove_dim_from_inputs(input_dims, "X")

        assert result == {"a": ("C", "Y"), "b": ("C",)}


class TestWrapWithVmap:
    """Tests for wrap_with_vmap function."""

    def test_no_core_dims(self):
        """Test that function is unchanged when no broadcast dims."""

        def simple_func(x):
            return x * 2

        input_dims = {"x": ("C",)}
        core_dims = {"x": ["C"]}
        arg_order = ["x"]

        wrapped = wrap_with_vmap(simple_func, input_dims, core_dims, arg_order)

        # Should return original function
        x = jnp.array([1.0, 2.0, 3.0])
        result = wrapped(x)
        expected = jnp.array([2.0, 4.0, 6.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_single_broadcast_dim(self):
        """Test vmap over single broadcast dimension."""

        def sum_over_c(production):
            """Sum over C dimension."""
            return jnp.sum(production)

        # production has shape (C, Y) but function works on (C,)
        input_dims = {"production": ("C", "Y")}
        core_dims = {"production": ["C"]}
        arg_order = ["production"]

        wrapped = wrap_with_vmap(sum_over_c, input_dims, core_dims, arg_order)

        # Input: (C=3, Y=2)
        production = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = wrapped(production)

        # Expected: sum over C for each Y
        # Y=0: 1+3+5=9, Y=1: 2+4+6=12
        expected = jnp.array([9.0, 12.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_two_broadcast_dims(self):
        """Test vmap over two broadcast dimensions."""

        def sum_over_c(production):
            """Sum over C dimension."""
            return jnp.sum(production)

        # production has shape (C, Y, X) but function works on (C,)
        input_dims = {"production": ("C", "Y", "X")}
        core_dims = {"production": ["C"]}
        arg_order = ["production"]

        wrapped = wrap_with_vmap(sum_over_c, input_dims, core_dims, arg_order)

        # Input: (C=2, Y=2, X=2)
        production = jnp.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )
        result = wrapped(production)

        # Expected: sum over C for each (Y, X)
        # (0,0): 1+5=6, (0,1): 2+6=8, (1,0): 3+7=10, (1,1): 4+8=12
        expected = jnp.array([[6.0, 8.0], [10.0, 12.0]])

        np.testing.assert_array_almost_equal(result, expected)

    def test_with_scalar_input(self):
        """Test vmap with scalar input (no dimensions)."""

        def scale(production, factor):
            """Scale production by factor."""
            return production * factor

        input_dims = {"production": ("C", "Y"), "factor": ()}
        core_dims = {"production": ["C"]}
        arg_order = ["production", "factor"]

        wrapped = wrap_with_vmap(scale, input_dims, core_dims, arg_order)

        production = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # (C=2, Y=2)
        factor = 2.0

        result = wrapped(production, factor)
        # vmap places the mapped dimension (Y) at position 0
        # Result shape is (Y, C) = (2, 2)
        expected = jnp.array([[2.0, 6.0], [4.0, 8.0]])

        np.testing.assert_array_almost_equal(result, expected)

    def test_multiple_inputs_different_dims(self):
        """Test vmap with inputs having different dimensions."""

        def combine(production, temperature):
            """Combine production and temperature."""
            # production: (C,), temperature: scalar after vmap
            return production * temperature

        input_dims = {"production": ("C", "Y", "X"), "temperature": ("Y", "X")}
        core_dims = {"production": ["C"]}
        arg_order = ["production", "temperature"]

        wrapped = wrap_with_vmap(combine, input_dims, core_dims, arg_order)

        # production: (C=2, Y=2, X=2)
        production = jnp.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )
        # temperature: (Y=2, X=2)
        temperature = jnp.array([[10.0, 20.0], [30.0, 40.0]])

        result = wrapped(production, temperature)

        # vmap places broadcast dims (Y, X) at the front
        # Result shape is (Y, X, C)
        # For (y=0, x=0): production[:, 0, 0] * temperature[0, 0] = [1, 5] * 10 = [10, 50]
        # For (y=0, x=1): production[:, 0, 1] * temperature[0, 1] = [2, 6] * 20 = [40, 120]
        # For (y=1, x=0): production[:, 1, 0] * temperature[1, 0] = [3, 7] * 30 = [90, 210]
        # For (y=1, x=1): production[:, 1, 1] * temperature[1, 1] = [4, 8] * 40 = [160, 320]
        expected = jnp.array(
            [
                [[10.0, 50.0], [40.0, 120.0]],
                [[90.0, 210.0], [160.0, 320.0]],
            ]
        )

        np.testing.assert_array_almost_equal(result, expected)

    def test_multi_output(self):
        """Test vmap with function returning multiple outputs."""

        def split(production):
            """Split production into two parts."""
            return production[:2], production[2:]

        input_dims = {"production": ("C", "Y")}
        core_dims = {"production": ["C"]}
        arg_order = ["production"]

        wrapped = wrap_with_vmap(split, input_dims, core_dims, arg_order)

        # production: (C=4, Y=2)
        production = jnp.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ]
        )

        first, second = wrapped(production)

        # vmap places Y at position 0
        # first: (Y=2, C=2), second: (Y=2, C=2)
        # For y=0: split([1, 3, 5, 7]) = ([1, 3], [5, 7])
        # For y=1: split([2, 4, 6, 8]) = ([2, 4], [6, 8])
        expected_first = jnp.array([[1.0, 3.0], [2.0, 4.0]])
        expected_second = jnp.array([[5.0, 7.0], [6.0, 8.0]])

        np.testing.assert_array_almost_equal(first, expected_first)
        np.testing.assert_array_almost_equal(second, expected_second)
