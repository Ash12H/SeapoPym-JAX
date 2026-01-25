"""Tests for the function registry."""

import pytest

from seapopym.blueprint import (
    FunctionNotFoundError,
    clear_registry,
    functional,
    get_function,
    list_functions,
)


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear registry before and after each test."""
    clear_registry()
    yield
    clear_registry()


class TestFunctionalDecorator:
    """Tests for the @functional decorator."""

    def test_register_simple_function(self):
        """Test registering a simple function."""

        @functional(name="test:simple", backend="jax")
        def simple_func(x):
            return x * 2

        # Function should be in registry
        assert "test:simple" in list_functions("jax")

        # Should be callable
        assert simple_func(5) == 10

    def test_register_with_metadata(self):
        """Test registering with full metadata."""

        @functional(
            name="test:growth",
            backend="jax",
            core_dims={"biomass": ["C"]},
            out_dims=["C"],
            units={"biomass": "g", "rate": "1/d", "return": "g/d"},
        )
        def growth(biomass, rate):
            return biomass * rate

        metadata = get_function("test:growth", "jax")

        assert metadata.name == "test:growth"
        assert metadata.backend == "jax"
        assert metadata.core_dims == {"biomass": ["C"]}
        assert metadata.out_dims == ["C"]
        assert metadata.units == {"biomass": "g", "rate": "1/d", "return": "g/d"}

    def test_register_multi_output(self):
        """Test registering a multi-output function."""

        @functional(
            name="test:predation",
            backend="jax",
            outputs=["prey_loss", "predator_gain"],
        )
        def predation(prey, predator, rate):
            flux = prey * predator * rate
            return -flux, +flux

        metadata = get_function("test:predation", "jax")

        assert metadata.is_multi_output
        assert metadata.output_names == ["prey_loss", "predator_gain"]

    def test_invalid_name_format(self):
        """Test that invalid name format raises error."""
        with pytest.raises(ValueError, match="namespace:function_name"):

            @functional(name="invalid_name", backend="jax")
            def bad_func(x):
                return x

    def test_invalid_backend(self):
        """Test that invalid backend raises error."""
        with pytest.raises(ValueError, match="Unknown backend"):

            @functional(name="test:func", backend="invalid")
            def bad_func(x):
                return x


class TestGetFunction:
    """Tests for get_function."""

    def test_get_existing_function(self):
        """Test retrieving an existing function."""

        @functional(name="test:exists", backend="jax")
        def exists(x):
            return x

        metadata = get_function("test:exists", "jax")
        assert metadata.name == "test:exists"
        assert metadata.func(10) == 10

    def test_get_nonexistent_function(self):
        """Test that nonexistent function raises error."""
        with pytest.raises(FunctionNotFoundError) as exc_info:
            get_function("test:nonexistent", "jax")

        assert "test:nonexistent" in str(exc_info.value)
        assert exc_info.value.code == "E101"

    def test_get_wrong_backend(self):
        """Test that getting from wrong backend raises error."""

        @functional(name="test:jax_only", backend="jax")
        def jax_only(x):
            return x

        with pytest.raises(FunctionNotFoundError):
            get_function("test:jax_only", "numpy")


class TestListFunctions:
    """Tests for list_functions."""

    def test_list_empty(self):
        """Test listing empty registry."""
        assert list_functions("jax") == []

    def test_list_single_backend(self):
        """Test listing functions for single backend."""

        @functional(name="test:a", backend="jax")
        def a(x):
            return x

        @functional(name="test:b", backend="jax")
        def b(x):
            return x

        @functional(name="test:c", backend="numpy")
        def c(x):
            return x

        jax_funcs = list_functions("jax")
        assert "test:a" in jax_funcs
        assert "test:b" in jax_funcs
        assert "test:c" not in jax_funcs

    def test_list_all_backends(self):
        """Test listing all functions across backends."""

        @functional(name="test:jax_func", backend="jax")
        def jax_func(x):
            return x

        @functional(name="test:numpy_func", backend="numpy")
        def numpy_func(x):
            return x

        all_funcs = list_functions()
        assert "test:jax_func" in all_funcs
        assert "test:numpy_func" in all_funcs


class TestFunctionMetadata:
    """Tests for FunctionMetadata."""

    def test_get_signature(self):
        """Test getting function signature."""

        @functional(name="test:sig", backend="jax")
        def sig_func(a, b, c=10):
            return a + b + c

        metadata = get_function("test:sig", "jax")
        sig = metadata.get_signature()

        params = list(sig.parameters.keys())
        assert params == ["a", "b", "c"]

    def test_get_required_inputs(self):
        """Test getting required inputs (no defaults)."""

        @functional(name="test:req", backend="jax")
        def req_func(required1, required2, optional=5):
            return required1 + required2 + optional

        metadata = get_function("test:req", "jax")
        required = metadata.get_required_inputs()

        assert required == ["required1", "required2"]
        assert "optional" not in required
