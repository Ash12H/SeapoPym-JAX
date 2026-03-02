"""Tests for the function registry."""

import pytest

from seapopym.blueprint import (
    FunctionNotFoundError,
    functional,
    get_function,
    list_functions,
)
from seapopym.blueprint.registry import REGISTRY


@pytest.fixture(autouse=True)
def clean_registry():
    """Save registry, clear for test, restore after."""
    saved = dict(REGISTRY)
    REGISTRY.clear()
    yield
    REGISTRY.clear()
    REGISTRY.update(saved)


class TestFunctionalDecorator:
    """Tests for the @functional decorator."""

    def test_register_simple_function(self):
        """Test registering a simple function."""

        @functional(name="test:simple")
        def simple_func(x):
            return x * 2

        # Function should be in registry
        assert "test:simple" in list_functions()

        # Should be callable
        assert simple_func(5) == 10

    def test_register_with_metadata(self):
        """Test registering with full metadata."""

        @functional(
            name="test:growth",
            core_dims={"biomass": ["C"]},
            out_dims=["C"],
            units={"biomass": "g", "rate": "1/d", "return": "g/d"},
        )
        def growth(biomass, rate):
            return biomass * rate

        metadata = get_function("test:growth")

        assert metadata.name == "test:growth"
        assert metadata.core_dims == {"biomass": ["C"]}
        assert metadata.out_dims == ["C"]
        assert metadata.units == {"biomass": "g", "rate": "1/d", "return": "g/d"}

    def test_register_multi_output(self):
        """Test registering a multi-output function."""

        @functional(
            name="test:predation",
            outputs=["prey_loss", "predator_gain"],
        )
        def predation(prey, predator, rate):
            flux = prey * predator * rate
            return -flux, +flux

        metadata = get_function("test:predation")

        assert metadata.is_multi_output
        assert metadata.output_names == ["prey_loss", "predator_gain"]

    def test_invalid_name_format(self):
        """Test that invalid name format raises error."""
        with pytest.raises(ValueError, match="namespace:function_name"):

            @functional(name="invalid_name")
            def bad_func(x):
                return x

    def test_overwrite_warns(self):
        """Test that re-registering the same name emits a warning."""

        @functional(name="test:dup")
        def first(x):
            return x

        with pytest.warns(UserWarning, match="already registered"):

            @functional(name="test:dup")
            def second(x):
                return x * 2


class TestGetFunction:
    """Tests for get_function."""

    def test_get_existing_function(self):
        """Test retrieving an existing function."""

        @functional(name="test:exists")
        def exists(x):
            return x

        metadata = get_function("test:exists")
        assert metadata.name == "test:exists"
        assert metadata.func(10) == 10

    def test_get_nonexistent_function(self):
        """Test that nonexistent function raises error."""
        with pytest.raises(FunctionNotFoundError) as exc_info:
            get_function("test:nonexistent")

        assert "test:nonexistent" in str(exc_info.value)
        assert exc_info.value.code == "E101"


class TestListFunctions:
    """Tests for list_functions."""

    def test_list_empty(self):
        """Test listing empty registry."""
        assert list_functions() == []

    def test_list_functions(self):
        """Test listing registered functions."""

        @functional(name="test:a")
        def a(x):
            return x

        @functional(name="test:b")
        def b(x):
            return x

        funcs = list_functions()
        assert "test:a" in funcs
        assert "test:b" in funcs


class TestFunctionMetadata:
    """Tests for FunctionMetadata."""

    def test_get_signature(self):
        """Test getting function signature."""

        @functional(name="test:sig")
        def sig_func(a, b, c=10):
            return a + b + c

        metadata = get_function("test:sig")
        sig = metadata.get_signature()

        params = list(sig.parameters.keys())
        assert params == ["a", "b", "c"]

    def test_get_required_inputs(self):
        """Test getting required inputs (no defaults)."""

        @functional(name="test:req")
        def req_func(required1, required2, optional=5):
            return required1 + required2 + optional

        metadata = get_function("test:req")
        required = metadata.get_required_inputs()

        assert required == ["required1", "required2"]
        assert "optional" not in required

    def test_output_names_default(self):
        """Test output_names fallback when outputs is None."""

        @functional(name="test:single")
        def single_func(x):
            return x

        metadata = get_function("test:single")
        assert metadata.outputs is None
        assert metadata.output_names == ["return"]
