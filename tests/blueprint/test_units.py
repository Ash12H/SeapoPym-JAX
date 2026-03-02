"""Tests for unit validation using Pint."""

from __future__ import annotations

import pytest

from seapopym.blueprint import Blueprint
from seapopym.blueprint.exceptions import UnitError
from seapopym.blueprint.registry import REGISTRY, functional
from seapopym.blueprint.units import UnitValidator


@pytest.fixture(autouse=True)
def clean_registry():
    """Save registry, clear for test, restore after."""
    saved = dict(REGISTRY)
    REGISTRY.clear()
    yield
    REGISTRY.clear()
    REGISTRY.update(saved)


# Sample functions for testing
def sample_growth(biomass: float, rate: float) -> float:
    """Growth function."""
    return biomass * rate


def sample_transport(biomass: float, velocity: float) -> float:
    """Transport function."""
    return biomass * velocity


class TestUnitValidator:
    """Test suite for UnitValidator."""

    def test_parse_unit_valid(self):
        """Test parsing valid units."""
        validator = UnitValidator()

        # Test various unit strings
        u1 = validator.parse_unit("m/s")
        u2 = validator.parse_unit("meter/second")
        u3 = validator.parse_unit("m * s**-1")

        # All should be equivalent canonical forms
        assert u1 == u2
        assert u2 == u3

    def test_parse_unit_dimensionless(self):
        """Test parsing dimensionless units."""
        validator = UnitValidator()

        u1 = validator.parse_unit("dimensionless")
        u2 = validator.parse_unit(None)

        assert u1 == u2
        assert str(u1) == "dimensionless"

    def test_parse_unit_invalid(self):
        """Test parsing invalid units."""
        validator = UnitValidator()

        with pytest.raises(UnitError, match="Invalid unit string"):
            validator.parse_unit("invalid_unit_xyz")

    def test_check_exact_match_success(self):
        """Test exact match with compatible canonical forms."""
        validator = UnitValidator()

        # These should all pass (same canonical form)
        validator.check_exact_match("m/s", "meter/second", context="test")
        validator.check_exact_match("m * s**-1", "m/s", context="test")
        validator.check_exact_match("1/s", "1/second", context="test")
        validator.check_exact_match("g", "gram", context="test")
        validator.check_exact_match(None, "dimensionless", context="test")

    def test_check_exact_match_failure_different_units(self):
        """Test exact match failure with different units."""
        validator = UnitValidator()

        # These should fail (different canonical forms)
        with pytest.raises(UnitError, match="Unit mismatch"):
            validator.check_exact_match("m/s", "km/s", context="test")

        with pytest.raises(UnitError, match="Unit mismatch"):
            validator.check_exact_match("1/d", "1/s", context="test")

        with pytest.raises(UnitError, match="Unit mismatch"):
            validator.check_exact_match("g", "kg", context="test")

    def test_check_exact_match_context_in_error(self):
        """Test that context appears in error message."""
        validator = UnitValidator()

        with pytest.raises(UnitError, match="in test_context"):
            validator.check_exact_match("m/s", "km/s", context="test_context")

    def test_validate_process_chain_success(self):
        """Test successful validation of a process chain."""
        functional(
            name="test:growth",
            units={"biomass": "g", "rate": "1/s", "return": "g/s"},
        )(sample_growth)

        # Create blueprint
        blueprint = Blueprint.from_dict(
            {
                "id": "test-model",
                "version": "1.0.0",
                "declarations": {
                    "state": {"biomass": {"units": "g"}},
                    "parameters": {"growth_rate": {"units": "1/s"}},
                },
                "process": [
                    {
                        "func": "test:growth",
                        "inputs": {
                            "biomass": "state.biomass",
                            "rate": "parameters.growth_rate",
                        },
                        "outputs": {"tendency": "derived.growth_flux"},
                    }
                ],
                "tendencies": {"biomass": [{"source": "derived.growth_flux"}]},
            }
        )

        # Resolve functions
        from seapopym.blueprint.registry import get_function

        resolved = {"test:growth": get_function("test:growth")}

        # Validate
        validator = UnitValidator()
        errors = validator.validate_process_chain(blueprint, resolved)

        assert len(errors) == 0

    def test_validate_process_chain_input_mismatch(self):
        """Test validation failure with input unit mismatch."""
        functional(
            name="test:transport",
            units={"biomass": "g", "velocity": "m/s", "return": "g*m/s"},
        )(sample_transport)

        # Create blueprint declaring velocity in km/s (incompatible)
        blueprint = Blueprint.from_dict(
            {
                "id": "test-model",
                "version": "1.0.0",
                "declarations": {
                    "state": {"biomass": {"units": "g"}},
                    "parameters": {"vel": {"units": "km/s"}},  # Wrong unit!
                },
                "process": [
                    {
                        "func": "test:transport",
                        "inputs": {
                            "biomass": "state.biomass",
                            "velocity": "parameters.vel",
                        },
                        "outputs": {"result": "derived.flux"},
                    }
                ],
            }
        )

        # Resolve functions
        from seapopym.blueprint.registry import get_function

        resolved = {"test:transport": get_function("test:transport")}

        # Validate
        validator = UnitValidator()
        errors = validator.validate_process_chain(blueprint, resolved)

        assert len(errors) == 1
        assert "km/s" in str(errors[0])
        assert "m/s" in str(errors[0])

    def test_validate_process_chain_tendency_must_have_time_dimension(self):
        """Test that tendencies must have /s time dimension."""
        functional(
            name="test:bad_growth",
            units={"biomass": "g", "rate": "dimensionless", "return": "g"},  # Missing /s!
        )(sample_growth)

        # Create blueprint
        blueprint = Blueprint.from_dict(
            {
                "id": "test-model",
                "version": "1.0.0",
                "declarations": {
                    "state": {"biomass": {"units": "g"}},
                    "parameters": {"rate": {"units": "dimensionless"}},
                },
                "process": [
                    {
                        "func": "test:bad_growth",
                        "inputs": {
                            "biomass": "state.biomass",
                            "rate": "parameters.rate",
                        },
                        "outputs": {"tendency": "derived.bad_growth_flux"},
                    }
                ],
                "tendencies": {"biomass": [{"source": "derived.bad_growth_flux"}]},
            }
        )

        # Resolve functions
        from seapopym.blueprint.registry import get_function

        resolved = {"test:bad_growth": get_function("test:bad_growth")}

        # Validate
        validator = UnitValidator()
        errors = validator.validate_process_chain(blueprint, resolved)

        # Should have error about missing time dimension
        assert len(errors) == 1
        assert "time dimension" in str(errors[0]).lower()
        assert "tendency" in str(errors[0]).lower()

    def test_validate_units_convenience_function(self):
        """Test the convenience function validate_units()."""
        from seapopym.blueprint.units import validate_units

        functional(
            name="test:conv_growth",
            units={"biomass": "g", "rate": "1/s", "return": "g/s"},
        )(sample_growth)

        # Create blueprint
        blueprint = Blueprint.from_dict(
            {
                "id": "test-model",
                "version": "1.0.0",
                "declarations": {
                    "state": {"biomass": {"units": "g"}},
                    "parameters": {"growth_rate": {"units": "1/s"}},
                },
                "process": [
                    {
                        "func": "test:conv_growth",
                        "inputs": {
                            "biomass": "state.biomass",
                            "rate": "parameters.growth_rate",
                        },
                        "outputs": {"tendency": "derived.conv_growth_flux"},
                    }
                ],
                "tendencies": {"biomass": [{"source": "derived.conv_growth_flux"}]},
            }
        )

        # Resolve functions
        from seapopym.blueprint.registry import get_function

        resolved = {"test:conv_growth": get_function("test:conv_growth")}

        # Validate
        errors = validate_units(blueprint, resolved)
        assert len(errors) == 0
