"""Tests for the models catalogue."""

import pytest

from seapopym.blueprint import list_functions
from seapopym.blueprint.schema import Blueprint
from seapopym.models import LMTL, LMTL_NO_TRANSPORT, load_model

# -- Transport-related forcing names in the full LMTL model --
_TRANSPORT_FORCINGS = {"u", "v", "D", "dx", "dy", "face_height", "face_width", "cell_area", "mask"}


class TestLoadPrebuiltModels:
    """Tests for pre-built model constants."""

    def test_load_lmtl(self):
        """LMTL loads correctly with expected id and step count."""
        assert isinstance(LMTL, Blueprint)
        assert LMTL.id == "seapodym-lmtl"
        assert len(LMTL.process) == 14

    def test_load_lmtl_no_transport(self):
        """LMTL_NO_TRANSPORT loads correctly with expected id and step count."""
        assert isinstance(LMTL_NO_TRANSPORT, Blueprint)
        assert LMTL_NO_TRANSPORT.id == "seapodym-lmtl-no-transport"
        assert len(LMTL_NO_TRANSPORT.process) == 10


class TestLoadModelFunction:
    """Tests for the load_model helper."""

    def test_load_model_returns_blueprint(self):
        """load_model returns a valid Blueprint for a known model."""
        bp = load_model("seapodym_lmtl")
        assert isinstance(bp, Blueprint)
        assert bp.id == "seapodym-lmtl"

    def test_load_model_not_found(self):
        """load_model raises an error for an unknown model name."""
        with pytest.raises(FileNotFoundError):
            load_model("inexistant")


class TestModelDeclarations:
    """Tests for model variable declarations."""

    @pytest.mark.parametrize("model", [LMTL, LMTL_NO_TRANSPORT], ids=["lmtl", "lmtl_no_transport"])
    def test_declares_state_variables(self, model: Blueprint):
        """Both models declare biomass and production state variables."""
        assert "biomass" in model.declarations.state
        assert "production" in model.declarations.state

    def test_lmtl_no_transport_has_no_transport_forcings(self):
        """The no-transport model does not declare transport-related forcings."""
        declared = set(LMTL_NO_TRANSPORT.declarations.forcings)
        assert declared.isdisjoint(_TRANSPORT_FORCINGS), (
            f"Unexpected transport forcings: {declared & _TRANSPORT_FORCINGS}"
        )


class TestFunctionRegistry:
    """Tests that all functions referenced in models are registered."""

    @pytest.mark.parametrize("model", [LMTL, LMTL_NO_TRANSPORT], ids=["lmtl", "lmtl_no_transport"])
    def test_all_functions_registered(self, model: Blueprint):
        """Every func referenced in process steps exists in the registry."""
        registered = set(list_functions())
        for step in model.process:
            assert step.func in registered, f"Function '{step.func}' not found in registry"
