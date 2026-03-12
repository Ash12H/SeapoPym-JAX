"""Tests for Blueprint.to_graphviz() DAG visualization."""

import pytest

from seapopym.blueprint import Blueprint


@pytest.fixture()
def growth_blueprint() -> Blueprint:
    """Simple growth + mortality blueprint."""
    return Blueprint.from_dict(
        {
            "id": "growth-mortality",
            "version": "1.0",
            "declarations": {
                "state": {"biomass": {"units": "g/m^2", "dims": ["Y", "X"]}},
                "parameters": {
                    "growth_rate": {"units": "1/s"},
                    "thermal_sensitivity": {"units": "1/delta_degC"},
                },
                "forcings": {"temperature": {"units": "degC", "dims": ["T", "Y", "X"]}},
            },
            "process": [
                {
                    "func": "eco:compute_growth",
                    "inputs": {"biomass": "state.biomass", "rate": "parameters.growth_rate"},
                    "outputs": {"return": "derived.growth_flux"},
                },
                {
                    "func": "eco:compute_mortality",
                    "inputs": {
                        "biomass": "state.biomass",
                        "temp": "forcings.temperature",
                        "gamma": "parameters.thermal_sensitivity",
                    },
                    "outputs": {"return": "derived.mortality_flux"},
                },
            ],
            "tendencies": {
                "biomass": [
                    {"source": "derived.growth_flux"},
                    {"source": "derived.mortality_flux"},
                ],
            },
        }
    )


@pytest.fixture()
def predator_prey_blueprint() -> Blueprint:
    """Predator-prey (Lotka-Volterra) blueprint with two state variables."""
    return Blueprint.from_dict(
        {
            "id": "predator-prey",
            "version": "1.0",
            "declarations": {
                "state": {
                    "prey": {"units": "g/m^2", "dims": ["Y", "X"]},
                    "predator": {"units": "g/m^2", "dims": ["Y", "X"]},
                },
                "parameters": {
                    "alpha": {"units": "1/s"},
                    "beta": {"units": "m^2/g/s"},
                    "delta": {"units": "dimensionless"},
                    "gamma": {"units": "1/s"},
                },
                "forcings": {},
            },
            "process": [
                {
                    "func": "eco:prey_growth",
                    "inputs": {"N": "state.prey", "alpha": "parameters.alpha"},
                    "outputs": {"return": "derived.prey_growth"},
                },
                {
                    "func": "eco:predation",
                    "inputs": {
                        "N": "state.prey",
                        "P": "state.predator",
                        "beta": "parameters.beta",
                        "delta": "parameters.delta",
                    },
                    "outputs": {"prey_loss": "derived.prey_loss", "predator_gain": "derived.predator_gain"},
                },
                {
                    "func": "eco:predator_death",
                    "inputs": {"P": "state.predator", "gamma": "parameters.gamma"},
                    "outputs": {"return": "derived.predator_death"},
                },
            ],
            "tendencies": {
                "prey": [
                    {"source": "derived.prey_growth"},
                    {"source": "derived.prey_loss"},
                ],
                "predator": [
                    {"source": "derived.predator_gain"},
                    {"source": "derived.predator_death"},
                ],
            },
        }
    )


class TestToGraphviz:
    """Tests for Blueprint.to_graphviz()."""

    def test_returns_digraph(self, growth_blueprint: Blueprint):
        """to_graphviz returns a graphviz.Digraph object."""
        import graphviz

        g = growth_blueprint.to_graphviz()
        assert isinstance(g, graphviz.Digraph)

    def test_svg_output(self, growth_blueprint: Blueprint):
        """The Digraph can render to SVG."""
        g = growth_blueprint.to_graphviz()
        svg = g.pipe(format="svg").decode()
        assert "<svg" in svg

    def test_contains_state_nodes(self, growth_blueprint: Blueprint):
        """State variables appear as nodes."""
        source = growth_blueprint.to_graphviz().source
        assert "biomass" in source

    def test_contains_function_nodes(self, growth_blueprint: Blueprint):
        """Process functions appear as nodes."""
        source = growth_blueprint.to_graphviz().source
        assert "compute_growth" in source
        assert "compute_mortality" in source

    def test_contains_parameter_nodes(self, growth_blueprint: Blueprint):
        """Parameters appear as nodes."""
        source = growth_blueprint.to_graphviz().source
        assert "growth_rate" in source
        assert "thermal_sensitivity" in source

    def test_contains_forcing_nodes(self, growth_blueprint: Blueprint):
        """Forcings appear as nodes."""
        source = growth_blueprint.to_graphviz().source
        assert "temperature" in source

    def test_contains_tendency_nodes(self, growth_blueprint: Blueprint):
        """Tendency accumulators appear as nodes."""
        source = growth_blueprint.to_graphviz().source
        assert "d biomass / dt" in source

    def test_no_tendencies(self, growth_blueprint: Blueprint):
        """show_tendencies=False hides tendency nodes."""
        source = growth_blueprint.to_graphviz(show_tendencies=False).source
        assert "tendency" not in source
        assert "d biomass / dt" not in source

    def test_direction_lr(self, growth_blueprint: Blueprint):
        """direction='LR' sets left-to-right layout."""
        source = growth_blueprint.to_graphviz(direction="LR").source
        assert "LR" in source

    def test_predator_prey_two_tendencies(self, predator_prey_blueprint: Blueprint):
        """Predator-prey model has two tendency nodes."""
        source = predator_prey_blueprint.to_graphviz().source
        assert "d prey / dt" in source
        assert "d predator / dt" in source

    def test_predator_prey_multi_output(self, predator_prey_blueprint: Blueprint):
        """Multi-output function (predation) creates edges to both derived variables."""
        source = predator_prey_blueprint.to_graphviz().source
        assert "prey_loss" in source
        assert "predator_gain" in source

    def test_duplicate_functions_unique_nodes(self):
        """Same function used twice creates two distinct nodes with same label."""
        bp = Blueprint.from_dict(
            {
                "id": "duplicate-func",
                "version": "1.0",
                "declarations": {
                    "state": {},
                    "parameters": {"a": {"units": "degC"}, "b": {"units": "degC"}},
                    "forcings": {},
                },
                "process": [
                    {
                        "func": "math:normalize",
                        "inputs": {"x": "parameters.a"},
                        "outputs": {"return": "derived.a_norm"},
                    },
                    {
                        "func": "math:normalize",
                        "inputs": {"x": "parameters.b"},
                        "outputs": {"return": "derived.b_norm"},
                    },
                ],
                "tendencies": {},
            }
        )
        source = bp.to_graphviz().source
        # Both nodes have the same label but different IDs
        assert source.count("label=normalize") == 2
        assert "process_0" in source
        assert "process_1" in source

    def test_lmtl_no_transport(self):
        """LMTL no-transport blueprint generates a valid graph."""
        from seapopym.models import LMTL_NO_TRANSPORT

        g = LMTL_NO_TRANSPORT.to_graphviz()
        svg = g.pipe(format="svg").decode()
        assert "<svg" in svg
        assert "mortality" in g.source
        assert "recruitment_flow" in g.source

    def test_lmtl_transport(self):
        """LMTL transport blueprint generates a valid graph."""
        from seapopym.models import LMTL

        g = LMTL.to_graphviz()
        svg = g.pipe(format="svg").decode()
        assert "<svg" in svg
        assert "transport_tendency" in g.source
