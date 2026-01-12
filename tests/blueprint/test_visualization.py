"""Tests for Blueprint visualization."""

import pytest

from seapopym.blueprint import Blueprint
from seapopym.blueprint.exceptions import ConfigurationError


def test_visualize_empty():
    bp = Blueprint()
    with pytest.raises(ConfigurationError):
        bp.visualize()


def test_visualize_simple(tmp_path):
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed")

    bp = Blueprint()
    bp.register_forcing("temp")

    def my_func(temp):
        return {"out": temp * 2}

    bp.register_unit(my_func, output_mapping={"out": "result"}, input_mapping={"temp": "temp"})

    # Smoke test for different layouts
    fig = bp.visualize(layout="hierarchical")
    assert fig is not None
    plt.close(fig)

    fig = bp.visualize(layout="spring")
    assert fig is not None
    plt.close(fig)

    fig = bp.visualize(layout="circular")
    assert fig is not None
    plt.close(fig)


def test_export_mermaid():
    bp = Blueprint()
    bp.register_forcing("temp")

    def my_func(temp):
        return {"out": temp * 2}

    bp.register_unit(my_func, output_mapping={"out": "result"}, input_mapping={"temp": "temp"})

    mm = bp.export_mermaid()
    assert "graph TD" in mm
    assert "temp" in mm
    assert "result" in mm
