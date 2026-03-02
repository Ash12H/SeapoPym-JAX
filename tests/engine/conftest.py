"""Shared fixtures for engine tests."""

import numpy as np
import pytest

from seapopym.blueprint import Blueprint, Config
from seapopym.blueprint.registry import REGISTRY


@pytest.fixture(autouse=True)
def clean_registry():
    """Save registry, clear for test, restore after."""
    saved = dict(REGISTRY)
    REGISTRY.clear()
    yield
    REGISTRY.clear()
    REGISTRY.update(saved)


@pytest.fixture
def simple_blueprint():
    """Create a simple blueprint for testing."""
    return Blueprint.from_dict(
        {
            "id": "test-model",
            "version": "0.1.0",
            "declarations": {
                "state": {
                    "biomass": {
                        "units": "g",
                        "dims": ["Y", "X"],
                    }
                },
                "parameters": {
                    "growth_rate": {
                        "units": "1/d",
                    }
                },
                "forcings": {
                    "temperature": {
                        "units": "degC",
                        "dims": ["T", "Y", "X"],
                    },
                    "mask": {
                        "dims": ["Y", "X"],
                    },
                },
            },
            "process": [
                {
                    "func": "test:growth",
                    "inputs": {
                        "biomass": "state.biomass",
                        "rate": "parameters.growth_rate",
                        "temp": "forcings.temperature",
                    },
                    "outputs": {
                        "tendency": "derived.growth_flux",
                    },
                }
            ],
            "tendencies": {
                "biomass": [{"source": "derived.growth_flux"}],
            },
        }
    )


@pytest.fixture
def simple_config():
    """Create a simple config for testing (10 timesteps, 5x5 grid)."""
    return Config.from_dict(
        {
            "parameters": {
                "growth_rate": {"value": 0.1},
            },
            "forcings": {
                "temperature": np.ones((10, 5, 5)) * 20.0,
                "mask": np.ones((5, 5)),
            },
            "initial_state": {
                "biomass": np.ones((5, 5)) * 100.0,
            },
            "execution": {
                "dt": "1d",
                "time_start": "2000-01-01",
                "time_end": "2000-01-11",
            },
        }
    )
