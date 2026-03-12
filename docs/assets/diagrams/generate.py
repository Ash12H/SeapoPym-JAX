#!/usr/bin/env python
"""Generate SVG DAG diagrams for documentation.

Usage:
    uv run python docs/assets/diagrams/generate.py

Requires: graphviz system binary + seapopym[viz]
"""

from pathlib import Path

from seapopym.blueprint import Blueprint
from seapopym.models import LMTL_NO_TRANSPORT

OUTPUT_DIR = Path(__file__).parent

# --- Example 1: Growth + Mortality ---
growth = Blueprint.from_dict(
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

# --- Example 2: Predator-Prey (Lotka-Volterra) ---
# dN/dt = αN - βNP
# dP/dt = δβNP - γP
predator_prey = Blueprint.from_dict(
    {
        "id": "predator-prey",
        "version": "1.0",
        "declarations": {
            "state": {
                "prey": {"units": "g/m^2", "dims": ["Y", "X"]},
                "predator": {"units": "g/m^2", "dims": ["Y", "X"]},
            },
            "parameters": {
                "alpha": {"units": "1/s", "description": "Prey growth rate"},
                "beta": {"units": "m^2/g/s", "description": "Predation rate"},
                "delta": {"units": "dimensionless", "description": "Conversion efficiency"},
                "gamma": {"units": "1/s", "description": "Predator death rate"},
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

# --- Generate all ---
diagrams = {
    "dag_growth": growth,
    "dag_predator_prey": predator_prey,
    "dag_lmtl_no_transport": LMTL_NO_TRANSPORT,
}

for name, blueprint in diagrams.items():
    path = OUTPUT_DIR / name
    blueprint.to_graphviz().render(str(path), cleanup=True)
    print(f"Generated {path}.svg")
