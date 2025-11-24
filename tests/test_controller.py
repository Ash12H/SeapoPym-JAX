from datetime import datetime, timedelta

import pytest
import xarray as xr

from seapopym.blueprint import Blueprint
from seapopym.controller import SimulationConfig, SimulationController


# --- Mocks ---
def grow_biomass(biomass):
    # Croissance simple : +1 par pas de temps
    # On produit 'next_biomass' pour éviter le cycle immédiat dans le Blueprint
    return {"next_biomass": biomass + 1}


def configure_model(bp: Blueprint):
    bp.register_forcing("biomass")
    bp.register_unit(grow_biomass, output_mapping={"next_biomass": "next_biomass"})


# --- Tests ---


def test_controller_lifecycle():
    """Test le cycle complet : Init -> Setup -> Run."""

    # 1. Config
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 4)  # 3 jours
    config = SimulationConfig(start_date=start, end_date=end, timestep=timedelta(days=1))

    controller = SimulationController(config)

    # 2. Setup
    initial_state = xr.Dataset({"biomass": (("x"), [10.0])}, coords={"x": [1]})

    controller.setup(configure_model, initial_state)

    assert controller.state is not None
    assert len(controller.groups) == 1
    assert "Global" in controller.groups

    # 3. Run
    controller.run()

    # Vérification finale
    # Comme on n'a pas de TimeIntegrator pour boucler next_biomass -> biomass,
    # biomass reste à 10.0, et next_biomass est calculé à 11.0 à chaque pas.
    # Le but ici est de vérifier que le run va au bout sans erreur.
    assert controller.state["biomass"].item() == 10.0
    assert controller.state["next_biomass"].item() == 11.0


def test_controller_setup_validation():
    """Vérifie que le setup valide l'état initial."""
    config = SimulationConfig(datetime(2020, 1, 1), datetime(2020, 1, 2))
    controller = SimulationController(config)

    # État vide, alors que le modèle attend 'biomass'
    empty_state = xr.Dataset({})

    from seapopym.gsm.exceptions import StateValidationError

    with pytest.raises(StateValidationError):
        controller.setup(configure_model, empty_state)


def test_controller_not_setup():
    """Vérifie qu'on ne peut pas run sans setup."""
    config = SimulationConfig(datetime(2020, 1, 1), datetime(2020, 1, 2))
    controller = SimulationController(config)

    with pytest.raises(RuntimeError):
        controller.run()


def test_invalid_config_dates():
    """Test que end_date <= start_date lève une erreur."""
    with pytest.raises(ValueError, match="end_date must be after start_date"):
        SimulationConfig(
            start_date=datetime(2020, 1, 2),
            end_date=datetime(2020, 1, 1),  # Avant start_date
        )


def test_invalid_config_timestep():
    """Test qu'un timestep négatif lève une erreur."""
    with pytest.raises(ValueError, match="timestep must be positive"):
        SimulationConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 1, 2),
            timestep=timedelta(days=-1),
        )


def test_controller_with_tendencies():
    """Test end-to-end avec TimeIntegrator et tendances."""

    def compute_tendency(biomass):
        # Croissance : +0.001 par seconde (constante)
        # Sur 1 jour (86400s) : +86.4
        return {"growth": xr.DataArray([0.001], dims=["x"])}

    def configure_model(bp: Blueprint):
        bp.register_forcing("biomass")
        bp.register_unit(
            compute_tendency,
            output_mapping={"growth": "growth_rate"},
            output_tendencies={"growth": "biomass"},  # Tendance de biomass
        )

    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 3)  # 2 jours
    config = SimulationConfig(start_date=start, end_date=end, timestep=timedelta(days=1))

    controller = SimulationController(config)

    # État initial : biomass = 100
    initial_state = xr.Dataset({"biomass": (("x"), [100.0])}, coords={"x": [1]})

    controller.setup(configure_model, initial_state)
    controller.run()

    # Calcul attendu :
    # dt = 86400 secondes, tendency = 0.001 /s
    # Jour 1 : 100 + 86400 * 0.001 = 186.4
    # Jour 2 : 186.4 + 86400 * 0.001 = 272.8
    final_biomass = controller.state["biomass"].item()
    assert final_biomass == pytest.approx(272.8, rel=0.01)
