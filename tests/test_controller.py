from datetime import datetime, timedelta

import pytest
import xarray as xr

from seapopym.blueprint import Blueprint
from seapopym.controller import SimulationConfig, SimulationController
from seapopym.standard import Coordinates


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
        bp.register_group(
            "",
            state_variables={"biomass": {"dims": ("x",), "units": "dimensionless"}},
            units=[],
        )
        bp.register_unit(
            compute_tendency,
            input_mapping={"biomass": "biomass"},
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


def test_controller_with_dynamic_forcings():
    """Test que les forçages dynamiques sont bien mis à jour à chaque pas de temps."""

    # 1. Configuration du modèle : output = temperature (forcing)
    def copy_temp(temperature):
        return {"out_temp": temperature}

    def configure(bp: Blueprint):
        bp.register_forcing("temperature")
        bp.register_unit(copy_temp, output_mapping={"out_temp": "recorded_temp"})

    # 2. Configuration Simulation
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 4)  # 3 jours
    config = SimulationConfig(start_date=start, end_date=end, timestep=timedelta(days=1))

    # 3. Forçages dynamiques
    # J1: 10°, J2: 20°, J3: 30°, J4: 40°
    times = [start + timedelta(days=i) for i in range(4)]
    temps = [10.0, 20.0, 30.0, 40.0]

    forcings = xr.Dataset(
        {"temperature": ((Coordinates.T, "x"), [[t] for t in temps])},
        coords={Coordinates.T: times, "x": [0]},
    )

    # 4. État initial (VIDE pour temperature, car fournie par forcing)
    # On met juste une variable dummy pour définir la grille spatiale si besoin,
    # mais ici x est déjà dans forcings.
    # Note: Le blueprint n'exige pas d'autres variables ici.
    initial_state = xr.Dataset(coords={"x": [0]})

    # 5. Run
    controller = SimulationController(config)
    controller.setup(configure, initial_state, forcings=forcings)

    # On exécute pas à pas pour vérifier

    # Step 1 (2020-01-01) -> Temp doit être 10.0
    controller.step()
    assert controller.state["temperature"].item() == 10.0
    assert controller.state["recorded_temp"].item() == 10.0

    controller._current_time += config.timestep

    # Step 2 (2020-01-02) -> Temp doit être 20.0
    controller.step()
    assert controller.state["temperature"].item() == 20.0
    assert controller.state["recorded_temp"].item() == 20.0


def test_controller_forcing_conflict():
    """Test que le controller lève une erreur si une variable est définie deux fois."""

    def configure(bp: Blueprint):
        bp.register_forcing("temperature")

    start = datetime(2020, 1, 1)
    config = SimulationConfig(
        start_date=start, end_date=start + timedelta(days=1), timestep=timedelta(days=1)
    )

    forcings = xr.Dataset(
        {"temperature": ((Coordinates.T, "x"), [[10.0]])}, coords={Coordinates.T: [start], "x": [0]}
    )

    initial_state = xr.Dataset({"temperature": (("x"), [0.0])}, coords={"x": [0]})

    controller = SimulationController(config)
    with pytest.raises(ValueError, match="Ambiguous definition"):
        controller.setup(configure, initial_state, forcings=forcings)


def test_controller_separate_state_and_forcings():
    """Test avec des variables distinctes dans state et forcings."""

    def configure(bp: Blueprint):
        bp.register_forcing("temperature")
        bp.register_forcing("biomass")  # Déclaré comme input externe (état initial)
        # Une unité qui utilise les deux
        # Note: output_tendencies n'est pas utilisé ici, on modifie directement (ce qui n'est pas standard mais ok pour ce test de structure)
        # Pour être propre, on devrait retourner une tendance, mais ici on teste juste l'accès aux variables.
        # On va faire simple : une unité qui produit une nouvelle variable dérivée "growth"
        bp.register_unit(
            lambda temperature, biomass: {"growth": biomass * temperature},
            output_mapping={"growth": "growth"},
        )

    start = datetime(2020, 1, 1)
    config = SimulationConfig(
        start_date=start, end_date=start + timedelta(days=1), timestep=timedelta(days=1)
    )

    # initial_state a biomass
    initial_state = xr.Dataset({"biomass": (("x"), [1.0])}, coords={"x": [0]})

    # forcings a temperature (constante sur 2 jours pour permettre l'interpolation)
    forcings = xr.Dataset(
        {"temperature": ((Coordinates.T, "x"), [[10.0], [10.0]])},
        coords={Coordinates.T: [start, start + timedelta(days=1)], "x": [0]},
    )

    # Devrait fonctionner sans erreur
    controller = SimulationController(config)
    controller.setup(configure, initial_state, forcings=forcings)

    # Vérification
    controller.step()

    # Après step, temperature doit être dans state (merge)
    assert "temperature" in controller.state
    assert controller.state["temperature"].item() == 10.0
    assert controller.state["growth"].item() == 10.0  # 1.0 * 10.0
