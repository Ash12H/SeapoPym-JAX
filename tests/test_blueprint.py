import pytest

from seapopym.blueprint.core import Blueprint
from seapopym.blueprint.exceptions import ConfigurationError, MissingInputError

# --- Fonctions Mock pour les tests ---


def compute_growth(temperature, nutrient):
    return {"biomass": temperature * nutrient}


def compute_mortality(biomass, temperature):
    return {"mortality": biomass * temperature}


def compute_advection(biomass, current):
    return {"flux": biomass * current}


def compute_with_params(biomass, param_a=1.0):
    return {"result": biomass * param_a}


# --- Tests ---


def test_simple_chain():
    """Test une chaîne simple : Forçage -> Unit A -> Unit B"""
    bp = Blueprint()

    # 1. Forçages
    bp.register_forcing("temperature")
    bp.register_forcing("nutrient")

    # 2. Unit A : Growth (produit 'biomass')
    bp.register_unit(compute_growth, output_mapping={"biomass": "biomass"})

    # 3. Unit B : Mortality (consomme 'biomass' et 'temperature')
    bp.register_unit(compute_mortality, output_mapping={"mortality": "mortality"})

    plan = bp.build()

    assert len(plan.task_sequence) == 2
    assert plan.task_sequence[0].name == "compute_growth"
    assert plan.task_sequence[1].name == "compute_mortality"
    assert "temperature" in plan.initial_variables
    assert "biomass" in plan.produced_variables
    assert "mortality" in plan.produced_variables


def test_explicit_mapping():
    """Test le renommage d'entrée via mapping"""
    bp = Blueprint()
    bp.register_forcing("sea_surface_temp")  # Nom différent de l'arg 'temperature'
    bp.register_forcing("nutrient")

    # On mappe 'temperature' -> 'sea_surface_temp'
    bp.register_unit(
        compute_growth,
        input_mapping={"temperature": "sea_surface_temp"},
        output_mapping={"biomass": "biomass"},
    )

    plan = bp.build()
    unit = plan.task_sequence[0]
    assert unit.input_mapping["temperature"] == "sea_surface_temp"


def test_missing_input():
    """Test qu'une erreur est levée si une entrée manque"""
    bp = Blueprint()
    bp.register_forcing("nutrient")
    # Manque 'temperature'

    with pytest.raises(MissingInputError):
        bp.register_unit(compute_growth, output_mapping={"biomass": "biomass"})


def test_cycle_detection():
    """Test la détection de cycle dans le graphe"""
    from seapopym.blueprint.exceptions import CycleError

    bp = Blueprint()
    bp.register_forcing("temperature")
    bp.register_forcing("biomass")

    # Premier : biomass + temperature -> mortality
    bp.register_unit(compute_mortality, output_mapping={"mortality": "mortality"})

    # Deuxième : mortality + temperature -> biomass (réutilise le nom "biomass")
    # Cela crée un cycle : biomass -> mortality -> biomass
    bp.register_unit(
        compute_mortality,
        input_mapping={"biomass": "mortality"},
        output_mapping={"result": "biomass"},  # Réutilise "biomass" comme sortie
    )

    # La détection de cycle doit se produire lors du build
    with pytest.raises(CycleError):
        bp.build()


def test_group_namespacing():
    """Test la résolution automatique dans un groupe"""
    bp = Blueprint()
    bp.register_forcing("temperature")
    bp.register_forcing("Tuna_nutrient")  # Spécifique au groupe

    # Groupe Tuna
    units = [
        {
            "func": compute_growth,
            "output_mapping": {"biomass": "biomass"},  # Deviendra Tuna_biomass
        },
        {
            "func": compute_mortality,
            # prend (biomass, temperature)
            # biomass -> Tuna_biomass (produit juste avant)
            "output_mapping": {"mortality": "mortality"},
        },
    ]

    bp.register_group("Tuna", units)

    plan = bp.build()

    # Vérifications
    growth_task = plan.task_sequence[0]
    mortality_task = plan.task_sequence[1]

    assert growth_task.name == "Tuna_compute_growth"
    assert growth_task.output_mapping["biomass"] == "Tuna_biomass"
    assert growth_task.input_mapping["nutrient"] == "Tuna_nutrient"
    assert growth_task.input_mapping["temperature"] == "temperature"  # Fallback global

    assert mortality_task.input_mapping["biomass"] == "Tuna_biomass"  # Résolu en interne du groupe


def test_default_params_ignored():
    """Test que les paramètres avec valeur par défaut sont ignorés lors de la résolution"""
    bp = Blueprint()
    bp.register_forcing("biomass")

    # compute_with_params(biomass, param_a=1.0)
    # param_a doit être ignoré, seul biomass est requis
    bp.register_unit(compute_with_params, output_mapping={"result": "result"})

    plan = bp.build()
    assert len(plan.task_sequence) == 1
    assert plan.task_sequence[0].name == "compute_with_params"


def test_inter_group_dependency():
    """
    Test complexe avec 2 groupes :
    - Tuna : Dépend de la température (Global). Produit Tuna_biomass.
    - Shark : Dépend de Tuna_biomass (Inter-groupe). Produit Shark_biomass.
    """
    bp = Blueprint()
    bp.register_forcing("temperature")

    # 1. Groupe Tuna
    def grow_tuna(temperature):
        return {"biomass": temperature}

    bp.register_group(
        "Tuna",
        [
            {
                "func": grow_tuna,
                "output_mapping": {"biomass": "biomass"},  # -> Tuna_biomass
            }
        ],
    )

    # 2. Groupe Shark
    def grow_shark(food):
        return {"biomass": food}

    bp.register_group(
        "Shark",
        [
            {
                "func": grow_shark,
                "input_mapping": {"food": "Tuna_biomass"},  # Mapping explicite vers l'autre groupe
                "output_mapping": {"biomass": "biomass"},  # -> Shark_biomass
            }
        ],
    )

    plan = bp.build()

    assert len(plan.task_sequence) == 2
    tuna_task = plan.task_sequence[0]
    shark_task = plan.task_sequence[1]

    # Vérif Tuna
    assert tuna_task.name == "Tuna_grow_tuna"
    assert tuna_task.output_mapping["biomass"] == "Tuna_biomass"
    assert tuna_task.input_mapping["temperature"] == "temperature"

    # Vérif Shark
    assert shark_task.name == "Shark_grow_shark"
    assert shark_task.output_mapping["biomass"] == "Shark_biomass"
    assert shark_task.input_mapping["food"] == "Tuna_biomass"


def test_multi_output():
    """Test une unité qui produit plusieurs sorties (dictionnaire)"""
    bp = Blueprint()
    bp.register_forcing("temperature")

    # Fonction qui retourne 2 valeurs
    def compute_bio_and_flux(temperature):
        return {"biomass": temperature * 2, "flux": temperature * 0.5}

    bp.register_unit(
        compute_bio_and_flux, output_mapping={"biomass": "my_biomass", "flux": "my_flux"}
    )

    plan = bp.build()

    assert len(plan.task_sequence) == 1
    task = plan.task_sequence[0]

    # Vérification du mapping
    assert task.output_mapping["biomass"] == "my_biomass"
    assert task.output_mapping["flux"] == "my_flux"

    # Vérification que les 2 variables sont produites
    assert "my_biomass" in plan.produced_variables
    assert "my_flux" in plan.produced_variables


def test_empty_output_mapping():
    """Test qu'un mapping de sortie vide lève une erreur"""
    bp = Blueprint()
    bp.register_forcing("temperature")

    with pytest.raises(ConfigurationError):
        bp.register_unit(compute_growth, output_mapping={})
