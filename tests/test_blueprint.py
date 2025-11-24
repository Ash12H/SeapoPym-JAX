import pytest

from seapopym.blueprint.core import Blueprint
from seapopym.blueprint.exceptions import MissingInputError


# --- Fonctions Mock pour les tests ---
def compute_growth(temperature, nutrient):
    return temperature * nutrient


def compute_mortality(biomass, temperature):
    return biomass * temperature


def compute_advection(biomass, current):
    return biomass * current


def compute_with_params(biomass, param_a=1.0):
    # param_a ne devrait pas être résolu par le graphe s'il a une valeur par défaut ?
    # Ou alors on force tout ? Pour l'instant le code force tout.
    # On va tester le comportement actuel.
    return biomass * param_a


# --- Tests ---


def test_simple_chain():
    """Test une chaîne simple : Forçage -> Unit A -> Unit B"""
    bp = Blueprint()

    # 1. Forçages
    bp.register_forcing("temperature")
    bp.register_forcing("nutrient")

    # 2. Unit A : Growth (produit 'biomass')
    bp.register_unit(compute_growth, output_name="biomass")

    # 3. Unit B : Mortality (consomme 'biomass' et 'temperature')
    bp.register_unit(compute_mortality, output_name="mortality")

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
        compute_growth, input_mapping={"temperature": "sea_surface_temp"}, output_name="biomass"
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
        bp.register_unit(compute_growth)


def test_cycle_detection():
    """Test la détection de cycle"""
    bp = Blueprint()
    bp.register_forcing("temperature")

    # A a besoin de B, B a besoin de A
    # Pour simuler ça, on doit "tricher" car register_unit vérifie l'existence des inputs.
    # Mais si on enregistre A (qui demande B) avant B, ça plante en MissingInputError.
    # C'est une propriété intéressante : notre implémentation actuelle empêche les cycles
    # par construction (on ne peut consommer que ce qui existe déjà) !
    # Sauf si on permettait la "forward reference".

    # Vérifions ce comportement :
    with pytest.raises(MissingInputError):
        bp.register_unit(compute_mortality, input_mapping={"biomass": "future_biomass"})


def test_group_namespacing():
    """Test la résolution automatique dans un groupe"""
    bp = Blueprint()
    bp.register_forcing("temperature")
    bp.register_forcing("Tuna_nutrient")  # Spécifique au groupe

    # Groupe Tuna
    # compute_growth prend (temperature, nutrient)
    # temperature -> global 'temperature' (match par défaut)
    # nutrient -> 'Tuna_nutrient' (match par namespace)

    units = [
        {
            "func": compute_growth,
            "output_name": "biomass",  # Deviendra Tuna_biomass
        },
        {
            "func": compute_mortality,
            # prend (biomass, temperature)
            # biomass -> Tuna_biomass (produit juste avant)
            "output_name": "mortality",
        },
    ]

    bp.register_group("Tuna", units)

    plan = bp.build()

    # Vérifications
    growth_task = plan.task_sequence[0]
    mortality_task = plan.task_sequence[1]

    assert growth_task.name == "Tuna_compute_growth"
    assert growth_task.output_name == "Tuna_biomass"
    assert growth_task.input_mapping["nutrient"] == "Tuna_nutrient"
    assert growth_task.input_mapping["temperature"] == "temperature"  # Fallback global

    assert mortality_task.input_mapping["biomass"] == "Tuna_biomass"  # Résolu en interne du groupe


def test_default_params_ignored():
    """Test que les paramètres avec valeur par défaut sont ignorés lors de la résolution"""
    bp = Blueprint()
    bp.register_forcing("biomass")

    # compute_with_params(biomass, param_a=1.0)
    # param_a doit être ignoré, seul biomass est requis
    bp.register_unit(compute_with_params, output_name="result")

    plan = bp.build()
    assert len(plan.task_sequence) == 1
    assert plan.task_sequence[0].name == "compute_with_params"
