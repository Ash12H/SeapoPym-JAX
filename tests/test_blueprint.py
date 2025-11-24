import pytest

from seapopym.blueprint import Blueprint
from seapopym.blueprint.exceptions import (
    ConfigurationError,
    CycleError,
    MissingInputError,
)


# --- Mocks ---
def compute_growth(temperature, nutrient):
    return {"biomass": temperature * nutrient}


def compute_mortality(biomass):
    return {"mortality": biomass * 0.1}


def compute_simple(temp):
    return {"out": temp}


# --- Tests ---


def test_simple_unit_registration():
    """Test l'enregistrement d'une unité simple et sa présence dans le plan."""
    bp = Blueprint()
    bp.register_forcing("temp")

    # On utilise une fonction simple à 1 argument pour éviter MissingInputError sur nutrient
    bp.register_unit(
        compute_simple, output_mapping={"out": "biomass"}, input_mapping={"temp": "temp"}
    )

    plan = bp.build()

    assert len(plan.task_groups) == 1
    group_name, tasks = plan.task_groups[0]
    assert group_name == "Global"
    assert len(tasks) == 1
    assert tasks[0].name == "compute_simple"
    assert "temp" in tasks[0].input_mapping
    assert tasks[0].input_mapping["temp"] == "temp"


def test_dependency_resolution():
    bp = Blueprint()
    bp.register_forcing("temp")
    bp.register_forcing("nutrient")

    # 1. Growth -> biomass
    bp.register_unit(
        compute_growth,
        output_mapping={"biomass": "biomass"},
        input_mapping={"temperature": "temp", "nutrient": "nutrient"},
    )

    # 2. Mortality -> mortality (dépend de biomass)
    bp.register_unit(compute_mortality, output_mapping={"mortality": "mortality"})

    plan = bp.build()

    assert len(plan.task_groups) == 1
    group_name, tasks = plan.task_groups[0]
    assert group_name == "Global"
    assert len(tasks) == 2
    # L'ordre topologique garantit que growth est avant mortality
    assert tasks[0].name == "compute_growth"
    assert tasks[1].name == "compute_mortality"


def test_cycle_detection():
    bp = Blueprint()
    bp.register_forcing("A")

    # A -> B
    bp.register_unit(compute_simple, output_mapping={"out": "B"}, input_mapping={"temp": "A"})
    # B -> A (Cycle !)
    bp.register_unit(compute_simple, output_mapping={"out": "A"}, input_mapping={"temp": "B"})

    with pytest.raises(CycleError):
        bp.build()


def test_missing_input_error():
    bp = Blueprint()
    # Pas de forcing enregistré

    with pytest.raises(MissingInputError):
        bp.register_unit(compute_simple, output_mapping={"out": "biomass"})


def test_multi_output():
    bp = Blueprint()
    bp.register_forcing("temp")

    def compute_multi(temp):
        return {"A": temp, "B": temp * 2}

    bp.register_unit(
        compute_multi, output_mapping={"A": "varA", "B": "varB"}, input_mapping={"temp": "temp"}
    )

    plan = bp.build()
    assert "varA" in plan.produced_variables
    assert "varB" in plan.produced_variables


def test_empty_output_mapping():
    bp = Blueprint()
    with pytest.raises(ConfigurationError):
        bp.register_unit(compute_simple, output_mapping={})


def test_group_registration():
    """Test l'enregistrement de groupes et le regroupement dans le plan."""
    bp = Blueprint()
    bp.register_forcing("temp")

    # Groupe Tuna
    bp.register_group(
        "Tuna",
        [
            {
                "func": compute_simple,
                "output_mapping": {"out": "biomass"},
                "input_mapping": {"temp": "temp"},
            }
        ],
    )

    # Groupe Shark (dépend de Tuna_biomass)
    bp.register_group(
        "Shark",
        [
            {
                "func": compute_mortality,
                "output_mapping": {"mortality": "mortality"},
                "input_mapping": {"biomass": "Tuna_biomass"},
            }
        ],
    )

    plan = bp.build()

    # On attend 2 groupes distincts car noms différents
    assert len(plan.task_groups) == 2

    g1_name, g1_tasks = plan.task_groups[0]
    assert g1_name == "Tuna"
    assert len(g1_tasks) == 1
    assert g1_tasks[0].name == "Tuna_compute_simple"

    g2_name, g2_tasks = plan.task_groups[1]
    assert g2_name == "Shark"
    assert len(g2_tasks) == 1
    assert g2_tasks[0].name == "Shark_compute_mortality"


def test_group_interleaving():
    """Test l'entrelacement des groupes."""
    bp = Blueprint()
    bp.register_forcing("temp")

    # Tuna 1
    bp.register_group(
        "Tuna",
        [
            {
                "func": compute_simple,
                "output_mapping": {"out": "tuna1"},
                "input_mapping": {"temp": "temp"},
            }
        ],
    )

    # Shark 1 (dépend de Tuna 1)
    bp.register_group(
        "Shark",
        [
            {
                "func": compute_simple,
                "output_mapping": {"out": "shark1"},
                "input_mapping": {"temp": "Tuna_tuna1"},
            }
        ],
    )

    # Tuna 2 (dépend de Shark 1)
    bp.register_group(
        "Tuna",
        [
            {
                "func": compute_simple,
                "output_mapping": {"out": "tuna2"},
                "input_mapping": {"temp": "Shark_shark1"},
            }
        ],
    )

    plan = bp.build()

    # On attend 3 groupes : Tuna -> Shark -> Tuna
    assert len(plan.task_groups) == 3
    assert plan.task_groups[0][0] == "Tuna"
    assert plan.task_groups[1][0] == "Shark"
    assert plan.task_groups[2][0] == "Tuna"
