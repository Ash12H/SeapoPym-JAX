import numpy as np
import pytest
import xarray as xr

from seapopym.blueprint import Blueprint
from seapopym.functional_group import FunctionalGroup
from seapopym.functional_group.exceptions import ExecutionError


# --- Mocks ---
def compute_growth(temp):
    return {"biomass": temp * 2}


def compute_mortality(biomass):
    return {"mortality": biomass * 0.1}


def compute_bad_return(temp):
    return temp * 2  # Pas un dict


def compute_missing_key(temp):
    return {"wrong_key": temp}


def compute_non_dataarray(temp):
    return {"biomass": 42}  # Retourne un scalaire au lieu d'un DataArray


# --- Tests ---


def test_simple_execution():
    """Test l'exécution d'une seule unité."""
    bp = Blueprint()
    bp.register_forcing("temp")
    bp.register_unit(compute_growth, output_mapping={"biomass": "biomass"})
    plan = bp.build()

    fg = FunctionalGroup("TestGroup", plan.task_groups[0][1])

    # State
    state = xr.Dataset({"temp": (("x"), [10, 20])})

    results = fg.compute(state)

    assert "biomass" in results
    np.testing.assert_array_equal(results["biomass"], [20, 40])


def test_chained_execution():
    """Test l'exécution chaînée : Growth -> Mortality."""
    bp = Blueprint()
    bp.register_forcing("temp")

    # 1. Growth -> biomass
    bp.register_unit(compute_growth, output_mapping={"biomass": "biomass"})

    # 2. Mortality(biomass) -> mortality
    # Le blueprint résout automatiquement biomass -> biomass
    bp.register_unit(compute_mortality, output_mapping={"mortality": "mortality"})

    plan = bp.build()
    fg = FunctionalGroup("TestGroup", plan.task_groups[0][1])

    state = xr.Dataset({"temp": (("x"), [10])})

    results = fg.compute(state)

    assert "biomass" in results
    assert "mortality" in results

    # Growth = 10 * 2 = 20
    # Mortality = 20 * 0.1 = 2.0
    assert results["biomass"].item() == 20
    assert results["mortality"].item() == 2.0


def test_execution_error_bad_return_type():
    """La fonction ne retourne pas un dict."""
    bp = Blueprint()
    bp.register_forcing("temp")
    bp.register_unit(compute_bad_return, output_mapping={"biomass": "biomass"})
    plan = bp.build()

    fg = FunctionalGroup("TestGroup", plan.task_groups[0][1])
    state = xr.Dataset({"temp": [10]})

    with pytest.raises(ExecutionError) as excinfo:
        fg.compute(state)

    assert "must return a dictionary" in str(excinfo.value)


def test_execution_error_missing_key():
    """La fonction retourne un dict mais sans la clé attendue."""
    bp = Blueprint()
    bp.register_forcing("temp")
    bp.register_unit(compute_missing_key, output_mapping={"biomass": "biomass"})
    plan = bp.build()

    fg = FunctionalGroup("TestGroup", plan.task_groups[0][1])
    state = xr.Dataset({"temp": [10]})

    with pytest.raises(ExecutionError) as excinfo:
        fg.compute(state)

    assert "did not return expected key" in str(excinfo.value)


def test_missing_input_runtime():
    """
    Cas théorique où le Blueprint a validé mais la variable n'est pas dans le state au runtime.
    (Ex: oubli d'initialisation du state)
    """
    bp = Blueprint()
    bp.register_forcing("temp")
    bp.register_unit(compute_growth, output_mapping={"biomass": "biomass"})
    plan = bp.build()

    fg = FunctionalGroup("TestGroup", plan.task_groups[0][1])

    # State vide !
    state = xr.Dataset({})

    with pytest.raises(ExecutionError) as excinfo:
        fg.compute(state)

    assert "not found in state" in str(excinfo.value)


def test_dynamic_task_execution():
    """Test l'exécution avec une liste de tâches fournie à compute()."""
    bp = Blueprint()
    bp.register_forcing("temp")
    bp.register_unit(compute_growth, output_mapping={"biomass": "biomass"})
    plan = bp.build()

    # Init sans tâches
    fg = FunctionalGroup("TestGroup")

    # State
    state = xr.Dataset({"temp": (("x"), [10])})

    # Compute avec tâches explicites
    # Note: plan.task_groups[0] est ("Global", [node])
    tasks = plan.task_groups[0][1]
    results = fg.compute(state, tasks=tasks)

    assert "biomass" in results
    assert results["biomass"].item() == 20


def test_empty_sequence_returns_empty():
    """Un groupe sans tâches retourne un dict vide."""
    fg = FunctionalGroup("Empty")
    state = xr.Dataset({"temp": [10]})

    with pytest.warns(UserWarning, match="executed with empty task sequence"):
        results = fg.compute(state)

    assert results == {}


def test_execution_error_non_dataarray():
    """La fonction retourne un dict mais avec une valeur qui n'est pas un DataArray."""
    bp = Blueprint()
    bp.register_forcing("temp")
    bp.register_unit(compute_non_dataarray, output_mapping={"biomass": "biomass"})
    plan = bp.build()

    fg = FunctionalGroup("TestGroup", plan.task_groups[0][1])
    state = xr.Dataset({"temp": [10]})

    with pytest.raises(ExecutionError) as excinfo:
        fg.compute(state)

    assert "must be a DataArray" in str(excinfo.value)
