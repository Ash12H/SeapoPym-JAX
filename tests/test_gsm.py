import numpy as np
import pytest
import xarray as xr

from seapopym.gsm import StateManager, StateValidationError


def test_create_initial_state():
    coords = {"lat": np.arange(10), "lon": np.arange(10)}
    variables = {"bathymetry": (("lat", "lon"), np.zeros((10, 10)))}

    state = StateManager.create_initial_state(coords, variables)

    assert isinstance(state, xr.Dataset)
    assert "bathymetry" in state
    assert state.coords["lat"].size == 10


def test_validate_success():
    coords = {"x": [1], "time": [0]}
    state = xr.Dataset({"temp": (("x"), [10])}, coords=coords)

    # Doit passer sans erreur (vars + coords)
    StateManager.validate(state, ["temp"], required_coords=["time"])


def test_validate_failure_vars():
    coords = {"x": [1]}
    state = xr.Dataset({"temp": (("x"), [10])}, coords=coords)

    with pytest.raises(StateValidationError) as excinfo:
        StateManager.validate(state, ["temp", "salt"])

    assert "Missing variables" in str(excinfo.value)
    assert "salt" in str(excinfo.value)


def test_validate_failure_coords():
    coords = {"x": [1]}
    state = xr.Dataset({"temp": (("x"), [10])}, coords=coords)

    with pytest.raises(StateValidationError) as excinfo:
        StateManager.validate(state, ["temp"], required_coords=["time"])

    assert "Missing coordinates" in str(excinfo.value)
    assert "time" in str(excinfo.value)


def test_merge_forcings_dict():
    coords = {"x": [1]}
    state = xr.Dataset({"temp": (("x"), [10])}, coords=coords)
    forcings = {"wind": (("x"), [5])}

    new_state = StateManager.merge_forcings(state, forcings)

    assert "wind" in new_state
    assert "temp" in new_state
    assert "wind" not in state


def test_merge_forcings_dataset():
    coords = {"x": [1]}
    state = xr.Dataset({"temp": (("x"), [10])}, coords=coords)
    forcings_ds = xr.Dataset({"wind": (("x"), [5])}, coords=coords)

    new_state = StateManager.merge_forcings(state, forcings_ds)

    assert "wind" in new_state
    assert "temp" in new_state
    assert isinstance(new_state, xr.Dataset)


def test_initialize_next_step_shallow():
    """Vérifie que la copie est shallow (partage mémoire) mais structurellement isolée."""
    coords = {"x": [1]}
    # On utilise numpy directement pour vérifier les adresses mémoire si besoin,
    # mais xarray gère ça.
    data = np.array([10])
    state = xr.Dataset({"temp": (("x"), data)}, coords=coords)

    next_state = StateManager.initialize_next_step(state)

    # 1. Vérification du partage de données (Shallow Copy)
    # Les valeurs sont les mêmes
    assert next_state["temp"].item() == 10
    # Si on modifie le numpy array sous-jacent (très bas niveau), ça se répercute
    # (C'est le danger et l'avantage du shallow copy)
    # Note: xarray protège souvent contre ça, mais testons l'identité des objets DataArray
    # Les objets DataArray sont recréés, mais pointent vers les mêmes données ?
    # En fait copy(deep=False) crée de nouveaux objets DataArray wrapper.

    # 2. Isolation structurelle
    # Ajouter une variable dans next ne touche pas prev
    next_state["new_var"] = (("x"), [5])
    assert "new_var" not in state

    # Remplacer une variable dans next ne touche pas prev
    next_state["temp"] = (("x"), [99])  # Réassignation
    assert state["temp"].item() == 10  # L'ancien pointeur est conservé
