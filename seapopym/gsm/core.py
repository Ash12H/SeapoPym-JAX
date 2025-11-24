"""Global State Manager (GSM) module for managing simulation state.

Provides utilities for creating, validating, and manipulating xarray Datasets
in a functional, immutable way.
"""

from collections.abc import Hashable, Iterable, Mapping
from typing import Any

import xarray as xr

from .exceptions import StateValidationError


class StateManager:
    """Gestionnaire de l'état global (Global State) de la simulation.

    Encapsule les opérations sur xarray.Dataset pour garantir la cohérence et l'immutabilité.
    """

    @staticmethod
    def create_initial_state(
        coords: Mapping[Hashable, Any], variables: Mapping[Hashable, Any] | None = None
    ) -> xr.Dataset:
        """Crée l'état initial de la simulation.

        Args:
            coords: Dictionnaire des coordonnées (ex: {'time': ..., 'lat': ..., 'lon': ...}).
            variables: Dictionnaire des variables initiales (ex: {'bathymetry': ...}).

        Returns:
            Un xarray.Dataset initialisé.
        """
        variables = variables or {}
        ds = xr.Dataset(data_vars=variables, coords=coords)
        return ds

    @staticmethod
    def validate(
        state: xr.Dataset,
        required_vars: Iterable[Hashable],
        required_coords: Iterable[Hashable] | None = None,
    ) -> None:
        """Valide que l'état contient toutes les variables et coordonnées requises.

        Args:
            state: Le Dataset à valider.
            required_vars: Liste des noms de variables attendues.
            required_coords: Liste des noms de coordonnées attendues (optionnel).

        Raises:
            StateValidationError: Si des variables ou coordonnées sont manquantes.
        """
        missing_vars = [var for var in required_vars if var not in state]
        if missing_vars:
            raise StateValidationError(f"Missing variables in state: {missing_vars}")

        if required_coords:
            missing_coords = [c for c in required_coords if c not in state.coords]
            if missing_coords:
                raise StateValidationError(f"Missing coordinates in state: {missing_coords}")

    @staticmethod
    def merge_forcings(
        state: xr.Dataset, forcings: xr.Dataset | Mapping[Hashable, Any]
    ) -> xr.Dataset:
        """Intègre les forçages du pas de temps courant dans l'état.

        Retourne un NOUVEAU Dataset (immutabilité fonctionnelle).

        Args:
            state: L'état actuel.
            forcings: Dataset ou Dictionnaire des forçages à ajouter/mettre à jour.

        Returns:
            Un nouveau Dataset contenant l'état + les forçages.
        """
        if isinstance(forcings, xr.Dataset):
            # xr.merge crée une copie et gère les alignements
            return xr.merge([state, forcings])
        else:
            # assign pour un dictionnaire simple
            return state.assign(**forcings)

    @staticmethod
    def initialize_next_step(current_state: xr.Dataset) -> xr.Dataset:
        """Prépare l'état pour le prochain pas de temps via une Shallow Copy.

        ATTENTION : Les DataArrays sous-jacents sont partagés (même mémoire).
        Cela permet d'éviter de dupliquer les données statiques (bathymétrie).
        Cependant, modifier les valeurs d'un tableau existant in-place affectera l'ancien état.
        L'usage prévu est d'ajouter/remplacer des variables entières pour le nouveau pas de temps.

        Args:
            current_state: L'état à la fin du pas de temps t.

        Returns:
            Un nouveau Dataset (structure isolée, données partagées) prêt pour t+1.
        """
        return current_state.copy(deep=False)
