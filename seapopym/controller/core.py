"""Core implementation of the Simulation Controller."""

import logging
from collections.abc import Callable, Hashable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pint
import pint_xarray  # noqa: F401
import xarray as xr

from seapopym.backend import ComputeBackend, DaskBackend, SequentialBackend
from seapopym.blueprint import Blueprint, ExecutionPlan
from seapopym.forcing import ForcingManager
from seapopym.gsm import StateManager
from seapopym.io.writer import BaseOutputWriter, MemoryWriter, ZarrWriter
from seapopym.time_integrator import TimeIntegrator

from .configuration import SimulationConfig

logger = logging.getLogger(__name__)


class SimulationController:
    """Orchestrateur de la simulation.

    Gère l'initialisation, la boucle temporelle et la coordination des composants.
    """

    def __init__(
        self,
        config: SimulationConfig,
        backend: ComputeBackend | str = "sequential",
    ):
        """Initialize the simulation controller.

        Args:
            config: Configuration parameters for the simulation.
            backend: The execution backend to use. Can be a string ('sequential', 'dask')
                    or a ComputeBackend instance. Defaults to 'sequential'.
        """
        self.config = config
        self.blueprint = Blueprint()
        self.state: xr.Dataset | None = None
        self.execution_plan: ExecutionPlan | None = None
        self.time_integrator: TimeIntegrator | None = None
        self.forcing_manager: ForcingManager | None = None
        self.writer: BaseOutputWriter | None = None
        self.backend: ComputeBackend

        if isinstance(backend, str):
            if backend == "sequential":
                self.backend = SequentialBackend()
            elif backend == "dask":
                self.backend = DaskBackend()
            else:
                raise ValueError(
                    f"Unknown backend type: '{backend}'. Supported: 'sequential', 'dask'."
                )
        else:
            self.backend = backend

        self._current_time = config.start_date

    def setup(
        self,
        model_configuration_func: Callable[[Blueprint], None],
        initial_state: xr.Dataset,
        forcings: xr.Dataset | None = None,
        parameters: Any | None = None,
        output_path: str | Path | None = None,
        output_variables: list[str] | None = None,
        output_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Configure et initialise la simulation.

        Args:
            model_configuration_func: Fonction utilisateur qui enregistre les unités dans le Blueprint.
            initial_state: État initial du monde (physique, biologie).
            forcings: Dataset optionnel contenant les variables de forçage temporel.
                     Doit contenir une dimension temporelle (Coordinates.T) et ne doit pas avoir
                     de variables en commun avec initial_state.
            parameters: Paramètres du modèle (Dataclass ou Dict).
            output_path: Path to save the output. If None, results are stored in memory.
            output_variables: List of variables to save. If None, all variables are saved.
            output_metadata: Metadata to add to the output dataset.
        """
        # 1. Configuration du modèle (Blueprint)
        model_configuration_func(self.blueprint)

        # 2. Ingestion des paramètres
        param_ds = xr.Dataset()
        if parameters is not None:
            param_ds = self._ingest_parameters(parameters)

        # 3. Compilation du plan d'exécution
        self.execution_plan = self.blueprint.build()

        # 4. Validation et stockage de l'état initial
        # On vérifie d'abord les conflits
        if forcings is not None:
            common_vars = set(initial_state.data_vars) & set(forcings.data_vars)
            if common_vars:
                raise ValueError(
                    f"Ambiguous definition: variables {common_vars} are defined in both initial state and forcings."
                )

        # On vérifie la couverture
        # On inclut les data_vars ET les coords car les fonctions peuvent demander des coordonnées (ex: latitude)
        provided_vars = (
            set(map(str, initial_state.data_vars))
            | set(map(str, initial_state.coords))
            | set(map(str, param_ds.data_vars))
        )

        if forcings is not None:
            provided_vars.update(map(str, forcings.data_vars))
            provided_vars.update(map(str, forcings.coords))

        missing_vars = set(self.execution_plan.initial_variables) - provided_vars
        if missing_vars:
            from seapopym.gsm.exceptions import StateValidationError

            raise StateValidationError(
                f"Missing required variables (data or coords): {missing_vars}"
            )

        # Fusion de l'état initial et des paramètres
        # Note: Les paramètres sont ajoutés à l'état initial
        self.state = xr.merge([initial_state, param_ds])

        # 5. Standardisation des unités
        # On valide et convertit les unités selon le Blueprint
        if forcings is not None:
            # On doit aussi standardiser les forçages
            # Mais ForcingManager s'attend à recevoir des forçages bruts et les interpoler
            # Idéalement, on standardise tout ce qui rentre.
            # Pour simplifier, on standardise l'état initial (qui contient les params)
            # et on demandera au ForcingManager de standardiser à la volée ou on le fait ici.
            # Faisons-le ici pour être sûr.
            forcings = self._standardize_units(forcings)

        self.state = self._standardize_units(self.state)

        # 6. Création du Time Integrator
        # Pour l'instant, pas de contrainte de positivité par défaut
        self.time_integrator = TimeIntegrator(scheme="euler")

        # 7. Création du Forcing Manager si des forçages sont fournis
        if forcings is not None:
            self.forcing_manager = ForcingManager(forcings)

        # 8. Setup Output Writer
        if output_path is None:
            self.writer = MemoryWriter(variables=output_variables, metadata=output_metadata)
        else:
            self.writer = ZarrWriter(
                path=output_path, variables=output_variables, metadata=output_metadata
            )

    def run(self) -> None:
        """Exécute la boucle de simulation complète."""
        if self.state is None:
            raise RuntimeError("Simulation not set up. Call setup() first.")

        logger.info(f"Starting simulation from {self.config.start_date} to {self.config.end_date}")

        try:
            # Save initial state
            if self.writer:
                self.writer.append(self.state)

            while self._current_time < self.config.end_date:
                self.step()
                self._current_time += self.config.timestep

                # Save state after step
                if self.writer:
                    self.writer.append(self.state)

            # Finalize writing
            if self.writer:
                self.writer.finalize()

            logger.info("Simulation completed.")
        except Exception as e:
            logger.error(f"Simulation failed at {self._current_time}: {e}")
            raise

    def step(self) -> None:
        """Exécute un pas de temps."""
        if self.state is None or self.execution_plan is None or self.time_integrator is None:
            raise RuntimeError("Simulation not set up.")

        # 1. Mise à jour des forçages pour self._current_time
        if self.forcing_manager:
            current_forcings = self.forcing_manager.get_forcings(self._current_time)
            self.state = StateManager.update_with_forcings(self.state, current_forcings)

        # 2. Exécution de la logique scientifique via le backend
        all_results = self.backend.execute(self.execution_plan.task_groups, self.state)

        # 3. Intégration temporelle (applique les tendances)
        dt = self.config.timestep.total_seconds()
        self.state = self.time_integrator.integrate(
            self.state, all_results, self.execution_plan.tendency_map, dt
        )

        # 4. Ajout des variables non-tendances (diagnostics, etc.)
        # On filtre pour ne pas réappliquer les tendances déjà intégrées
        tendency_vars = set(sum(self.execution_plan.tendency_map.values(), []))
        # 4. Fusion des diagnostics (variables produites mais non intégrées)
        # TODO: Filtrer ce qui doit être gardé ou non
        diagnostics: dict[Hashable, Any] = {
            k: v for k, v in all_results.items() if k not in tendency_vars
        }
        self.state = StateManager.update_with_forcings(self.state, diagnostics)

        # 5. Préparation du pas suivant
        self.state = StateManager.initialize_next_step(self.state)

    def _ingest_parameters(self, parameters: Any) -> xr.Dataset:
        """Convertit les paramètres en xarray.Dataset.

        Supporte:
        - Dataclass simple (pas de préfixe)
        - Dict simple {nom: valeur}
        - Dict imbriqué {groupe: Dataclass} ou {groupe: {nom: valeur}} (ajoute préfixe groupe/)
        """
        data_vars = {}

        def process_item(key_prefix: str, item: Any) -> None:
            # Si c'est une Dataclass, on la convertit en dict et on récure
            if is_dataclass(item):
                item_dict = asdict(item)
                for k, v in item_dict.items():
                    full_key = f"{key_prefix}/{k}" if key_prefix else k
                    process_item(full_key, v)

            # Si c'est un Dict, on récure
            elif isinstance(item, dict):
                for k, v in item.items():
                    full_key = f"{key_prefix}/{k}" if key_prefix else k
                    process_item(full_key, v)

            # Sinon c'est une valeur terminale (Leaf)
            else:
                name = key_prefix
                # Si c'est déjà un DataArray, on le garde tel quel
                if isinstance(item, xr.DataArray):
                    data_vars[name] = item
                # Si c'est une quantité Pint, on extrait valeur et unité
                elif hasattr(item, "magnitude") and hasattr(item, "units"):
                    data_vars[name] = xr.DataArray(
                        data=item.magnitude, attrs={"units": str(item.units)}
                    )
                # Sinon c'est une valeur brute (float, int)
                else:
                    data_vars[name] = xr.DataArray(data=item)

        process_item("", parameters)

        return xr.Dataset(data_vars)

    def _standardize_units(self, ds: xr.Dataset) -> xr.Dataset:
        """Valide et convertit les unités des variables du Dataset selon le Blueprint."""
        # On itère sur toutes les variables enregistrées dans le Blueprint
        # qui ont une unité définie.

        # Note: self.blueprint._data_nodes est privé mais accessible.
        # Idéalement on ajouterait une méthode publique.

        for name, node in self.blueprint._data_nodes.items():
            if name not in ds:
                continue

            target_unit = node.units
            if target_unit is None:
                continue

            var = ds[name]
            current_unit = var.attrs.get("units")

            # Si pas d'unité sur la variable, on assume qu'elle est bonne (ou on pourrait warning)
            if current_unit is None:
                # TODO: Log warning ?
                continue

            # Conversion avec pint-xarray
            try:
                # On utilise l'accessor pint pour convertir
                # Il faut d'abord "quantifier" le DataArray
                quantified = var.pint.quantify(unit_registry=pint.get_application_registry())
                converted = quantified.pint.to(target_unit)
                # On déquantifie pour revenir à des données brutes (float) avec l'unité mise à jour
                ds[name] = converted.pint.dequantify()
            except Exception as e:
                raise ValueError(
                    f"Unit conversion failed for variable '{name}': {current_unit} -> {target_unit}. Error: {e}"
                ) from e

        return ds

    @property
    def results(self) -> xr.Dataset:
        """Return the simulation results."""
        if self.writer is None:
            raise RuntimeError("Simulation not set up. Call setup() first.")
        return self.writer.finalize()
