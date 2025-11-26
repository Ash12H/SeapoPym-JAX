"""Core implementation of the Simulation Controller."""

import logging
from collections.abc import Callable, Hashable
from typing import Any

import xarray as xr

from seapopym.backend import ComputeBackend, DaskBackend, SequentialBackend
from seapopym.blueprint import Blueprint, ExecutionPlan
from seapopym.forcing import ForcingManager
from seapopym.gsm import StateManager
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
    ) -> None:
        """Configure et initialise la simulation.

        Args:
            model_configuration_func: Fonction utilisateur qui enregistre les unités dans le Blueprint.
            initial_state: État initial du monde (physique, biologie).
            forcings: Dataset optionnel contenant les variables de forçage temporel.
                     Doit contenir une dimension temporelle (Coordinates.T) et ne doit pas avoir
                     de variables en commun avec initial_state.
        """
        # 1. Configuration du modèle (Blueprint)
        model_configuration_func(self.blueprint)

        # 2. Compilation du plan d'exécution
        self.execution_plan = self.blueprint.build()

        # 3. Validation et stockage de l'état initial
        # On vérifie d'abord les conflits
        if forcings is not None:
            common_vars = set(initial_state.data_vars) & set(forcings.data_vars)
            if common_vars:
                raise ValueError(
                    f"Ambiguous definition: variables {common_vars} are defined in both initial state and forcings."
                )

        # On vérifie la couverture
        # On inclut les data_vars ET les coords car les fonctions peuvent demander des coordonnées (ex: latitude)
        provided_vars = set(map(str, initial_state.data_vars)) | set(map(str, initial_state.coords))

        if forcings is not None:
            provided_vars.update(map(str, forcings.data_vars))
            provided_vars.update(map(str, forcings.coords))

        missing_vars = set(self.execution_plan.initial_variables) - provided_vars
        if missing_vars:
            from seapopym.gsm.exceptions import StateValidationError

            raise StateValidationError(
                f"Missing required variables (data or coords): {missing_vars}"
            )

        self.state = initial_state

        # 5. Création du Time Integrator
        # Pour l'instant, pas de contrainte de positivité par défaut
        self.time_integrator = TimeIntegrator(scheme="euler")

        # 6. Création du Forcing Manager si des forçages sont fournis
        if forcings is not None:
            self.forcing_manager = ForcingManager(forcings)

    def run(self) -> None:
        """Exécute la boucle de simulation complète."""
        if self.state is None:
            raise RuntimeError("Simulation not set up. Call setup() first.")

        logger.info(f"Starting simulation from {self.config.start_date} to {self.config.end_date}")

        try:
            while self._current_time < self.config.end_date:
                self.step()
                self._current_time += self.config.timestep
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
