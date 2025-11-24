"""Core implementation of the Simulation Controller."""

import logging
from collections.abc import Callable

import xarray as xr

from seapopym.blueprint import Blueprint
from seapopym.functional_group import FunctionalGroup
from seapopym.gsm import StateManager
from seapopym.time_integrator import TimeIntegrator

from .configuration import SimulationConfig

logger = logging.getLogger(__name__)


class SimulationController:
    """Orchestrateur de la simulation.

    Gère l'initialisation, la boucle temporelle et la coordination des composants.
    """

    def __init__(self, config: SimulationConfig):
        """Initialize the simulation controller.

        Args:
            config: Configuration parameters for the simulation.
        """
        self.config = config
        self.blueprint = Blueprint()
        self.state: xr.Dataset | None = None
        self.groups: dict[str, FunctionalGroup] = {}
        self.execution_plan = None
        self.time_integrator: TimeIntegrator | None = None
        self._current_time = config.start_date

    def setup(
        self, model_configuration_func: Callable[[Blueprint], None], initial_state: xr.Dataset
    ) -> None:
        """Configure et initialise la simulation.

        Args:
            model_configuration_func: Fonction utilisateur qui enregistre les unités dans le Blueprint.
            initial_state: État initial du monde (physique, biologie).
        """
        # 1. Configuration du modèle (Blueprint)
        model_configuration_func(self.blueprint)

        # 2. Compilation du plan d'exécution
        self.execution_plan = self.blueprint.build()

        # 3. Validation et stockage de l'état initial
        StateManager.validate(initial_state, self.execution_plan.initial_variables)
        self.state = initial_state

        # 4. Création des groupes fonctionnels (Acteurs)
        # On identifie les groupes uniques nécessaires
        unique_group_names = {name for name, _ in self.execution_plan.task_groups}

        for name in unique_group_names:
            # On instancie le groupe sans séquence par défaut, car elle sera fournie dynamiquement
            self.groups[name] = FunctionalGroup(name=name)

        # 5. Création du Time Integrator
        # Pour l'instant, pas de contrainte de positivité par défaut
        self.time_integrator = TimeIntegrator(scheme="euler")

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

        # 1. (TODO) Mise à jour des forçages pour self._current_time

        # 2. Exécution de la logique scientifique par groupes ordonnés
        all_results = {}
        for group_name, tasks in self.execution_plan.task_groups:
            group = self.groups[group_name]

            # compute retourne un dict {var_name: DataArray}
            results = group.compute(self.state, tasks=tasks)
            all_results.update(results)

        # 3. Intégration temporelle (applique les tendances)
        dt = self.config.timestep.total_seconds()
        self.state = self.time_integrator.integrate(
            self.state, all_results, self.execution_plan.tendency_map, dt
        )

        # 4. Ajout des variables non-tendances (diagnostics, etc.)
        # On filtre pour ne pas réappliquer les tendances déjà intégrées
        tendency_vars = set(sum(self.execution_plan.tendency_map.values(), []))
        diagnostics = {k: v for k, v in all_results.items() if k not in tendency_vars}
        self.state = StateManager.merge_forcings(self.state, diagnostics)

        # 5. Préparation du pas suivant
        self.state = StateManager.initialize_next_step(self.state)
