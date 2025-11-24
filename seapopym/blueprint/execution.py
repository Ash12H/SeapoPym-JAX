"""Execution plan dataclass for compiled blueprints."""

from dataclasses import dataclass

from .nodes import ComputeNode


@dataclass
class ExecutionPlan:
    """Le résultat de la compilation du Blueprint.

    Contient tout ce dont le Controller a besoin pour exécuter la simulation.
    """

    task_sequence: list[ComputeNode]
    initial_variables: list[
        str
    ]  # Variables qui doivent être présentes au t=0 (Forçages + Init State)
    produced_variables: list[str]  # Variables générées par le système

    # Pour le futur : mapping des scopes pour l'optimisation
    # scope_map: Dict[str, str] = field(default_factory=dict)
