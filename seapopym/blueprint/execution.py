"""Execution plan dataclass for compiled blueprints."""

from dataclasses import dataclass, field

from .nodes import ComputeNode


@dataclass
class ExecutionPlan:
    """Résultat de la compilation du Blueprint.

    Contient la séquence ordonnée des groupes de tâches à exécuter.
    """

    task_groups: list[tuple[str, list[ComputeNode]]]  # [(group_name, [nodes]), ...]
    initial_variables: list[str]
    produced_variables: list[str]
    tendency_map: dict[str, list[str]] = field(default_factory=dict)  # {var_cible: [tendances]}
