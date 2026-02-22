"""Execution plan dataclass for compiled blueprints."""

from dataclasses import dataclass, field

from .nodes import ComputeNode


@dataclass
class ExecutionPlan:
    """Compiled execution plan with ordered task groups.

    Contains the ordered sequence of task groups to execute,
    along with variable tracking and tendency mapping.
    """

    task_groups: list[tuple[str, list[ComputeNode]]]  # [(group_name, [nodes]), ...]
    initial_variables: list[str]
    produced_variables: list[str]
    tendency_map: dict[str, list[str]] = field(default_factory=dict)  # {target_var: [tendency_sources]}
