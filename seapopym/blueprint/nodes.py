"""Node classes for the dependency graph."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, eq=False)
class DataNode:
    """A data variable in the dependency graph (e.g. 'state.biomass', 'forcings.temperature').

    Identity is based on `name` only (the variable path). Other fields are metadata.
    """

    name: str
    dims: tuple[str, ...] | None = None
    units: str | None = None
    is_tendency_of: str | None = None
    is_state: bool = False
    is_parameter: bool = False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataNode):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass(eq=False)
class ComputeNode:
    """A computation unit (function) in the dependency graph.

    Identity is based on `name` only (metadata for visualization/debug).

    Attributes:
        func: The function to execute.
        name: Unique identifier (auto-generated from outputs if not provided).
        output_mapping: Maps output keys to graph variable names.
        input_mapping: Maps function argument names to graph variable names.
        group: Functional group name (e.g. 'Tuna').
        core_dims: Dimensions operated on (not broadcast). Format: {"input_name": ["dim1"]}.
        input_dims: Actual dims per input after canonical transpose. Format: {"input_name": ("C", "Y", "X")}.
        out_dims: Output dimensions from function metadata.
    """

    func: Callable[..., Any]
    name: str
    output_mapping: dict[str, str] = field(default_factory=dict)
    input_mapping: dict[str, str] = field(default_factory=dict)
    group: str | None = None
    core_dims: dict[str, list[str]] = field(default_factory=dict)
    input_dims: dict[str, tuple[str, ...]] = field(default_factory=dict)
    out_dims: list[str] | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ComputeNode):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)
