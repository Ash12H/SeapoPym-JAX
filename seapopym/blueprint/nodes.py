"""Node classes for the dependency graph."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DataNode:
    """Représente une variable de données dans le graphe (ex: 'temperature', 'phyto_biomass')."""

    name: str
    dims: tuple[Any, ...] | None = None  # Pour validation future (ex: ('time', 'lat', 'lon'))
    units: str | None = None  # Unité attendue (ex: 'degC', 'm/s')
    is_tendency_of: str | None = None  # Si c'est une tendance, de quelle variable ?
    is_state: bool = False  # Si c'est une variable d'état (persistante)
    is_parameter: bool = False  # Si c'est un paramètre (constante/config)

    def __hash__(self) -> int:
        """Return hash based on node name."""
        return hash(self.name)


@dataclass
class ComputeNode:
    """Représente une unité de calcul (fonction) dans le graphe.

    Attributes:
        func: La fonction à exécuter.
        name: Identifiant unique.
        output_mapping: Mapping des sorties.
        input_mapping: Mapping des entrées.
        scope: Portée de l'unité.
        group: Nom du groupe fonctionnel auquel appartient cette unité.
        core_dims: Dimensions sur lesquelles la fonction opère (non broadcastées).
                   Format: {"input_name": ["dim1", "dim2"]}.
        input_dims: Dimensions réelles de chaque input après transposition canonique.
                    Format: {"input_name": ("C", "Y", "X")}.
    """

    func: Callable[..., Any]
    name: str  # Identifiant unique de l'étape (ex: 'compute_mortality_tuna')
    output_mapping: dict[str, str] = field(default_factory=dict)  # key_retour -> graph_var_name
    input_mapping: dict[str, str] = field(default_factory=dict)  # arg_name -> graph_var_name
    scope: str = "local"  # 'local' ou 'global'
    group: str | None = None  # Nom du groupe fonctionnel (ex: 'Tuna')
    core_dims: dict[str, list[str]] = field(default_factory=dict)
    input_dims: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __hash__(self) -> int:
        """Return hash based on node name."""
        return hash(self.name)
