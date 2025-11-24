"""Node classes for the dependency graph."""

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DataNode:
    """Représente une variable de données dans le graphe (ex: 'temperature', 'phyto_biomass')."""

    name: str
    dims: tuple | None = None  # Pour validation future (ex: ('time', 'lat', 'lon'))

    def __hash__(self) -> int:
        """Return hash based on node name."""
        return hash(self.name)


@dataclass
class ComputeNode:
    """Représente une unité de calcul (fonction) dans le graphe."""

    func: Callable
    name: str  # Identifiant unique de l'étape (ex: 'compute_mortality_tuna')
    output_name: str  # Nom de la variable produite dans le graphe
    input_mapping: dict[str, str] = field(default_factory=dict)  # arg_name -> graph_var_name
    scope: str = "local"  # 'local' ou 'global'

    def __hash__(self) -> int:
        """Return hash based on node name."""
        return hash(self.name)
