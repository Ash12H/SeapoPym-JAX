"""Blueprint module for dependency graph construction and execution planning."""

import inspect
import itertools
from collections.abc import Callable
from typing import Any

import networkx as nx

from .exceptions import ConfigurationError, CycleError, MissingInputError
from .execution import ExecutionPlan
from .nodes import ComputeNode, DataNode


class Blueprint:
    """Architecte du modèle. Construit le graphe de dépendances et compile le plan d'exécution."""

    def __init__(self) -> None:
        """Initialize the Blueprint with an empty dependency graph."""
        self.graph: nx.DiGraph[DataNode | ComputeNode] = nx.DiGraph()
        self.registered_variables: set[str] = set()  # Pour lookup rapide
        self._data_nodes: dict[str, DataNode] = {}  # Registre des noeuds de données
        self._group_context: str | None = None  # Pour le namespacing automatique

    def register_forcing(self, name: str, dims: tuple[Any, ...] | None = None) -> None:
        """Déclare une source de données externe (ex: température, courants).

        Raises:
            ConfigurationError: Si la variable est déjà enregistrée.

        """
        if name in self.registered_variables:
            raise ConfigurationError(f"Variable '{name}' is already registered.")

        node = DataNode(name=name, dims=dims)
        self.graph.add_node(node)
        self.registered_variables.add(name)
        self._data_nodes[name] = node

    def register_unit(
        self,
        func: Callable[..., Any],
        output_mapping: dict[str, str],
        input_mapping: dict[str, str] | None = None,
        output_tendencies: dict[str, str] | None = None,
        scope: str = "local",
        name: str | None = None,
    ) -> None:
        """Enregistre une unité de calcul.

        Args:
            func: La fonction Python à exécuter.
            output_mapping: Mapping des sorties {key_retour_fonction: nom_variable_graphe}.
            input_mapping: Surcharge des entrées {arg_name: graph_var_name}.
            output_tendencies: Mapping {key_retour: var_cible} pour marquer une sortie comme tendance.
                              Ex: {"rate": "biomass"} signifie que la clé "rate" est une tendance de "biomass".
                              Les tendances sont sommées par le TimeIntegrator lors de l'intégration.
            scope: 'local' ou 'global'.
            name: Nom unique de l'étape (optionnel, défaut = nom fonction).

        Example:
            >>> def compute_mortality(biomass):
            ...     return {"rate": biomass * -0.1}
            >>> bp.register_unit(
            ...     compute_mortality,
            ...     output_mapping={"rate": "mortality_rate"},
            ...     output_tendencies={"rate": "biomass"}  # mortality_rate est une tendance de biomass
            ... )
        """
        if not output_mapping:
            raise ConfigurationError("output_mapping must be provided and not empty.")

        input_mapping = input_mapping or {}

        func_name = func.__name__
        step_name = name or func_name

        if self._group_context:
            step_name = f"{self._group_context}_{step_name}"

        # Résolution des noms de sortie avec préfixe de groupe si nécessaire
        final_output_mapping = {}
        for key, var_name in output_mapping.items():
            if self._group_context:
                final_output_mapping[key] = f"{self._group_context}_{var_name}"
            else:
                final_output_mapping[key] = var_name

        compute_node = ComputeNode(
            func=func,
            name=step_name,
            output_mapping=final_output_mapping,
            input_mapping={},
            scope=scope,
            group=self._group_context,  # Enregistre le groupe
        )

        sig = inspect.signature(func)
        resolved_mapping = {}

        for arg_name, param in sig.parameters.items():
            if param.default != inspect.Parameter.empty:
                continue

            source_var = self._resolve_input(arg_name, input_mapping)

            if source_var and source_var in self.registered_variables:
                resolved_mapping[arg_name] = source_var
                source_node = self._data_nodes[source_var]
                self.graph.add_edge(source_node, compute_node)
            else:
                raise MissingInputError(
                    f"Argument '{arg_name}' for unit '{step_name}' could not be resolved (Source: '{source_var}')."
                )

        compute_node.input_mapping = resolved_mapping
        self.graph.add_node(compute_node)

        output_tendencies = output_tendencies or {}

        for key, var_name in final_output_mapping.items():
            # Déterminer si c'est une tendance
            is_tendency_of = output_tendencies.get(key)

            output_node = DataNode(name=var_name, is_tendency_of=is_tendency_of)
            self.graph.add_node(output_node)
            self.graph.add_edge(compute_node, output_node)
            self.registered_variables.add(var_name)
            self._data_nodes[var_name] = output_node

    def _resolve_input(self, arg_name: str, explicit_mapping: dict[str, str]) -> str | None:
        """Résout le nom de la variable dans le graphe selon la priorité.

        1. Mapping Explicite
        2. Namespacing (Groupe)
        3. Matching par défaut
        """
        # 1. Mapping Explicite
        if arg_name in explicit_mapping:
            return explicit_mapping[arg_name]

        # 2. Namespacing
        if self._group_context:
            prefixed_name = f"{self._group_context}_{arg_name}"
            if prefixed_name in self.registered_variables:
                return prefixed_name

        # 3. Matching par défaut (Global)
        if arg_name in self.registered_variables:
            return arg_name

        return None

    def register_group(self, group_prefix: str, units: list[dict[str, Any]]) -> None:
        """Helper pour enregistrer un groupe d'unités.

        Args:
            group_prefix: Le préfixe du groupe (ex: 'Tuna').
            units: Liste de dicts de configuration pour register_unit.
        """
        previous_context = self._group_context
        self._group_context = group_prefix

        try:
            for unit_conf in units:
                self.register_unit(
                    func=unit_conf["func"],
                    output_mapping=unit_conf["output_mapping"],
                    input_mapping=unit_conf.get("input_mapping"),
                    output_tendencies=unit_conf.get("output_tendencies"),
                    scope=unit_conf.get("scope", "local"),
                    name=unit_conf.get("name"),
                )
        finally:
            self._group_context = previous_context

    def build(self) -> ExecutionPlan:
        """Compile le graphe.

        Returns:
            ExecutionPlan: Le plan d'exécution compilé.

        Raises:
            CycleError: Si un cycle est détecté.
        """
        # 1. Tri Topologique pour l'ordre d'exécution
        # Le graphe contient des DataNodes et des ComputeNodes.
        # L'ordre d'exécution ne concerne que les ComputeNodes.
        # topological_sort lève NetworkXUnfeasible s'il y a un cycle.
        try:
            sorted_nodes = list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            raise CycleError("Graph contains a cycle (detected during topological sort).") from None

        task_sequence = [node for node in sorted_nodes if isinstance(node, ComputeNode)]

        # Regroupement des tâches contiguës par groupe
        task_groups = []
        for group_name, nodes in itertools.groupby(task_sequence, key=lambda n: n.group):
            final_group_name = group_name or "Global"
            task_groups.append((final_group_name, list(nodes)))

        # 3. Identification des variables initiales et produites
        # Initiales = DataNodes avec in_degree = 0 (pas produits par un calcul)
        initial_vars = [
            node.name
            for node in self.graph.nodes
            if isinstance(node, DataNode) and self.graph.in_degree(node) == 0
        ]

        # Produites = DataNodes avec in_degree > 0
        produced_vars = [
            node.name
            for node in self.graph.nodes
            if isinstance(node, DataNode) and self.graph.in_degree(node) > 0
        ]

        # 4. Construction du tendency_map
        from collections import defaultdict

        tendency_map = defaultdict(list)

        for node in self.graph.nodes:
            if isinstance(node, DataNode) and node.is_tendency_of:
                tendency_map[node.is_tendency_of].append(node.name)

        return ExecutionPlan(
            task_groups=task_groups,
            initial_variables=initial_vars,
            produced_variables=produced_vars,
            tendency_map=dict(tendency_map),
        )
