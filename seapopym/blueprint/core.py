"""Blueprint module for dependency graph construction and execution planning."""

import inspect
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
        self.graph: nx.DiGraph = nx.DiGraph()
        self.registered_variables: set[str] = set()  # Pour lookup rapide
        self._data_nodes: dict[str, DataNode] = {}  # Registre des noeuds de données
        self._group_context: str | None = None  # Pour le namespacing automatique

    def register_forcing(self, name: str, dims: tuple | None = None) -> None:
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
        func: Callable,
        input_mapping: dict[str, str] | None = None,
        output_name: str | None = None,
        scope: str = "local",
        name: str | None = None,
    ) -> None:
        """Enregistre une unité de calcul.

        Args:
            func: La fonction Python à exécuter.
            input_mapping: Surcharge des entrées {arg_name: graph_var_name}.
            output_name: Nom de la variable produite. Si None, utilise le nom de la fonction.
            scope: 'local' ou 'global'.
            name: Nom unique de l'étape (optionnel, défaut = nom fonction).

        """
        input_mapping = input_mapping or {}

        # 1. Identification de l'unité
        func_name = func.__name__
        step_name = name or func_name

        # Application du contexte de groupe au nom de l'étape si nécessaire
        if self._group_context:
            step_name = f"{self._group_context}_{step_name}"

        # 2. Détermination de la sortie
        # Si output_name n'est pas fourni, on utilise le nom de l'étape
        raw_output_name = output_name or func_name

        if self._group_context:
            # On préfixe toujours dans un groupe, sauf si on voulait permettre des sorties globales
            # Pour l'instant, on suit la logique du test : tout est scopé.
            final_output_name = f"{self._group_context}_{raw_output_name}"
        else:
            final_output_name = raw_output_name

        compute_node = ComputeNode(
            func=func,
            name=step_name,
            output_name=final_output_name,
            input_mapping={},  # Sera rempli après résolution
            scope=scope,
        )

        # 3. Introspection et Résolution des Entrées
        sig = inspect.signature(func)
        resolved_mapping = {}

        for arg_name, param in sig.parameters.items():
            # On ignore 'self' ou les args spécifiques si besoin (ex: params)
            # On ignore les paramètres avec valeur par défaut (considérés comme optionnels/config)
            if param.default != inspect.Parameter.empty:
                continue

            source_var = self._resolve_input(arg_name, input_mapping)

            # Validation stricte : la source doit exister
            if source_var and source_var in self.registered_variables:
                resolved_mapping[arg_name] = source_var
                # Création de l'arête Donnée -> Calcul
                # On récupère le DataNode existant depuis le registre
                source_node = self._data_nodes[source_var]
                self.graph.add_edge(source_node, compute_node)
            else:
                # Si on ne trouve pas, c'est une erreur
                raise MissingInputError(
                    f"Argument '{arg_name}' for unit '{step_name}' could not be resolved (Source: '{source_var}')."
                )

        compute_node.input_mapping = resolved_mapping
        self.graph.add_node(compute_node)

        # 4. Enregistrement de la Sortie
        output_node = DataNode(name=final_output_name)
        self.graph.add_node(output_node)
        self.graph.add_edge(compute_node, output_node)
        self.registered_variables.add(final_output_name)
        self._data_nodes[final_output_name] = output_node

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
            # On vérifie si cette variable est "connue" ou "produite" par le système
            # C'est délicat car l'ordre d'enregistrement compte.
            # Si l'unité A produit 'Tuna_biomass' et l'unité B (enregistrée après) la consomme,
            # registered_variables ne l'aura peut-être pas encore si on ne fait pas attention.
            # MAIS : register_unit est séquentiel. Donc si A est enregistré avant B, c'est bon.
            # Si B dépend de A, A doit être enregistré avant.
            if prefixed_name in self.registered_variables:
                return prefixed_name

            # Cas subtil : Peut-être que c'est une variable qui SERA produite par le groupe lui-même ?
            # Pour l'instant, on exige que la source soit déjà déclarée (Forçage ou Output précédent).

        # 3. Matching par défaut (Global)
        if arg_name in self.registered_variables:
            return arg_name

        return None

    def register_group(self, group_prefix: str, units: list[dict[str, Any]]) -> None:
        """Helper pour enregistrer un groupe d'unités."""
        previous_context = self._group_context
        self._group_context = group_prefix

        try:
            for unit_conf in units:
                self.register_unit(
                    func=unit_conf["func"],
                    input_mapping=unit_conf.get("input_mapping"),
                    output_name=unit_conf.get("output_name"),
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

        return ExecutionPlan(
            task_sequence=task_sequence,
            initial_variables=initial_vars,
            produced_variables=produced_vars,
        )
