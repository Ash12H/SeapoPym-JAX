"""Blueprint module for dependency graph construction and execution planning."""

import inspect
import itertools
from collections.abc import Callable
from functools import partial
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

    def register_forcing(
        self, name: str, dims: tuple[Any, ...] | None = None, units: str | None = None
    ) -> None:
        """Déclare une source de données externe (ex: température, courants).

        Raises:
            ConfigurationError: Si la variable est déjà enregistrée.

        """
        if name in self.registered_variables:
            raise ConfigurationError(f"Variable '{name}' is already registered.")

        node = DataNode(name=name, dims=dims, units=units)
        self.graph.add_node(node)
        self.registered_variables.add(name)
        self._data_nodes[name] = node

    def register_parameter(
        self, name: str, units: str | None = None, dims: tuple[Any, ...] | None = None
    ) -> None:
        """Déclare un paramètre du modèle (ex: taux de mortalité, coefficient).

        Args:
            name: Nom du paramètre.
            units: Unité attendue (ex: '1/day', 'm').
            dims: Dimensions optionnelles (si le paramètre est spatialisé).
        """
        if name in self.registered_variables:
            raise ConfigurationError(f"Variable '{name}' is already registered.")

        node = DataNode(name=name, dims=dims, units=units)
        self.graph.add_node(node)
        self.registered_variables.add(name)
        self._data_nodes[name] = node

    def register_unit(
        self,
        func: Callable[..., Any],
        output_mapping: dict[str, str],
        input_mapping: dict[str, str] | None = None,
        output_tendencies: dict[str, str] | None = None,
        output_units: dict[str, str] | None = None,
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
            output_units: Mapping {key_retour: unité} pour spécifier les unités des sorties.
                         Ex: {"output": "gC/m²/s"}. Les unités sont validées/converties par le Controller.
            scope: 'local' ou 'global'.
            name: Nom unique de l'étape (optionnel, défaut = nom fonction).

        Example:
            >>> def compute_mortality(biomass):
            ...     return {"rate": biomass * -0.1}
            >>> bp.register_unit(
            ...     compute_mortality,
            ...     output_mapping={"rate": "mortality_rate"},
            ...     output_tendencies={"rate": "biomass"},  # mortality_rate est une tendance de biomass
            ...     output_units={"rate": "gC/m²/s"}  # unité de la tendance
            ... )
        """
        if not output_mapping:
            raise ConfigurationError("output_mapping must be provided and not empty.")

        input_mapping = input_mapping or {}

        if isinstance(func, partial):
            func_name = func.func.__name__
        elif hasattr(func, "__name__"):
            func_name = func.__name__
        else:
            func_name = str(func)

        step_name = name or func_name

        if self._group_context:
            step_name = f"{self._group_context}/{step_name}"

        # Résolution des noms de sortie avec préfixe de groupe si nécessaire
        final_output_mapping = {}
        for key, var_name in output_mapping.items():
            if self._group_context:
                final_output_mapping[key] = f"{self._group_context}/{var_name}"
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
        output_units = output_units or {}

        for key, var_name in final_output_mapping.items():
            # Déterminer si c'est une tendance
            is_tendency_of = output_tendencies.get(key)
            if (
                is_tendency_of
                and is_tendency_of not in self.registered_variables
                and self._group_context
            ):
                # Résolution du nom de la cible de tendance
                # On utilise la même logique que pour les inputs :
                # 1. Si le nom existe tel quel (global), on le garde
                # 2. Sinon on le préfixe avec le groupe
                prefixed_target = f"{self._group_context}/{is_tendency_of}"
                # On préfère le nom préfixé si on est dans un groupe,
                # surtout pour les variables d'état locales.
                # Mais attention, la variable cible doit être déclarée (State ou Forcing).
                # Ici on fait juste la résolution de nom, la validation se fera au build().
                is_tendency_of = prefixed_target

            # Déterminer l'unité de sortie
            units = output_units.get(key)

            output_node = DataNode(name=var_name, is_tendency_of=is_tendency_of, units=units)
            self.graph.add_node(output_node)
            self.graph.add_edge(compute_node, output_node)
            self.registered_variables.add(var_name)
            self._data_nodes[var_name] = output_node

    def _resolve_input(self, arg_name: str, explicit_mapping: dict[str, str]) -> str | None:
        """Résout le nom de la variable dans le graphe selon la priorité.

        1. Mapping Explicite (Direct ou Préfixé)
        2. Namespacing (Groupe) sur arg_name
        3. Matching par défaut (Global) sur arg_name
        """
        # 1. Mapping Explicite
        if arg_name in explicit_mapping:
            mapped_name = explicit_mapping[arg_name]
            # Si le nom mappé existe tel quel (ex: forcing global), on le prend
            if mapped_name in self.registered_variables:
                return mapped_name

            # Sinon, si on est dans un groupe, on essaie de le préfixer
            if self._group_context:
                prefixed_mapped = f"{self._group_context}/{mapped_name}"
                if prefixed_mapped in self.registered_variables:
                    return prefixed_mapped

            # Si toujours pas trouvé, on retourne le nom mappé (pour l'erreur)
            return mapped_name

        # 2. Namespacing (sur le nom de l'argument)
        if self._group_context:
            prefixed_name = f"{self._group_context}/{arg_name}"
            if prefixed_name in self.registered_variables:
                return prefixed_name

        # 3. Matching par défaut (Global)
        if arg_name in self.registered_variables:
            return arg_name

        return None

    def register_group(
        self,
        group_prefix: str,
        units: list[dict[str, Any]],
        parameters: dict[str, dict[str, Any]] | None = None,
        state_variables: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Helper pour enregistrer un groupe d'unités.

        Args:
            group_prefix: Le préfixe du groupe (ex: 'Tuna').
            units: Liste de dicts de configuration pour register_unit.
            parameters: Dictionnaire de définition des paramètres du groupe.
                        Format: {param_name: {units: '...', dims: ...}}
                        Ces paramètres seront automatiquement enregistrés avec le préfixe du groupe.
            state_variables: Variables d'état maintenues par ce groupe.
                        Format: {var_name: {"dims": (...), "units": "..."}}
                        Ces variables sont persistantes et doivent être initialisées.
        """
        previous_context = self._group_context
        self._group_context = group_prefix

        try:
            # 1. Enregistrement automatique des paramètres du groupe
            if parameters:
                for param_name, param_spec in parameters.items():
                    full_name = f"{group_prefix}/{param_name}"
                    self.register_parameter(
                        name=full_name,
                        units=param_spec.get("units"),
                        dims=param_spec.get("dims"),
                    )

            # 2. Enregistrement des variables d'état explicites
            if state_variables:
                for var_name, var_spec in state_variables.items():
                    full_name = f"{group_prefix}/{var_name}"
                    if full_name in self.registered_variables:
                        raise ConfigurationError(
                            f"State variable '{full_name}' is already registered."
                        )

                    node = DataNode(
                        name=full_name,
                        dims=var_spec.get("dims"),
                        units=var_spec.get("units"),
                        is_state=True,
                    )
                    self.graph.add_node(node)
                    self.registered_variables.add(full_name)
                    self._data_nodes[full_name] = node

            # 3. Enregistrement des unités
            for unit_conf in units:
                func = unit_conf["func"]

                # Note: On a supprimé l'injection automatique via partial (parameters=...)
                # car maintenant les paramètres sont dans le State.
                # Le mapping se fera via _resolve_input qui cherchera {group_prefix}_{arg_name}

                self.register_unit(
                    func=func,
                    output_mapping=unit_conf["output_mapping"],
                    input_mapping=unit_conf.get("input_mapping"),
                    output_tendencies=unit_conf.get("output_tendencies"),
                    output_units=unit_conf.get("output_units"),
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
        # OU DataNodes marqués comme is_state (racines pour le pas de temps courant)
        initial_vars = [
            node.name
            for node in self.graph.nodes
            if isinstance(node, DataNode) and (self.graph.in_degree(node) == 0 or node.is_state)
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

        # 5. Validation stricte des tendances
        # Toute variable cible d'une tendance DOIT être une variable d'état déclarée
        declared_states = self.get_state_variables()
        tendency_targets = set(tendency_map.keys())

        invalid_targets = tendency_targets - declared_states
        if invalid_targets:
            raise ConfigurationError(
                f"The following variables are tendency targets but not declared as state variables: {invalid_targets}. "
                f"Declare them in register_group(..., state_variables={{...}})"
            )

        return ExecutionPlan(
            task_groups=task_groups,
            initial_variables=initial_vars,
            produced_variables=produced_vars,
            tendency_map=dict(tendency_map),
        )

    def visualize(
        self,
        figsize: tuple[int, int] = (14, 10),
        title: str = "Blueprint Dependency Graph",
        layout: str = "hierarchical",
    ) -> Any:
        """Visualise le graphe de dépendances du Blueprint.

        Args:
            figsize: Taille de la figure matplotlib.
            title: Titre du graphe.
            layout: Type de layout ('hierarchical', 'spring', 'kamada_kawai', 'circular').

        Returns:
            Figure matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for visualization. Install it with: pip install matplotlib"
            ) from None

        if not self.graph.nodes:
            raise ConfigurationError("Blueprint is empty. Register units before visualizing.")

        fig, ax = plt.subplots(figsize=figsize)

        # Layout selection
        if layout == "hierarchical":
            pos = self._hierarchical_layout()
        elif layout == "spring":
            pos = nx.spring_layout(self.graph, seed=42, k=2)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self.graph)
        elif layout == "circular":
            pos = nx.circular_layout(self.graph)
        else:
            raise ValueError(
                f"Unknown layout: {layout}. Use 'hierarchical', 'spring', 'kamada_kawai', or 'circular'."
            )

        # Séparation des noeuds par type
        data_nodes = [n for n in self.graph.nodes if isinstance(n, DataNode)]
        compute_nodes = [n for n in self.graph.nodes if isinstance(n, ComputeNode)]

        # Distinguer les variables initiales des produites
        initial_data_nodes = []
        produced_data_nodes = []
        tendency_data_nodes = []

        for node in data_nodes:
            if self.graph.in_degree(node) == 0:
                initial_data_nodes.append(node)
            elif node.is_tendency_of:
                tendency_data_nodes.append(node)
            else:
                produced_data_nodes.append(node)

        # Dessin des noeuds
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            nodelist=initial_data_nodes,
            node_color="lightgreen",
            node_shape="o",
            node_size=800,
            label="Data (Initial)",
            ax=ax,
        )
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            nodelist=produced_data_nodes,
            node_color="lightblue",
            node_shape="o",
            node_size=800,
            label="Data (Produced)",
            ax=ax,
        )
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            nodelist=tendency_data_nodes,
            node_color="yellow",
            node_shape="o",
            node_size=800,
            label="Tendency",
            ax=ax,
        )
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            nodelist=compute_nodes,
            node_color="orange",
            node_shape="s",
            node_size=1000,
            label="Compute",
            ax=ax,
        )

        # Edges
        nx.draw_networkx_edges(
            self.graph, pos, arrowstyle="->", arrowsize=20, edge_color="gray", width=2, ax=ax
        )

        # Labels
        labels = {n: n.name for n in self.graph.nodes}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=9, font_weight="bold", ax=ax)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", fontsize=10)
        ax.axis("off")
        plt.tight_layout()

        return fig

    def _hierarchical_layout(self) -> dict[Any, Any]:
        """Calcule un layout hiérarchique basé sur les niveaux topologiques.

        Les noeuds sont placés de haut en bas selon leur niveau de dépendance.

        Returns:
            dict: Positions {node: (x, y)}
        """
        # Calcul des niveaux (distance depuis les sources)
        levels = {}
        for node in nx.topological_sort(self.graph):
            if self.graph.in_degree(node) == 0:
                levels[node] = 0
            else:
                levels[node] = max(levels[pred] for pred in self.graph.predecessors(node)) + 1

        # Regroupement par niveau
        level_nodes: dict[int, list] = {}
        for node, level in levels.items():
            if level not in level_nodes:
                level_nodes[level] = []
            level_nodes[level].append(node)

        # Calcul des positions
        pos = {}
        max_level = max(levels.values())
        for level, nodes in level_nodes.items():
            y = 1 - (level / max_level)  # De haut (1) en bas (0)
            num_nodes = len(nodes)
            for i, node in enumerate(nodes):
                x = (i + 1) / (num_nodes + 1)  # Centré horizontalement
                pos[node] = (x, y)

        return pos

    def get_state_variables(self) -> set[str]:
        """Return the set of all declared state variable names."""
        return {name for name, node in self._data_nodes.items() if node.is_state}
