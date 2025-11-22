"""Blueprint: Graph-based dependency manager for the simulation.

This module provides the Blueprint class, which uses NetworkX to build, validate,
and order the execution of computational Units based on their data dependencies.
"""

import networkx as nx

from seapopym_message.core.unit import Unit


class Blueprint:
    """Dependency graph manager for simulation units.

    The Blueprint constructs a Directed Acyclic Graph (DAG) where nodes are either
    Variables (state/forcings) or Units (computations). Edges represent data flow:
    - Variable -> Unit: The unit requires this variable as input.
    - Unit -> Variable: The unit produces this variable as output.

    This structure allows for:
    1. Automatic topological sorting (execution order).
    2. Cycle detection (preventing infinite loops).
    3. Visualization of the model structure.
    """

    def __init__(self) -> None:
        """Initialize an empty Blueprint."""
        self.graph: nx.DiGraph = nx.DiGraph()
        self.units: dict[str, Unit] = {}

    def add_unit(self, unit: Unit) -> None:
        """Add a Unit to the dependency graph.

        Args:
            unit: The Unit instance to add.

        Raises:
            ValueError: If a unit with the same name already exists.
        """
        if unit.name in self.units:
            raise ValueError(f"Unit with name '{unit.name}' already exists in Blueprint.")

        self.units[unit.name] = unit

        # Add Unit node
        self.graph.add_node(unit.name, type="unit", obj=unit)

        # Add Input dependencies (Variable -> Unit)
        for input_name in unit.inputs:
            self.graph.add_node(input_name, type="variable")
            self.graph.add_edge(input_name, unit.name)

        # Add Forcing dependencies (Variable -> Unit)
        # Forcings are treated as external variables that must exist
        for forcing_name in unit.forcings:
            self.graph.add_node(forcing_name, type="variable", is_forcing=True)
            self.graph.add_edge(forcing_name, unit.name)

        # Add Output dependencies (Unit -> Variable)
        for output_name in unit.outputs:
            self.graph.add_node(output_name, type="variable")
            self.graph.add_edge(unit.name, output_name)

    def build(self) -> list[Unit]:
        """Validate the graph and return the execution order of Units.

        Returns:
            List of Units in topologically sorted order.

        Raises:
            ValueError: If the graph contains cycles (circular dependencies).
        """
        if not nx.is_directed_acyclic_graph(self.graph):
            # Find the cycle to report a helpful error
            try:
                cycle = nx.find_cycle(self.graph)
                raise ValueError(f"Dependency cycle detected in Blueprint: {cycle}")
            except nx.NetworkXNoCycle as e:
                # Should not happen if is_directed_acyclic_graph returned False
                raise ValueError("Dependency cycle detected (unknown path).") from e

        # Topological sort gives us a linear ordering of nodes (Variables and Units)
        sorted_nodes = list(nx.topological_sort(self.graph))

        # Filter to keep only Units, preserving the topological order
        sorted_units = []
        for node_name in sorted_nodes:
            if node_name in self.units:
                sorted_units.append(self.units[node_name])

        return sorted_units

    def visualize(self) -> str:
        """Generate a Graphviz DOT string representation of the blueprint.

        Returns:
            String containing the DOT definition.
        """
        lines = ["digraph Blueprint {"]
        lines.append("  rankdir=LR;")
        lines.append('  node [fontname="Helvetica"];')
        lines.append('  edge [fontname="Helvetica"];')

        # Style for Units
        lines.append('  node [shape=box, style=filled, fillcolor="#E0F7FA", color="#006064"];')
        for unit_name in self.units:
            lines.append(f'  "{unit_name}";')

        # Style for Variables
        lines.append('  node [shape=ellipse, style=filled, fillcolor="#FFF3E0", color="#E65100"];')
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "variable":
                label = node
                if data.get("is_forcing"):
                    label += " (Forcing)"
                    lines.append(
                        f'  "{node}" [label="{label}", fillcolor="#E8F5E9", color="#1B5E20"];'
                    )
                else:
                    lines.append(f'  "{node}" [label="{label}"];')

        # Edges
        for u, v in self.graph.edges():
            lines.append(f'  "{u}" -> "{v}";')

        lines.append("}")
        return "\n".join(lines)

    def get_required_inputs(self) -> set[str]:
        """Identify variables that are required but not produced by any unit.

        These are typically initial conditions or external forcings.

        Returns:
            Set of variable names.
        """
        required = set()
        for node, data in self.graph.nodes(data=True):
            # If in-degree is 0, it's not produced by any unit in the graph
            if data.get("type") == "variable" and self.graph.in_degree(node) == 0:
                required.add(node)
        return required
