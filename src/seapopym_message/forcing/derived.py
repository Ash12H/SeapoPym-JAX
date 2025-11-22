"""Derived forcings: Compute new forcings from base forcings.

The @derived_forcing decorator allows users to create custom forcings
that are computed from base forcings (loaded from files) or other
derived forcings.

This enables:
- Modular forcing definitions (like @unit for kernels)
- Dependency resolution (automatic ordering)
- Extensibility (users can add custom forcings without modifying core)

Example:
    >>> @derived_forcing(
    ...     name="recruitment",
    ...     inputs=["primary_production"],
    ...     params=["transfer_coefficient", "day_of_year"],
    ... )
    ... def compute_recruitment(primary_production, transfer_coefficient, day_of_year):
    ...     seasonal_factor = 1.0 + 0.3 * jnp.sin(2 * jnp.pi * day_of_year / 365)
    ...     return primary_production * transfer_coefficient * seasonal_factor
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class DerivedForcing:
    """Metadata for a derived forcing.

    A derived forcing is a function that computes a new forcing variable
    from one or more input forcings and parameters.

    Args:
        name: Name of the derived forcing.
        inputs: List of input forcing names (dependencies).
        params: List of parameter names required for computation.
        func: The computation function.

    Example:
        >>> def compute_rec(pp, coef):
        ...     return pp * coef
        >>> derived = DerivedForcing(
        ...     name="recruitment",
        ...     inputs=["primary_production"],
        ...     params=["transfer_coefficient"],
        ...     func=compute_rec,
        ... )
    """

    name: str
    inputs: list[str]
    params: list[str]
    func: Callable

    def compute(self, forcings: dict[str, Any], params: dict[str, Any]) -> Any:
        """Compute the derived forcing.

        Args:
            forcings: Dictionary of available forcings (base + derived).
            params: Dictionary of parameters.

        Returns:
            Computed forcing (usually xarray.DataArray or jnp.ndarray).

        Raises:
            KeyError: If required inputs or params are missing.
        """
        # Gather input forcings
        input_values = {name: forcings[name] for name in self.inputs}

        # Gather parameters
        param_values = {name: params[name] for name in self.params}

        # Call the function with inputs and params as kwargs
        result = self.func(**input_values, **param_values)

        return result

    def __repr__(self) -> str:
        """String representation."""
        return f"DerivedForcing(name='{self.name}', " f"inputs={self.inputs}, params={self.params})"


def derived_forcing(
    name: str,
    inputs: list[str],
    params: list[str] | None = None,
) -> Callable:
    """Decorator to create a derived forcing.

    This decorator wraps a function to create a DerivedForcing instance.
    Similar to @unit for kernels, but for forcing computations.

    Args:
        name: Name of the derived forcing.
        inputs: List of input forcing names (dependencies).
        params: List of parameter names (optional).

    Returns:
        Decorator function.

    Example:
        >>> @derived_forcing(
        ...     name="recruitment",
        ...     inputs=["primary_production"],
        ...     params=["transfer_coefficient"],
        ... )
        ... def compute_recruitment(primary_production, transfer_coefficient):
        ...     return primary_production * transfer_coefficient
        ...
        >>> # compute_recruitment is now a DerivedForcing instance
        >>> compute_recruitment.name
        'recruitment'
    """
    if params is None:
        params = []

    def decorator(func: Callable) -> DerivedForcing:
        """Wrap function as DerivedForcing."""
        derived = DerivedForcing(
            name=name,
            inputs=inputs,
            params=params,
            func=func,
        )

        # Attach original function as attribute for introspection
        derived.func = func

        return derived

    return decorator


def resolve_dependencies(derived_forcings: dict[str, DerivedForcing]) -> list[str]:
    """Resolve dependency order for derived forcings.

    Uses topological sort to determine the order in which derived forcings
    should be computed, ensuring all dependencies are available.

    Args:
        derived_forcings: Dictionary mapping names to DerivedForcing instances.

    Returns:
        List of forcing names in dependency order.

    Raises:
        ValueError: If circular dependencies are detected.

    Example:
        >>> forcings = {
        ...     "recruitment": DerivedForcing(
        ...         name="recruitment",
        ...         inputs=["primary_production"],
        ...         params=[],
        ...         func=lambda pp: pp * 0.1,
        ...     ),
        ...     "biomass_growth": DerivedForcing(
        ...         name="biomass_growth",
        ...         inputs=["recruitment", "temperature"],
        ...         params=[],
        ...         func=lambda r, t: r * t,
        ...     ),
        ... }
        >>> order = resolve_dependencies(forcings)
        >>> order
        ['recruitment', 'biomass_growth']
    """
    # Build dependency graph
    graph: dict[str, list[str]] = {}
    in_degree: dict[str, int] = {}

    for name in derived_forcings:
        graph[name] = []
        in_degree[name] = 0

    # Count dependencies (only from other derived forcings)
    for name, forcing in derived_forcings.items():
        for dep in forcing.inputs:
            if dep in derived_forcings:
                graph[dep].append(name)
                in_degree[name] += 1

    # Topological sort (Kahn's algorithm)
    queue = [name for name, degree in in_degree.items() if degree == 0]
    sorted_order = []

    while queue:
        current = queue.pop(0)
        sorted_order.append(current)

        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check for cycles
    if len(sorted_order) != len(derived_forcings):
        remaining = [name for name in derived_forcings if name not in sorted_order]
        msg = f"Circular dependency detected in derived forcings: {remaining}"
        raise ValueError(msg)

    return sorted_order
