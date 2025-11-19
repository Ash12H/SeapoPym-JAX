"""Visualization utilities for simulation results.

This module provides functions to visualize diagnostics time series,
spatial fields, and create animations.
"""

from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure

from seapopym_message.utils.grid import SphericalGridInfo


def plot_diagnostics_timeseries(
    diagnostics: list[dict[str, Any]],
    variable: str = "biomass",
    figsize: tuple[float, float] = (10, 6),
) -> Figure:
    """Plot time series of diagnostics.

    Args:
        diagnostics: List of diagnostic dictionaries from simulation.
        variable: Variable to plot (e.g., "biomass", "temperature").
        figsize: Figure size (width, height) in inches.

    Returns:
        Matplotlib figure.

    Example:
        >>> diagnostics = run_simulation(workers, dt=0.1, t_max=10.0)
        >>> fig = plot_diagnostics_timeseries(diagnostics, variable="biomass")
        >>> plt.show()
    """
    # Extract time and values
    times = [d["t"] for d in diagnostics]
    mean_values = [d[f"{variable}_global_mean"] for d in diagnostics]
    min_values = [d[f"{variable}_global_min"] for d in diagnostics]
    max_values = [d[f"{variable}_global_max"] for d in diagnostics]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot mean with min/max envelope
    ax.plot(times, mean_values, "b-", linewidth=2, label="Mean")
    ax.fill_between(times, min_values, max_values, alpha=0.3, color="b", label="Min-Max range")

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel(f"{variable.capitalize()}", fontsize=12)
    ax.set_title(f"{variable.capitalize()} Evolution", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_spatial_field(
    field: jnp.ndarray,
    grid: SphericalGridInfo,
    title: str = "Spatial Field",
    cmap: str = "viridis",
    figsize: tuple[float, float] = (10, 6),
    vmin: float | None = None,
    vmax: float | None = None,
) -> Figure:
    """Plot a 2D spatial field.

    Args:
        field: 2D array (nlat, nlon) to plot.
        grid: Grid information for coordinates.
        title: Plot title.
        cmap: Colormap name.
        figsize: Figure size (width, height) in inches.
        vmin: Minimum value for colorbar (default: field min).
        vmax: Maximum value for colorbar (default: field max).

    Returns:
        Matplotlib figure.

    Example:
        >>> final_state = get_global_state(workers, patches)
        >>> fig = plot_spatial_field(
        ...     final_state["biomass"], grid, title="Final Biomass"
        ... )
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get coordinates
    lon = grid.lon_coords
    lat = grid.lat_coords

    # Create meshgrid for pcolormesh
    LON, LAT = np.meshgrid(lon, lat)

    # Plot
    im = ax.pcolormesh(
        LON,
        LAT,
        np.array(field),
        cmap=cmap,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(title, fontsize=12)

    # Labels
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Aspect ratio
    ax.set_aspect("equal")

    plt.tight_layout()
    return fig


def plot_convergence(
    diagnostics: list[dict[str, Any]],
    expected_value: float,
    variable: str = "biomass",
    figsize: tuple[float, float] = (10, 6),
) -> Figure:
    """Plot convergence to expected equilibrium.

    Args:
        diagnostics: List of diagnostic dictionaries.
        expected_value: Expected equilibrium value.
        variable: Variable to plot.
        figsize: Figure size.

    Returns:
        Matplotlib figure.

    Example:
        >>> diagnostics = run_simulation(workers, dt=0.1, t_max=50.0)
        >>> fig = plot_convergence(diagnostics, expected_value=100.0)
        >>> plt.show()
    """
    times = [d["t"] for d in diagnostics]
    mean_values = [d[f"{variable}_global_mean"] for d in diagnostics]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Top panel: actual values
    ax1.plot(times, mean_values, "b-", linewidth=2, label="Simulated")
    ax1.axhline(
        expected_value, color="r", linestyle="--", linewidth=2, label="Expected equilibrium"
    )
    ax1.set_ylabel(f"{variable.capitalize()}", fontsize=12)
    ax1.set_title("Convergence to Equilibrium", fontsize=14, fontweight="bold")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Bottom panel: relative error
    relative_error = [abs(v - expected_value) / expected_value * 100 for v in mean_values]
    ax2.semilogy(times, relative_error, "g-", linewidth=2)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.set_ylabel("Relative Error (%)", fontsize=12)
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    return fig


def create_animation(
    states: list[dict[str, jnp.ndarray]],
    grid: SphericalGridInfo,
    variable: str = "biomass",
    interval: int = 100,
    figsize: tuple[float, float] = (10, 6),
    cmap: str = "viridis",
) -> FuncAnimation:
    """Create animation of spatial field evolution.

    Args:
        states: List of state dictionaries (one per timestep).
        grid: Grid information.
        variable: Variable to animate.
        interval: Milliseconds between frames.
        figsize: Figure size.
        cmap: Colormap.

    Returns:
        Matplotlib FuncAnimation object.

    Example:
        >>> # Collect states during simulation
        >>> states = []
        >>> for i in range(0, len(diagnostics), 10):  # Every 10 steps
        ...     state = get_global_state(workers, patches)
        ...     states.append(state)
        >>> anim = create_animation(states, grid, variable="biomass")
        >>> anim.save("biomass_evolution.mp4")
    """
    lon = grid.lon_coords
    lat = grid.lat_coords
    LON, LAT = np.meshgrid(lon, lat)

    # Determine global min/max for consistent colorbar
    all_values = [np.array(state[variable]) for state in states]
    vmin = min(v.min() for v in all_values)
    vmax = max(v.max() for v in all_values)

    fig, ax = plt.subplots(figsize=figsize)

    def update(frame: int) -> tuple:
        ax.clear()
        field = np.array(states[frame][variable])

        im = ax.pcolormesh(LON, LAT, field, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)

        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)
        ax.set_title(f"{variable.capitalize()} - Frame {frame}", fontsize=14, fontweight="bold")
        ax.set_aspect("equal")

        return (im,)

    anim = FuncAnimation(fig, update, frames=len(states), interval=interval, blit=False)

    return anim


def plot_multi_variable(
    diagnostics: list[dict[str, Any]],
    variables: list[str],
    figsize: tuple[float, float] = (12, 8),
) -> Figure:
    """Plot multiple variables in subplots.

    Args:
        diagnostics: List of diagnostic dictionaries.
        variables: List of variable names to plot.
        figsize: Figure size.

    Returns:
        Matplotlib figure.

    Example:
        >>> diagnostics = run_simulation(workers, dt=0.1, t_max=10.0)
        >>> fig = plot_multi_variable(diagnostics, ["biomass", "R", "mortality_rate"])
        >>> plt.show()
    """
    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 1, figsize=figsize, sharex=True)

    if n_vars == 1:
        axes = [axes]

    times = [d["t"] for d in diagnostics]

    for ax, var in zip(axes, variables, strict=True):
        mean_key = f"{var}_global_mean"
        min_key = f"{var}_global_min"
        max_key = f"{var}_global_max"

        if mean_key in diagnostics[0]:
            mean_values = [d[mean_key] for d in diagnostics]
            min_values = [d[min_key] for d in diagnostics]
            max_values = [d[max_key] for d in diagnostics]

            ax.plot(times, mean_values, "b-", linewidth=2, label="Mean")
            ax.fill_between(times, min_values, max_values, alpha=0.3, color="b")

            ax.set_ylabel(var.capitalize(), fontsize=12)
            ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time", fontsize=12)
    fig.suptitle("Multi-Variable Evolution", fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig
