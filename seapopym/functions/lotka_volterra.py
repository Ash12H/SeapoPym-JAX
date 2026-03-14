"""Lotka-Volterra predator-prey functions.

Classic two-species ODE system:
    dN/dt = alpha * N - beta * N * P
    dP/dt = delta * beta * N * P - gamma * P

All functions use the @functional decorator for Blueprint integration
and support automatic differentiation via JAX.
"""

from __future__ import annotations

from seapopym.blueprint import functional


@functional(
    name="lv:prey_growth",
    units={"N": "dimensionless", "alpha": "1/s", "return": "1/s"},
)
def prey_growth(N, alpha):
    """Prey exponential growth: +alpha * N."""
    return alpha * N


@functional(
    name="lv:predation",
    units={
        "N": "dimensionless",
        "P": "dimensionless",
        "beta": "1/s",
        "delta": "dimensionless",
        "prey_loss": "1/s",
        "predator_gain": "1/s",
    },
    outputs=["prey_loss", "predator_gain"],
)
def predation(N, P, beta, delta):
    """Predation interaction: prey loses beta*N*P, predator gains delta*beta*N*P."""
    interaction = beta * N * P
    return -interaction, delta * interaction


@functional(
    name="lv:predator_death",
    units={"P": "dimensionless", "gamma": "1/s", "return": "1/s"},
)
def predator_death(P, gamma):
    """Predator natural mortality: -gamma * P."""
    return -gamma * P
