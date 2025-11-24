"""Core implementation of the Time Integrator."""

import xarray as xr

from .constraints import enforce_positivity
from .schemes import euler_forward


class TimeIntegrator:
    """Responsable de l'intégration temporelle de l'état."""

    def __init__(
        self,
        scheme: str = "euler",
        positive_vars: list[str] | None = None,
    ):
        """Initialise le Time Integrator.

        Args:
            scheme: Nom du schéma d'intégration ('euler', 'rk4', etc.).
            positive_vars: Variables à contraindre >= 0 (ex: biomasses).
        """
        self.scheme = scheme
        self.positive_vars = positive_vars or []

    def integrate(
        self,
        state: xr.Dataset,
        all_results: dict[str, xr.DataArray],
        tendency_map: dict[str, list[str]],
        dt: float,
    ) -> xr.Dataset:
        """Fait évoluer l'état de t à t+dt.

        Args:
            state: État à t.
            all_results: Tous les résultats calculés (tendances + diagnostics).
            tendency_map: Mapping des tendances vers leurs variables cibles.
            dt: Pas de temps (en secondes).

        Returns:
            État à t+dt.
        """
        # 1. Application du schéma numérique
        if self.scheme == "euler":
            new_state = euler_forward(state, all_results, tendency_map, dt)
        else:
            raise ValueError(f"Unknown scheme: {self.scheme}")

        # 2. Application des contraintes
        new_state = enforce_positivity(new_state, self.positive_vars)

        return new_state
