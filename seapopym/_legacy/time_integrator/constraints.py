"""Gestion des contraintes pour le Time Integrator."""

import xarray as xr


def enforce_positivity(state: xr.Dataset, variables: list[str]) -> xr.Dataset:
    """Force les variables spécifiées à être >= 0.

    Args:
        state: L'état à contraindre.
        variables: Liste des variables qui doivent être positives (ex: biomasses).

    Returns:
        État avec contraintes appliquées.
    """
    constrained_state = state.copy()

    for var_name in variables:
        if var_name in constrained_state:
            constrained_state[var_name] = constrained_state[var_name].clip(min=0)

    return constrained_state
