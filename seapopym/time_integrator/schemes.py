"""Schémas d'intégration numérique pour le Time Integrator."""

import xarray as xr


def euler_forward(
    state: xr.Dataset,
    all_results: dict[str, xr.DataArray],
    tendency_map: dict[str, list[str]],
    dt: float,
) -> xr.Dataset:
    """Schéma d'Euler explicite : S(t+dt) = S(t) + dt * sum(tendances).

    Args:
        state: État actuel.
        all_results: Tous les résultats des groupes (tendances + autres variables).
        tendency_map: Mapping {var_cible: [liste_tendances]}.
        dt: Pas de temps (en secondes).

    Returns:
        Nouvel état.
    """
    new_state = state.copy()

    for target_var, tendency_names in tendency_map.items():
        if target_var in state:
            # Collecter les tendances disponibles
            tendencies = [all_results[t] for t in tendency_names if t in all_results]

            if tendencies:
                # Somme de toutes les tendances affectant cette variable
                total_tendency = sum(tendencies)
                new_state[target_var] = state[target_var] + dt * total_tendency
            # Sinon, la variable reste inchangée (pas de tendance appliquée)

    return new_state
