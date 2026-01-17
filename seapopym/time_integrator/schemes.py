"""Schémas d'intégration numérique pour le Time Integrator."""

import xarray as xr


def euler_forward(
    state: xr.Dataset,
    all_results: dict[str, xr.DataArray],
    tendency_map: dict[str, list[str]],
    dt: float,
) -> xr.Dataset:
    """Schéma d'Euler explicite (optimisé) : S(t+dt) = S(t) + dt * sum(tendances).

    Optimizations:
    - Avoids full state.copy() by using state.assign() at the end
    - Avoids creating intermediate list of tendencies
    - Uses incremental sum instead of Python sum() over list

    Args:
        state: État actuel.
        all_results: Tous les résultats des groupes (tendances + autres variables).
        tendency_map: Mapping {var_cible: [liste_tendances]}.
        dt: Pas de temps (en secondes).

    Returns:
        Nouvel état.
    """
    # Collect all updates to apply at once (avoids multiple Dataset copies)
    updates: dict[str, xr.DataArray] = {}

    for target_var, tendency_names in tendency_map.items():
        if target_var not in state:
            continue

        # Incremental sum without building a list
        total_tendency: xr.DataArray | None = None
        for t_name in tendency_names:
            if t_name in all_results:
                if total_tendency is None:
                    total_tendency = all_results[t_name]
                else:
                    # In-place style addition (xarray handles this efficiently)
                    total_tendency = total_tendency + all_results[t_name]

        if total_tendency is not None:
            updates[target_var] = state[target_var] + dt * total_tendency

    # Single assignment operation (more efficient than repeated __setitem__)
    if updates:
        return state.assign(updates)
    return state
