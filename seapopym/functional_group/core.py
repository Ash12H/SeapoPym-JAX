"""Core implementation of the Functional Group."""
import warnings
from collections.abc import Hashable

import xarray as xr

from seapopym.blueprint.nodes import ComputeNode

from .exceptions import ExecutionError


class FunctionalGroup:
    """Représente un groupe fonctionnel (ex: une espèce) capable d'exécuter sa propre logique.

    Il orchestre l'exécution séquentielle des unités de calcul (ComputeNodes).
    """

    def __init__(self, name: str, task_sequence: list[ComputeNode] | None = None):
        """Initialise le groupe fonctionnel.

        Args:
            name: Nom du groupe (ex: 'Tuna').
            task_sequence: Liste ordonnée des tâches par défaut (optionnel).
        """
        self.name = name
        self.task_sequence = task_sequence or []

    def compute(
        self, state: xr.Dataset, tasks: list[ComputeNode] | None = None
    ) -> dict[Hashable, xr.DataArray]:
        """Exécute une séquence de tâches sur l'état donné.

        IMPORTANT: Cette méthode ne modifie PAS le state d'entrée (lecture seule).
        Les résultats intermédiaires sont stockés dans un contexte local.

        Args:
            state: L'état global (ou une vue) contenant les données nécessaires.
            tasks: Liste des tâches à exécuter. Si None, utilise la séquence par défaut du groupe.

        Returns:
            Dictionnaire {nom_variable_graphe: DataArray} de TOUS les résultats produits.

        Raises:
            ValueError: Si aucune tâche n'est fournie (ni dans tasks, ni dans self.task_sequence).
            ExecutionError: Si une erreur survient pendant l'exécution.
        """
        sequence = tasks if tasks is not None else self.task_sequence

        if not sequence:
            warnings.warn(
                f"FunctionalGroup '{self.name}' executed with empty task sequence",
                UserWarning,
                stacklevel=2,
            )
            return {}

        # Contexte local pour stocker les résultats intermédiaires
        # Cela permet aux unités suivantes d'utiliser les résultats des précédentes
        local_context: dict[Hashable, xr.DataArray] = {}

        results: dict[Hashable, xr.DataArray] = {}

        for node in sequence:
            try:
                # 1. Résolution des entrées
                inputs = {}
                for arg_name, graph_var_name in node.input_mapping.items():
                    if graph_var_name in local_context:
                        inputs[arg_name] = local_context[graph_var_name]
                    elif graph_var_name in state:
                        inputs[arg_name] = state[graph_var_name]
                    else:
                        # Cela ne devrait pas arriver si le Blueprint a bien fait son travail de validation
                        # Mais on garde une sécurité au runtime
                        raise KeyError(
                            f"Variable '{graph_var_name}' not found in state or local context."
                        )

                # 2. Exécution de la fonction
                # La fonction retourne un dictionnaire {key_retour: data}
                unit_output = node.func(**inputs)

                if not isinstance(unit_output, dict):
                    raise TypeError(
                        f"Function '{node.name}' must return a dictionary, got {type(unit_output)}."
                    )

                # 3. Mapping des sorties vers le graphe
                for key_retour, graph_var_name in node.output_mapping.items():
                    if key_retour not in unit_output:
                        raise KeyError(
                            f"Function '{node.name}' did not return expected key '{key_retour}'."
                        )

                    data = unit_output[key_retour]

                    if not isinstance(data, xr.DataArray):
                        raise TypeError(
                            f"Output '{key_retour}' from unit '{node.name}' must be a DataArray, "
                            f"got {type(data)}."
                        )

                    local_context[graph_var_name] = data
                    results[graph_var_name] = data

            except Exception as e:
                raise ExecutionError(f"Error executing unit '{node.name}': {str(e)}") from e

        return results
