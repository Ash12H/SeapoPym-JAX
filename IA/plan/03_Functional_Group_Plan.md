# Plan d'Implémentation : Functional Group

Ce document détaille l'implémentation du composant **Functional Group**, responsable de l'exécution de la logique scientifique (unités de calcul) pour un groupe donné (ex: Thon, Requin).

## 1. Objectifs

Le Functional Group est un "travailleur". Il reçoit un état global (ou une vue), exécute une séquence d'opérations définies par le Blueprint, et retourne les modifications à appliquer.

Il doit :
*   Être initialisé avec une séquence de tâches (`ComputeNode`).
*   Exécuter ces tâches dans l'ordre.
*   Gérer le flux de données entre les tâches (résultats intermédiaires).
*   Être indépendant de Ray (pour l'instant), mais conçu pour être encapsulé dans un Ray Actor plus tard.

## 2. Structure du Module

Création du package `seapopym.functional_group`.

*   `seapopym/functional_group/__init__.py` : Exports.
*   `seapopym/functional_group/core.py` : Classe `FunctionalGroup`.
*   `seapopym/functional_group/exceptions.py` : Erreurs spécifiques.

## 3. Implémentation de la Classe FunctionalGroup

### 3.1. Initialisation
```python
class FunctionalGroup:
    def __init__(self, name: str, task_sequence: list[ComputeNode]):
        self.name = name
        self.task_sequence = task_sequence
```

### 3.2. Méthode Compute
```python
def compute(self, state: xr.Dataset, tasks: list[ComputeNode] | None = None) -> dict[str, xr.DataArray]:
    """
    Exécute la séquence de tâches sur l'état donné.
    Si `tasks` est fourni, exécute cette sous-séquence. Sinon, utilise la séquence par défaut.
    Retourne un dictionnaire des variables produites (nom graphe -> DataArray).
    """
```

**Logique interne :**
1.  Sélectionner la séquence de tâches (argument `tasks` ou `self.task_sequence`).
2.  Créer un dictionnaire local `local_context` pour stocker les résultats intermédiaires.
3.  Pour chaque `node` dans la séquence :
    a.  **Résoudre les entrées** : Priorité au `local_context`, puis `state`.
    b.  **Exécuter** : Appeler `node.func(**inputs)`.
    c.  **Valider et Mapper** : Vérifier que la sortie est un `DataArray` (ou scalaire convertible) et la stocker.
4.  Retourner le dictionnaire des résultats.

## 4. Tests Unitaires

Créer `tests/test_functional_group.py` :
*   **Simple Execution** : Une seule unité, vérification de la sortie.
*   **Chained Execution** : Unit A -> Unit B. Vérifier que B reçoit la sortie de A.
*   **State Access** : Vérifier que les unités accèdent bien aux variables du `state` (ex: température).
*   **Output Mapping** : Vérifier que le mapping de sortie est respecté.

## 5. Dépendances
*   `xarray`
*   `seapopym.blueprint` (pour `ComputeNode`)

## Ordre de Développement
1.  Structure de fichiers.
2.  Implémentation de `FunctionalGroup`.
3.  Tests unitaires.
