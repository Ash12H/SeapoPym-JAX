# Plan d'Implémentation : Global State Manager (GSM)

Ce document détaille les étapes pour implémenter le composant **Global State Manager**, responsable de la structure de données centrale de la simulation (le `State`).

## 1. Objectifs

Le GSM n'est pas un acteur actif, mais un ensemble d'outils pour manipuler la "Vérité Terrain" de la simulation.
Il doit garantir :
*   **Structure Standardisée** : Toutes les données (biologie, physique) vivent dans un unique `xarray.Dataset`.
*   **Cohérence** : Alignement des dimensions (lat, lon, depth).
*   **Immutabilité** : Le passage de $t$ à $t+1$ se fait par création d'un nouvel état, facilitant le parallélisme (Zero-Copy reads).

## 2. Structure du Module

Création du package `seapopym.gsm`.

*   `seapopym/gsm/__init__.py` : Exports.
*   `seapopym/gsm/core.py` : Classe `StateManager`.
*   `seapopym/gsm/exceptions.py` : Erreurs spécifiques (StateValidationError).

## 3. Implémentation du StateManager

Dans `core.py`, implémenter la classe `StateManager`. C'est une classe utilitaire (méthodes statiques ou de classe) ou un singleton léger.

### 3.1. Création de l'État Initial
*   Méthode `create_initial_state(coords: dict, variables: dict) -> xr.Dataset`
    *   Initialise un `xr.Dataset` propre.
    *   Vérifie que les coordonnées sont valides.
    *   Ajoute les variables initiales.

### 3.2. Validation
*   Méthode `validate(state: xr.Dataset, required_vars: List[str])`
    *   Vérifie que toutes les variables demandées par le Blueprint (`ExecutionPlan.initial_variables`) sont présentes.
    *   Vérifie la cohérence des dimensions (ex: toutes les variables 2D ont bien (lat, lon)).

### 3.3. Gestion des Forçages
*   Méthode `merge_forcings(state: xr.Dataset, forcings: dict) -> xr.Dataset`
    *   Retourne un **nouveau** Dataset (ou une vue mise à jour) contenant l'état actuel + les forçages du jour.
    *   Attention à la performance : utiliser `xr.merge` ou l'assignation optimisée.

### 3.4. Opérations de Cycle de Vie
*   Méthode `initialize_next_step(current_state: xr.Dataset) -> xr.Dataset`
    *   Prépare le squelette pour $t+1$.
    *   Copie superficielle (`copy(deep=False)`) pour ne pas dupliquer les données statiques (bathymétrie, mask) inutilement, tout en permettant de modifier les variables d'état.

## 4. Tests Unitaires

Créer `tests/test_gsm.py` :
*   **Création** : Vérifier qu'on peut créer un state valide.
*   **Validation** : Vérifier qu'il rejette un state incomplet.
*   **Immutabilité** : Vérifier que `initialize_next_step` ne modifie pas l'original.
*   **Merge** : Vérifier l'intégration correcte des forçages.

## 5. Dépendances
*   `xarray`
*   `numpy`

## Ordre de Développement
1.  Structure de fichiers.
2.  `StateManager.create_initial_state`.
3.  `StateManager.validate`.
4.  Tests de base.
5.  Fonctions avancées (merge, next_step).
