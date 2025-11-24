# État de l'Architecture Seapopym

**Date :** 24 Novembre 2025
**Statut :** Architecture de base opérationnelle (Séquentielle).

## Composants Implémentés

### 1. Blueprint (`seapopym.blueprint`)
*   **Rôle** : Architecte et compilateur du modèle.
*   **Fonctionnalités** :
    *   Enregistrement des unités de calcul et forçages.
    *   Résolution des dépendances (Mapping explicite, Namespacing, Matching par défaut).
    *   Détection de cycles.
    *   **Compilation** : Produit un `ExecutionPlan` structuré en **groupes de tâches** (`task_groups`).
*   **Tests** : Couverture complète, validation des graphes complexes et entrelacés.

### 2. Global State Manager (GSM) (`seapopym.gsm`)
*   **Rôle** : Gestionnaire de l'état (`xarray.Dataset`).
*   **Fonctionnalités** :
    *   Création de l'état initial.
    *   Validation (variables, coordonnées).
    *   Immutabilité fonctionnelle (Merge, Copy).
*   **Tests** : Couverture complète.

### 3. Functional Group (`seapopym.functional_group`)
*   **Rôle** : Exécuteur de logique ("Worker").
*   **Fonctionnalités** :
    *   Exécution dynamique d'une liste de tâches (`compute(state, tasks=...)`).
    *   Validation stricte des sorties (doit être `xr.DataArray`).
    *   Prêt pour être encapsulé en Ray Actor.
*   **Tests** : Couverture complète.

### 4. Controller (`seapopym.controller`)
*   **Rôle** : Orchestrateur.
*   **Fonctionnalités** :
    *   Setup : Instancie les groupes uniques (Acteurs).
    *   Run : Boucle temporelle.
    *   Step : Itère sur le plan d'exécution et délègue aux groupes.
*   **Tests** : Validation du cycle de vie complet.

## Prochaines Étapes Critiques

### 1. Time Integrator (Priorité Haute)
*   **Problème actuel** : La mise à jour de l'état est un simple remplacement (`merge`). Il n'y a pas de gestion physique des flux (biomasse $t+1 = t + dt \times \text{tendency}$).
*   **Objectif** : Implémenter un composant qui reçoit les tendances de tous les groupes, résout les conflits (ex: biomasse négative), et met à jour l'état.

### 2. Ray Integration (Priorité Moyenne)
*   **Objectif** : Transformer `FunctionalGroup` en `@ray.remote`.
*   **Changement** : Le Controller devra gérer des `ObjectRef` et utiliser `ray.get`/`ray.wait`.

### 3. Forcing Manager (Priorité Basse)
*   **Objectif** : Charger dynamiquement les données climatiques (NetCDF) au fur et à mesure de la simulation (Lazy Loading / Streaming).
