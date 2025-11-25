# Plan d'Intégration Dask

## Objectif
Implémenter `DaskBackend` pour permettre l'exécution distribuée des groupes de tâches via `dask.delayed`.

## 1. Architecture

### 1.1. Réutilisation de la Logique Core
Nous réutiliserons `seapopym.backend.core.execute_task_sequence` qui est déjà une fonction pure et testée.

### 1.2. Le `DaskBackend`
Il utilisera `dask.delayed` pour construire un graphe de tâches.

```python
import dask

class DaskBackend(ComputeBackend):
    def execute(self, task_groups, state):
        # Liste pour stocker les objets Delayed
        delayed_groups = []

        # Pour gérer les dépendances séquentielles entre groupes,
        # on doit passer le résultat du groupe précédent au suivant.
        # Mais execute_task_sequence attend un dict complet.

        # Approche :
        # On construit une chaîne de dépendances.
        # Chaque groupe prend en entrée le résultat fusionné des précédents.

        accumulated_results = {} # Au départ vide (ou delayed vide)

        # Problème : dask.delayed sur un dict mutable est délicat.
        # Solution : On définit une fonction wrapper qui fait la fusion ET l'exécution.

        previous_result = None

        results_list = []

        for group_name, tasks in task_groups:
            # On crée une tâche delayed
            # Elle dépend implicitement de previous_result si on le passe en argument
            current_result = dask.delayed(execute_wrapper)(
                tasks, state, previous_result
            )

            results_list.append(current_result)
            previous_result = current_result # Le prochain dépendra de celui-ci

        # Exécution parallèle
        # Note : Si on chaîne strictement (previous_result), on perd le parallélisme ENTRE groupes.
        # Mais on a dit que les groupes étaient séquentiels (dépendances topologiques).
        # Le parallélisme viendra de Dask si on arrive à exprimer que G1 et G2 sont indépendants.

        # MAIS : Le Controller nous donne une liste ORDONNÉE (topologique).
        # Si G1 et G2 sont indépendants, ils peuvent tourner en parallèle.
        # Comment le savoir ? Le Controller a linéarisé le graphe.

        # Si on veut du parallélisme, il faut que execute_task_sequence n'attende pas TOUT le contexte,
        # mais seulement ce dont il a besoin.
        # Or, on ne sait pas ce dont il a besoin sans analyser les inputs des tâches.

        # Pour l'instant, on va implémenter une exécution "Lazy" qui respecte l'ordre donné.
        # L'avantage de Dask ici sera surtout de pouvoir décharger le calcul sur un cluster Dask
        # et de gérer les gros calculs (chunked arrays) efficacement.

        final_results = dask.compute(results_list)[0]

        # Fusion finale
        all_results = {}
        for res in final_results:
            all_results.update(res)

        return all_results
```

### 1.3. Optimisation du Parallélisme
Pour permettre à Dask de paralléliser `G1` et `G2` s'ils sont indépendants (même si la liste est ordonnée), il faudrait passer à chaque groupe **uniquement** les dépendances dont il a besoin.
Mais `execute_task_sequence` prend tout le contexte.

**Amélioration future** : Analyser les `input_mapping` des tâches du groupe pour extraire les clés nécessaires, et ne passer que celles-là (sous forme de `dask.delayed`). Ainsi Dask verra que G2 ne dépend pas de G1.

Pour cette première version, on va faire simple :
1.  On lance tous les groupes via `dask.delayed`.
2.  On leur passe `accumulated_results` (qui sera résolu au moment du compute).
    *   Attention : Si on passe un dict mutable qui se remplit au fur et à mesure, ça casse le modèle fonctionnel de Dask.
    *   Il vaut mieux que chaque groupe retourne son petit dict de résultats, et qu'on fusionne tout à la fin.
    *   Mais pour les dépendances ?

    **Approche Robuste :**
    On passe à chaque groupe la liste des résultats delayed précédents.
    Dans le wrapper, on `dask.compute` (ou on attend) ces résultats, on les fusionne, et on exécute.

    Ou mieux : On laisse Dask gérer. On passe une liste d'objets delayed. Dask les résoudra avant d'appeler la fonction.

## 2. Étapes d'Implémentation

### Étape 1 : Nettoyage Ray
1.  Supprimer `seapopym/backend/ray.py`.
2.  Supprimer `tests/test_backend_ray.py`.
3.  Supprimer `tests/backend_test_helpers.py`.

### Étape 2 : Implémentation Dask
1.  Créer `seapopym/backend/dask.py`.
2.  Implémenter `DaskBackend`.

### Étape 3 : Tests
1.  Créer `tests/test_backend_dask.py`.
2.  Adapter les tests séquentiels.

## 3. Dépendances
- `dask` (déjà présent via xarray ?) -> À vérifier.
- `distributed` (optionnel, pour le cluster local).
