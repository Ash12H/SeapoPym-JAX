# Plan d'Implémentation : Optimisation du Transport par Parallélisme de Données (Dask Chunking)

**Date** : 2026-01-09
**Objectif** : Résoudre le goulot d'étranglement de performance identifié dans le manuscrit (80% du temps alloué au "Transport Production") en activant le parallélisme sur la dimension des cohortes, sans modifier la structure sémantique du Blueprint.

---

## 1. Diagnostic et Stratégie

### Le Problème

-   Le modèle actuel exécute le transport de la production (variable 3D : `time, cohort, y, x`) comme un bloc monolithique.
-   Bien que le `Blueprint` supporte le parallélisme de tâches, ce nœud unique de transport sature un seul thread (ou bloque les autres), limitant le speedup global à ~1.25x (Loi d'Amdahl).

### La Solution Choisie : Chunking par Cohorte

-   Utiliser la capacité native de **xarray** et **Dask** pour diviser les données.
-   Configurer les variables d'état (notamment `production`) pour qu'elles soient "chunkées" selon la dimension `cohort`.
-   Chaque cohorte devient un bloc de données indépendant.
-   Les opérations de transport (Numba kernels) sont appliquées automatiquement en parallèle sur chaque chunk par le scheduler Dask.

**Avantages** :

-   Aucun changement dans la topologie du Graphe (Blueprint).
-   Aucun changement dans les équations (Kernels Numba).
-   Gain de performance attendu quasi-linéaire avec le nombre de workers (jusqu'au nombre de cohortes).

---

## 2. Étapes d'Implémentation

### Étape 1 : Modification de la Configuration (`SimulationController`)

**Cible** : `seapopym/controller/core.py`.

-   Ajouter un argument `chunks: dict[str, int] | None = None` à la méthode `setup()`.
-   Dans `_ingest_initial_state`, si `chunks` est fourni, l'appliquer à l'état initial (`ds.chunk(chunks)`).
-   **Stratégie** : L'utilisateur a le contrôle total. Il définit quelles dimensions chunker (ex: `{"cohort": 1}`) et comment.
-   **Flexibilité** : Cette approche est agnostique au modèle (LMTL, future extension) et permet de s'adapter à la topologie de la machine.
-   **Parallélisme unifié** : Dask gère la distribution des chunks (intra-tâche) en concurrence avec les autres tâches du graphe (inter-tâches) via le work-stealing.

**Exemple d'utilisation** :

```python
sim.setup(
    ...,
    chunks={"cohort": 1}  # Parallélisme total sur les cohortes
)
```

### Étape 2 : Vérification de la Compatibilité des Fonctions de Transport

**Cible** : `seapopym/transport/core.py` et `numba_kernels.py`.

-   S'assurer que `xr.apply_ufunc` dans `compute_transport_numba` est configuré pour accepter des entrées chunkées (`dask='parallelized'`).
-   Vérifier que les dimensions "core" (`input_core_dims`) sont correctement définies pour exclure `cohort` (le transport agit sur Y, X, pas sur Cohort). Si `cohort` n'est pas dans les core dims, `apply_ufunc` map automatiquement sur cette dimension.

### Étape 3 : Configuration du Client Dask (Controller)

**Cible** : `seapopym/controller/core.py`.

-   Vérifier que le `SimulationController` instancie bien un `LocalCluster` ou un `ThreadPoolExecutor` capable de traiter ces chunks.
-   S'assurer que la configuration par défaut n'écrase pas le chunking personnalisé.

### Étape 4 : Validation (Reproduction des Benchmarks)

Nous devrons relancer deux tests pour valider l'optimisation :

1.  **Test "Time Decomposition" (Profilage)** :

    -   Le temps "Wall clock" du nœud Transport devrait diminuer proportionnellement aux workers.
    -   Le ratio 80% devrait baisser significativement.

2.  **Test "Strong Scaling" (Modèle Complet)** :
    -   C'est ici que nous pourrons enfin générer le graphique de Strong Scaling du _modèle réel_.
    -   Nous nous attendons à voir une courbe de speedup qui décolle (au lieu de stagner à 1.25x), validant l'approche pour le papier.

---

## 3. Risques et Attentions

-   **Overhead Graphe** : Avec 50 cohortes et beaucoup de pas de temps, le graphe de tâches Dask (de bas niveau) peut devenir très gros. Il faudra surveiller le temps de génération du graphe.
-   **Mémoire** : Le traitement en parallèle de 12 cohortes (si 12 cœurs) consomme 12x plus de mémoire vive instantanée pour les buffers temporaires. Vérifier que la machine de test tient la charge.

---

## 4. Prochaine Action Immédiate

Auditer le fichier d'initialisation de l'état pour localiser où appliquer le `.chunk()`.
