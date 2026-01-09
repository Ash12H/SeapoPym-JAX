# Plan de Refactorisation : Architecture des Backends de Parallélisme

**Date** : 2026-01-09
**Objectif** : Clarifier et séparer les stratégies de parallélisme (Task vs Data) en éliminant l'incohérence architecturale actuelle.

---

## 1. Diagnostic du Problème Actuel

### 1.1 Incohérence Sémantique

**SequentialBackend** (Comportement trompeur) :
- **Nom** : "Sequential" suggère aucun parallélisme
- **Réalité** : Laisse passer les Dask arrays chunked → parallélisme de données **implicite**
- **Problème** : Utilisé comme workaround pour le parallélisme de données

**DaskBackend** (Conflit structurel) :
- **Mécanisme** : `dask.delayed` pour paralléliser les **tâches**
- **Limitation** : Matérialise tous les inputs → **incompatible** avec le chunking
- **Problème** : Impossible de combiner task parallelism + data parallelism

### 1.2 Conflit Fondamental : Deux Graphes Dask Concurrents

```
Dask Delayed (Task Graph - Haut Niveau)
    Parallélise les FONCTIONS
    Graph : Task A → Task B → Task C
                ⚠️ CONFLIT
Dask Arrays (Data Graph - Bas Niveau)
    Parallélise les CHUNKS de données
    Graph : Chunk 1, Chunk 2, ... Chunk N
```

**Conséquence** : Quand on mélange les deux, le `delayed` force la matérialisation des Dask arrays, annulant le chunking.

---

## 2. Vision Cible : 3 Backends Distincts

### 2.1 SequentialBackend (Vrai séquentiel)

**Comportement** :
- Exécution strictement séquentielle, tâche par tâche
- Matérialise TOUTES les données (eager computation)
- Aucun parallélisme (ni task, ni data)

**Implémentation** :
- Appelle `execute_task_sequence` (existant)
- Ajoute `_materialize()` pour forcer `.compute()` sur les Dask arrays
- Matérialise l'état initial ET les résultats intermédiaires

**Usage** :
- Debugging
- Simulations de petite taille
- Garantir comportement déterministe
- Tests unitaires

**Configuration** :
```python
controller = SimulationController(config, backend="sequential")
controller.setup(configure_model, initial_state=state)
# ⚠️ Pas de chunking autorisé (sera matérialisé)
```

---

### 2.2 TaskParallelBackend (Ancien DaskBackend renommé)

**Comportement** :
- Parallélise les **tâches indépendantes** du graphe DAG
- Utilise `dask.delayed` pour construire le graphe de tâches
- Matérialise l'état initial (chunking incompatible)

**Implémentation** :
- Garde la logique actuelle du `DaskBackend`
- Ajoute validation : rejette les données chunked avec erreur explicite
- Ajoute `_validate_no_chunked_data()` pour détecter les chunks

**Usage** :
- Modèle avec plusieurs groupes fonctionnels **indépendants**
- Speedup attendu : 1-12× selon le nombre de tâches parallélisables
- Ne convient PAS au transport massif (pas de chunking)

**Configuration** :
```python
controller = SimulationController(config, backend="task_parallel")
controller.setup(configure_model, initial_state=state)
# ⚠️ Erreur si state contient des Dask arrays chunked
```

**Validation** :
```python
BackendConfigurationError: TaskParallelBackend detected chunked data.
Variable 'production' has chunks: (1920, 4320, 1).
→ Use backend="data_parallel" for chunked data parallelism.
```

---

### 2.3 DataParallelBackend (Nouveau)

**Comportement** :
- Parallélise les **chunks de données** au sein de chaque tâche
- Laisse les Dask arrays chunked traverser le backend sans matérialisation
- Les fonctions utilisent `xr.apply_ufunc(..., dask="parallelized")`

**Implémentation** :
- Similaire au `SequentialBackend` actuel (exécution séquentielle des tâches)
- **Différence clé** : Ne matérialise PAS les données
- Ajoute option `persist_intermediates` pour gérer l'explosion du graphe
- Ajoute `_validate_chunked_data()` pour vérifier la présence de chunks

**Usage** :
- Transport de production (80% du temps de calcul)
- Grandes grilles spatiales
- Nombreuses cohortes (50+)
- Out-of-core computation (données > RAM)

**Configuration** :
```python
controller = SimulationController(config, backend="data_parallel")
controller.setup(
    configure_model,
    initial_state=state,
    chunks={"cohort": 1}  # Chunking REQUIS
)
```

**Options** :
```python
# Option 1 : Laisser le graphe s'accumuler (défaut)
backend = DataParallelBackend(persist_intermediates=False)

# Option 2 : Persister après chaque groupe (évite explosion du graphe)
backend = DataParallelBackend(persist_intermediates=True)
```

**Validation** :
```python
# Warning si pas de chunks détectés
DataParallelBackendWarning: No chunked data found in state.
DataParallelBackend is optimized for chunked arrays.
→ Consider using backend="sequential" or chunking your data.
```

---

## 3. Tableau Comparatif

| Caractéristique | Sequential | TaskParallel | DataParallel |
|----------------|------------|--------------|--------------|
| **Parallélisme** | ❌ Aucun | ✅ Inter-tâches | ✅ Intra-tâches (chunks) |
| **Mécanisme** | Eager exec | `dask.delayed` | Dask arrays chunked |
| **Données Dask chunked** | ⚠️ Matérialisées | ⚠️ **Rejetées** (erreur) | ✅ **Requises** |
| **RAM requise** | Tout en mémoire | Tout en mémoire | Out-of-core possible |
| **Speedup typique** | 1× (baseline) | 1-12× (selon tâches) | 1-50× (selon chunks) |
| **Use case primaire** | Debug, tests | Multi-groupes indép. | Transport, grandes grilles |
| **Graphe Dask** | Aucun | Task graph | Data graph |
| **Scheduler Dask** | N/A | ThreadPool/Distributed | ThreadPool/Distributed |
| **Validation** | Aucune | Rejette chunks | Warning si pas de chunks |

---

## 4. Plan d'Implémentation

### Phase 1 : Préparation (Non-breaking)

#### 4.1 Créer les exceptions et helpers
**Fichiers** :
- `seapopym/backend/exceptions.py` (modifier)
- `seapopym/backend/validation.py` (créer)

**Contenu** :
```python
# exceptions.py
class BackendConfigurationError(Exception):
    """Raised when backend is misconfigured for the data type."""
    pass

# validation.py
def has_chunked_arrays(dataset: xr.Dataset) -> bool:
    """Check if dataset contains chunked Dask arrays."""
    ...

def validate_no_chunks(dataset: xr.Dataset, backend_name: str) -> None:
    """Raise error if dataset has chunks (for TaskParallel)."""
    ...

def validate_has_chunks(dataset: xr.Dataset, backend_name: str) -> None:
    """Warn if dataset has no chunks (for DataParallel)."""
    ...
```

#### 4.2 Créer DataParallelBackend
**Fichier** : `seapopym/backend/data_parallel.py` (créer)

**Implémentation** :
```python
class DataParallelBackend(ComputeBackend):
    """Data parallelism via Dask array chunking (intra-task parallelism)."""

    def __init__(self, persist_intermediates: bool = False):
        self.persist_intermediates = persist_intermediates

    def execute(self, task_groups, state):
        # Validation : warn si pas de chunks
        validate_has_chunks(state, "DataParallelBackend")

        all_results = {}
        for group_name, tasks in task_groups:
            # Exécution séquentielle MAIS données lazy
            group_results = execute_task_sequence(tasks, state, all_results)

            # Option : persist pour éviter explosion du graphe
            if self.persist_intermediates:
                group_results = self._persist_results(group_results)

            all_results.update(group_results)

        return all_results
```

#### 4.3 Ajouter alias TaskParallelBackend
**Fichier** : `seapopym/backend/task_parallel.py` (créer)

**Contenu** : Copie de `dask.py` avec nouveau nom + validation

```python
class TaskParallelBackend(ComputeBackend):
    """Task parallelism via dask.delayed (inter-task parallelism)."""

    def execute(self, task_groups, state):
        # Validation : rejette les chunks
        validate_no_chunks(state, "TaskParallelBackend")

        # Logique existante du DaskBackend...
```

---

### Phase 2 : Modification du SequentialBackend

#### 4.4 Ajouter matérialisation dans SequentialBackend
**Fichier** : `seapopym/backend/sequential.py` (modifier)

**Changements** :
```python
class SequentialBackend(ComputeBackend):
    """Pure sequential execution with eager computation (no parallelism)."""

    def execute(self, task_groups, state):
        # Matérialise l'état initial
        state = self._materialize_dataset(state)

        all_results = {}
        for group_name, tasks in task_groups:
            group_results = execute_task_sequence(tasks, state, all_results)

            # Matérialise les résultats intermédiaires
            group_results = self._materialize_results(group_results)
            all_results.update(group_results)

        return all_results

    def _materialize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Force computation of all lazy arrays in dataset."""
        if hasattr(ds, 'compute'):
            return ds.compute()
        return ds

    def _materialize_results(self, results: dict) -> dict:
        """Force computation of all lazy arrays in results."""
        return {
            k: v.compute() if hasattr(v, 'compute') else v
            for k, v in results.items()
        }
```

---

### Phase 3 : Intégration dans SimulationController

#### 4.5 Modifier le constructeur du SimulationController
**Fichier** : `seapopym/controller/core.py` (modifier)

**Changements** :
```python
def __init__(self, config, backend: ComputeBackend | str = "sequential"):
    ...
    if isinstance(backend, str):
        if backend == "sequential":
            self.backend = SequentialBackend()
        elif backend == "dask":
            # Deprecation warning
            warnings.warn(
                "backend='dask' is deprecated. Use 'task_parallel' for task parallelism "
                "or 'data_parallel' for data chunking parallelism.",
                DeprecationWarning, stacklevel=2
            )
            self.backend = TaskParallelBackend()
        elif backend == "task_parallel":
            self.backend = TaskParallelBackend()
        elif backend == "data_parallel":
            self.backend = DataParallelBackend()
        else:
            raise ValueError(
                f"Unknown backend: '{backend}'. "
                f"Supported: 'sequential', 'task_parallel', 'data_parallel'."
            )
```

#### 4.6 Ajouter validation dans setup()
**Fichier** : `seapopym/controller/core.py` (modifier)

**Changements** :
```python
def setup(self, ..., chunks=None):
    ...
    # Validation backend vs chunking
    if chunks is not None and isinstance(self.backend, TaskParallelBackend):
        raise BackendConfigurationError(
            "TaskParallelBackend is incompatible with chunked data. "
            "Use backend='data_parallel' for data chunking parallelism."
        )

    # Ingest initial state avec chunking
    state_ds = self._ingest_initial_state(initial_state, chunks=chunks)
    ...
```

---

### Phase 4 : Mise à jour __init__.py

#### 4.7 Exporter les nouveaux backends
**Fichier** : `seapopym/backend/__init__.py` (modifier)

**Changements** :
```python
from seapopym.backend.base import ComputeBackend
from seapopym.backend.sequential import SequentialBackend
from seapopym.backend.dask import DaskBackend  # Deprecated
from seapopym.backend.task_parallel import TaskParallelBackend
from seapopym.backend.data_parallel import DataParallelBackend
from seapopym.backend.exceptions import BackendConfigurationError, ExecutionError

__all__ = [
    "ComputeBackend",
    "SequentialBackend",
    "DaskBackend",  # Deprecated, kept for compatibility
    "TaskParallelBackend",
    "DataParallelBackend",
    "BackendConfigurationError",
    "ExecutionError",
]
```

---

### Phase 5 : Correction Bug Transport

#### 4.8 Ajouter dask_gufunc_kwargs manquant
**Fichier** : `seapopym/transport/core.py` (modifier)

**Ligne 669** : Ajouter le paramètre manquant dans la fonction de diffusion

```python
flux_diff_e, flux_diff_w, flux_diff_n, flux_diff_s = xr.apply_ufunc(
    diffusion_flux_numba,
    ...,
    dask="parallelized",
    output_dtypes=[state_clean.dtype] * 4,
    dask_gufunc_kwargs={"allow_rechunk": True},  # <-- AJOUTER
)
```

---

### Phase 6 : Tests et Documentation

#### 4.9 Créer tests unitaires
**Fichiers** : `tests/backend/test_*.py`

**Tests à créer** :
- `test_sequential_backend.py` : Vérifier matérialisation
- `test_task_parallel_backend.py` : Vérifier rejet des chunks
- `test_data_parallel_backend.py` : Vérifier préservation des chunks
- `test_backend_validation.py` : Vérifier les erreurs de configuration

#### 4.10 Mettre à jour les notebooks
**Fichiers** :
- `notebook/demo_dask_chunking.ipynb`
- `notebook/demo_seapopym_chunking.ipynb`

**Changements** :
```python
# Avant (trompeur)
controller = SimulationController(config, backend="sequential")
controller.setup(..., chunks={"cohort": 1})

# Après (clair)
controller = SimulationController(config, backend="data_parallel")
controller.setup(..., chunks={"cohort": 1})
```

#### 4.11 Documentation utilisateur
**Fichier** : `docs/backends.md` (créer)

**Contenu** :
- Quand utiliser chaque backend
- Exemples de configuration
- Tableau comparatif
- FAQ : "Quel backend choisir ?"

---

## 5. Migration Path (Rétrocompatibilité)

### 5.1 Deprecation de DaskBackend

**Stratégie** :
1. Garder `DaskBackend` comme alias de `TaskParallelBackend`
2. Émettre un `DeprecationWarning` avec message explicite
3. Prévoir suppression dans v2.0

```python
class DaskBackend(TaskParallelBackend):
    """Deprecated: Use TaskParallelBackend instead."""

    def __init__(self):
        warnings.warn(
            "DaskBackend is deprecated and will be removed in v2.0. "
            "Use TaskParallelBackend for task parallelism or "
            "DataParallelBackend for data chunking parallelism.",
            DeprecationWarning, stacklevel=2
        )
        super().__init__()
```

### 5.2 Migration guidée par erreurs explicites

**Scénario 1** : Utilisateur utilise `backend="dask"` avec chunks
```python
# Code utilisateur (ancien)
controller = SimulationController(config, backend="dask")
controller.setup(..., chunks={"cohort": 1})

# Comportement
→ DeprecationWarning + BackendConfigurationError avec suggestion :
  "Use backend='data_parallel' for chunked data."
```

**Scénario 2** : Utilisateur utilise `backend="sequential"` avec chunks (actuel workaround)
```python
# Code utilisateur (ancien workaround)
controller = SimulationController(config, backend="sequential")
controller.setup(..., chunks={"cohort": 1})

# Comportement (après refactorisation)
→ Warning : "SequentialBackend will materialize chunked data.
            Use backend='data_parallel' to preserve chunking."
```

---

## 6. Risques et Mitigations

### 6.1 Risques Identifiés

| Risque | Probabilité | Impact | Mitigation |
|--------|-------------|--------|------------|
| **Breaking change** pour utilisateurs du workaround | Élevée | Moyen | Deprecation warnings + migration guide |
| **Complexité accrue** (3 backends au lieu de 2) | Moyenne | Faible | Documentation claire + exemples |
| **Confusion utilisateurs** (quel backend choisir ?) | Moyenne | Moyen | Validation proactive + messages d'erreur clairs |
| **Bug dans la matérialisation** (SequentialBackend) | Faible | Élevé | Tests unitaires exhaustifs |
| **Régression performance** | Faible | Élevé | Benchmarks avant/après |

### 6.2 Plan de Test

**Tests unitaires** :
- Chaque backend avec données Numpy (non-chunked)
- Chaque backend avec données Dask chunked
- Validation des erreurs de configuration
- Matérialisation correcte dans SequentialBackend

**Tests d'intégration** :
- Simulation complète avec chaque backend
- Comparaison résultats : tous les backends doivent donner les mêmes résultats numériques
- Benchmarks performance : vérifier les speedups attendus

**Tests de régression** :
- Relancer tous les notebooks d'expériences
- Vérifier que les figures du manuscrit sont reproduites

---

## 7. Planning et Ordre d'Exécution

### Sprint 1 : Infrastructure (2-3h)
1. ✅ Créer `validation.py` avec helpers
2. ✅ Modifier `exceptions.py` avec `BackendConfigurationError`
3. ✅ Créer `data_parallel.py`
4. ✅ Créer `task_parallel.py`

### Sprint 2 : Modification SequentialBackend (1h)
5. ✅ Ajouter méthodes `_materialize_*` dans `sequential.py`
6. ✅ Tests unitaires pour matérialisation

### Sprint 3 : Intégration Controller (1-2h)
7. ✅ Modifier `SimulationController.__init__` pour supporter les nouveaux backends
8. ✅ Ajouter validation dans `setup()`
9. ✅ Modifier `__init__.py` pour exporter les nouveaux backends

### Sprint 4 : Bugfix Transport (15 min)
10. ✅ Ajouter `dask_gufunc_kwargs` manquant dans `transport/core.py`

### Sprint 5 : Tests (2-3h)
11. ✅ Tests unitaires pour chaque backend
12. ✅ Tests d'intégration
13. ✅ Benchmarks performance

### Sprint 6 : Documentation et Migration (2h)
14. ✅ Mise à jour notebooks démo
15. ✅ Créer documentation `docs/backends.md`
16. ✅ Ajouter exemples dans `README.md`

**Temps total estimé** : 8-12 heures

---

## 8. Résultat Attendu

### 8.1 Après Refactorisation

**Clarté** :
- Chaque backend a un nom qui décrit précisément son comportement
- Utilisateurs comprennent immédiatement quel backend choisir

**Validation** :
- Erreurs explicites en cas de mauvaise configuration
- Messages d'aide pointent vers la solution

**Extensibilité** :
- Facile d'ajouter un `HybridBackend` plus tard
- Architecture propre pour expérimenter avec d'autres schedulers (Ray, etc.)

**Performance** :
- DataParallelBackend résout le problème de RAM du transport
- Speedup mesurable sur le cas réel (50 cohortes)

### 8.2 Exemple d'Utilisation Final

```python
# Cas 1 : Debug / petit modèle
controller = SimulationController(config, backend="sequential")
controller.setup(configure_model, initial_state=state)

# Cas 2 : Modèle multi-groupes indépendants
controller = SimulationController(config, backend="task_parallel")
controller.setup(configure_model, initial_state=state)
# → Parallélise les groupes fonctionnels

# Cas 3 : Transport massif avec chunking
controller = SimulationController(config, backend="data_parallel")
controller.setup(configure_model, initial_state=state, chunks={"cohort": 1})
# → Parallélise les cohortes dans le transport

# Cas 4 : Configuration avancée
backend = DataParallelBackend(persist_intermediates=True)
controller = SimulationController(config, backend=backend)
controller.setup(configure_model, initial_state=state, chunks={"cohort": 5})
# → Groupes de 5 cohortes + persist pour éviter explosion du graphe
```

---

## 9. Notes pour l'Article

Cette refactorisation **améliore la rigueur scientifique** de l'article :

**Section "Implémentation Logicielle"** (à mettre à jour) :
- Clarifier que le modèle supporte 3 stratégies d'exécution
- Expliquer que le parallélisme de données (chunking) est la clé pour le transport
- Ajouter un tableau comparatif dans les méthodes

**Section "Performances - Strong Scaling"** (nouveau) :
- Enfin possible de présenter un vrai graphique de strong scaling pour le modèle complet
- Comparer les 3 backends sur la simulation LMTL réelle
- Démontrer que le DataParallelBackend résout le goulot d'étranglement du transport

**Figure à ajouter** :
- Strong Scaling du modèle complet (1, 2, 4, 8, 12 workers)
- Avec DataParallelBackend, devrait approcher la scalabilité linéaire jusqu'au nombre de cohortes

---

## 10. Prochaines Étapes

1. **Valider ce plan** avec l'équipe
2. **Créer une TODO list exhaustive** pour le suivi
3. **Commencer le développement** par Sprint 1
4. **Tester incrémentalement** après chaque sprint
5. **Documenter au fur et à mesure**

---

## Références

- Dask Documentation: Task Scheduling vs Array Chunking
- Xarray apply_ufunc: dask="parallelized" behavior
- Loi d'Amdahl: Limites théoriques du parallélisme
