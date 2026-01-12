# Plan de Tests : Chunking et Performance Dask

**Objectif** : Garantir que les fonctions LMTL se comportent correctement avec des données chunkées et éviter les explosions de tâches Dask (problème `reindex()` rencontré).

**Date** : 2026-01-12
**Statut** : 📋 Planification

---

## 🎯 Contexte et Motivation

### Problème Identifié
- **Symptôme** : Fonction utilisant `reindex()` génère un nombre excessif de tâches Dask
- **Impact** : Saturation du scheduler, mémoire insuffisante, performance dégradée
- **Cause probable** : Opérations qui forcent l'alignement/broadcast sur toutes les combinaisons de chunks

### Risques sans Tests
1. **Régression silencieuse** : Modifications futures peuvent réintroduire le problème
2. **Scalabilité compromise** : Code correct en local mais inutilisable sur cluster
3. **Débogage difficile** : Problèmes apparaissent seulement à grande échelle

### Bénéfices Attendus
- ✅ Détection précoce des anti-patterns Dask
- ✅ Documentation du comportement attendu
- ✅ Confiance pour l'exécution distribuée
- ✅ Base pour optimisations futures

---

## 📊 Architecture des Tests (4 Niveaux)

Basée sur les bonnes pratiques industrielles (Xarray, Dask-ML, Pangeo).

### Niveau 1 : Functional Correctness ✅ (Existant)
**Fichiers** : `tests/test_lmtl.py`, `tests/test_lmtl_units.py`
**Objectif** : Vérifier la correction mathématique
**Statut** : ✅ Déjà implémenté, coverage 90%

### Niveau 2 : Chunking Correctness 🎯 (À implémenter)
**Fichier** : `tests/test_lmtl_chunking.py` (nouveau)
**Objectif** : Vérifier que chunking ne change pas le résultat
**Priorité** : **HAUTE** - Bloquant pour utilisation distribuée

**Tests à implémenter** :
```python
def test_production_dynamics_chunked_vs_unchunked():
    """Résultat identique avec/sans chunking."""

def test_mortality_with_spatial_chunks():
    """Chunking spatial (x, y) préserve les résultats."""

def test_recruitment_with_cohort_chunks():
    """Chunking cohorte préserve les résultats."""
```

### Niveau 3 : Task Graph Efficiency 🔍 (À implémenter)
**Fichier** : `tests/test_lmtl_dask_efficiency.py` (nouveau)
**Objectif** : Détecter les explosions de tâches
**Priorité** : **HAUTE** - Résout le problème reindex()

**Tests à implémenter** :
```python
def test_production_dynamics_task_count():
    """Nombre de tâches proportionnel aux chunks."""

def test_no_implicit_rechunking():
    """Pas de rechunking non sollicité."""

def test_reindex_operations_bounded():
    """Opérations reindex() ne causent pas d'explosion."""
```

### Niveau 4 : Performance Benchmarks 📈 (Optionnel)
**Fichier** : `benchmarks/bench_lmtl_scaling.py` (nouveau)
**Objectif** : Mesurer la scalabilité réelle
**Priorité** : **BASSE** - Utile mais non bloquant

---

## 🛠️ Implémentation Détaillée

### Phase 1 : Chunking Correctness Tests

#### Fichier : `tests/test_lmtl_chunking.py`

**Structure proposée** :
```python
"""Tests de correction avec données chunkées.

Objectif : Garantir que chunking ne change pas les résultats numériques.
"""

import pytest
import xarray as xr
import numpy as np
from seapopym.lmtl.core import (
    compute_production_dynamics,
    compute_mortality_tendency,
    compute_recruitment_age,
)

# Fixture pour données réalistes
@pytest.fixture
def lmtl_data():
    """Données LMTL réalistes pour tests."""
    return {
        "production": xr.DataArray(...),
        "biomass": xr.DataArray(...),
        "temperature": xr.DataArray(...),
    }

# Test Pattern 1 : Cohort chunking
def test_production_dynamics_cohort_chunks(lmtl_data):
    """Production dynamics avec chunking cohortes."""
    prod = lmtl_data["production"]

    # Référence sans chunks
    result_ref = compute_production_dynamics(prod, ...)

    # Avec chunks cohortes (use case réaliste)
    prod_chunked = prod.chunk({"cohort": 5})
    result_chunked = compute_production_dynamics(prod_chunked, ...)

    # Vérification
    xr.testing.assert_allclose(
        result_chunked.compute(),
        result_ref,
        rtol=1e-10
    )

# Test Pattern 2 : Spatial chunking
def test_mortality_spatial_chunks(lmtl_data):
    """Mortalité avec chunking spatial."""
    biomass = lmtl_data["biomass"]

    result_ref = compute_mortality_tendency(biomass, ...)

    # Chunking spatial (x, y) - peut causer problèmes
    biomass_chunked = biomass.chunk({"x": 10, "y": 10})
    result_chunked = compute_mortality_tendency(biomass_chunked, ...)

    xr.testing.assert_allclose(
        result_chunked.compute(),
        result_ref,
        rtol=1e-10
    )

# Test Pattern 3 : Mixed chunking
def test_production_mixed_chunks(lmtl_data):
    """Production avec chunking mixte (cohort + spatial)."""
    prod = lmtl_data["production"]

    result_ref = compute_production_dynamics(prod, ...)

    # Chunking mixte (cas réaliste cluster)
    prod_chunked = prod.chunk({"cohort": 5, "x": 20, "y": 20})
    result_chunked = compute_production_dynamics(prod_chunked, ...)

    xr.testing.assert_allclose(
        result_chunked.compute(),
        result_ref,
        rtol=1e-10
    )
```

**Critères de succès** :
- ✅ Tous les tests passent avec `rtol=1e-10` (précision machine)
- ✅ Pas de warnings Dask sur rechunking
- ✅ Temps d'exécution raisonnable (< 2x version non-chunkée)

---

### Phase 2 : Task Graph Efficiency Tests

#### Fichier : `tests/test_lmtl_dask_efficiency.py`

**Structure proposée** :
```python
"""Tests d'efficacité des task graphs Dask.

Objectif : Détecter les explosions de tâches et rechunking excessif.
"""

import pytest
import xarray as xr
import dask
from seapopym.lmtl.core import compute_production_dynamics

def count_tasks(dask_array):
    """Compte le nombre de tâches dans le graph Dask."""
    return len(dask_array.__dask_graph__())

def test_production_task_count_linear():
    """Le nombre de tâches doit être linéaire en nombre de chunks."""
    prod = create_production_array(n_cohorts=100, shape=(100, 100))

    # Test avec différentes stratégies de chunking
    chunks_strategies = [
        {"cohort": 10},    # 10 chunks
        {"cohort": 20},    # 5 chunks
        {"cohort": 50},    # 2 chunks
    ]

    task_counts = []
    for chunks in chunks_strategies:
        prod_chunked = prod.chunk(chunks)
        result = compute_production_dynamics(prod_chunked, ...)
        task_counts.append(count_tasks(result))

    # Vérifier linéarité : tasks_10chunks ≈ 2 * tasks_20chunks
    assert task_counts[0] / task_counts[1] == pytest.approx(2, rel=0.2)
    assert task_counts[1] / task_counts[2] == pytest.approx(2.5, rel=0.2)

def test_no_unexpected_rechunking():
    """Détecter rechunking non sollicité (coûteux)."""
    prod = create_production_array().chunk({"cohort": 10})

    result = compute_production_dynamics(prod, ...)

    # Analyser le graph pour détecter opérations "rechunk"
    graph = result.__dask_graph__()
    rechunk_ops = [k for k in graph.keys() if "rechunk" in str(k)]

    assert len(rechunk_ops) == 0, \
        f"Unexpected rechunking detected: {len(rechunk_ops)} operations"

def test_reindex_bounded_complexity():
    """Vérifier que reindex() ne cause pas d'explosion."""
    # Scénario : aligner deux arrays avec chunks différents
    arr1 = xr.DataArray(...).chunk({"cohort": 5})
    arr2 = xr.DataArray(...).chunk({"cohort": 3})

    # Opération potentiellement problématique
    result = arr1.reindex_like(arr2)

    # Compter les tâches
    n_tasks = count_tasks(result)
    n_chunks_arr1 = len(arr1.chunks[0])
    n_chunks_arr2 = len(arr2.chunks[0])

    # Seuil : O(n_chunks) au lieu de O(n_chunks^2)
    max_expected = (n_chunks_arr1 + n_chunks_arr2) * 5  # Facteur sécurité

    assert n_tasks < max_expected, \
        f"Task explosion: {n_tasks} tasks (expected < {max_expected})"

@pytest.mark.parametrize("chunk_size", [1, 5, 10, 20])
def test_task_count_scales_properly(chunk_size):
    """Vérifier que task count évolue correctement."""
    prod = create_production_array(n_cohorts=100).chunk({"cohort": chunk_size})
    result = compute_production_dynamics(prod, ...)

    n_tasks = count_tasks(result)
    n_chunks = 100 // chunk_size

    # Formule attendue : n_tasks ≈ n_chunks * complexity_factor
    # Ajuster complexity_factor selon profiling réel
    complexity_factor = 10  # À calibrer
    expected_tasks = n_chunks * complexity_factor

    assert n_tasks < expected_tasks * 2, \
        f"Too many tasks for chunk_size={chunk_size}: {n_tasks} (expected ~{expected_tasks})"
```

**Critères de succès** :
- ✅ Nombre de tâches O(n_chunks) et non O(n_chunks²)
- ✅ Pas de rechunking implicite détecté
- ✅ Scalabilité linéaire avec taille des chunks

---

### Phase 3 : Benchmarks de Performance (Optionnel)

#### Fichier : `benchmarks/bench_lmtl_scaling.py`

**Structure proposée** :
```python
"""Benchmarks de scalabilité LMTL avec Dask.

Usage: pytest benchmarks/ --benchmark-only
"""

import pytest
from seapopym.lmtl.core import compute_production_dynamics

@pytest.mark.benchmark(group="scaling-cohorts")
def test_scaling_cohorts(benchmark):
    """Mesurer temps vs nombre de cohortes."""
    def run(n_cohorts):
        prod = create_production_array(n_cohorts=n_cohorts).chunk({"cohort": 10})
        return compute_production_dynamics(prod, ...).compute()

    benchmark(run, n_cohorts=100)

@pytest.mark.benchmark(group="scaling-spatial")
def test_scaling_spatial_resolution(benchmark):
    """Mesurer temps vs résolution spatiale."""
    def run(grid_size):
        prod = create_production_array(shape=(grid_size, grid_size))
        prod = prod.chunk({"x": 50, "y": 50, "cohort": 10})
        return compute_production_dynamics(prod, ...).compute()

    benchmark(run, grid_size=500)
```

---

## 📅 Roadmap d'Implémentation

### Sprint 1 : Chunking Correctness (3-4h)
**Objectif** : Garantir correction numérique

- [ ] Créer `tests/test_lmtl_chunking.py`
- [ ] Fixture pour données réalistes LMTL
- [ ] Tests cohort chunking pour toutes fonctions LMTL
- [ ] Tests spatial chunking
- [ ] Tests mixed chunking
- [ ] **Validation** : Tous tests passent, coverage +5%

### Sprint 2 : Task Graph Efficiency (4-5h)
**Objectif** : Détecter explosions de tâches

- [ ] Créer `tests/test_lmtl_dask_efficiency.py`
- [ ] Helper `count_tasks()` et analyse graph
- [ ] Test task count linéarité
- [ ] Test détection rechunking
- [ ] Test spécifique `reindex()` operations
- [ ] **Validation** : Détecter le problème reindex() existant

### Sprint 3 : Fix Problèmes Détectés (variable)
**Objectif** : Corriger les anti-patterns

- [ ] Identifier fonctions problématiques (via tests Sprint 2)
- [ ] Refactoring pour éviter rechunking
- [ ] Remplacer `reindex()` par alternatives efficaces
- [ ] **Validation** : Tests efficiency passent

### Sprint 4 : Benchmarks (2-3h, optionnel)
**Objectif** : Documentation performance

- [ ] Créer `benchmarks/bench_lmtl_scaling.py`
- [ ] Benchmarks cohort scaling
- [ ] Benchmarks spatial scaling
- [ ] **Validation** : Baseline établi pour détection régression

---

## 🎓 Bonnes Pratiques Apprises

### Références Industrielles
1. **Xarray** : `tests/test_dask.py` - Pattern de test chunking
2. **Dask-ML** : Task graph analysis pour ML distribué
3. **Pangeo** : Performance testing pour geosciences

### Patterns à Suivre
- ✅ Séparer tests fonctionnels des tests performance
- ✅ Utiliser `xr.testing.assert_allclose()` pour comparaisons
- ✅ Parameterize chunk strategies pour couverture
- ✅ Documenter les seuils de performance attendus

### Anti-Patterns à Éviter
- ❌ `reindex()` sur données chunkées différemment
- ❌ Opérations qui forcent alignment global
- ❌ Broadcasting implicite sur dimensions chunkées
- ❌ `.values` qui force compute() prématuré

---

## 📈 Métriques de Succès

### Coverage
- **Actuel** : 78% global, 90% LMTL core
- **Objectif** : 82% global (après ajout tests chunking)
- **Note** : Numba kernels (2%) non traçables, acceptable

### Performance
- **Task Count** : O(n_chunks) confirmé par tests
- **No Rechunking** : 0 opérations rechunk implicites
- **Scalabilité** : Linéaire jusqu'à 500 cohortes

### Confiance Déploiement
- ✅ Simulations distribuées validées
- ✅ Problèmes reindex() résolus et testés
- ✅ Documentation complète du comportement chunking

---

## 🔄 Maintenance Continue

### CI/CD Integration
```yaml
# .github/workflows/tests.yml (proposition)
- name: Run chunking tests
  run: pytest tests/test_lmtl_chunking.py -v

- name: Run efficiency tests
  run: pytest tests/test_lmtl_dask_efficiency.py -v --tb=short
```

### Revue Périodique
- **Mensuel** : Vérifier seuils task count restent valides
- **Avant release** : Run benchmarks complets
- **Après optimisation** : Update baselines dans tests

---

## 📚 Ressources Complémentaires

### Documentation Dask
- [Dask Best Practices](https://docs.dask.org/en/latest/best-practices.html)
- [Debugging Performance](https://docs.dask.org/en/latest/debugging-performance.html)
- [Task Graph Optimization](https://docs.dask.org/en/latest/optimize.html)

### Articles Académiques
- Rocklin (2015) : "Dask: Parallel Computation with Blocked algorithms and Task Scheduling"
- Hoyer & Hamman (2017) : "xarray: N-D labeled Arrays and Datasets in Python"

---

**Prochaine étape** : Implémenter Sprint 1 (Chunking Correctness Tests)

**Questions à résoudre** :
1. Quelles fonctions LMTL utilisent `reindex()` actuellement ?
2. Quelle stratégie de chunking est prévue pour production ?
3. Cluster Dask specs (workers, memory) pour calibrer seuils ?
