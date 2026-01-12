# Plan de Résolution : Problèmes d'Efficacité Dask

**Date** : 2026-01-12
**Statut** : 📋 Planification
**Priorité** : 🔴 **HAUTE** - Bloquant pour utilisation distribuée

---

## 🎯 Résumé Exécutif

### Problèmes Identifiés
Les tests de chunking révèlent que la **correctness est validée** (résultats numériques corrects), mais des **problèmes d'efficacité critiques** empêchent la scalabilité :

1. **LMTL `compute_production_dynamics`** : Rechunking causé par `reindex()` et `shift()` sur dimension cohorte
2. **Transport `compute_transport_xarray`** : Rechunking causé par `shift()`/`roll()` sur dimensions spatiales (y, x)

### Impact
- ⚠️ **Graph de tâches explosif** : O(N²) au lieu de O(N) tâches
- ⚠️ **Saturation mémoire** : Réorganisation massive des données entre workers
- ⚠️ **Performance dégradée** : Temps de calcul prohibitif sur cluster

### Objectif
Éliminer le rechunking implicite pour atteindre une scalabilité linéaire O(N chunks).

---

## 📊 Analyse Technique Détaillée

### Problème 1 : LMTL Production Dynamics

**Fichier** : `seapopym/lmtl/core.py`, fonction `compute_production_dynamics` (lignes 230-361)

#### Opérations Problématiques

**1. Ligne 293 : `reindex()` avec `method="nearest"`**
```python
d_tau = cohort_ages.diff("cohort")
# diff() réduit la taille de 1
d_tau = d_tau.reindex(cohort=cohort_ages.coords["cohort"], method="nearest")
# ⚠️ PROBLÈME : reindex() force l'alignement de tous les chunks
```

**Pourquoi c'est problématique :**
- `reindex()` avec `method="nearest"` force Dask à évaluer la position de chaque élément dans tous les chunks
- Avec N chunks de cohortes, génère O(N²) tâches de réindexation
- Cause principale de l'explosion du graph détectée dans le rapport

**2. Ligne 305 : `shift(cohort=1)`**
```python
influx_flux = total_outflow_flux.shift(cohort=1, fill_value=0.0)
# ⚠️ PROBLÈME : shift() sur dimension chunkée force rechunking
```

**3. Ligne 342 : `shift(cohort=1)` sur masque**
```python
prev_is_recruited = is_recruited.shift(cohort=1, fill_value=False)
# ⚠️ PROBLÈME : Deuxième shift() sur même dimension
```

**Pourquoi `shift()` est problématique :**
- `shift()` doit accéder à des éléments dans des chunks adjacents
- Dask doit créer des tâches de communication inter-chunks
- Avec chunking sur cohortes, chaque shift génère N-1 communications

#### Impact Mesuré
D'après le rapport de tests :
- ✅ Correctness : Résultats numériques corrects
- ❌ Efficiency : Rechunking détecté lors du calcul
- 🔍 Cause confirmée : Opérations `reindex()` et `shift()`

---

### Problème 2 : Transport Xarray

**Fichiers** :
- `seapopym/transport/core.py`, fonction `compute_transport_xarray` (lignes 479-594)
- `seapopym/transport/boundary.py`, fonction `get_neighbors_with_bc` (lignes 70-142)

#### Opérations Problématiques

**1. `get_neighbors_with_bc()` utilise `shift()` et `roll()`**

Dans `boundary.py` (lignes 109-142) :
```python
def get_neighbors_with_bc(data, boundary):
    # PERIODIC boundaries
    if boundary.east == BoundaryType.PERIODIC:
        data_west = data.roll({dim_x: 1}, roll_coords=False)  # ⚠️ PROBLÈME
        data_east = data.roll({dim_x: -1}, roll_coords=False) # ⚠️ PROBLÈME
    else:
        # NON-PERIODIC boundaries
        data_west = data.shift({dim_x: 1})  # ⚠️ PROBLÈME
        data_east = data.shift({dim_x: -1}) # ⚠️ PROBLÈME
    # Répété pour North-South...
```

**Pourquoi c'est problématique :**
- `shift()` et `roll()` sur dimensions spatiales (y, x) forcent le rechunking
- Si y ou x sont chunkées, Dask doit réorganiser les données entre workers
- Chaque appel à `get_neighbors_with_bc()` génère 4 opérations de shift/roll

**2. Multiples appels dans `_prepare_transport_data()`**

Dans `core.py`, `_prepare_transport_data()` appelle `get_neighbors_with_bc()` plusieurs fois :
- Ligne 230-232 : `get_neighbors_with_bc(state_clean, ...)` → 4 shifts
- Ligne 233 : `get_neighbors_with_bc(u_clean, ...)` → 4 shifts
- Ligne 234 : `get_neighbors_with_bc(v_clean, ...)` → 4 shifts
- Ligne 245 : `get_neighbors_with_bc(D_spatial, ...)` → 4 shifts
- Ligne 253 : `get_neighbors_with_bc(dx_spatial, ...)` → 4 shifts
- Ligne 258 : `get_neighbors_with_bc(dy_spatial, ...)` → 4 shifts
- Ligne 271-273 : `get_neighbors_with_bc(mask, ...)` → 4 shifts (si masque fourni)

**Total : 24-28 opérations shift/roll par appel à `compute_transport_xarray`**

#### Protection Existante

Le code inclut déjà un **warning de performance** (`_validate_chunking_for_transport`, lignes 38-101) :
```python
def _validate_chunking_for_transport(state: xr.DataArray) -> None:
    """Validate that chunking is optimal for transport computation."""
    # Détecte si Y ou X sont chunkées
    if y_is_chunked or x_is_chunked:
        warnings.warn(
            "Transport computation: Core dimensions (Y, X) are chunked, "
            "which will trigger expensive rechunking.",
            PerformanceWarning
        )
```

**Implications :**
- ✅ Le problème est **connu et documenté** dans le code
- ✅ Les utilisateurs sont **avertis** si chunking spatial est détecté
- ❌ Mais aucune **solution alternative** n'est fournie pour la version Xarray

#### Comparaison avec Version Numba

`compute_transport_numba` (lignes 602-777) **évite ce problème** :
- Utilise `xr.apply_ufunc` avec `input_core_dims=[[dim_y, dim_x], ...]`
- Spécifie `dask_gufunc_kwargs={"allow_rechunk": False}` (ligne 724)
- Les dimensions y, x sont traitées comme "core dimensions" par le kernel Numba
- Dask n'a **pas besoin** de faire des shifts car Numba accède directement aux voisins

**Conclusion :** Version Numba déjà optimale, problème limité à version Xarray.

---

## 🛠️ Solutions Proposées

### Solution 1 : LMTL Production Dynamics (PRIORITÉ HAUTE)

#### 1.1 Remplacer `reindex()` par Padding Manuel

**Problème actuel (ligne 293) :**
```python
d_tau = cohort_ages.diff("cohort")  # Size: N-1
d_tau = d_tau.reindex(cohort=cohort_ages.coords["cohort"], method="nearest")  # ⚠️ RECHUNK
```

**Solution proposée :**
```python
d_tau = cohort_ages.diff("cohort")  # Size: N-1

# Option A : Padding avec xr.concat (préserve chunks)
last_d_tau = d_tau.isel(cohort=-1)  # Dernier élément
d_tau = xr.concat([d_tau, last_d_tau], dim="cohort")  # Size: N

# Option B : Padding avec xr.pad (plus explicite)
d_tau = d_tau.pad(cohort=(0, 1), mode="edge")  # Répète dernière valeur

# Option C : Calcul direct sans reindex (optimal)
# Calculer d_tau avec même taille que cohort_ages dès le départ
d_tau_vals = np.diff(cohort_ages.values, append=cohort_ages.values[-1] - cohort_ages.values[-2])
d_tau = xr.DataArray(d_tau_vals, coords=cohort_ages.coords, dims=cohort_ages.dims)
```

**Avantages :**
- ✅ Évite `reindex()` complètement
- ✅ Préserve la structure des chunks
- ✅ Opération O(N) au lieu de O(N²)

#### 1.2 Remplacer `shift()` par Indexing Direct

**Problème actuel (lignes 305, 342) :**
```python
influx_flux = total_outflow_flux.shift(cohort=1, fill_value=0.0)  # ⚠️ RECHUNK
prev_is_recruited = is_recruited.shift(cohort=1, fill_value=False)  # ⚠️ RECHUNK
```

**Solution proposée :**
```python
# Option A : Slicing + concat (préserve chunks)
zero_flux = xr.zeros_like(total_outflow_flux.isel(cohort=0))
influx_flux = xr.concat(
    [zero_flux, total_outflow_flux.isel(cohort=slice(None, -1))],
    dim="cohort"
)

false_mask = xr.full_like(is_recruited.isel(cohort=0), False)
prev_is_recruited = xr.concat(
    [false_mask, is_recruited.isel(cohort=slice(None, -1))],
    dim="cohort"
)

# Option B : Padding (plus concis)
influx_flux = total_outflow_flux.pad(cohort=(1, 0), constant_values=0.0).isel(cohort=slice(None, -1))
prev_is_recruited = is_recruited.pad(cohort=(1, 0), constant_values=False).isel(cohort=slice(None, -1))
```

**Avantages :**
- ✅ Évite `shift()` et son rechunking
- ✅ Slicing est une opération lazy efficace pour Dask
- ✅ `concat` respecte la structure des chunks originaux

#### 1.3 Stratégie d'Implémentation

**Étapes :**
1. Créer une fonction helper `_safe_cohort_shift()` :
   ```python
   def _safe_cohort_shift(data: xr.DataArray, shift: int, fill_value: float) -> xr.DataArray:
       """Shift along cohort dimension without triggering rechunking."""
       if shift > 0:
           # Shift forward: add zeros at beginning
           pad = xr.full_like(data.isel(cohort=slice(None, shift)), fill_value)
           return xr.concat([pad, data.isel(cohort=slice(None, -shift))], dim="cohort")
       elif shift < 0:
           # Shift backward: add zeros at end
           pad = xr.full_like(data.isel(cohort=slice(shift, None)), fill_value)
           return xr.concat([data.isel(cohort=slice(-shift, None)), pad], dim="cohort")
       else:
           return data
   ```

2. Refactoriser `compute_production_dynamics()` :
   - Remplacer `reindex()` par calcul direct de `d_tau`
   - Remplacer tous les `shift()` par `_safe_cohort_shift()`
   - Ajouter assertions pour vérifier que la taille est préservée

3. Valider avec tests existants :
   - `tests/test_lmtl_chunking.py` doit passer (correctness)
   - `tests/test_lmtl_dask_efficiency.py` doit détecter amélioration (efficiency)

---

### Solution 2 : Transport Xarray (PRIORITÉ MOYENNE)

#### 2.1 Option A : `map_overlap` pour Gestion des Voisins (Recommandé)

Xarray/Dask fournit `map_overlap` pour opérations nécessitant accès aux voisins sans rechunking.

**Nouveau design :**
```python
def _compute_transport_with_overlap(
    state: xr.DataArray,
    u: xr.DataArray,
    v: xr.DataArray,
    # ... autres paramètres
) -> dict[str, xr.DataArray]:
    """Compute transport using map_overlap to avoid rechunking."""

    dim_y = Coordinates.Y.value
    dim_x = Coordinates.X.value

    # Define kernel that processes one chunk + overlap
    def transport_kernel(state_block, u_block, v_block, ...):
        """Process a single spatial block with 1-cell overlap."""
        # Accès direct aux voisins dans le bloc padé
        state_east = state_block[..., :, 1:]   # Shift x
        state_west = state_block[..., :, :-1]
        state_north = state_block[..., 1:, :]  # Shift y
        state_south = state_block[..., :-1, :]

        # Calculer flux, divergence, tendances...
        return advection_rate, diffusion_rate

    # Apply kernel with overlap
    result = xr.map_blocks(
        transport_kernel,
        state, u, v, ...,
        kwargs={...},
        template=state,  # Output shape
    )

    return result
```

**Avantages :**
- ✅ Pas de rechunking global : chaque chunk traité indépendamment
- ✅ Overlap géré automatiquement par Dask
- ✅ Scalabilité O(N chunks)

**Inconvénients :**
- ⚠️ Refactoring significatif de `_prepare_transport_data()` et flux functions
- ⚠️ Gestion des boundary conditions plus complexe (edges du domaine global)

#### 2.2 Option B : Ajouter Avertissement + Recommander Numba (Simple)

**Changement minimal :**
1. Améliorer le warning existant dans `_validate_chunking_for_transport()` :
   ```python
   warnings.warn(
       "Transport computation: Core dimensions (Y, X) are chunked, "
       "which will trigger expensive rechunking.\n\n"
       "RECOMMENDATION: Use compute_transport_numba() instead, which "
       "handles spatial chunking efficiently via apply_ufunc.\n\n"
       "To rechunk your data optimally:\n"
       "  state.chunk({cohort: 5, y: -1, x: -1})",
       PerformanceWarning
   )
   ```

2. Ajouter une note dans la docstring de `compute_transport_xarray()` :
   ```python
   def compute_transport_xarray(...):
       """Compute transport tendencies using Xarray implementation.

       Performance Note:
           This implementation does NOT support chunking along Y or X dimensions
           due to shift/roll operations. For distributed computing with spatial
           chunking, use `compute_transport_numba()` instead.

           Optimal chunking strategy:
           - Chunk along cohort/time dimensions: chunk({cohort: 5})
           - Keep Y and X dimensions un-chunked: chunk({y: -1, x: -1})
       """
   ```

**Avantages :**
- ✅ Effort minimal (documentation uniquement)
- ✅ Version Numba déjà optimale et testée
- ✅ Users guidés vers la solution performante

**Inconvénients :**
- ⚠️ Version Xarray reste limitée (mais c'est accepté et documenté)

#### 2.3 Recommandation pour Transport

**Stratégie proposée : Option B (Avertissement + Recommander Numba)**

**Justification :**
1. **Version Numba déjà optimale** : Pas besoin de dupliquer l'effort
2. **Cas d'usage clair** :
   - Xarray : Développement, prototypage, petites grilles
   - Numba : Production, simulations distribuées, grandes grilles
3. **Coût/bénéfice** : Refactoring de la version Xarray avec `map_overlap` est lourd pour un gain limité

**Si refactoring souhaité plus tard :**
- Implémenter Option A (`map_overlap`) dans une **nouvelle fonction** `compute_transport_xarray_distributed()`
- Garder `compute_transport_xarray()` actuel pour compatibilité backward
- Ajouter tests de performance comparatifs

---

## 📅 Roadmap d'Implémentation

### Phase 1 : LMTL Production Dynamics (2-3h)
**Priorité : 🔴 CRITIQUE**

#### Tâches
- [x] **T1.1** : Créer helper `_safe_cohort_shift()` dans `lmtl/core.py`
- [x] **T1.2** : Remplacer `reindex()` par calcul direct de `d_tau`
- [x] **T1.3** : Remplacer `shift()` par `_safe_cohort_shift()` (2 occurrences)
- [x] **T1.4** : Exécuter tests de correctness : `pytest tests/test_lmtl_chunking.py -v`
- [x] **T1.5** : Exécuter tests d'efficacité : `pytest tests/test_lmtl_dask_efficiency.py -v`
- [x] **T1.6** : Vérifier que rechunking n'est plus détecté

#### Critères de Succès
- ✅ Tous les tests de correctness passent (résultats inchangés)
- ✅ Tests d'efficacité ne détectent plus de rechunking
- ✅ Task count linéaire O(N chunks) confirmé
- ✅ Pas de warnings de performance

#### Livrable
- Code refactorisé dans `seapopym/lmtl/core.py`
- Tests verts confirmant correctness + efficiency

---

### Phase 2 : Transport Documentation (1h)
**Priorité : 🟡 HAUTE**

#### Tâches
- [x] **T2.1** : Améliorer warning dans `_validate_chunking_for_transport()`
- [x] **T2.2** : Ajouter Performance Note dans docstring `compute_transport_xarray()`
- [x] **T2.3** : Ajouter exemple de chunking optimal dans docstring
- [x] **T2.4** : Créer guide utilisateur dans documentation :
  - Quand utiliser Xarray vs Numba
  - Stratégies de chunking recommandées
  - Exemples de rechunking avant transport

#### Critères de Succès
- ✅ Warning guide les utilisateurs vers Numba
- ✅ Documentation claire sur limitations Xarray
- ✅ Exemples de code fonctionnels

#### Livrable
- Code mis à jour : `seapopym/transport/core.py`
- Documentation : `docs/performance/chunking_guide.md` (nouveau fichier)

---

### Phase 3 : Validation Complète (1-2h)
**Priorité : 🟢 IMPORTANTE**

#### Tâches
- [x] **T3.1** : Exécuter suite complète de tests chunking
  ```bash
  pytest tests/test_lmtl_chunking.py tests/test_transport_chunking.py -v
  ```
- [x] **T3.2** : Exécuter suite complète de tests efficiency
  ```bash
  pytest tests/test_lmtl_dask_efficiency.py tests/test_transport_dask_efficiency.py -v
  ```
- [x] **T3.3** : Générer rapport de coverage
  ```bash
  pytest --cov=seapopym.lmtl --cov=seapopym.transport --cov-report=html
  ```
- [x] **T3.4** : Profiler exemple réel avec Dask dashboard
  - Lancer cluster Dask local
  - Exécuter simulation avec chunking
  - Vérifier graph de tâches (linéarité)
  - Capturer screenshots dashboard
- [x] **T3.5** : Mettre à jour rapport de tests `IA/Rapport_Tests_Chunking.md`

#### Critères de Succès
- ✅ Tous les tests passent (correctness + efficiency)
- ✅ Coverage stable ou amélioré
- ✅ Dashboard Dask confirme scalabilité linéaire
- ✅ Rapport mis à jour avec statut ✅ RÉSOLU

#### Livrable
- Rapport mis à jour : `IA/Rapport_Tests_Chunking.md`
- Screenshots dashboard Dask : `IA/profiling/dask_graphs_after_fix/`

---

### Phase 4 : Transport Map_Overlap (OPTIONNEL, 5-8h)
**Priorité : 🔵 BASSE - Seulement si besoin critique identifié**

#### Prérequis
- Validation que solution Numba ne suffit pas
- Cas d'usage concret nécessitant version Xarray avec chunking spatial

#### Tâches
- [ ] **T4.1** : Design détaillé `compute_transport_xarray_distributed()`
- [ ] **T4.2** : Implémenter kernel avec `map_overlap`
- [ ] **T4.3** : Gérer boundary conditions aux edges du domaine global
- [ ] **T4.4** : Tests de correctness vs version originale
- [ ] **T4.5** : Tests d'efficacité avec chunking spatial
- [ ] **T4.6** : Benchmarks comparatifs Xarray-overlap vs Numba

#### Critères de Succès
- ✅ Version Xarray distribué supporte chunking spatial
- ✅ Performance comparable à Numba
- ✅ Backward compatible (ancienne fonction conservée)

#### Livrable
- Nouvelle fonction : `compute_transport_xarray_distributed()`
- Tests : `tests/test_transport_distributed.py`
- Benchmarks : `benchmarks/bench_transport_methods.py`

**Note :** Cette phase sera décidée après Phase 3, selon les besoins réels.

---

## 📈 Métriques de Succès

### Métriques Techniques

| Métrique                          | Avant (Baseline)      | Objectif Après Fix    | Validation                    |
| :-------------------------------- | :-------------------- | :-------------------- | :---------------------------- |
| **Task Count (LMTL)**             | O(N²) ou instable     | O(N chunks)           | Tests efficiency              |
| **Rechunking Ops (LMTL)**         | Détecté (multiple)    | 0 rechunking implicite| Analyse graph Dask            |
| **Task Count (Transport Xarray)** | O(N²) spatial chunks  | Warning si chunked    | Validation warning            |
| **Warnings Performance**          | Non bloquant          | Clair + actionnable   | Review user feedback          |
| **Correctness Tests**             | 100% pass             | 100% pass (inchangé)  | CI doit rester vert           |
| **Efficiency Tests**              | Fail (rechunking)     | Pass (no rechunking)  | `test_*_dask_efficiency.py`   |

### Métriques Utilisateur

| Aspect                   | Avant                 | Après                             |
| :----------------------- | :-------------------- | :-------------------------------- |
| **Simulations Distribuées** | Bloqué (task explosion) | ✅ Scalabilité linéaire          |
| **Temps Calcul (100 cohorts)** | Prohibitif (timeout)    | Linéaire avec nombre de workers  |
| **Memory Usage**         | Saturation scheduler  | Stable, prévisible                |
| **Guidance Utilisateur** | Manquante             | Documentation claire chunking     |

---

## 🔍 Risques et Mitigation

### Risque 1 : Régression Correctness
**Probabilité :** 🟡 Moyenne
**Impact :** 🔴 Critique

**Mitigation :**
- ✅ Tests de correctness exhaustifs existants (`test_lmtl_chunking.py`)
- ✅ Validation chunk par chunk avec `rtol=1e-10`
- ✅ CI doit rester verte à chaque commit

### Risque 2 : Performances Dégradées
**Probabilité :** 🟢 Faible
**Impact :** 🟡 Moyen

**Mitigation :**
- ✅ Solutions proposées sont optimisations pures (pas de calculs supplémentaires)
- ✅ `concat` et slicing sont opérations lazy Dask efficaces
- ✅ Benchmarks pour confirmer amélioration

### Risque 3 : Cas Edge Non Testés
**Probabilité :** 🟡 Moyenne
**Impact :** 🟡 Moyen

**Mitigation :**
- ✅ Tests existants couvrent multiple chunk strategies
- ✅ Ajouter tests edge cases :
  - Cohort unique (N=1)
  - Chunking extrême (chunk_size=1)
  - Boundary conditions variées (PERIODIC, CLOSED)

### Risque 4 : Refactoring Transport Map_Overlap Complexe
**Probabilité :** 🔴 Haute (si Phase 4)
**Impact :** 🟡 Moyen

**Mitigation :**
- ✅ Phase 4 est **optionnelle** et basse priorité
- ✅ Version Numba couvre les besoins critiques
- ✅ Si implémenté : nouvelle fonction séparée (pas de modification version existante)

---

## 📚 Références et Ressources

### Documentation Dask
- [Dask Array Slicing](https://docs.dask.org/en/latest/array-slicing.html) - Opérations lazy efficaces
- [Dask Map Overlap](https://docs.dask.org/en/latest/array-overlap.html) - Opérations avec voisinage
- [Avoiding Rechunking](https://docs.dask.org/en/latest/array-best-practices.html#avoid-rechunking) - Best practices

### Xarray + Dask
- [Xarray Chunking Guide](https://docs.xarray.dev/en/stable/user-guide/dask.html#chunking)
- [Xarray Apply UFUNC](https://docs.xarray.dev/en/stable/user-guide/computation.html#wrapping-custom-computation) - Pour Numba integration

### Patterns Similaires dans l'Écosystème
- **Pangeo** : Ocean modeling with Xarray + Dask - Patterns de chunking spatial
- **Xarray-Simlab** : Modèles couplés avec Dask - Gestion des voisinages
- **Xgcm** : Grid operations preserving chunk structure

### Articles Techniques
- Hoyer & Hamman (2017) : "xarray: N-D labeled Arrays and Datasets in Python"
- Rocklin (2015) : "Dask: Parallel Computation with Blocked algorithms"

---

## 🎯 Prochaines Étapes Immédiates

### Action 1 : Validation du Plan
- [ ] Review du plan par l'équipe
- [ ] Validation des priorités
- [ ] Confirmation de la roadmap

### Action 2 : Setup Environnement
- [ ] Créer branche `fix/dask-rechunking-efficiency`
- [ ] Setup Dask dashboard pour profiling
- [ ] Préparer données test réalistes (100+ cohorts)

### Action 3 : Démarrage Phase 1
- [ ] Implémenter `_safe_cohort_shift()`
- [ ] Refactoriser `compute_production_dynamics()`
- [ ] Exécuter tests de validation

---

## 📝 Notes d'Implémentation

### Principes de Design
1. **Préservation de la Correction** : Zéro régression numérique acceptée
2. **Opérations Lazy** : Favoriser slicing/concat vs operations eagerly evaluated
3. **Chunk-Aware** : Toujours vérifier l'impact sur le graph Dask
4. **Testabilité** : Chaque changement validé par tests automatisés
5. **Documentation** : Guide l'utilisateur vers les bonnes pratiques

### Code Review Checklist
- [ ] Aucun `reindex()` avec `method=` sur données chunkées
- [ ] Aucun `shift()` sur dimensions chunkées sans raison impérieuse
- [ ] `concat` utilisé pour combiner chunks (pas de rechunking forcé)
- [ ] Warnings de performance clairs et actionnables
- [ ] Docstrings mentionnent stratégies de chunking recommandées

---

**Plan créé le** : 2026-01-12
**Dernière mise à jour** : 2026-01-12
**Auteur** : Claude (analyse basée sur tests et code source)

**Prêt pour implémentation :** ✅ OUI - Phase 1 peut démarrer immédiatement
