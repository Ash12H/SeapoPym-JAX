# Workflow State - Transport JAX

## Informations générales

- **Projet** : Transport JAX (advection + diffusion)
- **Étape courante** : Terminé
- **Rôle actif** : -
- **Dernière mise à jour** : 2026-01-29

## Résumé du besoin

Implémenter une fonction de transport (advection + diffusion) en **JAX** compatible avec la différentiation automatique.

### Spécifications techniques

**Méthode numérique :**
- **Volumes finis** avec schéma **upwind** pour l'advection
- Diffusion par différences centrées
- Équations :
  - Flux advectif : `F_adv = u_face × c_upwind × face_area`
  - Flux diffusif : `F_diff = -D_face × ∂c/∂n × face_area`
  - Tendance = `-divergence / cell_area × mask`

**Dimensions :**
- Core dims : `X`, `Y` (2D horizontal)
- Broadcasting sur dimension `C` (cohorts/production)

**Conditions aux bords :**
- CLOSED (0) : Pas de flux (terres, bords fermés)
- OPEN (1) : Gradient nul (flux peuvent sortir)
- PERIODIC (2) : Wrap-around (cylindre est-ouest)

**Masque :** Terres bloquant les flux

**Contraintes :**
- Compatible JAX
- Compatible différentiation automatique (jax.grad)
- Backend JAX préféré

### Périmètre
- **In scope** : Transport 2D, upwind, conditions aux bords, masque, broadcasting
- **Out of scope** : Schémas d'ordre supérieur (TVD, Lax-Wendroff), transport 3D

## Rapport d'analyse

### Structure du projet

```
seapopym/
├── blueprint/       # Définition déclarative de modèles (schemas, registry, validation)
├── compiler/        # Compilation de blueprints → CompiledModel (préprocessing, unités)
├── engine/          # Exécution (backends JAX/NumPy, runners, vectorize, step)
├── functions/       # Fonctions biologiques (@functional decorées)
├── optimization/    # Gradient, évolutionnaire, hybride (JAX+optax)
└── _legacy/         # Ancien code (transport Numba, etc.)
```

### Technologies identifiées

- **Langage** : Python 3.10+
- **Framework principal** : JAX (jax.numpy, jax.lax.scan, jax.vmap, jax.grad)
- **Données** : xarray, zarr, netCDF4
- **Optimisation** : optax, evosax (optionnel)
- **Legacy** : Numba (transport actuel)
- **Tests** : pytest, pytest-cov
- **Linter** : ruff, pyright

### Patterns et conventions

- **Nommage** : snake_case pour fonctions/variables, PascalCase pour classes
- **Fonctions** : Décorateur `@functional(name="namespace:function", backend="jax", ...)`
- **Dimensions canoniques** : `("E", "T", "F", "C", "Z", "Y", "X")`
- **Broadcasting** : `jax.vmap` automatique via `engine/vectorize.py`
- **Backend** : `jax.lax.scan` pour boucles temporelles, JIT compilation
- **Core dims** : Dimensions explicitement traitées (non broadcast)

### Points d'attention pour le transport JAX

1. **Incompatibilité Numba ↔ JAX** : Le code legacy utilise `@guvectorize` (Numba) avec boucles explicites. JAX interdit les boucles indexées mutables.

2. **Différentiabilité** : Le schéma upwind utilise `if u > 0 then c_center else c_neighbor`. En JAX, cela nécessite `jnp.where()` pour rester différentiable.

3. **Conditions aux bords** : Le legacy utilise des indices dynamiques (`ip1 = -1` pour CLOSED). En JAX, on utilise `jnp.roll()` pour PERIODIC et `jnp.pad()` ou slicing pour OPEN/CLOSED.

4. **Masque** : Multiplication par le masque (`* mask`) est compatible JAX et différentiable.

5. **Broadcasting C → (C, Y, X)** : La convention actuelle utilise `vmap` sur Y et X avec `core_dims={"state": ["C"]}`. Le transport devra définir `core_dims={"state": ["Y", "X"]}` car il opère sur la grille spatiale.

6. **Géométrie sphérique** : Le legacy calcule `dx(lat)`, `dy`, `cell_areas`, `face_areas`. Ces fonctions peuvent être réutilisées ou réimplémentées en JAX.

### Emplacement recommandé

- **Nouveau module** : `seapopym/functions/transport.py`
- **Même pattern que** : `seapopym/functions/biology.py`
- **Tests** : `tests/functions/test_transport.py`

## Décisions d'architecture

### Choix techniques

| Domaine | Choix | Justification |
|---------|-------|---------------|
| Backend | JAX pur | Différentiable, JIT compilable, compatible optimisation |
| Schéma advection | Upwind + `jnp.where` | Simple, stable, différentiable |
| Schéma diffusion | Différences centrées | Standard, différentiable |
| Conditions bords | `jnp.roll` + masquage | Compatible JAX, pas de boucles |
| Géométrie | Paramètres explicites | Agnostique de la grille (lat/lon, ORCA, etc.) |
| Broadcasting | `core_dims=["Y", "X"]` | vmap sur C pour multi-cohorts |

### Interface validée

```python
@functional(
    name="phys:transport_tendency",
    backend="jax",
    core_dims={all: ["Y", "X"]},
    outputs=["advection_rate", "diffusion_rate"],
)
def transport_tendency(
    state,        # Concentration (Y, X)
    u, v,         # Vitesses [m/s]
    D,            # Diffusion [m²/s]
    dx, dy,       # Distances entre centres [m]
    face_height,  # Hauteur faces E/W [m] (=dy pour grille simple, =e2u pour ORCA)
    face_width,   # Largeur faces N/S [m] (=dx pour grille simple, =e1v pour ORCA)
    cell_area,    # Aire cellules [m²]
    mask,         # Masque (1=océan, 0=terre)
    bc_north=0, bc_south=0, bc_east=0, bc_west=0,  # 0=CLOSED, 1=OPEN, 2=PERIODIC
) -> tuple[advection_rate, diffusion_rate]
```

### Structure proposée

```
seapopym/functions/
├── __init__.py        # Ajouter export
├── biology.py         # Existant
└── transport.py       # NOUVEAU

tests/functions/
└── test_transport.py  # NOUVEAU
```

### Risques identifiés

| Risque | Impact | Mitigation |
|--------|--------|------------|
| Différentiabilité upwind | Moyen | `jnp.where(u > 0, c_center, c_neighbor)` |
| Performance vs Numba | Bas | JIT compense ; acceptable pour optimisation |
| Masque aux bords | Bas | Multiplier flux par mask des deux cellules |

## Todo List

| État | ID | Nom | Description | Dépendances | Résolution |
|------|----|-----|-------------|-------------|------------|
| ☑ | T1 | Créer transport.py (base) | Créer `seapopym/functions/transport.py` avec : BoundaryType enum, fonctions `_get_neighbor_*` (east, west, north, south) utilisant `jnp.roll` et padding selon BC | - | Fichier créé avec BoundaryType + 4 helpers voisins + 4 helpers masques BC |
| ☑ | T2 | Ajouter flux advection | Ajouter fonction `_compute_advection_fluxes` dans transport.py : calcul des 4 flux (E,W,N,S) avec schéma upwind via `jnp.where` | T1 | Fonction ajoutée avec upwind différentiable |
| ☑ | T3 | Ajouter flux diffusion | Ajouter fonction `_compute_diffusion_fluxes` dans transport.py : calcul des 4 flux avec gradient centré | T1 | Fonction ajoutée avec gradient centré |
| ☑ | T4 | Ajouter transport_tendency | Ajouter fonction principale `transport_tendency` avec décorateur `@functional`, assemblage des flux et calcul divergence | T2, T3 | Fonction ajoutée avec @functional, docstring complète |
| ☑ | T5 | Mettre à jour __init__.py | Modifier `seapopym/functions/__init__.py` pour exporter `transport_tendency` | T4 | Export ajouté, import vérifié OK |
| ☑ | T6 | Créer tests transport | Créer `tests/functions/test_transport.py` avec tests : grille simple, conditions aux bords (CLOSED, OPEN, PERIODIC), masque, différentiabilité | T4 | 16 tests créés, tous passent |

## Historique des transitions

| De | Vers | Raison | Date |
|----|------|--------|------|
| 1. Initialisation | 2. Analyse | Besoin validé par l'utilisateur | 2026-01-29 |
| 2. Analyse | 3. Architecture | Analyse complétée | 2026-01-29 |
| 3. Architecture | 4. Planification | Architecture validée par l'utilisateur | 2026-01-29 |
| 4. Planification | 5. Execution | Todo list complétée | 2026-01-29 |
| 5. Execution | 6. Revue | Tâches T1-T5 complétées, code compile | 2026-01-29 |

## Rapport de revue

### Vérifications automatiques

| Outil | Résultat | Erreurs | Warnings |
|-------|----------|---------|----------|
| ruff check | ✅ | 0 | 0 |
| pyright | ✅ | 0 | 0 |
| ruff format | ❌ | 1 | - |

### Issues identifiées

| ID | Sévérité | Description | Fichier | Action |
|----|----------|-------------|---------|--------|
| I1 | Mineure | Formatage non conforme | seapopym/functions/transport.py | ☑ Corrigé avec ruff format |

### Analyse des tâches échouées

Aucune tâche échouée.

### Cohérence avec la codebase

- ✅ Convention de nommage snake_case respectée
- ✅ Structure du fichier cohérente avec biology.py
- ✅ Décorateur @functional utilisé correctement
- ✅ Docstrings complètes avec exemples
- ✅ Type hints présents

### Décision

1 issue mineure (formatage) → Passer à Resolution

## Historique des transitions (suite)

| De | Vers | Raison | Date |
|----|------|--------|------|
| 6. Revue | 7. Resolution | 1 issue mineure à corriger | 2026-01-29 |
| 7. Resolution | 8. Test | Issue I1 corrigée | 2026-01-29 |

## Tests

### Tests créés

| Fichier | Fonctionnalité testée | Nb tests | Types |
|---------|----------------------|----------|-------|
| tests/functions/test_transport.py | BoundaryType, neighbors, advection, diffusion, mask, différentiabilité, JIT, BC | 16 | Unitaire |

### Résultats d'exécution

- **Date** : 2026-01-29
- **Commande** : `uv run pytest tests/functions/test_transport.py -v`

| Statut | Nombre |
|--------|--------|
| ✅ Passés | 16 |
| ❌ Échoués | 0 |
| ⏭ Ignorés | 0 |
| **Total** | 16 |

### Couverture des tests

- ✅ BoundaryType enum
- ✅ Fonctions voisins (4 directions × 2 BC types)
- ✅ Champ uniforme → tendance nulle
- ✅ Diffusion lisse les gradients
- ✅ Advection déplace la masse
- ✅ Masque bloque les flux
- ✅ Différentiabilité par rapport à state
- ✅ Différentiabilité par rapport à D
- ✅ Compilation JIT
- ✅ Conditions aux bords PERIODIC, CLOSED, OPEN

## Historique des transitions (suite Test)

| De | Vers | Raison | Date |
|----|------|--------|------|
| 8. Test | 9. Finalisation | Tous les tests passent | 2026-01-29 |
| 9. Finalisation | Terminé | Commit c667fd9 créé | 2026-01-29 |

## Résumé final

### Ce qui a été réalisé

Implémentation d'une fonction de transport (advection + diffusion) en JAX pur :
- Schéma volumes finis avec advection upwind
- Diffusion par différences centrées
- Support des conditions aux bords (CLOSED, OPEN, PERIODIC)
- Masque pour les terres
- Interface agnostique de la grille (compatible lat/lon, ORCA, etc.)
- Entièrement différentiable (compatible jax.grad)

### Fichiers impactés

| Action | Fichier |
|--------|---------|
| Créé | `seapopym/functions/transport.py` |
| Modifié | `seapopym/functions/__init__.py` |
| Créé | `tests/functions/__init__.py` |
| Créé | `tests/functions/test_transport.py` |

### Statistiques

- Tâches planifiées : 6
- Tâches réussies : 6
- Tâches échouées : 0
- Tests créés : 16
- Tests passés : 16

### Actions de sauvegarde effectuées

- [x] git commit c667fd9 "feat(functions): add JAX transport function for advection and diffusion"
