# Workflow State

## Informations générales

- **Projet** : SeapoPym-JAX - Phase 3 (Engine)
- **Étape courante** : 9. Finalisation ✓
- **Rôle actif** : Développeur
- **Dernière mise à jour** : 2026-01-26
- **Statut** : COMPLÉTÉ

## Résumé du besoin

**Objectif** : Implémenter le module `seapopym/engine/` qui orchestre la boucle temporelle, gère les I/O et supporte deux backends (JAX/NumPy).

**Livrables attendus** (selon SPEC_03) :

- Step kernel (`step_fn`) : logique d'un pas de temps
- `StreamingRunner` : mode production avec chunking et I/O asynchrone
- `GradientRunner` : mode optimisation compatible autodiff
- `JAXBackend` : boucle via `jax.lax.scan`
- `NumpyBackend` : boucle via `for` Python
- I/O asynchrone entre chunks

**Dépendances amont** :

- Phase 2 (Compiler) fournit le `CompiledModel` ✓

## Rapport d'analyse

### Structure du projet

```
seapopym/
├── blueprint/           # Phase 1 - Schémas Pydantic, registry @functional, validation
├── compiler/            # Phase 2 - CompiledModel, inference, transpose, preprocessing
├── functions/           # Fonctions @functional (biology.py)
└── _legacy/             # Ancien code archivé
tests/
├── blueprint/           # 40 tests
├── compiler/            # 66 tests
└── _legacy/             # Tests archivés
```

### Technologies identifiées

- **Langage** : Python 3.10+
- **Schémas** : Pydantic v2 (BaseModel, ConfigDict)
- **Arrays** : JAX/NumPy dual backend, xarray pour I/O
- **Graphes** : NetworkX (DiGraph)
- **Tests** : pytest avec coverage
- **Qualité** : Ruff + Pyright (strict)

### Patterns et conventions

- **Nommage** : snake_case (fonctions), PascalCase (classes)
- **Type alias** : `Array = Any` pour compatibilité JAX/NumPy
- **Exceptions** : Hiérarchie avec codes (E1xx Blueprint, E2xx Compiler)
- **Exports** : `__init__.py` explicites avec docstrings

### Interfaces clés pour Engine

| Interface | Module | Usage |
| --------- | ------ | ----- |
| `CompiledModel` | compiler/model.py | Données pytrees (state, forcings, parameters, shapes, graph) |
| `get_function(name, backend)` | blueprint/registry.py | Récupère une fonction enregistrée |
| `FunctionMetadata` | blueprint/registry.py | Métadonnées (func, core_dims, outputs) |
| `ExecutionPlan` | blueprint/execution.py | Groupes de tâches ordonnées |

### Points d'attention

- Le mask est dans `forcings["mask"]` (convention Compiler)
- Les paramètres peuvent être `float` ou `jnp.ndarray`
- Le graphe dans `CompiledModel.graph` vient de la validation Phase 1

## Décisions d'architecture

| Domaine | Choix | Rationale |
| ------- | ----- | --------- |
| Pattern Backend | Protocol (`Backend`) | Mock facile, extensibilité |
| Boucle JAX | `jax.lax.scan` | Compilation XLA, gradient-friendly |
| Boucle NumPy | `for` Python | Debug facile, pas de dépendance JAX |
| I/O async | `ThreadPoolExecutor` | GIL libéré pendant JAX |
| Step fn | Closure sur graph/registry | JIT-friendly |
| Intégrateur | Euler explicite (V1) | Simple, suffisant |
| Exceptions | E3xx | Continuité avec E1xx (Blueprint), E2xx (Compiler) |

### Structure proposée

```
seapopym/engine/
├── __init__.py          # Exports publics
├── exceptions.py        # E300-E304
├── backends.py          # Protocol Backend, JAXBackend, NumpyBackend
├── step.py              # build_step_fn()
├── runners.py           # StreamingRunner, GradientRunner
└── io.py                # AsyncWriter
```

### Interfaces

```python
class Backend(Protocol):
    def scan(self, step_fn, init, xs) -> tuple[Carry, Y]: ...

def build_step_fn(model: CompiledModel) -> Callable[[State, Forcings_t], tuple[State, Outputs]]

class StreamingRunner:
    def run(self, output_path: str | Path) -> None

class GradientRunner:
    def run(self) -> tuple[State, Outputs]
```

### Risques identifiés

| Risque | Impact | Mitigation |
| ------ | ------ | ---------- |
| Graph complexe | Moyen | Test sur modèle toy d'abord |
| Memory GradientRunner | Moyen | Limiter T en V1, checkpointing Phase 5 |

## Todo List

| État | ID  | Nom                     | Description                                                              | Dépendances | Résolution |
| ---- | --- | ----------------------- | ------------------------------------------------------------------------ | ----------- | ---------- |
| ✓    | T1  | Créer structure package | Créer `seapopym/engine/__init__.py` (vide)                               | -           | Fait       |
| ✓    | T2  | Implémenter exceptions  | Créer `exceptions.py` avec E300-E304                                     | T1          | Fait       |
| ✓    | T3  | Implémenter backends    | Créer `backends.py` avec Protocol, JAXBackend, NumpyBackend              | T2          | Fait       |
| ✓    | T4  | Implémenter step        | Créer `step.py` avec `build_step_fn()`                                   | T2, T3      | Fait       |
| ✓    | T5  | Implémenter io          | Créer `io.py` avec `AsyncWriter`                                         | T2          | Fait       |
| ✓    | T6  | Implémenter runners     | Créer `runners.py` avec `StreamingRunner`, `GradientRunner`              | T3, T4, T5  | Fait       |
| ✓    | T7  | Exporter API publique   | Mettre à jour `__init__.py` avec exports                                 | T6          | Fait       |
| ✓    | T8  | Tests backends          | Créer `tests/engine/test_backends.py`                                    | T3          | Fait       |
| ✓    | T9  | Tests step              | Créer `tests/engine/test_step.py`                                        | T4          | Fait       |
| ✓    | T10 | Tests io                | Créer `tests/engine/test_io.py`                                          | T5          | Fait       |
| ✓    | T11 | Tests runners           | Créer `tests/engine/test_runners.py`                                     | T6          | Fait       |
| ✓    | T12 | Test intégration E2E    | Test Blueprint → Compile → Run sur modèle toy                            | T7          | Fait       |
| ✓    | T13 | Vérification qualité    | Ruff + Pyright sans erreurs                                              | T12         | 149 tests, 0 erreurs |

## Historique des transitions

| De | Vers | Raison | Date |
| -- | ---- | ------ | ---- |
| - | 1. Initialisation | Démarrage Phase 3 | 2026-01-26 |
| 1. Initialisation | 2. Analyse | Besoin validé par l'utilisateur | 2026-01-26 |
| 2. Analyse | 3. Architecture | Analyse complétée | 2026-01-26 |
| 3. Architecture | 4. Planification | Architecture validée | 2026-01-26 |
| 4. Planification | 5. Execution | Todo list complétée (13 tâches) | 2026-01-26 |
| 5. Execution | 9. Finalisation | Toutes les tâches terminées, 149 tests passent | 2026-01-26 |

## Résumé de l'implémentation

### Fichiers créés

- `seapopym/engine/__init__.py` - Exports publics
- `seapopym/engine/exceptions.py` - Exceptions E300-E304
- `seapopym/engine/backends.py` - Protocol Backend, JAXBackend, NumpyBackend
- `seapopym/engine/step.py` - build_step_fn() avec intégration Euler
- `seapopym/engine/io.py` - AsyncWriter avec Zarr
- `seapopym/engine/runners.py` - StreamingRunner, GradientRunner
- `tests/engine/__init__.py`
- `tests/engine/test_backends.py` - 7 tests
- `tests/engine/test_step.py` - 7 tests
- `tests/engine/test_io.py` - 9 tests
- `tests/engine/test_runners.py` - 10 tests
- `tests/engine/test_integration.py` - 10 tests E2E

### Corrections appliquées

1. Static forcings (mask) slicing - Fixed with `np.broadcast_to`
2. Race condition in async zarr writes - Fixed with `threading.Lock`
3. Empty outputs from step_fn - Fixed by including state in outputs
4. Graph nodes as objects not dicts - Fixed by using `isinstance(node, ComputeNode)`
5. Registry isolation between tests - Fixed with `autouse=True` fixtures
