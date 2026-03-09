# Workflow State

## Informations generales

- **Projet** : Decomposition du Runner en fonctions
- **Etape courante** : 9. Finalisation
- **Role actif** : Facilitateur
- **Derniere mise a jour** : 2026-03-09

## Resume du besoin

### Quoi

Decomposer la classe `Runner` en fonctions pures et unifier le mecanisme de sortie via le Protocol `OutputWriter`.

Aujourd'hui le Runner melange deux responsabilites :
- **Moteur d'execution** : chunking, scan, step_fn, chargement des forcings
- **Orchestration** : merge de params, vmap, writer lifecycle, choix disk/memory

Le refactoring produit :
- **`run()`** : fonction moteur unique — execute N chunks de `lax.scan`, delegue la sortie au writer
- **`simulate()`** : sucre syntaxique — construit step_fn + writer, appelle `run()`
- **`WriterRaw`** : nouveau writer pour les sorties JAX brutes (remplace le pattern `collected`)
- **`build_writer()`** : fonction I/O extraite de `Runner._build_writer()`
- La classe `Runner` et `RunnerConfig` disparaissent

### Pourquoi

- **Separation des responsabilites** : le moteur ne doit pas connaitre vmap, le merge de params, ni le choix du format de sortie
- **Composabilite** : l'optimiseur, le futur Sobol, ou tout consommateur peut appeler `run()` directement et composer ses propres transforms JAX (vmap, grad)
- **Simplification** : `RunnerConfig` disparait — chaque fonction prend ses parametres directement
- **Unification** : `run_full` = `run_chunked` avec 1 seul chunk → une seule fonction `run()`
- **API unifiee via Writer** : le format de sortie (raw JAX, xr.Dataset, Zarr) est determine par le writer, pas par la fonction d'execution

### Concepts cles

#### `export_variables` est fondamental

`export_variables` determine ce que `lax.scan` accumule a chaque pas. Sans filtrage, scan stocke TOUS les intermediaires + state a chaque timestep — multiplicateur de memoire direct. Il doit TOUJOURS etre passe a `build_step_fn()`, que ce soit pour simulation ou optimisation.

#### Trois niveaux de sortie via le Protocol OutputWriter

| Writer | `append()` | `finalize()` | JAX-traceable | Usage |
|--------|-----------|-------------|---------------|-------|
| `WriterRaw` | stocke arrays JAX dans une liste | `jnp.concatenate` | **oui** | optimisation, bas niveau |
| `WriterMemory` | stocke arrays JAX dans une liste | concat + `xr.Dataset` | non | simulation en memoire |
| `WriterDisk` | `jax.device_get()` + ecriture Zarr | retourne None | non | gros jeux de donnees |

- **WriterRaw** est le seul writer JAX-traceable → seul utilisable dans `jax.vmap`/`jax.grad`
- **WriterDisk** utilise `jax.device_get()` pour le transfert device→host (pas `np.asarray()`)
- `run()` utilise `WriterRaw` par defaut si aucun writer n'est fourni

### Cycles d'execution cibles

**`run()` — fonction moteur unique** :
```python
def run(step_fn, model, state, params, chunk_size=None, writer=None):
    """Moteur d'execution pur. Une seule boucle, writer pluggable."""
    if writer is None:
        writer = WriterRaw()
    chunk_size = chunk_size or model.n_timesteps

    for start, end in chunk_ranges(model.n_timesteps, chunk_size):
        forcings = model.forcings.get_chunk(start, end)
        (state, params), outputs = lax.scan(step_fn, (state, params), forcings)
        writer.append(outputs)

    return state, writer.finalize()
```

**`simulate()` — sucre syntaxique** :
```python
def simulate(model, chunk_size=None, output_path=None, export_variables=None):
    export_variables = export_variables or list(model.state.keys())
    step_fn = build_step_fn(model, export_variables=export_variables)
    writer = build_writer(model, output_path, export_variables)
    return run(step_fn, model, dict(model.state), dict(model.parameters),
               chunk_size=chunk_size, writer=writer)
```

**Optimisation (l'optimiseur gere vmap)** :
```python
step_fn = build_step_fn(model, export_variables=["biomass"])

def eval_one(single_free):
    merged = {**model.parameters, **single_free}
    state, outputs = run(step_fn, model, dict(model.state), merged, chunk_size=5)
    return outputs

jax.vmap(eval_one)(population_params)  # WriterRaw par defaut, JAX-traceable
```

### API publique cible

```python
from seapopym.engine import build_step_fn, run, simulate
from seapopym.engine import WriterRaw, WriterMemory, WriterDisk
```

Six exports. Pas de classe Runner. Pas de RunnerConfig.

### Perimetre (in scope)

- Creer `WriterRaw` (conforme au Protocol `OutputWriter`)
- Creer `run()` comme fonction moteur unique (unifie `_run_chunked` et `_run_full`)
- Creer `simulate()` comme sucre syntaxique
- Extraire `build_writer()` dans `io.py`
- Remplacer `np.asarray()` par `jax.device_get()` dans `DiskWriter.append()`
- Supprimer `Runner`, `RunnerConfig`, `_scan()`
- Adapter les tests existants
- Adapter les imports dans `engine/__init__.py`

### Hors perimetre

- Modification de l'optimiseur (workflow dedie — il continuera a utiliser Runner via un shim temporaire ou sera adapte separement)
- Modification de `step.py` (build_step_fn inchange)
- Modification du Protocol `OutputWriter` (WriterRaw le respecte tel quel)
- Ajout de `grad`, `pmap`, `checkpoint` (architecture compatible, pas implemente)

### Contraintes

- Breaking change OK (utilisateur unique)
- L'optimiseur utilise `Runner.optimization()` et `runner(model, params)` — necessite un shim temporaire ou une adaptation dans un workflow dedie
- Les tests existants (456 passed) doivent continuer a passer apres adaptation
- `WriterRaw` doit etre JAX-traceable (utilisable dans `jax.vmap`)

### Risques

- L'optimiseur reference `Runner` directement — risque de casser l'optimisation tant que le workflow dedie n'est pas fait
- `_scan()` (wrapper jit) est utilise pour le chunking mais pas pour le full scan — a decider si on unifie avec ou sans jit dans `run()`

## Rapport d'analyse

### Structure du perimetre

Le refactoring touche 1 module principal (`seapopym/engine/`) avec 4 fichiers :

| Fichier | Role | Statut |
|---------|------|--------|
| `engine/runner.py` | Runner + RunnerConfig + _scan | **a remplacer** |
| `engine/io.py` | OutputWriter Protocol, DiskWriter, MemoryWriter | **a modifier** (ajout WriterRaw, extraction build_writer, np.asarray→device_get) |
| `engine/__init__.py` | Exports publics | **a modifier** (Runner→run/simulate, ajout WriterRaw/MemoryWriter) |
| `engine/step.py` | build_step_fn | **inchange** |
| `engine/vectorize.py` | vmap wrapping | **inchange** |
| `engine/exceptions.py` | EngineError, ChunkingError, EngineIOError | **inchange** |

### Consommateurs identifies

| Consommateur | Usage actuel | Impact |
|-------------|-------------|--------|
| `optimization/_common.py` | `runner(model, free_params, export_variables=...)` dans `build_loss_fn` | **shim ou adaptation** — utilise Runner comme callable |
| `optimization/cmaes.py` | `Runner` en TYPE_CHECKING, `self.runner` stocke | idem |
| `optimization/ga.py` | idem | idem |
| `optimization/ipop.py` | idem | idem |
| `optimization/gradient_optimizer.py` | idem | idem |
| `tests/engine/test_runner.py` | Tests Runner config, simulation, optimization | **a reecrire** |
| `tests/engine/test_integration.py` | Tests E2E avec Runner | **a adapter** |
| `tests/compiler/test_optimization_runner.py` | Tests Runner en mode optimization | **a adapter** |
| `tests/optimization/test_*.py` | Tests optimiseurs avec Runner | **hors perimetre** (workflow optimiseur) |
| `examples/01..09` | Divers usages Runner | **a adapter** |
| `article/notebooks/` | Usages Runner | **a adapter** |
| `examples/09_memory_profile_simulation.py` | Utilise `_scan` directement + writers manuellement | **deja proche de l'API cible** |

### Patterns identifies

1. **Pattern simulation** (Runner.simulation → runner.run) :
   - Utilise dans tests, exemples, articles
   - Construit step_fn + writer, dispatch chunked/full, lifecycle writer (init/append/finalize/close)

2. **Pattern optimization** (Runner.optimization → runner.__call__) :
   - Utilise par les optimiseurs via `build_loss_fn` dans `_common.py`
   - Le runner est passe en parametre, appele comme `runner(model, free_params)`
   - L'optimiseur CMA-ES fait son propre `jax.vmap(eval_one)` AU-DESSUS du runner
   - Le runner fait aussi `jax.vmap(eval_one)` si `vmap_params=True` — double vmap potentiel

3. **Pattern bas niveau** (example 09) :
   - Construit step_fn, writer, fait la boucle manuellement avec `_scan`
   - Deja tres proche de `run()` cible

4. **`_scan` (JIT wrapper)** :
   - Utilise UNIQUEMENT dans `_run_chunked`, pas dans `_run_full`
   - `_run_full` appelle `lax.scan` directement
   - Decision : dans `run()`, utiliser `lax.scan` directement (pas de JIT explicite — JAX le fait automatiquement via tracing)

### Points d'attention

1. **Optimiseur fait deja son propre vmap** : Dans `cmaes.py:130`, `eval_population = jax.jit(jax.vmap(eval_one))` — si le runner a aussi `vmap_params=True`, c'est un double vmap. Le refactoring clarifie ca : `run()` ne fait JAMAIS de vmap, c'est le consommateur qui decide.

2. **DiskWriter utilise `np.asarray()`** (`io.py:190`) : A remplacer par `jax.device_get()` pour un transfert device→host propre et intentionnel.

3. **MemoryWriter.finalize() utilise `np.asarray(jnp.concatenate())`** (`io.py:282`) : OK, pas a changer — c'est le bon pattern pour la conversion finale.

4. **`_scan` JIT wrapper** : Utilisee seulement dans `_run_chunked`. Dans `run()`, on peut utiliser `lax.scan` directement — JAX jit-compile automatiquement lors du tracing. Si besoin de JIT explicite pour le chunking, le consommateur peut wraper `run()` dans `jax.jit`.

5. **Writer lifecycle** : Actuellement `Runner.run()` gere `initialize/append/finalize/close` avec try/finally. La fonction `run()` ne fera que `append/finalize`. `simulate()` gerera `initialize/close`. Pour l'optimisation, `WriterRaw` n'a pas besoin d'initialisation lourde.

6. **Shim optimiseur** : `build_loss_fn` dans `_common.py` appelle `runner(model, free_params)`. On peut soit :
   - (a) Adapter `_common.py` pour utiliser `run()` directement (HORS PERIMETRE — workflow dedie)
   - (b) Fournir un shim `Runner` temporaire qui delègue a `run()` (IN SCOPE)
   - Decision : option (b) pour ne pas casser l'optimiseur

### Technologies

- Langage : Python 3.12
- Framework : JAX (lax.scan, jax.vmap, jax.jit, jax.device_get)
- I/O : Zarr, xarray
- Build : Hatchling + uv
- Tests : pytest (456 passed)
- Linting : Ruff, Pyright

### Conventions

- snake_case, type hints, 120 chars, Google docstrings
- Protocol (typing) pour les interfaces (OutputWriter)
- Composition fonctionnelle JAX preferee au pattern Strategy OO
- `export_variables` toujours passe a `build_step_fn()`

## Decisions d'architecture

### Choix techniques

| Domaine | Choix | Justification |
|---------|-------|---------------|
| Execution | `lax.scan` direct (pas de wrapper JIT) | JAX jit-compile via tracing ; `_scan` ajoute une indirection inutile |
| Writer par defaut | `WriterRaw` dans `run()` | Seul writer JAX-traceable, necessaire pour vmap/grad |
| Transfert device→host | `jax.device_get()` | Transfert explicite et intentionnel (remplace `np.asarray()`) |
| Retrocompat optimiseur | Shim `Runner` temporaire | Ne casse pas les optimiseurs, adaptes dans un workflow dedie |

### Structure proposee

```
seapopym/engine/
├── __init__.py          # exports: run, simulate, build_step_fn, build_writer,
│                        #          WriterRaw, WriterMemory, WriterDisk
├── run.py               # NEW — run(), simulate(), chunk_ranges()
├── io.py                # MODIFIED — ajout WriterRaw, ajout build_writer(), device_get
├── step.py              # INCHANGE
├── vectorize.py         # INCHANGE
├── exceptions.py        # INCHANGE
└── runner.py            # DEPRECATED — shim Runner/RunnerConfig qui delegue a run()
```

### Interfaces et contrats

#### `run()` — moteur d'execution

```python
def run(
    step_fn: Callable,
    model: CompiledModel,
    state: State,
    params: Params,
    chunk_size: int | None = None,
    writer: OutputWriter | None = None,
) -> tuple[State, Any]:
```

Contrat :
- Si `writer is None` → utilise `WriterRaw()` (pas d'initialisation necessaire)
- Si `chunk_size is None` → 1 seul chunk = tous les timesteps
- Retourne `(final_state, writer.finalize())`
- Ne fait JAMAIS de vmap — c'est le consommateur qui decide
- Ne gere PAS le lifecycle writer (initialize/close) — c'est `simulate()` ou le consommateur

#### `simulate()` — sucre syntaxique

```python
def simulate(
    model: CompiledModel,
    chunk_size: int | None = None,
    output_path: str | Path | None = None,
    export_variables: list[str] | None = None,
) -> tuple[State, Any]:
```

Contrat :
- Construit `step_fn` via `build_step_fn(model, export_variables=...)`
- Construit `writer` via `build_writer(model, output_path, export_variables)`
- Gere le lifecycle writer (initialize + close via try/finally)
- Appelle `run()` avec le writer

#### `build_writer()` — construction du writer

```python
def build_writer(
    model: CompiledModel,
    output_path: str | Path | None,
    export_variables: list[str],
) -> OutputWriter:
```

Contrat :
- `output_path is not None` → `DiskWriter` (initialise)
- `output_path is None` → `MemoryWriter` (initialise)
- Appelle `writer.initialize(shapes, variables, coords, var_dims)`
- Extrait de `Runner._build_writer()` — meme logique

#### `WriterRaw` — writer JAX-traceable

```python
class WriterRaw:
    def append(self, data: dict[str, Array]) -> None: ...
    def finalize(self) -> dict[str, Array]: ...
    def close(self) -> None: ...
```

Contrat :
- `append()` : stocke le dict dans une liste Python (trace-time, pas run-time)
- `finalize()` : `jax.tree.map(lambda *a: jnp.concatenate(a, axis=0), *chunks)` si N>1, sinon retourne le seul chunk
- Pas de `initialize()` necessaire (pas de coords/shapes a pre-configurer)
- JAX-traceable → utilisable dans `jax.vmap` et `jax.grad`

#### Shim `Runner` (temporaire)

```python
class Runner:
    """Deprecated — use run() and simulate() directly."""

    @classmethod
    def simulation(cls, chunk_size=None, output=None) -> Runner: ...

    @classmethod
    def optimization(cls, vmap=False, chunk_size=None) -> Runner: ...

    def run(self, model, output_path=None, export_variables=None):
        return simulate(model, chunk_size=..., output_path=..., export_variables=...)

    def __call__(self, model, free_params, export_variables=None):
        step_fn = build_step_fn(model, export_variables=export_variables)
        def eval_one(single_free):
            merged = {**model.parameters, **single_free}
            _, outputs = run(step_fn, model, dict(model.state), merged, chunk_size=...)
            return outputs
        if self._vmap:
            return jax.vmap(eval_one)(free_params)
        return eval_one(free_params)
```

Contrat :
- Meme interface que l'ancien Runner
- Delegue a `run()` et `simulate()`
- Emettra un `DeprecationWarning` a l'instanciation
- Supprime dans le workflow optimiseur

### Risques identifies

| Risque | Impact | Mitigation |
|--------|--------|------------|
| Shim Runner ne reproduit pas exactement le comportement | Moyen | Tests existants servent de validation |
| WriterRaw pas reellement JAX-traceable | Haut | Valide dans le workflow Runner (test chunked+vmap passe) |
| `lax.scan` sans JIT explicite plus lent en chunked | Bas | JAX trace+compile automatiquement ; benchmark si doute |
| DiskWriter casse avec `jax.device_get` | Bas | Meme semantique que `np.asarray`, test existant le valide |

## Todo List

| Etat | ID | Nom | Description | Dependances | Resolution |
|------|-----|-----|-------------|-------------|------------|
| ✅ | T1 | WriterRaw | Creer `WriterRaw` dans `io.py` — append stocke dans liste, finalize concatene via `jax.tree.map` + `jnp.concatenate`, close no-op. Pas de `initialize()`. | - | |
| ✅ | T2 | build_writer | Extraire `build_writer(model, output_path, export_variables)` dans `io.py` — logique identique a `Runner._build_writer()`. Retourne DiskWriter ou MemoryWriter initialise. | - | |
| ✅ | T3 | device_get | Dans `io.py`, remplacer `np.asarray(v)` par `jax.device_get(v)` dans `DiskWriter.append()`. | - | |
| ✅ | T4 | run.py | Creer `engine/run.py` avec `run(step_fn, model, state, params, chunk_size, writer)` et helper `chunk_ranges()`. Logique unifiee : 1 boucle de chunks, lax.scan direct, writer.append. | T1 | |
| ✅ | T5 | simulate | Ajouter `simulate(model, chunk_size, output_path, export_variables)` dans `run.py`. Construit step_fn + writer, gere lifecycle (try/finally close), appelle run(). | T2, T4 | |
| ✅ | T6 | shim Runner | Remplacer le contenu de `runner.py` par un shim : Runner/RunnerConfig delegent a run()/simulate(). DeprecationWarning a l'instanciation. | T4, T5 | |
| ✅ | T7 | __init__.py | Mettre a jour `engine/__init__.py` : exporter run, simulate, build_writer, WriterRaw, WriterMemory, DiskWriter. Garder Runner/RunnerConfig (shim). | T6 | |
| ✅ | T8 | tests runner | Adapter `tests/engine/test_runner.py` : les tests existants doivent passer avec le shim. Ajouter tests directs pour `run()` et `simulate()`. | T7 | |
| ✅ | T9 | tests integration | Adapter `tests/engine/test_integration.py` : remplacer `Runner` par `run()`/`simulate()` dans les tests E2E. Garder un test avec le shim. | T7 | |
| ✅ | T10 | tests optimization | Adapter `tests/compiler/test_optimization_runner.py` pour fonctionner avec le shim. | T6 | |
| ✅ | T11 | examples | Adapter les imports dans `examples/01..09` et `article/notebooks/` pour utiliser la nouvelle API ou le shim. | T7 | |

## Historique des transitions

| De | Vers | Raison | Date |
|----|------|--------|------|
| 1. Initialisation | 2. Analyse | Besoin valide | 2026-03-09 |
| 2. Analyse | 3. Architecture | Analyse completee | 2026-03-09 |
| 3. Architecture | 4. Planification | Architecture validee | 2026-03-09 |
| 4. Planification | 5. Execution | Todo list completee | 2026-03-09 |
| 5. Execution | 6. Revue | T1-T11 implementees | 2026-03-09 |
| 6. Revue | 8. Test | Code revu, redondance corrigee | 2026-03-09 |
| 8. Test | 9. Finalisation | 465 passed, 0 failed | 2026-03-09 |
