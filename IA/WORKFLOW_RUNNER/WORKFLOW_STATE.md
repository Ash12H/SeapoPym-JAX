# Workflow State

## Informations générales

- **Projet** : Refactoring du Runner (Engine)
- **Étape courante** : 3. Architecture
- **Rôle actif** : Architecte
- **Dernière mise à jour** : 2026-03-09

## Résumé du besoin

### Quoi

Refactoring du Runner : corriger l'intégration des statics (workflow Compiler), consolider les chemins d'exécution, et ouvrir l'architecture aux transforms JAX (vmap, grad, pmap, checkpoint).

### Pourquoi

- **Bug statics** : `get_chunk()` ne retourne plus les statics (mask) → `step.py` fallback silencieux sur `mask=1.0`
- **Dead code** : `param_mode` et `loop_mode` dans RunnerConfig sont déclarés mais jamais lus
- **3 méthodes privées dupliquées** : `_run_simulation`, `_run_optimization`, `_run_optimization_vmap` partagent la même construction de step_fn
- **Chunked + vmap possible** : contrairement à ce que le docstring actuel affirme, on peut mettre vmap DANS la boucle Python (pas l'inverse) → gain mémoire significatif pour l'optimisation évolutionnaire

### Périmètre (in scope)

- Corriger l'intégration statics dans `build_step_fn()` (2 xfails à résoudre)
- Consolider les 3 chemins en 2 : `_run_chunked` et `_run_full`
- Retirer les options mortes de RunnerConfig (`param_mode`, `loop_mode`)
- Composition fonctionnelle des transforms JAX (vmap, grad, pmap, checkpoint)
- Support chunked + vmap (optimisation évolutionnaire avec mémoire réduite)
- Préserver les interfaces existantes (factory methods, OutputWriter Protocol)

### Hors périmètre

- Blueprint et Configuration (stables)
- Applications (Optimiseur, Sobol, etc.)
- Refactoring du ForcingStore (complété dans le workflow Compiler)
- Implémentation effective de `fori_loop`, `pmap`, `grad` (architecture compatible, pas implémentés dans ce workflow)

### Contraintes

- JAX comme backend de calcul (jit, vmap, scan, etc.)
- Xarray pour la gestion des données (le Runner consomme ce que le Compiler/ForcingStore fournit)
- Pas de régression de performance (benchmarks existants dans `examples/` à utiliser comme référence)

### Risques

- Les benchmarks existants ne sont peut-être pas fiables pour détecter des régressions
- Le refactoring touche le coeur de l'exécution — toute erreur impacte toutes les applications en aval

## Rapport d'analyse

### Structure du projet

```
seapopym/
├── blueprint/       # Déclaration du modèle (Pydantic, YAML) — STABLE
├── compiler/        # Blueprint+Config → CompiledModel (JAX arrays) — STABLE
├── engine/          # Runner, step, I/O, vectorize — CIBLE DU REFACTORING
├── optimization/    # Optimiseurs (CMA-ES, GA, Gradient) — HORS SCOPE
├── functions/       # Fonctions physiques (LMTL, transport) — STABLE
├── models/          # Blueprints YAML pré-définis — STABLE
├── types.py         # Array, State, Params, Forcings, Outputs
└── dims.py          # Ordre canonique des dimensions (E,T,F,C,Z,Y,X)
```

### Technologies identifiées

- Langage : Python 3.12
- Build : Hatchling, uv
- Calcul : JAX 0.4.20+, Xarray 2023.12+, NumPy 1.26+
- Validation : Pydantic 2.0+
- I/O : Zarr 2.16+, netCDF4, Dask (lazy loading)
- Optimisation : Optax (gradient), evosax (évolutionnaire)
- Tests : pytest, markers (slow, integration, unit, gpu)
- Linting : Ruff, Pyright, pydocstyle (Google convention)
- Conventions : snake_case, type hints complets, 120 chars/ligne

### Fichiers du Runner (cible du refactoring)

| Fichier                | Lignes | Rôle                                                |
| ---------------------- | ------ | --------------------------------------------------- |
| `engine/runner.py`     | 368    | Runner + RunnerConfig, 3 chemins d'exécution        |
| `engine/step.py`       | 306    | build_step_fn, résolution inputs, intégration Euler |
| `engine/io.py`         | 317    | DiskWriter (Zarr), MemoryWriter (xarray)            |
| `engine/vectorize.py`  | 227    | wrap_with_vmap, calcul d'axes broadcast             |
| `engine/exceptions.py` | 46     | Exceptions spécifiques                              |

### Architecture actuelle du Runner

**RunnerConfig** (frozen dataclass) :

- `param_mode`: "closure" | "carry"
- `loop_mode`: "scan" | "fori_loop"
- `vmap_params`: bool
- `chunk_size`: int | None
- `output_mode`: "disk" | "memory" | "raw"

**Factory methods** :

- `Runner.simulation(chunk_size, output)` → mode simulation
- `Runner.optimization(vmap)` → mode optimisation

**3 chemins d'exécution** :

1. `_run_simulation()` : chunking temporel + writer (DiskWriter ou MemoryWriter)
2. `_run_optimization()` : scan complet, un jeu de paramètres
3. `_run_optimization_vmap()` : scan complet + vmap sur population de paramètres

### Flux de données

```
CompiledModel
├── state: dict[str, Array]         → évolue à chaque pas
├── parameters: dict[str, Array]    → constants
├── forcings: ForcingStore           → lazy Xarray, get_chunk() / get_all_dynamic() / get_statics()
└── compute_nodes: list[ComputeNode] → chaîne de calcul ordonnée
         ↓
    build_step_fn(model)
         ↓
    lax.scan(step_fn, (state, params), forcings)
         ↓
    Per-timestep: resolve_inputs → compute_nodes → euler_integration → mask → outputs
         ↓
    writer.append(outputs)   [simulation]
    return outputs            [optimisation]
```

**Xarray** : les forcings sont stockés en DataArray (lazy via Dask/netCDF). `get_chunk(start, end)` matérialise et interpole les dynamiques. `get_all_dynamic()` charge tous les dynamiques (obligatoire pour grad, optionnel pour vmap — chunked+vmap est possible). `get_statics()` retourne les statiques (mask, etc.) en JAX arrays — à capturer en closure.

### Patterns et conventions

- Fonctions pures JAX-compatible (pas de side effects dans step_fn)
- Protocol pour OutputWriter (structural typing)
- @functional decorator pour les fonctions physiques avec metadata (core_dims, units)
- Namespace-prefixed function names ("biol:growth", "phys:advection")
- Frozen dataclass pour config immutable

### Points d'attention

1. **Triple chemin d'exécution** : duplication significative entre `_run_simulation`, `_run_optimization`, `_run_optimization_vmap`. Le `step_fn` est construit de la même façon mais les 3 chemins diffèrent par le wrapping (chunked loop, scan simple, vmap+scan).

2. **Validation combinatoire** dans `RunnerConfig.__post_init__` : les règles de compatibilité entre options (vmap+chunk interdit, disk+fori_loop interdit, etc.) croissent linéairement. Fragile à l'ajout de nouvelles combinaisons.

3. **Résolution d'inputs par strings** (`step.py:190-218`) : dispatch basé sur des chaînes ("state", "forcings", "parameters", "derived") avec recherche séquentielle. Fragile aux typos.

4. ~~**Incompatibilité structurelle chunking/vmap**~~ — PARTIELLEMENT FAUX. Le nesting `vmap(Python loop)` est impossible, mais `Python loop(vmap(scan))` fonctionne. On peut mettre vmap DANS chaque itération de la boucle Python. Les forcings sont en closure (partagés, pas dupliqués par individu). Seul `grad + chunked` reste impossible (accumulation de gradients inter-chunks complexe).

5. **Mémoire des forcings** : `get_all_dynamic()` charge tous les dynamiques en mémoire. Avec chunked+vmap, on peut réduire à `chunk_size × Y × X` au lieu de `n_timesteps × Y × X`. Gain proportionnel à `n_timesteps / chunk_size`.

6. **Clamping hard-codé** : `jnp.maximum(value, 0.0)` dans `_integrate_euler()` (`step.py:282`). Non configurable.

7. **Export variables filtré à chaque pas** : dict comprehension (`step.py:117`) dans la boucle interne du scan. Overhead mineur mais évitable.

8. ~~**Flux Xarray lazy/eager opaque**~~ — RÉSOLU par workflow Compiler. Le ForcingStore gère la transition lazy→JAX de manière explicite. `get_chunk()` et `get_statics()` matérialisent et valident (NaN) au moment de l'appel.

9. **Régression de performance suspectée** : l'utilisateur a observé des baisses de performances par rapport aux premières versions. Non confirmé, à valider avec des benchmarks.

10. **Mask récupéré depuis `forcings_t`** (`step.py:88`) : `mask = forcings_t.get("mask", 1.0)`. Depuis le workflow Compiler, `get_chunk()` ne fournit plus le mask (statique). Le fallback `1.0` masque silencieusement le bug — tous les tests E2E passent sauf `test_mask_zeros_state` (xfail). Le mask doit être injecté via la closure de `build_step_fn()`.

11. **`_resolve_inputs` cherche dans `forcings_t`** (`step.py:197,212`) : si un input de type `forcings.mask` est référencé par un process, `_resolve_inputs` le cherche dans `forcings_t` (dynamiques). Or les statics ne sont plus dans `forcings_t`. Il faut ajouter les statics comme source de résolution.

### Anatomie de step.py (fichier central)

`build_step_fn(model)` construit une closure `step_fn` compatible `lax.scan` :

```
build_step_fn(model)
│
├── Closure captures:
│   ├── compute_nodes        (list[ComputeNode])
│   ├── tendency_map         (dict[str, list[TendencySource]])
│   ├── dt                   (float, seconds)
│   └── vmapped_funcs        (dict, pre-computed vmap wrappers)
│       ⚠ NE CAPTURE PAS les statics (mask, etc.)
│
└── step_fn(carry=(state, params), forcings_t) → ((new_state, params), outputs)
    │
    ├── mask = forcings_t.get("mask", 1.0)      ← L88, BUG: mask n'est plus dans forcings_t
    ├── _resolve_inputs(mapping, state, forcings_t, params, intermediates)
    │   └── category="forcings" → forcings_t[var_name]   ← L212, BUG: statics absents de forcings_t
    ├── execute compute_nodes (process chain)
    ├── _integrate_euler(state, intermediates, tendency_map, dt)
    │   └── jnp.maximum(value + total * dt, 0.0)          ← L282, clamping hard-codé
    ├── _apply_mask(new_state, mask)
    └── filter export_variables                            ← L117
```

**Deux bugs liés au workflow Compiler** :
- `L88` : `mask = forcings_t.get("mask", 1.0)` — mask est statique, absent de `forcings_t`
- `L212` : `_resolve_inputs` cherche `forcings.*` dans `forcings_t` — les statics n'y sont plus

**Solution** : `build_step_fn()` doit appeler `model.forcings.get_statics()` et capturer le résultat en closure. Les statics doivent être accessibles dans `_execute_step` (pour mask) et dans `_resolve_inputs` (pour les inputs de type `forcings.mask`).

### API utilisateur actuelle (via examples/)

```python
# Simulation
runner = Runner.simulation(chunk_size=32, output="memory")
state, dataset = runner.run(model, export_variables=["biomass"])

# Optimisation
runner = Runner.optimization()            # single
runner = Runner.optimization(vmap=True)   # population
outputs = runner(model, free_params)

# Bas niveau (benchmarks)
step_fn = build_step_fn(model)
(state, params), outputs = _scan(step_fn, (state, params), forcings, n_steps)
```

## Pré-requis (workflow Compiler) — COMPLÉTÉ

Le workflow Compiler (`IA/WORKFLOW_COMPILER/`) est terminé (32 tâches, commit `6b63471`). Changements effectifs à intégrer côté Runner :

### API ForcingStore finalisée

```python
class ForcingStore:
    def get_chunk(start, end) -> dict[str, Array]    # dynamiques SEULEMENT (avec dim T)
    def get_statics() -> dict[str, Array]             # statiques (sans dim T) — à appeler 1 fois
    def get_all_dynamic() -> dict[str, Array]         # = get_chunk(0, n_timesteps)
```

### Tests xfailed à corriger (2)

1. **`tests/compiler/test_forcing.py::test_get_chunk_includes_statics`** — `get_chunk()` ne retourne plus les statics
2. **`tests/engine/test_integration.py::test_mask_zeros_state`** — mask absent de `forcings_t`, `step.py:88` tombe sur `mask = forcings_t.get("mask", 1.0)` → pas de masquage

### Chemins d'exécution impactés dans runner.py

| Chemin | Ligne | Appel actuel | Ce qu'il manque |
|---|---|---|---|
| `_run_simulation()` | 320 | `model.forcings.get_chunk(start_t, end_t)` | `get_statics()` non appelé |
| `_run_optimization()` | 344 | `model.forcings.get_all_dynamic()` | `get_statics()` non appelé |
| `_run_optimization_vmap()` | 361 | `model.forcings.get_all_dynamic()` | `get_statics()` non appelé |

### Changement central à implémenter

`build_step_fn()` dans `step.py` doit **capturer les statics en closure** :
- Appeler `model.forcings.get_statics()` une seule fois à la construction du step_fn
- Injecter les statics (notamment `mask`) dans la closure du step_fn
- `step.py:88` ne doit plus fallback sur `1.0` — le mask est toujours disponible via la closure
- Les statics ne transitent plus par `lax.scan` (pas de broadcast inutile, moins de mémoire)

### Validation NaN

Gérée automatiquement par `get_chunk()` et `get_statics()` — le Runner n'a pas à s'en occuper.

## Décisions d'architecture

### 1. Pas de pattern Strategy — Composition fonctionnelle

Les transforms JAX (vmap, grad, pmap, checkpoint) sont des **wrappers fonctionnels one-liner**
(`fn = jax.vmap(fn)`). Les encapsuler dans des classes Strategy ajouterait de l'indirection
sans réduire la complexité. La composition fonctionnelle est le paradigme natif de JAX.

**Patterns utilisés** :
- **Factory Method** : `Runner.simulation()`, `Runner.optimization()` — encapsulent les combinaisons valides
- **Protocol** (déjà en place) : `OutputWriter` → `DiskWriter`, `MemoryWriter` — logique substantielle par implémentation
- **Composition fonctionnelle** : `jax.vmap(jax.value_and_grad(fn))` — natif JAX, zéro overhead

### 2. Deux chemins de données au lieu de trois

| Chemin | Quand | Chargement forcings | Compatible avec |
|---|---|---|---|
| `_run_chunked` | `chunk_size is not None` et pas `grad` | `get_chunk()` par itération | simulation, vmap, pmap, fori_loop |
| `_run_full` | `chunk_size is None` ou `grad=True` | `get_all_dynamic()` une fois | simulation, vmap, pmap, grad, fori_loop |

`checkpoint` n'est **pas** un chemin de données — c'est un modificateur du `step_fn`
(`step_fn = jax.checkpoint(step_fn)`) appliqué en amont, avant le choix du chemin.

### 3. Chunked + vmap est possible

Le docstring actuel du runner dit `vmap → Python for-loop ❌`. C'est faux pour le nesting inverse :

```
❌  jax.vmap(lambda: for chunk: lax.scan(...))     — vmap AUTOUR de la boucle Python
✅  for chunk: jax.vmap(lambda: lax.scan(...))(…)  — vmap DANS chaque itération
```

Les forcings du chunk sont en **closure** → partagés entre individus, pas dupliqués.
Gain mémoire : `n_timesteps / chunk_size` fois moins de mémoire forcing.

Seul `grad + chunked` reste impossible (l'accumulation de gradients inter-chunks nécessite
une gestion spéciale hors scope de ce workflow).

### 4. Statics capturés en closure dans `build_step_fn()`

```python
def build_step_fn(model, ...):
    statics = model.forcings.get_statics()   # appelé UNE fois, capturé en closure

    def _execute_step(state, forcings_t, parameters):
        all_forcings = {**statics, **forcings_t}   # fusion à chaque pas
        mask = all_forcings.get("mask", 1.0)       # mask désormais disponible
        ...
        _resolve_inputs(..., all_forcings, ...)     # aucun changement nécessaire
```

- Les statics sont des **constantes JAX** dans la closure → pas de coût mémoire dans le scan
- `_resolve_inputs` et `_apply_mask` ne changent **pas** — ils reçoivent le dict fusionné
- Le fallback `mask=1.0` est conservé (un modèle peut ne pas déclarer de mask)

### 5. RunnerConfig simplifié

**Retiré** (dead code, jamais lu dans la logique d'exécution) :
- `param_mode` (toujours "carry", "closure" jamais implémenté)
- `loop_mode` (toujours "scan", "fori_loop" jamais implémenté)

**Conservé** :
- `vmap_params: bool`
- `chunk_size: int | None`
- `output_mode: Literal["disk", "memory", "raw"]`

**Contraintes dans `__post_init__`** :
- `grad → chunk_size must be None` (grad incompatible avec chunking)
- `vmap/pmap + output_mode != "raw"` → erreur (outputs ont une dim supplémentaire)

### 6. Matrice des combinaisons valides

```
                    run_chunked         run_full
                    (boucle Python)     (scan unique)
────────────────────────────────────────────────────
simulation            ✅                  ✅
vmap                  ✅                  ✅
pmap                  ✅                  ✅
grad                  ❌                  ✅ (+checkpoint optionnel)
fori_loop             ✅                  ✅
```

### 7. Structure cible du code

```
_execute(model, params, ...)
│
├── step_fn = build_step_fn(model, ...)
│   └── statics capturés en closure (get_statics() appelé 1 fois)
│
├── if checkpoint: step_fn = jax.checkpoint(step_fn)
│
├── if grad → _run_full (forcément)
├── elif chunk_size → _run_chunked
├── else → _run_full
│
├── _run_chunked(model, step_fn, params)
│   ├── core = scan or fori_loop
│   ├── if vmap: core = jax.vmap(core, in_axes=(0, 0, None))
│   ├── if pmap: core = jax.pmap(core, in_axes=(0, 0, None))
│   └── for chunk: core(states, params, get_chunk(...))
│
└── _run_full(model, step_fn, params)
    ├── forcings = get_all_dynamic()
    ├── def run_core(p): lax.scan(step_fn, ..., forcings)
    ├── if vmap: run_core = jax.vmap(run_core)
    ├── if grad:  run_core = jax.value_and_grad(run_core)
    ├── if pmap: run_core = jax.pmap(run_core)
    └── return run_core(params)
```

## Todo List

### Phase 1 : Statics en closure (correction critique)

| État | ID | Nom | Description | Dépendances | Résolution |
| ---- | -- | --- | ----------- | ----------- | ---------- |
| ☐ | T1 | Capturer statics dans `build_step_fn` | Dans `engine/step.py` : au début de `build_step_fn()`, appeler `statics = model.forcings.get_statics()`. Cette variable est capturée dans la closure de `_execute_step`. | - | |
| ☐ | T2 | Fusionner statics+forcings_t dans `_execute_step` | Dans `engine/step.py`, dans `_execute_step()` (L77) : remplacer l'usage direct de `forcings_t` par `all_forcings = {**statics, **forcings_t}`. Utiliser `all_forcings` pour le mask (L88) et dans l'appel à `_resolve_inputs` (L95). Les fonctions `_resolve_inputs`, `_apply_mask`, `_integrate_euler` ne changent PAS — elles reçoivent le dict fusionné. Le fallback `mask = all_forcings.get("mask", 1.0)` est conservé (modèles sans mask). | T1 | |

### Phase 2 : Consolidation des chemins d'exécution

| État | ID | Nom | Description | Dépendances | Résolution |
| ---- | -- | --- | ----------- | ----------- | ---------- |
| ☐ | T3 | Créer `_run_chunked` unifié | Dans `engine/runner.py` : remplacer `_run_simulation()` (L285-331) par `_run_chunked()`. La méthode construit une fonction `core(states, params, forcings_chunk)` qui appelle `lax.scan(step_fn, ...)`. Si `self.config.vmap_params`, wrapper avec `jax.vmap(core, in_axes=(0, 0, None))` — les forcings sont en closure (axis=None), state+params sont vmappés (axis=0). La boucle Python itère sur les chunks : `forcings_chunk = model.forcings.get_chunk(start, end)`, appelle `core(states, params, forcings_chunk)`, et passe les outputs au writer si présent. | T2 | |
| ☐ | T4 | Créer `_run_full` unifié | Dans `engine/runner.py` : fusionner `_run_optimization()` (L335-348) et `_run_optimization_vmap()` (L350-367) en une seule méthode `_run_full()`. Charger `forcings = model.forcings.get_all_dynamic()` une fois. Construire `run_core(p)` qui fait `lax.scan(step_fn, (state, merged), forcings)`. Composer les transforms : `if vmap: run_core = jax.vmap(run_core)`. Retourner `run_core(params)`. | T2 | |
| ☐ | T5 | Créer `_execute` orchestrateur | Dans `engine/runner.py` : créer une méthode `_execute(model, params, export_variables)` qui : (1) construit `step_fn = build_step_fn(model, export_variables)`, (2) applique checkpoint si configuré : `step_fn = jax.checkpoint(step_fn)`, (3) dispatch vers `_run_chunked` ou `_run_full` selon `chunk_size` et `grad`. Les méthodes publiques `run()` et `__call__()` délèguent à `_execute`. | T3, T4 | |
| ☐ | T6 | Supprimer les anciennes méthodes | Dans `engine/runner.py` : supprimer `_run_simulation()`, `_run_optimization()`, `_run_optimization_vmap()`. Vérifier que `run()` et `__call__()` passent par `_execute()`. | T5 | |

### Phase 3 : Nettoyage RunnerConfig

| État | ID | Nom | Description | Dépendances | Résolution |
| ---- | -- | --- | ----------- | ----------- | ---------- |
| ☐ | T7 | Retirer `param_mode` de RunnerConfig | Dans `engine/runner.py` : supprimer le champ `param_mode: Literal["closure", "carry"]` de RunnerConfig (L110). Supprimer sa mention dans le docstring (L96-98). Supprimer les affectations `param_mode="carry"` dans les factory methods `simulation()` (L185) et `optimization()` (L202). | T6 | |
| ☐ | T8 | Retirer `loop_mode` de RunnerConfig | Dans `engine/runner.py` : supprimer le champ `loop_mode: Literal["scan", "fori_loop"]` (L111). Supprimer sa mention dans le docstring (L99-101). Supprimer la validation `if self.output_mode == "disk" and self.loop_mode != "scan"` dans `__post_init__` (L117-118). Supprimer les affectations `loop_mode="scan"` dans les factory methods (L186, L203). | T7 | |
| ☐ | T9 | Mettre à jour le docstring du module | Dans `engine/runner.py` : mettre à jour le docstring du module (L1-64) pour refléter la nouvelle architecture : 2 chemins (`_run_chunked`, `_run_full`), composition fonctionnelle des transforms, chunked+vmap possible. Retirer les mentions de `get_all()` (remplacé par `get_all_dynamic()`). Corriger le passage qui dit `vmap → Python for-loop ❌` — expliquer que le nesting inverse fonctionne. | T6 | |

### Phase 4 : Tests

| État | ID | Nom | Description | Dépendances | Résolution |
| ---- | -- | --- | ----------- | ----------- | ---------- |
| ☐ | T10 | Supprimer xfail `test_get_chunk_includes_statics` | Dans `tests/compiler/test_forcing.py` : supprimer entièrement le test `test_get_chunk_includes_statics` (L38-50). Ce test validait l'ancien comportement (statics dans get_chunk). Le nouveau comportement est correct — les statics sont accessibles via `get_statics()`. | T2 | |
| ☐ | T11 | Retirer xfail de `test_mask_zeros_state` | Dans `tests/engine/test_integration.py` : retirer le décorateur `@pytest.mark.xfail(...)` (L163-169) du test `test_mask_zeros_state`. Le test doit désormais passer car les statics (mask) sont capturés en closure par `build_step_fn()` et fusionnés avec `forcings_t` à chaque pas. | T2 | |
| ☐ | T12 | Adapter les tests RunnerConfig | Dans `tests/engine/test_runner.py` : supprimer les tests qui référencent `param_mode` ou `loop_mode` (ex: `test_invalid_disk_fori_loop` L38-41, assertions sur `loop_mode` L23). Adapter les assertions restantes pour le RunnerConfig simplifié. | T7, T8 | |
| ☐ | T13 | Ajouter test chunked+vmap | Dans `tests/engine/test_integration.py` : ajouter un test E2E qui vérifie que `Runner.optimization(vmap=True, chunk_size=5)` fonctionne. Créer un modèle toy avec population de 3 individus (params vmappés avec shape `(3,)`), 10 timesteps, chunk_size=5. Vérifier que les outputs ont la dim population et que les résultats sont identiques à `Runner.optimization(vmap=True)` sans chunking (même résultat, moins de mémoire). | T3 | |
| ☐ | T14 | Ajouter test statics en closure | Dans `tests/engine/test_runner.py` ou `test_integration.py` : ajouter un test qui vérifie explicitement que les statics sont accessibles dans le step_fn. Construire un modèle toy avec un forcing statique `mask` contenant des zéros. Vérifier via `Runner.simulation()` que les régions masquées restent à zéro (ce qui échouait avant la correction). Ce test est similaire à `test_mask_zeros_state` mais sans xfail. | T2 | |
| ☐ | T15 | Run full test suite | Lancer `uv run python -m pytest -x -q` et vérifier : 0 xfailed (les 2 précédents sont corrigés), tous les tests passent. Vérifier aussi que les exemples existants ne sont pas cassés par les changements de RunnerConfig. | T10, T11, T12, T13, T14 | |
| ✅ | T16 | Exposer `chunk_size` dans `optimization()` | Dans `engine/runner.py` : ajouter `chunk_size: int | None = None` à `Runner.optimization()` et le passer au `RunnerConfig`. Chunked+vmap est supporté depuis T3. | T3 | |

## Historique des transitions

| De                | Vers            | Raison                          | Date       |
| ----------------- | --------------- | ------------------------------- | ---------- |
| 1. Initialisation | 2. Analyse      | Besoin validé par l'utilisateur | 2026-03-08 |
| 2. Analyse        | 3. Architecture | Analyse complétée               | 2026-03-08 |
