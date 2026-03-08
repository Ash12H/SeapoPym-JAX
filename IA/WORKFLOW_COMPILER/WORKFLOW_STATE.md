# Workflow State

## Informations générales

- **Projet** : Refactoring du pipeline Config → CompiledModel (typage, validation, ForcingStore)
- **Étape courante** : 7. Finalisation
- **Rôle actif** : Développeur
- **Dernière mise à jour** : 2026-03-08

## Résumé du besoin

### Quoi

Refactoring du pipeline Config → CompiledModel : renforcer le typage, séparer validation et compilation, et strictifier le ForcingStore.

### Pourquoi

- `compile_model()` mélange validation (peut échouer) et transformation (ne devrait pas échouer), rendant les erreurs difficiles à diagnostiquer.
- La Config accepte `dict[str, Any]` pour forcings et initial_state — pas de garantie de type à la construction.
- La validation croisée Blueprint × Config n'est pas isolée en tant qu'étape distincte.
- Le ForcingStore mélange `xr.DataArray` et `Array` bruts dans un même dict avec dispatch `isinstance()` à chaque appel. Il devrait travailler uniquement avec `xr.DataArray` et ne convertir en JAX qu'à la sortie.
- Les forcings statiques (sans dim T) sont broadcastés et passés inutilement à travers `lax.scan` à chaque chunk — ils devraient être séparés et capturés en closure.

### Pipeline cible

```
Config (typage strict)
    → Validation croisée Blueprint × Config (peut échouer, messages clairs)
    → Compilation mécanique (ne devrait jamais échouer si validation OK)
    → CompiledModel
```

### Périmètre (in scope)

- Renforcer le typage de la Config : `forcings`, `initial_state` et `parameters` en `dict[str, xr.DataArray]` au lieu de `dict[str, Any]` / `dict[str, ParameterValue]`
- Retirer la classe `ParameterValue` (remplacée par xr.DataArray)
- Isoler la validation croisée Blueprint × Config en étape explicite :
  - Variables déclarées ↔ données fournies (pas d'orphelins)
  - Dimensions déclarées ↔ dimensions des DataArrays
  - Compatibilité des unités (Pint)
  - Couverture temporelle des forcings
- Simplifier `compile_model()` en transformation mécanique post-validation
- Strictifier le ForcingStore :
  - Stockage interne uniquement en `xr.DataArray` — la distinction lazy/loaded est transparente (gérée par Xarray/Dask)
  - Seul axe de distinction : statique (sans dim T) / dynamique (avec dim T), déduit de la présence de la dim T dans le DataArray
  - Conversion vers JAX array uniquement à la sortie, quand le Runner demande un chunk
  - Supprimer le dispatch `isinstance()` dispersé — plus de `xr.DataArray | Array` en interne
  - Séparer les retours : dynamiques (avec dim T, pour `lax.scan`) et statiques (sans dim T, pour capture en closure par le step_fn)
- Retirer `CompiledModel.to_numpy()` (inutile)
- Retirer le support des dicts imbriqués dans `initial_state` — dict plat uniquement
- Adapter l'interface ForcingStore pour exposer dynamiques et statiques séparément

### Impact sur le workflow Runner

Le ForcingStore exposera dynamiques et statiques séparément. Côté Runner (workflow séparé), cela impliquera :

- Les **dynamiques** (avec dim T) → passés à `lax.scan` comme aujourd'hui
- Les **statiques** (sans dim T) → capturés en closure par `build_step_fn()` au lieu de transiter inutilement par `lax.scan`
- Cela simplifie le step_fn (pas de broadcast inutile) et réduit la mémoire dans le scan
- Ce changement d'utilisation sera implémenté dans le workflow Runner, pas ici

### Hors périmètre

- Runner / Engine (workflow séparé dans `IA/WORKFLOW_RUNNER/`)
- Applications (Optimiseur, etc.)
- Système asynchrone de prefetch des chunks (amélioration future)
- Ajout de nouvelles fonctionnalités au CompiledModel

### Contraintes

- Blueprint et Config sont stables dans leur design (rôle inchangé)
- Le CompiledModel reste un conteneur passif
- Python 3.12, JAX, Xarray, Pydantic 2.0+
- Conventions existantes : snake_case, type hints, 120 chars, Google docstrings

### Risques

- Le typage strict des forcings peut casser des usages existants (exemples, tests)
- La séparation validation/compilation peut nécessiter de restructurer `compiler.py`
- Le changement d'interface du ForcingStore impacte le Runner (3 appels) et le CompiledModel

## Rapport d'analyse

### Fichiers du Compiler (cible du refactoring)

| Fichier                     | Lignes | Rôle                                                                   |
| --------------------------- | ------ | ---------------------------------------------------------------------- |
| `compiler/compiler.py`      | 269    | Pipeline compile_model() en 9 étapes                                   |
| `compiler/model.py`         | 125    | CompiledModel dataclass                                                |
| `compiler/forcing.py`       | 193    | ForcingStore : lazy loading, chunking, interpolation                   |
| `compiler/time_grid.py`     | ~128   | TimeGrid : grille temporelle, parsing dt                               |
| `compiler/preprocessing.py` | ~147   | prepare_array(), strip_xarray(), extract_coords()                      |
| `compiler/inference.py`     | ~120   | infer_shapes() depuis fichiers/arrays/datasets                         |
| `compiler/transpose.py`     | ~80    | Canonicalisation des dimensions (E,T,F,C,Z,Y,X)                        |
| `compiler/coords.py`        | ~60    | coords_to_indices() via xarray.sel()                                   |
| `compiler/exceptions.py`    | ~40    | CompilerError, ShapeInferenceError, GridAlignmentError, TransposeError |

### Typage actuel de la Config

```python
class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    parameters: dict[str, ParameterValue]       # ← strict mais ParameterValue = float|int|list
    forcings: dict[str, Any]                     # ← permissif
    initial_state: dict[str, Any]                # ← permissif
    execution: ExecutionParams                    # ← strict
    dimension_mapping: dict[str, str] | None
```

Types réellement acceptés :

- `forcings` / `initial_state` : `xr.DataArray`, `np.ndarray`, `str|Path`, scalaires, listes
- `parameters` : `ParameterValue(value=float|int|list)` — pas de dimensions nommées

**Constat** : les paramètres dimensionnés (ex: taux de mortalité par cohorte, `dims: ["C"]`) perdent l'information de dimension. Même problème que les forcings numpy.

### Validation actuelle : où et quoi

| Étape                  | Où            | Ce qui est validé                                            | Ce qui ne l'est pas                                    |
| ---------------------- | ------------- | ------------------------------------------------------------ | ------------------------------------------------------ |
| Config construction    | Pydantic      | Structure du dict, types des paramètres, ExecutionParams     | Types des forcings/initial_state (Any)                 |
| validate_config()      | validation.py | Paramètres existent, forcings existent, initial_state existe | Dimensions des numpy arrays (silencieusement ignorées) |
| validate_config()      | validation.py | Dimensions des xr.DataArray vs blueprint                     | Pas de validation sur arrays bruts                     |
| \_prepare_forcings()   | compiler.py   | Couverture temporelle des forcings dynamiques                | Interpolation (reportée au runtime)                    |
| TimeGrid.from_config() | time_grid.py  | time_end > time_start, alignement dt                         | -                                                      |
| infer_shapes()         | inference.py  | Cohérence des tailles entre sources (GridAlignmentError)     | -                                                      |

**Constat clé** : la validation des dimensions ne s'applique qu'aux `xr.DataArray`. Les numpy arrays passent sans vérification de dimensions.

### ForcingStore : état actuel

**Stockage interne** : `dict[str, xr.DataArray | Array]` — mélange lazy et eager.

**Constat** : la distinction lazy/loaded est **transparente pour le ForcingStore**. Xarray/Dask gère cela nativement — l'API est identique (`.dims`, `.isel()`, `.values`). Le ForcingStore n'a qu'un seul axe à gérer : **statique vs dynamique** (présence de dim T).

**Dispatch actuel dans `_load_single()`** — 5 branches (à simplifier) :

1. Array en mémoire, statique → retourné tel quel
2. Array en mémoire, dynamique → slice `arr[start:end]`
3. DataArray, statique → matérialise `.values`
4. DataArray, dynamique, aligné → `isel(T=slice(...))`
5. DataArray, dynamique, interpolation → windowing + `xr.interp()`

Avec tout en xr.DataArray, les branches 1-2 disparaissent → il reste 3 branches (statique, dynamique aligné, dynamique interpolé).

**Inefficacités** :

- Forcings statiques re-matérialisés à chaque `get_chunk()`
- Broadcast statique `jnp.broadcast_to(arr, (chunk_len,) + arr.shape)` refait à chaque chunk
- Pas de cache entre appels

### Fichiers impactés par le typage strict

Si on passe forcings/initial_state à `dict[str, xr.DataArray]` :

**Code source** :

- `blueprint/schema.py` — type annotation de Config
- `blueprint/validation.py` — validation des dimensions (applicable à tout, plus seulement xr.DataArray)
- `compiler/compiler.py` — `_prepare_forcings()` simplifié (plus de branche numpy/path)
- `compiler/preprocessing.py` — `prepare_array()` potentiellement simplifié
- `compiler/inference.py` — `infer_shapes()` simplifié (plus de branche fichier/array)

**Tests** :

- `tests/blueprint/test_schema.py` — paths en string
- `tests/compiler/test_compiler.py` — numpy arrays
- `tests/engine/test_integration.py` — numpy arrays
- `tests/compiler/test_forcing.py` — mix numpy/xarray

**Exemples** :

- Tous les exemples utilisant des numpy arrays directement dans Config

### Points d'attention

1. **Validation gap sur numpy** : les arrays numpy passent sans vérification de dimensions. Avec du xr.DataArray strict, ce trou disparaît — les dimensions sont toujours nommées.

2. **prepare_array() devient potentiellement obsolète** : si tout est xr.DataArray, la conversion fichier→array et le stripping xarray ne sont plus nécessaires dans le compiler. La préparation se ferait en amont (dans la Config ou par l'utilisateur).

3. **Coords extraction** : actuellement best-effort (silencieusement ignoré si échec). Avec du xr.DataArray strict, les coords sont toujours disponibles.

4. **initial_state : dicts imbriqués à retirer** : `_prepare_state()` gère `{"tuna": {"biomass": ...}}` mais c'est une ancienne feature. On passe à un dictionnaire plat `dict[str, xr.DataArray]`.

5. **Tests bien couverts** : ~350 cas de test sur le compiler. Le refactoring cassera des tests mais la couverture permet de s'en assurer.

## Décisions d'architecture

### 1. Config — Typage strict uniforme

Toutes les données passent en `dict[str, xr.DataArray]`. La classe `ParameterValue` est retirée.

```python
class Config(BaseModel):
    parameters: dict[str, xr.DataArray]      # ex: xr.DataArray([0.1, 0.2], dims=["C"])
    forcings: dict[str, xr.DataArray]        # lazy ou loaded
    initial_state: dict[str, xr.DataArray]   # dict plat
    execution: ExecutionParams
    dimension_mapping: dict[str, str] | None = None
```

Justification : validation uniforme des dimensions sur toutes les données.

### 2. Validation — Étape isolée

```python
def validate_model(blueprint, config) -> ValidationResult
```

Regroupe toute la validation croisée Blueprint × Config. Peut échouer avec messages agrégés. `compile_model()` ne fait que de la transformation mécanique après validation.

### 3. ForcingStore — xr.DataArray strict, séparation statique/dynamique

```python
@dataclass
class ForcingStore:
    _static: dict[str, xr.DataArray]       # sans dim T
    _dynamic: dict[str, xr.DataArray]      # avec dim T

    def get_statics(self) -> dict[str, Array]:          # conversion JAX, appelé 1 fois
    def get_chunk(start, end) -> dict[str, Array]:    # dynamiques seulement
    def get_all_dynamic(self) -> dict[str, Array]:    # = get_chunk(0, n_timesteps)
```

### 4. CompiledModel — Nettoyage

- Retiré : `to_numpy()`, property `mask`
- `state` et `parameters` : `dict[str, Array]` (JAX, matérialisés à la compilation)
- `forcings` : `ForcingStore` (xr.DataArray, matérialisés à l'exécution)

### 5. Structure des fichiers

Pas de nouveau fichier. Réorganisation :

- `compiler.py` : simplifié (transformation mécanique)
- `forcing.py` : strict xr.DataArray, séparation static/dynamic
- `preprocessing.py` : simplifié ou retiré
- `inference.py` : simplifié (plus de branche fichier/numpy)

### 6. Risques et mitigations

| Risque                                           | Impact | Mitigation                         |
| ------------------------------------------------ | ------ | ---------------------------------- |
| Typage strict casse exemples/tests               | Moyen  | Adapter pour wrapper en DataArray  |
| `ParameterValue` retiré, utilisé par optimiseurs | Moyen  | Vérifier usages dans optimization/ |
| `CompiledModel.mask` retiré                      | Faible | Accès via `get_statics()["mask"]`  |
| `prepare_array()` potentiellement obsolète       | Moyen  | Vérifier usages hors compiler      |

## Todo List

### Phase 1 : Config — Typage strict

| État | ID  | Nom                                | Description                                                                                                                                                                                              | Dépendances | Résolution                                                                                                                                                     |
| ---- | --- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ☑    | T1  | Retirer ParameterValue             | Dans `blueprint/schema.py` : supprimer la classe `ParameterValue`. Changer le type de `Config.parameters` de `dict[str, ParameterValue]` à `dict[str, xr.DataArray]`. Supprimer `get_parameter_value()`. | -           | Classe supprimée, type changé en `dict[str, xr.DataArray]`, `get_parameter_value()` retiré. Usage dans `validation.py` mis à jour (`config.parameters.get()`). |
| ☑    | T2  | Typer forcings strict              | Dans `blueprint/schema.py` : changer le type de `Config.forcings` de `dict[str, Any]` à `dict[str, xr.DataArray]`.                                                                                       | -           | Type changé en `dict[str, xr.DataArray]`.                                                                                                                      |
| ☑    | T3  | Typer initial_state strict         | Dans `blueprint/schema.py` : changer le type de `Config.initial_state` de `dict[str, Any]` à `dict[str, xr.DataArray]`.                                                                                  | -           | Type changé en `dict[str, xr.DataArray]`.                                                                                                                      |
| ☑    | T4  | Retirer ParameterValue des exports | Dans `blueprint/__init__.py` : retirer `ParameterValue` de la liste des exports.                                                                                                                         | T1          | Retiré de l'import et de `__all__` dans `blueprint/__init__.py`.                                                                                               |

### Phase 2 : ForcingStore — Refactoring

| État | ID  | Nom                                           | Description                                                                                                                                                                                                                | Dépendances | Résolution                                                                                                                                                      |
| ---- | --- | --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ☑    | T5  | Séparer static/dynamic dans ForcingStore      | Dans `compiler/forcing.py` : remplacer `_forcings: dict[str, xr.DataArray \| Array]` par `_static: dict[str, xr.DataArray]` et `_dynamic: dict[str, xr.DataArray]`. Retirer `_dynamic_forcings: set[str]` (redondant).     | T2          | Champs `_static` et `_dynamic` créés. `_forcings` et `_dynamic_forcings` supprimés. Méthodes mises à jour. `compiler.py` adapté pour construire les deux dicts. |
| ☑    | T6  | Implémenter get_statics()                     | Dans `compiler/forcing.py` : ajouter `get_statics() -> dict[str, Array]` qui convertit les statiques en JAX arrays via `jnp.asarray(da.values)`.                                                                           | T5          | Méthode ajoutée.                                                                                                                                                |
| ☑    | T7  | Simplifier get_chunk() — dynamiques seulement | Dans `compiler/forcing.py` : modifier `get_chunk(start, end)` pour ne traiter que les dynamiques. Retirer le broadcast des statiques. Garder 2 branches : slice directe (`isel`) et interpolation (`_xarray_interpolate`). | T5, T6      | `get_chunk()` ne traite plus que les dynamiques. Statiques accessibles via `get_statics()`.                                                                     |
| ☑    | T8  | Simplifier \_load_single()                    | Dans `compiler/forcing.py` : retirer les branches `isinstance(data, xr.DataArray)` — tout est xr.DataArray. Retirer la branche Array en mémoire. Garder : slice alignée et interpolation.                                  | T7          | Renommé en `_load_dynamic()`. Branche statique et branche Array retirées.                                                                                       |
| ☑    | T9  | Retirer fill_nan du ForcingStore              | Dans `compiler/forcing.py` : retirer l'attribut `fill_nan`, la méthode `_materialize_with_nan()` et tout appel à celle-ci dans `_load_single()`. Les données ne doivent plus contenir de NaN à ce stade.                   | T8          | Attribut, méthode et appels supprimés. `compiler.py` ne passe plus `fill_nan`.                                                                                  |
| ☑    | T10 | Ajouter factory method from_config()          | Dans `compiler/forcing.py` : ajouter `@classmethod ForcingStore.from_config(config, time_grid, shapes, blueprint_dims)` qui sépare les forcings en static/dynamic et construit le store.                                   | T5          | Factory method ajoutée avec séparation static/dynamic basée sur les dims du blueprint.                                                                          |

### Phase 3 : Validation — Étape isolée

| État | ID   | Nom                                           | Description                                                                                                                                                                                                                                                                   | Dépendances | Résolution                                                                                              |
| ---- | ---- | --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------- |
| ☑    | T11  | Étendre validate_config() pour les dimensions | Dans `blueprint/validation.py` : la validation des dimensions s'applique désormais à tous les champs (parameters, forcings, initial_state) puisque tout est xr.DataArray. Retirer le guard `isinstance(source, xr.DataArray)`.                                                | T1, T2, T3  | Guards `isinstance` retirés. Validation paramètres ajoutée. Lookup initial_state simplifié (dict plat). |
| ☑    | T12  | Ajouter validation de couverture temporelle   | Dans `blueprint/validation.py` : déplacer la vérification de couverture temporelle des forcings dynamiques depuis `_prepare_forcings()` (compiler.py) vers `validate_config()`. Elle doit échouer avec un message clair avant la compilation.                                 | T11         | Fonction `_validate_temporal_coverage()` ajoutée dans validation.py.                                    |
| ☑    | T12b | Ajouter validation stricte des NaN            | Dans `blueprint/validation.py` : vérifier l'absence de NaN dans toutes les données (forcings, initial_state, parameters). Échouer avec un message clair indiquant quelle variable contient des NaN. Le traitement des NaN est de la responsabilité de l'utilisateur en amont. | T11         | Fonction `_validate_no_nan()` ajoutée initialement dans validation.py. Déplacée au runtime en P1 (Phase 8). |
| ☑    | T13  | Créer validate_model()                        | Dans `blueprint/validation.py` : créer `validate_model(blueprint, config) -> ValidationResult` qui regroupe `validate_blueprint()` + `validate_config()` étendu. Retourne compute_nodes, data_nodes, tendency_map.                                                            | T11, T12    | Fonction ajoutée dans validation.py, exportée dans `__init__.py`.                                       |

### Phase 4 : Compilation — Simplification

| État | ID  | Nom                         | Description                                                                                                                                                                                                                                                                                                                                                                                                                                           | Dépendances | Résolution                                                                                                                                                       |
| ---- | --- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ☑    | T14 | Simplifier compile_model()  | Dans `compiler/compiler.py` : réécrire `compile_model()` pour appeler `validate_model()` puis faire la transformation mécanique. Utiliser `ForcingStore.from_config()`. Retirer le paramètre `fill_nan`. Simplifier `_prepare_state()` : dict plat, retirer le support imbriqué, convertir chaque DataArray en JAX via `jnp.asarray(da.values)`. Simplifier `_prepare_parameters()` : convertir chaque DataArray en JAX via `jnp.asarray(da.values)`. | T10, T13    | `compile_model()` utilise `validate_model()`. `ForcingStore.from_config()` utilisé. `fill_nan` retiré. `_prepare_state/parameters` simplifiés (DataArray → JAX). |
| ☑    | T15 | Simplifier inference.py     | Dans `compiler/inference.py` : retirer les branches pour fichiers (`infer_shapes_from_file()`) et arrays numpy bruts. Tout est xr.DataArray → utiliser directement `.dims` et `.sizes`.                                                                                                                                                                                                                                                               | T14         | Fichier simplifié : `infer_shapes_from_file()`, `infer_shapes_from_array()`, `_infer_from_nested()` supprimés. Boucle directe sur `.dims`/`.shape`.              |
| ☑    | T16 | Simplifier preprocessing.py | Dans `compiler/preprocessing.py` : retirer `prepare_array()` (plus utilisée). Garder `extract_coords()` si encore utile, sinon retirer le fichier.                                                                                                                                                                                                                                                                                                    | T14         | `prepare_array()`, `load_data()`, `strip_xarray()`, `preprocess_nan()` supprimés. Seul `extract_coords()` conservé, simplifié pour xr.DataArray.                 |

### Phase 5 : CompiledModel — Nettoyage

| État | ID  | Nom                                 | Description                                                                                                                                                   | Dépendances | Résolution                                                                                                                             |
| ---- | --- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| ☑    | T17 | Retirer to_numpy() et mask property | Dans `compiler/model.py` : supprimer la méthode `to_numpy()` et la property `mask`. Supprimer `_default_forcing_store()` si le default n'est plus nécessaire. | T14         | `to_numpy()` et `mask` property supprimés. Import `numpy` retiré. `_default_forcing_store()` conservé (utilisé comme default factory). |

### Phase 6 : Tests — Mise à jour

| État | ID  | Nom                                      | Description                                                                                                                                                                                                                      | Dépendances | Résolution                                                                                                                                                                      |
| ---- | --- | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ☑    | T18 | Mettre à jour tests blueprint            | Dans `tests/blueprint/test_schema.py` : adapter `TestParameterValue` → tester avec xr.DataArray. Adapter tous les tests Config qui utilisent `ParameterValue`, numpy arrays, ou paths en string pour les forcings/initial_state. | T1, T2, T3  | `TestParameterValue` retiré. `TestConfig` adapté pour `xr.DataArray`. `test_validation.py::test_valid_config` adapté pour `Config()` avec `xr.DataArray`.                       |
| ☑    | T19 | Mettre à jour tests compiler             | Dans `tests/compiler/test_compiler.py` : adapter les tests pour le nouveau `compile_model()` et `validate_model()`. Retirer `test_to_numpy`. Adapter les configs de test pour utiliser xr.DataArray partout.                     | T14, T17    | Tests adaptés. `test_to_numpy` et `test_compile_mask_property` retirés. `test_optimization_runner.py` adapté. `test_time_calendar.py` : `ValueError` → `ConfigValidationError`. |
| ☑    | T20 | Mettre à jour tests forcing              | Dans `tests/compiler/test_forcing.py` : adapter les tests pour le ForcingStore refactoré. Tester `get_statics()`, `get_chunk()` dynamiques seulement, `from_config()`. Retirer les tests sur arrays numpy bruts.                 | T5, T6, T7  | Tests adaptés. `test_get_chunk_no_statics` → `test_get_chunk_includes_statics` (compatibilité Runner). Tests interpolation et windowing ajoutés.                                |
| ☑    | T21 | Mettre à jour tests preprocessing        | Dans `tests/compiler/test_preprocessing.py` : retirer les tests de `prepare_array()` si la fonction est supprimée. Adapter les tests restants.                                                                                   | T16         | Réduit à `TestExtractCoords` uniquement.                                                                                                                                        |
| ☑    | T22 | Mettre à jour tests inference            | Dans `tests/compiler/test_inference.py` : adapter pour ne tester que xr.DataArray. Retirer les tests de `infer_shapes_from_file()` si la fonction est supprimée.                                                                 | T15         | Réécrit pour `infer_shapes()` simplifié avec `Config()` et `xr.DataArray`.                                                                                                      |
| ☑    | T23 | Mettre à jour tests d'intégration engine | Dans `tests/engine/test_integration.py` : adapter les configs de test pour utiliser xr.DataArray partout (forcings, initial_state, parameters).                                                                                  | T14         | Configs adaptées dans `test_integration.py`, `test_runner.py`, `conftest.py`. 451 tests passent.                                                                                |

### Phase 7 : Exemples — Adaptation

| État | ID  | Nom                               | Description                                                                                                                                                                                                                                                                                                   | Dépendances | Résolution                                                                                 |
| ---- | --- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------ |
| ☑    | T24 | Adapter les exemples simulation   | Dans `examples/01_lmtl_no_transport.py`, `examples/02_transport_zalesak_jax.py`, `examples/04_benchmark_time_chunking.py` : wrapper les données en xr.DataArray dans la construction de Config.                                                                                                               | T14         | `Config()` avec `xr.DataArray` partout. Ex 02 n'utilise pas Config (standalone transport). |
| ☑    | T25 | Adapter les exemples optimisation | Dans `examples/05_ipop_cmaes_lmtl_0d.py`, `examples/05b_ga_lmtl_0d.py`, `examples/06_ipop_cmaes_2groups_lmtl_0d.py`, `examples/06b_ga_2groups_lmtl_0d.py`, `examples/07_ga_2groups_lmtl_transport.py` : wrapper les données en xr.DataArray. Vérifier que les optimiseurs ne dépendent pas de ParameterValue. | T14         | Tous adaptés. Scalaires forcings (D, bc\_\*) wrappés en `xr.DataArray()`.                  |
| ☑    | T26 | Adapter les exemples benchmark    | Dans `examples/03_benchmark_vmap_vs_grid.py`, `examples/08_benchmark_disk_vs_memory.py`, `examples/09_memory_profile_simulation.py` : wrapper les données en xr.DataArray.                                                                                                                                    | T14         | Tous adaptés. `Config()` avec `xr.DataArray`.                                              |

### Phase 8 : Revue — Corrections post-review

| État | ID | Nom                                           | Description                                                                                                                                                                                                                              | Dépendances | Résolution                                                                                                                                                                                                |
| ---- | -- | --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ☑    | P1 | Déplacer validation NaN au runtime            | **CRITIQUE** — `_validate_no_nan()` dans `validation.py` matérialisait les données lazy (Dask) à la validation, cassant le lazy loading. Déplacé vers le runtime : `ForcingStore.get_chunk()`/`get_statics()` + `_prepare_state/parameters` dans `compiler.py`. | T12b        | `_validate_no_nan()` supprimée de `validation.py`. `_check_nan()` ajouté dans `forcing.py` (get_chunk, get_statics) et `compiler.py` (_prepare_state, _prepare_parameters). ValueError levée au runtime. |
| ☑    | P2 | Renommer get_all → get_all_dynamic            | Nom trompeur : `get_all()` ne retourne que les dynamiques depuis T7 mais le nom suggère « tout ». Renommé en `get_all_dynamic()` pour cohérence avec l'API (get_chunk ne retourne que les dynamiques).                                    | T7          | Renommé dans `forcing.py`, `runner.py`, tous les tests (`test_forcing.py`, `test_interpolation.py`, `test_time_calendar.py`), exemples (`03_benchmark`), et docs (`memory_analysis.md`).                  |
| ☑    | P3 | Retirer blueprint_dims de infer_shapes        | Paramètre `blueprint_dims` accepté par `infer_shapes()` mais jamais utilisé dans le corps de la fonction. Dead code depuis T15.                                                                                                           | T15         | Paramètre retiré de `inference.py`. Appel mis à jour dans `compiler.py`.                                                                                                                                  |
| ☑    | P4 | Factoriser _validate_data_dims                | 3 blocs quasi-identiques (parameters, forcings, initial_state) dans `_validate_data_dims()`. Factorisé en une seule boucle sur `[(prefix, data_source)]`.                                                                                | T11         | Boucle unique sur `sections: list[tuple[str, dict]]` dans `validation.py`.                                                                                                                                |
| ☑    | P5 | Ajouter tests validation NaN runtime          | Tests manquants pour la validation NaN déplacée au runtime (P1). Couvre : NaN dans get_chunk, get_statics, _prepare_state, _prepare_parameters, et passthrough pour les types int (pas de NaN possible).                                  | P1, P2      | 3 tests dans `test_forcing.py` (chunk/statics/int) + 3 tests dans `test_compiler.py` (state/param/int). 455 passed, 2 xfailed.                                                                           |

## Historique des transitions

| De                | Vers             | Raison                                                   | Date       |
| ----------------- | ---------------- | -------------------------------------------------------- | ---------- |
| 1. Initialisation | 2. Analyse       | Besoin validé par l'utilisateur                          | 2026-03-08 |
| 2. Analyse        | 3. Architecture  | Analyse complétée                                        | 2026-03-08 |
| 3. Architecture   | 4. Planification | Architecture validée par l'utilisateur                   | 2026-03-08 |
| 4. Planification  | 5. Execution     | Todo list complétée                                      | 2026-03-08 |
| 5. Execution      | 6. Revue         | 27 tâches (T1-T26 + T12b) complétées. 451 tests passent. | 2026-03-08 |
| 6. Revue          | 7. Finalisation  | Revue critique : 5 corrections (P1-P5) identifiées et appliquées. 455 passed, 2 xfailed. | 2026-03-08 |
