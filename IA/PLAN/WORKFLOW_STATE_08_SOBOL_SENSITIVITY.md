# Workflow State - Analyse de Sensibilite Sobol

## Informations generales

- **Projet** : Module d'analyse de sensibilite de Sobol pour SeapoPym
- **Etape courante** : 9. Finalisation
- **Role actif** : Finaliseur
- **Derniere mise a jour** : 2026-02-06

## Resume du besoin

### Objectif

Implementer un module d'analyse de sensibilite de Sobol permettant d'evaluer l'impact de chaque parametre du modele sur differentes Quantities of Interest (QoI) extraites en des points specifiques de la grille de simulation.

### Workflow utilisateur

1. L'utilisateur definit une grille de simulation 2D (~20x20 deg) autour de points d'interet (stations oceanographiques)
2. Il selectionne les parametres a analyser avec leurs bornes
3. Il fournit les points d'extraction (liste de positions lat/lon)
4. SALib genere la matrice de Saltelli : N*(D+2) (1er ordre) ou N*(2D+2) (2nd ordre)
5. Le modele tourne en batch (vmap sur les echantillons de parametres + GPU), avec mini-batch temporel pour gerer la memoire
6. A chaque mini-batch temporel, les valeurs aux points d'interet sont extraites (reduction massive de memoire)
7. Les QoI sont calculees a partir de la serie temporelle complete aux points extraits : mean, var, argmax (jour du pic), median, etc.
8. Les indices de Sobol (S1, ST, optionnellement S2) sont calcules par QoI et par point
9. Sauvegarde incrementale en Parquet apres chaque batch de parametres (reprise possible)

### Architecture memoire (deux niveaux de batch)

- **Batch (outer)** : echantillons Sobol (ex: 256 param sets a la fois)
  - Sauvegarde Parquet a chaque fin de batch
  - Permet pause/reprise
- **Mini-batch (inner)** : chunks temporels (ex: 365 pas de temps)
  - Extraction des points d'interet a chaque chunk
  - Accumulation des series temporelles extraites (memoire faible)
  - La grille spatiale complete n'est gardee que pour le chunk courant

### Budget memoire type

- Etat GPU par mini-batch : batch_size * T_chunk * Y * X * 4 bytes
- Points accumules : batch_size * T_total * n_points * 4 bytes (negligeable)
- Exemple : 256 * 365 * 180 * 360 * 4 ~ 23 GB (ajustable via batch_size et T_chunk)

### Perimetre technique

- Nouveau sous-module : `seapopym/sensitivity/`
- Dependance : SALib en groupe optionnel (pyproject.toml)
- Performance : `jax.vmap` sur axe parametres + `lax.scan` pour le temps + GPU
- Parametres : selection explicite par l'utilisateur avec bornes
- Persistence : Parquet avec index sample, valeurs parametres, QoI par point

### Hors perimetre

- Pas d'autres methodes de sensibilite (Morris, FAST)
- Pas de visualisation integree
- Pas de multi-GPU (pmap)

## Rapport d'analyse

### Structure pertinente du projet

```
seapopym/
  engine/
    step.py          # build_step_fn(model, params_as_argument=True/False)
    runners.py       # StreamingRunner (chunked) + GradientRunner (full scan)
    backends.py      # JAXBackend (lax.scan + jit) + NumpyBackend
    vectorize.py     # wrap_with_vmap() pour broadcast spatial
    io.py            # DiskWriter (zarr) + MemoryWriter (xarray)
  compiler/
    model.py         # CompiledModel dataclass
  optimization/
    gradient.py      # GradientRunner.run_with_params() + SparseObservations
    optimizer.py     # Optimizer (optax wrapper)
    evolutionary.py  # CMA-ES (evosax, optionnel)
    hybrid.py        # CMA-ES + gradient
    loss.py          # mse, rmse, nrmse
  functions/
    lmtl.py          # fonctions LMTL (biologie)
    transport.py     # advection-diffusion JAX
```

### Technologies identifiees

- Langage : Python 3.12+
- Framework calcul : JAX (jit, lax.scan, vmap, grad)
- Build : hatchling
- Tests : pytest (markers: slow, integration, unit, gpu)
- Linter/Formatter : ruff (line-length=120, double quotes)
- Dependances cles : jax[cuda12], optax, networkx, pydantic, xarray, zarr, scipy

### Patterns et conventions

- **Nommage** : snake_case partout, classes CamelCase
- **Type aliases** : `Array = Any`, `State = dict[str, Array]`, `Params = dict[str, Array]`
- **Docstrings** : Google style avec Args/Returns
- **Deps optionnelles** : try/except dans `__init__.py` + message d'erreur utile + groupe dans pyproject.toml
- **Tests** : classes pytest (TestXxx), pas de fixtures complexes
- **Dimension canonique** : `(E, T, F, C, Z, Y, X)` dans CANONICAL_DIMS

### Points cles pour Sobol

#### 1. `build_step_fn(model, params_as_argument=True)` (step.py:35)

C'est la brique fondamentale. Produit une step function avec signature :
```
((state, params), forcings_t) -> ((new_state, params), outputs)
```
Les params sont dans le carry et non captures par closure. Compatible avec `lax.scan` ET `jax.vmap`.

#### 2. `GradientRunner.run_with_params()` (optimization/gradient.py:88)

Utilise deja `params_as_argument=True` + `lax.scan`. **MAIS** : mute `self.model.parameters` in-place (lignes 276-290). Ce pattern n'est PAS compatible avec vmap. Pour Sobol, il faut une approche purement fonctionnelle.

#### 3. StreamingRunner (runners.py:36)

Gere deja le chunking temporel avec `_slice_forcings(start, end)`. Le pattern de decoupe forcings est reutilisable pour les mini-batches temporels.

#### 4. Systeme vmap spatial (vectorize.py)

Le vmap existant broadcast sur (Y, X) pour les fonctions avec core_dims. C'est interne au step_fn et ne concerne PAS le vmap sur les params Sobol. Les deux niveaux de vmap sont independants :
- vmap interne (spatial) = deja dans step_fn
- vmap externe (Sobol params) = A AJOUTER

#### 5. JAXBackend.scan() (backends.py:56)

Cree un `@jax.jit` a chaque appel. Pour Sobol avec mini-batches, il faudra gerer le JIT plus finement (compiler une fois, reutiliser).

#### 6. Dependances disponibles

- `scipy>=1.15.3` est deja en dep → `scipy.stats.qmc.Sobol` disponible sans rien ajouter
- SALib serait une dep optionnelle supplementaire (pour sampling Saltelli + calcul indices)
- `pyarrow` / `pandas` pour Parquet : pandas est deja en dep

### Points d'attention

1. **Purete fonctionnelle** : Le pattern de `GradientRunner` (mutation in-place) ne marchera pas avec `jax.vmap`. Il faut une approche fonctionnelle pure ou le scan prend les params en argument.

2. **JIT et recompilation** : `JAXBackend.scan()` JIT-compile a chaque appel. Pour le Sobol (beaucoup d'appels identiques), il faut compiler une fois et reutiliser. Le vmapped scan devrait etre JIT-compile une seule fois.

3. **Broadcasting forcings statiques** : Les forcings sans dimension temporelle (ex: mask) doivent etre broadcast AVANT le vmap. Pattern deja present dans `GradientRunner._prepare_forcings()` et `StreamingRunner._slice_forcings()`.

4. **Extraction de points** : Fancy indexing JAX (`output[:, y_indices, x_indices]`) fonctionne mais genere des gather ops sur GPU. Pour un petit nombre de points (~5-20), c'est negligeable.

5. **Parquet + reprise** : pandas est deja une dep. Le schema Parquet doit inclure l'index de sample pour permettre la reprise.

6. **scipy.stats.qmc.Sobol** : N doit etre une puissance de 2. La dimension du Sobol est 2*D (pour la matrice A et B de Saltelli). SALib simplifie tout ca.

## Decisions d'architecture

### Choix techniques

| Domaine          | Choix                              | Justification                                                    |
|------------------|------------------------------------|------------------------------------------------------------------|
| Sampling         | SALib (dep optionnelle)            | Gere Saltelli + indices Sobol, bien teste, standard en SA        |
| Persistence      | Parquet via pandas                 | Deja en dep, append efficace, lisible, reprise possible          |
| Batch eval       | `jax.vmap` + `lax.scan`           | Parallelisme sur params, compose avec scan existant              |
| JIT              | Compile une fois le vmapped scan   | Evite recompilation a chaque batch                               |
| QoI              | Fonctions JAX pures               | Executables sur GPU, vmappables                                  |
| Step function    | `params_as_argument=True`          | Params dynamiques dans le carry, compatible vmap                 |

### Structure proposee

```
seapopym/sensitivity/
    __init__.py        # API publique : SobolAnalyzer, SobolResult
    sobol.py           # SobolAnalyzer (orchestration principale)
    runner.py          # SobolRunner (vmap + scan chunke + extraction points)
    qoi.py             # Fonctions QoI (mean, var, argmax, median...)
    checkpoint.py      # Persistence Parquet (save/resume)
```

### Interfaces et contrats

#### SobolAnalyzer (sobol.py) - point d'entree utilisateur

```python
class SobolAnalyzer:
    def __init__(self, model: CompiledModel): ...

    def analyze(
        self,
        param_bounds: dict[str, tuple[float, float]],
        extraction_points: list[tuple[int, int]],  # indices (y, x)
        output_variable: str = "biomass",
        n_samples: int = 1024,           # N, puissance de 2
        calc_second_order: bool = False,
        qoi: list[str] | None = None,    # ["mean","var","argmax","median"]
        batch_size: int = 256,
        chunk_size: int = 365,
        checkpoint_path: Path | None = None,
    ) -> SobolResult: ...
```

#### SobolResult (sobol.py) - resultat structure

```python
@dataclass
class SobolResult:
    S1: pd.DataFrame        # index=(qoi, point), columns=param_names
    ST: pd.DataFrame
    S2: pd.DataFrame | None
    S1_conf: pd.DataFrame
    ST_conf: pd.DataFrame
    n_samples: int
    problem: dict           # SALib problem dict (reproductibilite)
```

#### SobolRunner (runner.py) - execution batched (interne)

```python
class SobolRunner:
    def __init__(self, model, extraction_points, output_variable, chunk_size): ...

    def run_batch(self, params_batch: dict[str, Array]) -> Array:
        """Retourne (batch_size, T_total, n_points)"""
```

Compile `jax.jit(jax.vmap(lax.scan))` une seule fois. Gere le chunking temporel
et l'extraction de points a chaque chunk. Padding du dernier batch si incomplet.

#### QoI (qoi.py) - fonctions pures

```python
def compute_qoi(time_series: Array, qoi_names: list[str]) -> dict[str, Array]:
    """time_series: (batch_size, T, n_points) -> dict[qoi: (batch_size, n_points)]"""
```

QoI disponibles : mean, var, std, median, argmax (jour du pic), min, max.

#### SobolCheckpoint (checkpoint.py) - persistence incrementale

```python
class SobolCheckpoint:
    def __init__(self, path: Path, problem: dict): ...
    def save_batch(self, sample_indices, params_values, qoi_values): ...
    def load(self) -> tuple[int, pd.DataFrame]: ...
```

### Risques identifies

| Risque                                    | Impact | Mitigation                                            |
|-------------------------------------------|--------|-------------------------------------------------------|
| OOM GPU (batch_size x chunk_size x Y x X) | Haut   | Log budget memoire estime + params ajustables         |
| JIT recompile si shapes changent           | Moyen  | Padding du dernier batch a batch_size                 |
| SALib API change entre versions            | Bas    | Fixer version min dans pyproject.toml                 |
| Checkpoint incompatible si QoI changent    | Bas    | Stocker metadata (qoi_names, problem) dans le Parquet |

## Todo List

| Etat | ID  | Nom                            | Description                                                                                                                                  | Dependances | Resolution |
|------|-----|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|-------------|------------|
| ☐    | T0  | Unifier params_as_argument     | Supprimer le mode closure (params_as_argument=False) et utiliser params_as_argument=True partout. Refactoring hors perimetre Sobol.          | -           | -          |
| ☑    | T1  | Creer `sensitivity/__init__.py`| Creer `seapopym/sensitivity/__init__.py` avec API publique (imports de SobolAnalyzer, SobolResult) et import optionnel SALib avec message d'erreur. | -           | Cree avec lazy imports et message d'erreur SALib |
| ☑    | T2  | Creer `sensitivity/qoi.py`    | Creer `seapopym/sensitivity/qoi.py` avec `compute_qoi(time_series, qoi_names)`. Fonctions JAX pures : mean, var, std, median, argmax, min, max. Input: (batch, T, n_points) → Output: dict[qoi: (batch, n_points)]. | -           | Cree avec registre de 7 QoI + compute_qoi() |
| ☑    | T3  | Creer `sensitivity/checkpoint.py` | Creer `seapopym/sensitivity/checkpoint.py` avec `SobolCheckpoint`. Methodes: `save_batch(sample_indices, params_values, qoi_values)`, `load() -> (n_done, DataFrame)`. Schema Parquet avec metadata (problem, qoi_names). | -           | Cree avec save/load/validate + metadata JSON |
| ☑    | T4  | Creer `sensitivity/runner.py`  | Creer `seapopym/sensitivity/runner.py` avec `SobolRunner`. Utilise `build_step_fn(model, params_as_argument=True)`. Compile `jit(vmap(lax.scan))` une fois. `run_batch(params_batch) -> (batch_size, T_total, n_points)`. Gere chunking temporel + extraction points + padding dernier batch. | T2          | Cree avec jit(vmap(scan)) cache + chunking temporel |
| ☑    | T5  | Creer `sensitivity/sobol.py`   | Creer `seapopym/sensitivity/sobol.py` avec `SobolAnalyzer` et `SobolResult`. Orchestre les 3 phases : SALib sampling → SobolRunner batches → SALib analyze. Gere reprise via SobolCheckpoint. | T2, T3, T4  | Cree avec 3 phases + resume + validation |
| ☑    | T6  | Ajouter dep SALib pyproject.toml | Ajouter le groupe optionnel `sensitivity = ["SALib>=1.4.0"]` dans `[project.optional-dependencies]` de `pyproject.toml`. | -           | Groupe sensitivity ajoute |
| ☑    | T7  | Creer `tests/sensitivity/__init__.py` | Creer le package de tests `tests/sensitivity/__init__.py`.                                                                            | -           | Cree |
| ☑    | T8  | Creer `tests/sensitivity/test_qoi.py` | Tests unitaires pour `compute_qoi` : verifier mean, var, argmax, median sur des series temporelles connues.                           | T2          | 12/12 pass |
| ☑    | T9  | Creer `tests/sensitivity/test_checkpoint.py` | Tests unitaires pour `SobolCheckpoint` : save/load round-trip, reprise partielle, metadata coherentes.                          | T3          | 6/6 pass |
| ☑    | T10 | Creer `tests/sensitivity/test_runner.py` | Tests unitaires pour `SobolRunner` : run_batch sur modele minimal (0D), verifier shapes, extraction points, padding.                | T4          | 4/4 pass (fix: chunk padding + 0D n_points) |
| ☑    | T11 | Creer `tests/sensitivity/test_sobol.py` | Test d'integration pour `SobolAnalyzer.analyze()` sur modele minimal. Verifier que S1+ST sont calcules, checkpoint fonctionne.       | T5          | 6/6 pass (fix: n_points inference for 0D) |

## Historique des transitions

| De                | Vers            | Raison                                     | Date       |
|-------------------|-----------------|--------------------------------------------|------------|
| 1. Initialisation | 2. Analyse      | Besoin valide par l'utilisateur            | 2026-02-06 |
| 2. Analyse        | 3. Architecture | Analyse completee                          | 2026-02-06 |
| 3. Architecture   | 4. Planification| Architecture validee par l'utilisateur     | 2026-02-06 |
| 4. Planification  | 5. Execution    | Todo list completee                        | 2026-02-06 |
| 5. Execution      | 6. Revue        | Taches T1-T6 implementees                 | 2026-02-06 |
| 6. Revue          | 8. Test         | 0 issues (2 mineures corrigees: ruff)      | 2026-02-06 |
| 8. Test           | 9. Finalisation | 29/29 tests pass (2 bugs fixes: chunk padding, 0D n_points) | 2026-02-06 |
