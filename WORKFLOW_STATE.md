# Workflow State

## Informations generales

- **Projet** : Redesign du pipeline d'optimisation
- **Etape courante** : 9. Finalisation
- **Role actif** : Finalisateur
- **Derniere mise a jour** : 2026-03-02

## Resume du besoin

### Constat

`SparseObservations` est une classe rigide (indices `T, Y, X` hardcodes) utilisee a un seul endroit (`GradientRunner.make_loss_fn`). `GradientRunner` cumule execution du modele et utilitaires de gradient redondants avec JAX. Le pipeline d'optimisation est inutilement couple.

### Objectif

Simplifier le pipeline d'optimisation en :

1. **Supprimant `SparseObservations`** — l'utilisateur construit sa loss/likelihood lui-meme
2. **Supprimant `GradientRunner`** — deplacer `run_with_params` sur `CompiledModel`
3. **Ajoutant un helper `coords_to_indices`** — conversion coordonnees physiques (lat, lon, date) vers indices entiers, frontiere xarray/JAX

### Pipeline cible

```
1. Charger les donnees (forcings + observations terrain en coordonnees)
2. Charger le blueprint
3. Compiler le modele -> CompiledModel (avec .run_with_params())
4. Convertir observations coordonnees -> indices (helper coords_to_indices, utilise la grille du CompiledModel)
5. Construire la loss/likelihood (l'utilisateur ecrit sa fonction, appelle model.run_with_params dedans)
6. Choisir un optimizer et lancer : optimizer.run(loss_fn, initial_params)
```

### Decisions prises pendant l'initialisation

- **Perimetre** : chaine complete (dataclass, runner, construction loss, documentation)
- **Retrocompatibilite** : breaking change OK
- **Public cible** : developpeur integrateur, pas utilisateur final. On fournit les briques de base, pas de helpers de commodite. Un developpeur tiers peut construire sa surcouche.
- **Pas de classe Observation generique** : l'utilisateur ecrit la comparaison lui-meme
- **Selection des parametres** : a la charge de l'utilisateur dans sa loss (merge params libres + fixes)
- **Differentiabilite** : depend de l'optimizer choisi (gradient = JAX pur, CMA-ES = libre)
- **Frontiere xarray/JAX** : la conversion coordonnees -> indices se fait AVANT la construction de la loss (etape 4), plus aucun xarray dans le pipeline d'optimisation apres cette etape

### Deux chemins pour l'optimisation

**Chemin loss (gradient, CMA-ES, hybride) :**
L'utilisateur ecrit une `loss_fn(params) -> scalar` qui appelle `model.run_with_params`, extrait les predictions, et compare avec ses observations via les fonctions de loss (mse, rmse, nrmse).

**Chemin bayesien (NUTS) — deux modes dans `make_log_posterior` :**

1. **Proxy** : l'utilisateur fournit une `loss_fn` existante + priors → `log_post = -loss + log_prior`. Approximatif, suffisant pour MAP.
2. **Gaussien complet** : l'utilisateur fournit une `predict_fn(params) -> Array` + `obs_values: Array` (meme shape) + priors + modele d'erreur (sigma libre ou fixe). Statistiquement correct pour NUTS.

Contrat de shape : `predict_fn(params)` et `obs_values` doivent retourner des arrays de meme shape (comparaison de vecteurs). Erreur explicite si mismatch.

### Ce qui disparait

| Composant                                   | Raison                                                                        |
| ------------------------------------------- | ----------------------------------------------------------------------------- |
| `SparseObservations`                        | Rigide, utilisee a un seul endroit, l'utilisateur peut faire mieux            |
| `GradientRunner`                            | `run_with_params` -> `CompiledModel`, le reste est redondant (1 ligne de JAX) |
| `GradientRunner.make_loss_fn`               | Remplace par la loss ecrite par l'utilisateur                                 |
| `GradientRunner.compute_gradient`           | `jax.grad(loss_fn)(params)`                                                   |
| `GradientRunner.compute_value_and_gradient` | `jax.value_and_grad(loss_fn)(params)`                                         |
| `GradientRunner.optimize`                   | Redondant avec `Optimizer.run()`                                              |
| `gradient.py` (fichier entier)              | Plus rien dedans apres suppression de GradientRunner et SparseObservations    |
| `tests/optimization/test_gradient.py`       | Tests du code supprime                                                        |

### Ce qui reste / s'ajoute

| Composant                            | Role                                               |
| ------------------------------------ | -------------------------------------------------- |
| `CompiledModel.run_with_params()`    | Execution du modele (unique point d'entree)        |
| `coords_to_indices()`                | Helper conversion coordonnees -> indices (nouveau) |
| `loss.py` (mse, rmse, nrmse)         | Briques composables pour les loss                  |
| `likelihood.py` (make_log_posterior) | Construction log-posterior bayesien                |
| `Optimizer`                          | Gradient-based (Adam, SGD via optax)               |
| `EvolutionaryOptimizer`              | CMA-ES, GA via evosax                              |
| `HybridOptimizer`                    | CMA-ES + gradient                                  |
| `run_nuts`                           | MCMC via BlackJAX                                  |

## Decisions d'architecture

### Q1 — Classes vs fonctions : statu quo

Garder les classes pour les optimizers avec etat (GradientOptimizer, EvolutionaryOptimizer, HybridOptimizer) et les fonctions pour les appels stateless (run_nuts, run_ipop). Pas de Protocol commun — les interfaces sont trop differentes. Documenter clairement les deux niveaux :
- **Briques de base** : GradientOptimizer, EvolutionaryOptimizer (un seul run)
- **Strategies composees** : HybridOptimizer (compose les briques), run_ipop (boucle sur EvolutionaryOptimizer)
- **Autre paradigme** : run_nuts (echantillonnage MCMC, pas de l'optimisation)

### Q2 — Renommer Optimizer → GradientOptimizer

Coherent avec EvolutionaryOptimizer et HybridOptimizer. Explicite sur la methode.

### Q3 — Pas de Protocol transversal

Les interfaces sont trop differentes (n_steps vs n_samples, OptimizeResult vs NUTSResult). Forcer un Protocol serait de la sur-architecture. Les type aliases (Q4) suffisent pour le contrat d'entree.

### Q4 — Type aliases dans seapopym/types.py

Ajouter :
- `LossFn = Callable[[Params], Array]` (params -> scalar a minimiser)
- `PredictFn = Callable[[Params], Array]` (params -> array de predictions)
- `LogPosteriorFn = Callable[[Params], Array]` (params -> scalar a maximiser)

### Q5 — coords_to_indices : fonction standalone dans seapopym/compiler/coords.py

Fonction standalone, pas methode sur CompiledModel. Placee dans `seapopym/compiler/coords.py` proche de la source des coordonnees.

Interface : kwargs avec noms de dimensions canoniques (E, T, F, C, Z, Y, X). Dimension-agnostique — fonctionne pour toute dimension presente dans `model.coords`. L'utilisateur passe `grid=model.coords` + les dimensions qu'il veut indexer. Nearest-neighbor matching.

Matching : delegue a xarray `.sel()`. Parametre `sel_kwargs: dict | None` (defaut `{"method": "nearest"}`), passe tel quel via `**sel_kwargs` a xarray. API 100% compatible xarray, zero maintenance.

Validation :
- Chaque kwarg doit correspondre a une cle dans `model.coords` → erreur explicite sinon
- Valeurs hors range de la grille → erreur explicite (geree par xarray + tolerance)

### Q6 — Pas de shape check dans la couche de base

Le mismatch pred/obs casse a l'execution JAX. C'est une surcouche que le developpeur integrateur peut ajouter. On documente le contrat de shape.

### Structure des fichiers apres redesign

```
seapopym/compiler/
├── model.py           CompiledModel + run_with_params() (nouveau)
├── coords.py          coords_to_indices() (nouveau)
└── ...

seapopym/optimization/
├── __init__.py        Exports mis a jour (sans GradientRunner, SparseObservations)
├── optimizer.py       GradientOptimizer (renomme), OptimizeResult
├── loss.py            mse, rmse, nrmse (inchange)
├── likelihood.py      GaussianLikelihood, make_log_posterior (inchange)
├── prior.py           Priors, PriorSet (inchange)
├── evolutionary.py    EvolutionaryOptimizer (inchange)
├── hybrid.py          HybridOptimizer (ref interne vers GradientOptimizer)
├── nuts.py            run_nuts, NUTSResult (inchange)
└── ipop.py            run_ipop, IPOPResult (inchange)
                       gradient.py SUPPRIME

seapopym/types.py      + LossFn, PredictFn, LogPosteriorFn
```

### Risques identifies

| Risque | Impact | Mitigation |
|---|---|---|
| `build_step_fn` import dans compiler/ cree un couplage engine → compiler | Moyen | Verifier que le couplage est unidirectionnel (compiler importe engine, pas l'inverse) |
| Renommage Optimizer → GradientOptimizer casse HybridOptimizer en interne | Bas | Mettre a jour les refs internes (hybrid.py L100) |
| Integration test utilise GradientRunner | Bas | Reecrire avec CompiledModel.run_with_params |
| Metadata F dimension perdue a la compilation | Hors perimetre | Limitation connue, a traiter dans un redesign du compiler |

## Rapport d'analyse

### Structure du module optimization/

10 fichiers, 2227 LOC, 151 tests. Organisation 1:1 source/test.

```
seapopym/optimization/
├── __init__.py      (82 LOC)   Exports + lazy loading optional deps
├── gradient.py      (294 LOC)  GradientRunner, SparseObservations  ← A SUPPRIMER
├── optimizer.py     (345 LOC)  Optimizer (optax), OptimizeResult
├── loss.py          (152 LOC)  mse, rmse, nrmse
├── likelihood.py    (182 LOC)  GaussianLikelihood, make_log_posterior
├── prior.py         (291 LOC)  Uniform, Normal, LogNormal, HalfNormal, TruncatedNormal, PriorSet
├── evolutionary.py  (291 LOC)  EvolutionaryOptimizer (evosax)
├── hybrid.py        (252 LOC)  HybridOptimizer (CMA-ES + gradient)
├── nuts.py          (152 LOC)  run_nuts, NUTSResult (blackjax)
└── ipop.py          (186 LOC)  run_ipop, run_ipop_cmaes, IPOPResult
```

### Technologies

- **Langage** : Python >=3.12, pyright target 3.10
- **Core** : JAX, jax.numpy, jax.lax.scan
- **Gradient** : optax (Adam, SGD, RMSprop, Adagrad)
- **Evolutionnaire** : evosax (CMA-ES, SimpleGA) — optionnel
- **Bayesien** : blackjax (NUTS) — optionnel
- **Types** : `seapopym/types.py` → `Array = Any`, `Params = dict[str, Array]`
- **Build** : hatchling, uv
- **Lint/format** : ruff (120 chars, double quotes, isort)
- **Type check** : pyright basic
- **Tests** : pytest, classes `Test*`, Google docstrings

### Graphe de dependances internes

```
loss.py, prior.py, optimizer.py, nuts.py  ← independants (feuilles)
likelihood.py  ← prior.py
gradient.py    ← loss.py, optimizer.py, engine.step, compiler.CompiledModel
evolutionary.py ← optimizer.py (OptimizeResult)
hybrid.py      ← evolutionary.py, optimizer.py
ipop.py        ← evolutionary.py
__init__.py    ← tout
```

### CompiledModel (etat actuel)

Fichier : `seapopym/compiler/model.py`. Dataclass data-centric (pytree-friendly).

**N'a PAS de `run_with_params()`**. L'execution est faite par `GradientRunner` qui :

1. Appelle `build_step_fn(model)` depuis `seapopym.engine.step`
2. Execute via `jax.lax.scan(step_fn, (state, params), forcings)`

Pour deplacer `run_with_params` sur `CompiledModel`, il faut importer `build_step_fn` dans `compiler/model.py` ou un module adjacent.

### Usage externe de gradient.py

| Consommateur                          | Importe                            | Impact suppression                                 |
| ------------------------------------- | ---------------------------------- | -------------------------------------------------- |
| `tests/optimization/test_gradient.py` | GradientRunner, SparseObservations | Supprimer le fichier test                          |
| `tests/engine/test_integration.py`    | GradientRunner                     | Adapter : utiliser `CompiledModel.run_with_params` |
| `__init__.py`                         | GradientRunner, SparseObservations | Retirer des exports                                |
| Exemples (04, 07)                     | Pas d'import direct                | Pas d'impact                                       |

### Incoherences relevees entre optimizers

**Interfaces :**

| Composant                 | Type     | Run method   | Iteration param | Retour           |
| ------------------------- | -------- | ------------ | --------------- | ---------------- |
| Optimizer                 | classe   | `.run()`     | `n_steps`       | `OptimizeResult` |
| EvolutionaryOptimizer     | classe   | `.run()`     | `n_generations` | `OptimizeResult` |
| HybridOptimizer           | classe   | `.run()`     | `n_generations` | `OptimizeResult` |
| run_ipop / run_ipop_cmaes | fonction | appel direct | `n_generations` | `IPOPResult`     |
| run_nuts                  | fonction | appel direct | `n_samples`     | `NUTSResult`     |

**Resultats :**

| Champ          | OptimizeResult              | NUTSResult                         | IPOPResult                          |
| -------------- | --------------------------- | ---------------------------------- | ----------------------------------- |
| params/samples | `params: Params`            | `samples: Params`                  | `modes: list[OptimizeResult]`       |
| score          | `loss: float`               | `log_posterior_values: Array`      | via modes                           |
| historique     | `loss_history: list[float]` | —                                  | `all_results: list[OptimizeResult]` |
| convergence    | `converged: bool`           | `divergences: Array`               | `n_restarts: int`                   |
| metadata       | `message: str`              | `acceptance_rate`, `kernel_params` | —                                   |

**Bounds :**

- Optimizer : optionnel, scaling "none"/"bounds"/"log"
- EvolutionaryOptimizer : optionnel mais recommande (defaut [0,1])
- IPOP : obligatoire
- NUTS : via PriorSet.get_bounds()

**Composition interne :**

- HybridOptimizer utilise EvolutionaryOptimizer (phase 1) + Optimizer (phase 2)
- run_ipop utilise EvolutionaryOptimizer en boucle avec population doublante

### Points d'attention

1. **`build_step_fn` couplage** : gradient.py est le seul consommateur de `seapopym.engine.step.build_step_fn`. Deplacer `run_with_params` implique de deplacer ce couplage vers `compiler/`.
2. **`extract_trainable_params`** : methode utilitaire sur GradientRunner (L287-294) qui lit `model.trainable_params`. Sera perdue a la suppression — peut etre triviale a reproduire.
3. **Integration test** : `tests/engine/test_integration.py` utilise GradientRunner pour valider le pipeline Blueprint → Compile → Optimize. Devra etre reecrit.
4. **3 types de retour differents** : OptimizeResult, NUTSResult, IPOPResult sans ancetre commun.
5. **Prior Protocol** : `prior.py` utilise deja un pattern Protocol pour les distributions. Precedent pour Q3.

## Todo List

| Etat | ID  | Nom | Description | Dependances | Resolution |
| ---- | --- | --- | ----------- | ----------- | ---------- |
| ☑ | T1 | Type aliases | Ajouter `LossFn`, `PredictFn`, `LogPosteriorFn` dans `seapopym/types.py` | - | Ajout de 3 aliases + import Callable |
| ☑ | T2 | run_with_params | Ajouter methode `run_with_params()` sur `CompiledModel` dans `seapopym/compiler/model.py`. Importer `build_step_fn` depuis `seapopym.engine.step`, executer via `jax.lax.scan`. Signature : `run_with_params(params, initial_state=None, forcings=None) -> (final_state, outputs)` | - | Methode ajoutee, import lazy dans le body |
| ☑ | T3 | coords_to_indices | Creer `seapopym/compiler/coords.py` avec `coords_to_indices(grid, sel_kwargs=None, **dims) -> tuple[Array, ...]`. Delegue le matching a `xr.DataArray.sel(**sel_kwargs)`. Default `sel_kwargs={"method": "nearest"}`. Valider que chaque kwarg existe dans grid | - | Fichier cree, delegation a xr.DataArray.sel |
| ☑ | T4 | Export coords | Ajouter `coords_to_indices` dans `seapopym/compiler/__init__.py` et `__all__` | T3 | Exporte ajoutee |
| ☑ | T5 | Rename Optimizer | Renommer `Optimizer` → `GradientOptimizer` dans `seapopym/optimization/optimizer.py` (classe + docstrings + refs internes) | - | Classe renommee + exemples docstring |
| ☑ | T6 | Update hybrid.py | Mettre a jour `from seapopym.optimization.optimizer import Optimizer` → `GradientOptimizer` dans `seapopym/optimization/hybrid.py` | T5 | Import + 2 refs internes mises a jour |
| ☑ | T7 | Update __init__.py | Dans `seapopym/optimization/__init__.py` : retirer import `GradientRunner`/`SparseObservations`, remplacer `Optimizer` par `GradientOptimizer`, mettre a jour `__all__` et docstring | T5, T6 | Imports et exports mis a jour |
| ☑ | T8 | Delete gradient.py | Supprimer `seapopym/optimization/gradient.py` | T2, T7 | Fichier supprime |
| ☑ | T9 | Update test_optimizer | Renommer toutes les refs `Optimizer` → `GradientOptimizer` dans `tests/optimization/test_optimizer.py` | T5 | Import + 30 refs + classes + docstrings |
| ☑ | T10 | Delete test_gradient | Supprimer `tests/optimization/test_gradient.py` | T8 | Fichier supprime |
| ☑ | T11 | Adapt test_integration | Dans `tests/engine/test_integration.py` : remplacer `GradientRunner(model)` par `model.run_with_params(dict(model.parameters))` directement | T2, T8 | Adapte, 4/4 tests passent |
| ☑ | T12 | Tests coords | Creer `tests/compiler/test_coords.py` : tester `coords_to_indices` — cas nominal (nearest match), erreur dim inconnue, erreur valeur hors range, sel_kwargs custom (method=None pour exact match) | T3 | 13 tests, 4 classes (Nominal, Errors, SelKwargs, Datetime) |
| ☑ | T13 | Tests run_with_params | Creer `tests/compiler/test_run_with_params.py` : tester `CompiledModel.run_with_params` — execution nominale, override state/forcings, retour (final_state, outputs) | T2 | 7 tests, 2 classes (Nominal, Overrides). Fixture isolee (pop au lieu de clear_registry) |
| ☑ | T14 | Cleanup doc | Supprimer `sparse-observations-redesign.md` (contenu integre dans WORKFLOW_STATE.md) | T8 | Fichier supprime |

## Rapport de revue

### Verifications automatiques

| Outil | Resultat | Erreurs | Warnings |
|-------|----------|---------|----------|
| ruff check | ✅ | 0 | 0 |
| pyright | ✅ | 0 | 0 |
| ruff format | ⚠️ | 3 pre-existantes | - |

### Issues identifiees

| ID | Severite | Description | Fichier | Action |
|----|----------|-------------|---------|--------|
| I1 | Info | 3 docstrings/commentaires mentionnent encore "Optimizer" (sans prefix "Gradient") | evolutionary.py:3, prior.py:228, optimizer.py:1 | ✅ Corrige |
| I2 | Info | `build_step_fn(self)` appele a chaque `run_with_params` (pas de cache comme dans l'ancien GradientRunner) | compiler/model.py:112 | ⏭ Acceptable — JAX gere le cache JIT |
| I3 | Info | Formatage pre-existant (espaces dans f-strings) | hybrid.py:192, optimizer.py:325, __init__.py:75 | ✅ Corrige |

### Analyse des taches echouees

Aucune tache echouee. T12 et T13 reportees a l'etape 8 (Test) conformement au protocole.

### Decision

3 issues Info → Passer a Resolution

### Verifications post-correction

| Outil | Resultat |
|-------|----------|
| ruff check | ✅ 0 erreurs (fichiers modifies) |
| ruff format | ✅ 0 fichiers a reformater |
| pyright | ✅ 0 erreurs |
| pytest | ✅ 146 passed, 0 failed |

## Rapport de test

### Nouveaux tests crees

| Fichier | Tests | Classes | Resultat |
|---------|-------|---------|----------|
| `tests/compiler/test_coords.py` | 13 | TestCoordsToIndicesNominal (6), TestCoordsToIndicesErrors (2), TestCoordsToIndicesSelKwargs (4), TestCoordsToIndicesDatetime (1) | ✅ 13/13 |
| `tests/compiler/test_run_with_params.py` | 7 | TestRunWithParamsNominal (5), TestRunWithParamsOverrides (2) | ✅ 7/7 |

### Verifications

| Outil | Resultat |
|-------|----------|
| ruff check | ✅ 0 erreurs |
| pyright | ✅ 0 erreurs |
| pytest (20 nouveaux tests) | ✅ 20/20 passed |
| pytest (suite complete) | ✅ 541 passed, 2 failed (pre-existants sur main) |

### Note sur les 2 echecs pre-existants

`tests/models/test_models.py::TestFunctionRegistry::test_all_functions_registered[lmtl|lmtl_no_transport]` echouent egalement sur `main` (verifie). Cause : d'autres fixtures (`test_compiler.py`, `test_registry.py`, `conftest.py` engine) utilisent `clear_registry()` et suppriment les fonctions LMTL globales. Non lie a notre redesign.

## Historique des transitions

| De                | Vers       | Raison                          | Date       |
| ----------------- | ---------- | ------------------------------- | ---------- |
| 1. Initialisation | 2. Analyse | Besoin valide par l'utilisateur | 2026-03-02 |
| 2. Analyse | 3. Architecture | Analyse completee | 2026-03-02 |
| 3. Architecture | 4. Planification | Architecture validee | 2026-03-02 |
| 4. Planification | 5. Execution | Todo list completee (14 taches) | 2026-03-02 |
| 5. Execution | 6. Revue | 12/14 taches completees, 2 reportees a Test | 2026-03-02 |
| 6. Revue | 7. Resolution | 3 issues Info | 2026-03-02 |
| 7. Resolution | 8. Test | Corrections validees, 146 tests passent | 2026-03-02 |
| 8. Test | 9. Finalisation | 20 nouveaux tests passent, 0 regression | 2026-03-02 |
