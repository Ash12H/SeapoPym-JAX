# Audit qualité — `seapopym/optimization/`

## Vue d'ensemble

Le module contient ~2 200 LOC répartis en 10 fichiers, avec des tests 1:1.
Dépendances optionnelles : `evosax` (CMA-ES, GA), `blackjax` (NUTS), `flax` (inutilisé ?).

---

## Milestone 1 — Fondations (`loss`, `prior`, `optimizer`)

Socle commun utilisé par tous les autres modules.

- [x] **`loss.py`** — Fix pyright `jnp.where` type, +3 tests masked nrmse (coverage 100%)
- [x] **`prior.py`** — Suppr. `_bounds_arrays()` dupliqué, fix `TruncatedNormal.sample` (CDF inverse), +4 tests PriorSet (coverage 96%)
- [x] **`optimizer.py`** — Fix import order (ruff E402), +1 test mixed bounded/unbounded (coverage 93%)

## Milestone 2 — Inférence bayésienne (`likelihood`, `nuts`)

Composants pour estimation MAP et échantillonnage MCMC.

- [x] **`likelihood.py`** — Fix `_bounds_arrays()` → `get_bounds()`, +3 tests (reparameterize, jacobian, missing sigma KeyError)
- [x] **`nuts.py`** — Fix import order (ruff E402), `# type: ignore` BlackJAX justifiés

## Milestone 3 — Optimisation gradient (`gradient`)

Runner qui connecte le modèle compilé aux loss functions.

- [x] **`gradient.py`** (294 LOC) — `GradientRunner`, `SparseObservations`
  - SparseObservations limitation (T,Y,X rigide) documentée → `docs/design/sparse-observations-redesign.md`
  - +3 tests GradientRunner (unknown loss_type, rmse/nrmse, compute_value_and_gradient)

## Milestone 4 — Optimisation évolutionnaire (`evolutionary`, `ipop`)

Stratégies population-based via evosax.

- [x] **`evolutionary.py`** (291 LOC) — `EvolutionaryOptimizer`
  - Fix ruff E402 (imports after logger)
  - Fix tests: `_flatten` API mismatch (2 vs 4 return values), `_apply_bounds` removed → `TestNormalization`
  - Fix `test_run_minimizes_quadratic` / `test_run_multivariate` (missing bounds → targets unreachable)
- [x] **`ipop.py`** (185 LOC) — `run_ipop`, `run_ipop_cmaes`
  - Fix ruff E402 (imports after logger)
  - Fix pyright: `strategy` param `str` → `Literal["cma_es", "simple_ga"]`
  - Fix `test_bimodal_finds_two_modes` (distance_threshold trop élevé pour bounds normalisés)

## Milestone 5 — Hybride + module entry (`hybrid`, `__init__`)

Combinaison CMA-ES + gradient, et point d'entrée du package.

- [x] **`hybrid.py`** (252 LOC) — `HybridOptimizer`
  - Fix ruff E402/I001 (imports after logger, unsorted)
  - Fix pyright: `_flatten` 4→2 return values, `_apply_bounds` → `jnp.clip`
  - Fix tests: missing bounds, loss_history length assertion
- [x] **`__init__.py`** (82 LOC) — exports, lazy loading
  - Fix ruff I001 (import block unsorted)
  - `flax` non référencé dans le module — confirmé inutile
  - `__all__` / `__getattr__` cohérents

---

## Vérification globale (après chaque milestone)

```bash
uv run ruff check seapopym/optimization/
uv run pyright seapopym/optimization/
uv run pytest tests/optimization/ -v
```
