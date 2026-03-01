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

- [ ] **`likelihood.py`** (182 LOC) — `GaussianLikelihood`, `make_log_posterior`
  - Vérifier correction Jacobienne dans `reparameterize_log_posterior`
  - Ruff / pyright / coverage
- [ ] **`nuts.py`** (152 LOC) — `NUTSResult`, `run_nuts`
  - `# type: ignore` sur BlackJAX — acceptable ?
  - Vérifier warmup reuse pattern
  - Ruff / pyright / coverage

## Milestone 3 — Optimisation gradient (`gradient`)

Runner qui connecte le modèle compilé aux loss functions.

- [ ] **`gradient.py`** (294 LOC) — `GradientRunner`, `SparseObservations`
  - Gestion cohorts hardcodée `[..., 0]` — à rendre flexible ?
  - Pas de validation des indices temporels des observations
  - Ruff / pyright / coverage

## Milestone 4 — Optimisation évolutionnaire (`evolutionary`, `ipop`)

Stratégies population-based via evosax.

- [ ] **`evolutionary.py`** (291 LOC) — `EvolutionaryOptimizer`
  - Early stopping near-zero : `min_fitness < best_loss * (1 - tol_fun)` trop strict ?
  - Vérifier `_build_strategy()` / `_init_state()` API evosax
  - Ruff / pyright / coverage
- [ ] **`ipop.py`** (185 LOC) — `run_ipop`, `run_ipop_cmaes`
  - Mode detection via distance euclidienne normalisée
  - Ruff / pyright / coverage

## Milestone 5 — Hybride + module entry (`hybrid`, `__init__`)

Combinaison CMA-ES + gradient, et point d'entrée du package.

- [ ] **`hybrid.py`** (252 LOC) — `HybridOptimizer`
  - Référence à `_apply_bounds()` inexistant sur `EvolutionaryOptimizer` ?
  - Combinaison loss history depuis le 1er candidat uniquement
  - Ruff / pyright / coverage
- [ ] **`__init__.py`** (82 LOC) — exports, lazy loading
  - `flax` dans les dépendances optionnelles mais inutilisé — supprimer ?
  - Vérifier cohérence `__all__` / `__getattr__`
  - Ruff / pyright / coverage

---

## Vérification globale (après chaque milestone)

```bash
uv run ruff check seapopym/optimization/
uv run pyright seapopym/optimization/
uv run pytest tests/optimization/ -v
```
