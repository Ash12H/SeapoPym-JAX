# Workflow Optimizer — Contexte et guide

## Objectif

Adapter le module `seapopym/optimization/` pour utiliser l'API fonctionnelle du moteur (`run()`, `build_step_fn`) au lieu du `Runner` deprecated.

## Contexte issu des workflows precedents

### Workflow Runner (TERMINE)

- `build_step_fn()` capture les statics en closure, fusionne avec `forcings_t` a chaque pas
- `export_variables` est fondamental : determine ce que `lax.scan` accumule. Doit TOUJOURS etre passe a `build_step_fn()`
- vmap est gere par le consommateur, pas par le moteur

### Workflow Engine (TERMINE)

Le Runner a ete decompose en fonctions pures :

```python
from seapopym.engine import run, simulate, build_step_fn, WriterRaw
```

- **`run(step_fn, model, state, params, chunk_size=None, writer=None)`** : moteur pur, ne fait jamais de vmap
- **`simulate(model, chunk_size, output_path, export_variables)`** : sucre syntaxique pour simulation
- **`WriterRaw`** : writer JAX-traceable, utilise par defaut dans `run()`. Compatible vmap/grad.
- **`Runner`** : shim deprecated qui delegue a `run()`/`simulate()`. Emet `DeprecationWarning`.

### Decouverte cle : double vmap

L'optimiseur CMA-ES fait deja son propre `jax.vmap(eval_one)` au-dessus du runner :

```python
# Dans cmaes.py:130
eval_population = jax.jit(jax.vmap(eval_one))
```

Si le runner a aussi `vmap_params=True`, c'est un double vmap. Avec la nouvelle API, `run()` ne fait jamais de vmap — c'est l'optimiseur qui decide. Cela clarifie la responsabilite.

## Etat actuel du module optimization

### Fichiers

| Fichier | Role | References Runner |
|---------|------|-------------------|
| `_common.py` | `build_loss_fn(runner, model, ...)` — appelle `runner(model, free_params)` | **point central** |
| `cmaes.py` | CMAESOptimizer — `self.runner` stocke un Runner | oui |
| `ga.py` | GAOptimizer — idem | oui |
| `ipop.py` | IPOPCMAESOptimizer — idem | oui |
| `gradient_optimizer.py` | GradientOptimizer — idem | oui |
| `loss.py` | MSE, RMSE, NRMSE | non |
| `objective.py` | Objective, PreparedObjective | non |
| `prior.py` | PriorSet, Uniform, Normal, LogNormal | non |

### Pattern actuel

Tous les optimiseurs suivent le meme pattern :

```python
class XxxOptimizer:
    def __init__(self, runner: Runner, objectives, bounds, ...):
        self.runner = runner

    def run(self, model, ...):
        loss_fn = build_loss_fn(self.runner, model, prepared, ...)
        # ... strategie d'optimisation
```

Et dans `_common.py` :

```python
def build_loss_fn(runner, model, prepared_objectives, priors, export_variables):
    def loss_fn(free_params):
        outputs = runner(model, free_params, export_variables=export_variables)
        # ... calcul de la loss
        return total
    return loss_fn
```

### Ce que fait `runner(model, free_params)` (via le shim)

```python
step_fn = build_step_fn(model, export_variables=export_variables)
merged = {**model.parameters, **single_free}
state = dict(model.state)
_, outputs = run(step_fn, model, state, merged, chunk_size=self.config.chunk_size)
return outputs
```

## Pistes de refactoring

### Option A : Remplacer Runner par run() dans _common.py

`build_loss_fn` prend directement les parametres du moteur :

```python
def build_loss_fn(model, prepared_objectives, priors, export_variables, chunk_size=None):
    step_fn = build_step_fn(model, export_variables=export_variables)

    def loss_fn(free_params):
        merged = {**model.parameters, **free_params}
        _, outputs = run(step_fn, model, dict(model.state), merged, chunk_size=chunk_size)
        # ... calcul de la loss
        return total
    return loss_fn
```

Les optimiseurs n'ont plus besoin de `runner` dans leur `__init__` — ils prennent `chunk_size` directement.

### Option B : Callable generique

`build_loss_fn` prend un callable `eval_fn(free_params) -> outputs` :

```python
def build_loss_fn(eval_fn, prepared_objectives, priors):
    def loss_fn(free_params):
        outputs = eval_fn(free_params)
        # ...
    return loss_fn
```

L'optimiseur construit `eval_fn` lui-meme. Plus flexible mais plus de code dans chaque optimiseur.

## Contraintes

- Les tests existants (465 passed) doivent continuer a passer
- Breaking change OK (utilisateur unique)
- Le shim Runner reste disponible pendant la transition
- `export_variables` et `chunk_size` doivent etre propages a `build_step_fn` / `run()`
- L'optimiseur gere vmap, pas le moteur

## Tests existants

```
tests/optimization/test_cmaes.py
tests/optimization/test_ga.py
tests/optimization/test_gradient_optimizer.py
tests/optimization/test_ipop_cmaes.py
tests/optimization/test_common.py
tests/compiler/test_optimization_runner.py
```
