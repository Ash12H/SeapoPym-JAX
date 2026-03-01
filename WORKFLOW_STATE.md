# Workflow State

## Informations generales

- **Projet** : Redesign du pipeline d'optimisation
- **Etape courante** : 1. Initialisation (en attente de validation utilisateur)
- **Role actif** : Facilitateur
- **Derniere mise a jour** : 2026-03-01

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
- **Approche** : hybride (chemin user-driven + loss functions composables comme briques)
- **Pas de classe Observation generique** : l'utilisateur ecrit la comparaison lui-meme
- **Differentiabilite** : depend de l'optimizer choisi (gradient = JAX pur, CMA-ES = libre)
- **Loss vs Likelihood** : meme contrat `Callable[[Params], Array]`, likelihood necessite en plus sigma + priors
- **Frontiere xarray/JAX** : la conversion coordonnees -> indices se fait AVANT la construction de la loss (etape 4), plus aucun xarray dans le pipeline d'optimisation apres cette etape

### Ce qui disparait

| Composant | Raison |
|---|---|
| `SparseObservations` | Rigide, utilisee a un seul endroit, l'utilisateur peut faire mieux |
| `GradientRunner` | `run_with_params` -> `CompiledModel`, le reste est redondant (1 ligne de JAX) |
| `GradientRunner.make_loss_fn` | Remplace par la loss ecrite par l'utilisateur |
| `GradientRunner.compute_gradient` | `jax.grad(loss_fn)(params)` |
| `GradientRunner.compute_value_and_gradient` | `jax.value_and_grad(loss_fn)(params)` |
| `GradientRunner.optimize` | Redondant avec `Optimizer.run()` |

### Ce qui reste / s'ajoute

| Composant | Role |
|---|---|
| `CompiledModel.run_with_params()` | Execution du modele (unique point d'entree) |
| `coords_to_indices()` | Helper conversion coordonnees -> indices (nouveau) |
| `loss.py` (mse, rmse, nrmse) | Briques composables pour les loss |
| `likelihood.py` (make_log_posterior) | Construction log-posterior bayesien |
| `Optimizer` | Gradient-based (Adam, SGD via optax) |
| `EvolutionaryOptimizer` | CMA-ES, GA via evosax |
| `HybridOptimizer` | CMA-ES + gradient |
| `run_nuts` | MCMC via BlackJAX |

## Decisions d'architecture

_A definir a l'etape 3_

## Todo List

| Etat | ID | Nom | Description | Dependances | Resolution |
|------|----|-----|-------------|-------------|------------|

## Historique des transitions

| De | Vers | Raison | Date |
|----|------|--------|------|
