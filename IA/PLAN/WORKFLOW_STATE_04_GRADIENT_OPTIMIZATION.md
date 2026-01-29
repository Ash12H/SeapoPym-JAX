# Workflow State

## Informations générales

- **Projet** : Optimisation de paramètres avec calcul de gradient
- **Étape courante** : 9. Finalisation
- **Rôle actif** : Facilitateur
- **Dernière mise à jour** : 2026-01-28

## Résumé du besoin

**Objectif** : Permettre l'optimisation de paramètres du modèle LMTL en minimisant l'écart (RMSE/NRMSE) entre les sorties simulées et des observations.

**Fonctionnalités souhaitées** :
- Choix libre des paramètres à optimiser parmi ceux fournis à la simulation
- Support d'observations ponctuelles ou en série temporelle, potentiellement incomplètes en espace/temps
- Comparaison flexible (biomasse vs biomasse, production vs production)
- Choix de l'algorithme d'optimisation
- Contraintes sur les paramètres (au moins bornes simples)

**Périmètre initial** :
- Prototype sur un modèle simple (0D ou grille réduite)
- Bornes simples sur les paramètres
- Un ou deux algorithmes (ex: Adam, L-BFGS)

**Hors périmètre initial** (pour plus tard) :
- Contraintes complexes (inégalités entre paramètres)
- Optimisation sur grandes grilles avec longues séries temporelles
- Checkpointing avancé

## Rapport d'analyse

### Structure pertinente du projet

| Composant | Fichier | Rôle |
|-----------|---------|------|
| Définition paramètres | `seapopym/blueprint/schema.py` | `ParameterValue`, `Config` |
| Compilation paramètres | `seapopym/compiler/compiler.py` | `_prepare_parameters()` |
| Stockage compilé | `seapopym/compiler/model.py` | `CompiledModel.parameters`, `.trainable_params` |
| Step function | `seapopym/engine/step.py` | `build_step_fn()`, `_resolve_inputs()` |
| Runners | `seapopym/engine/runners.py` | `GradientRunner` (existe déjà) |
| Backend JAX | `seapopym/engine/backends.py` | `JAXBackend.scan()` |

### Flux actuel des paramètres

```
Config.parameters → Compiler._prepare_parameters() → CompiledModel.parameters
                                                            ↓
                                                    build_step_fn() [CLOSURE]
                                                            ↓
                                                    step_fn(state, forcings)
```

**Problème principal** : Les paramètres sont capturés par **closure** dans `build_step_fn()`. Le step_fn ne les reçoit pas en argument, ce qui empêche `jax.grad()` de calculer les gradients.

### Patterns existants (réutilisables)

| Pattern | Statut | Fichier |
|---------|--------|---------|
| `trainable_params` list | ✅ Existe | `compiler.py` |
| Pytree dict structure | ✅ Compatible JAX | `model.py` |
| `jax.lax.scan` | ✅ Différentiable | `backends.py` |
| State en argument | ✅ Fonctionnel | `step.py` |
| `GradientRunner` | ⚠️ Incomplet | `runners.py` |

### Points bloquants pour jax.grad

| Problème | Impact | Solution |
|----------|--------|----------|
| Closure parameters | Bloquant | Passer params en argument |
| Mutation in-place | Bloquant | Utiliser pattern fonctionnel |
| Scan signature | Mineur | Wrapper avec params |

### Compatibilité gradient

**Déjà compatible :**
- [x] Backend JAX utilise `lax.scan` (différentiable)
- [x] Structure pytree pour paramètres
- [x] Tracking des paramètres trainable
- [x] Fonctions ComputeNode stateless
- [x] State passé en argument
- [x] Vmap disponible

**À modifier :**
- [ ] Step function : closure → arguments
- [ ] Pattern mutation → fonctionnel
- [ ] Signature scan avec params
- [ ] Loss function threading

## Décisions d'architecture

### Choix techniques

| Domaine | Choix | Justification |
|---------|-------|---------------|
| Optimisation | Optax | Standard JAX, large choix d'optimiseurs (Adam, SGD), interface simple |
| L-BFGS (optionnel) | JAXopt | Pour convergence rapide sur petits problèmes |
| Pattern params | Argument explicite | Nécessaire pour `jax.grad()`, remplace closure |
| Observations éparses | Indexation | Pas de matrice pleine, O(n_obs) au lieu de O(grille) |
| Backward compatible | Mode dual | Simulation normale inchangée, gradient opt-in |

### Structure proposée

```
seapopym/
├── engine/
│   ├── step.py           # MODIFIER: support params en argument (optionnel)
│   └── backends.py       # OK
│
└── optimization/         # NOUVEAU MODULE
    ├── __init__.py
    ├── loss.py           # Fonctions de coût (RMSE, NRMSE)
    ├── optimizer.py      # Wrapper Optax/JAXopt
    └── gradient.py       # Runner différentiable
```

### Interfaces et contrats

**Loss functions** (`loss.py`):
- `rmse(pred, obs, mask=None) -> scalar`
- `nrmse(pred, obs, mask=None, mode="std") -> scalar`

**Optimizer** (`optimizer.py`):
- `Optimizer(algorithm, learning_rate, bounds)`
- `optimizer.step(params, grads) -> new_params`
- `optimizer.run(loss_fn, params, n_steps) -> OptimizeResult`

**GradientRunner** (`gradient.py`):
- `make_loss_fn(observations, loss_type, variables) -> Callable`
- `compute_gradient(params, loss_fn) -> grads`
- `optimize(observations, params_to_optimize, optimizer, n_steps) -> OptimizeResult`

### Gestion des observations éparses

Format d'entrée : indices + valeurs (pas de matrice pleine)
```python
observations = {
    "times": array([5, 12, 30]),    # indices temporels
    "y": array([10, 50, 30]),       # indices spatiaux Y
    "x": array([20, 60, 40]),       # indices spatiaux X
    "values": array([1.2, 0.8, 1.5]) # valeurs observées
}
```

### Risques identifiés

| Risque | Impact | Mitigation |
|--------|--------|------------|
| Mémoire (longues séries) | Haut | Prototype sur 0D, checkpointing hors périmètre |
| NaN dans gradients | Moyen | Clip values, vérification bornes |
| Convergence lente | Bas | Plusieurs algorithmes disponibles |

### Impact performances

| Mode | Impact |
|------|--------|
| Simulation normale | Aucun (backward compatible) |
| Calcul gradient | ~2-3× temps, ~N× mémoire (inhérent autodiff) |

## Todo List

| État | ID | Nom | Description | Dépendances | Résolution |
|------|-----|-----|-------------|-------------|------------|
| ☑ | T1 | Créer module optimization | Créer `seapopym/optimization/__init__.py` avec exports publics | - | Module créé avec exports |
| ☑ | T2 | Créer loss.py | Créer `seapopym/optimization/loss.py` avec `rmse()` et `nrmse()` | T1 | Implémenté rmse, nrmse, mse avec masques |
| ☑ | T3 | Créer optimizer.py | Créer `seapopym/optimization/optimizer.py` avec classe `Optimizer` wrappant Optax | T1 | Implémenté avec Adam, SGD, RMSprop, Adagrad + bornes |
| ☑ | T4 | Modifier step.py | Ajouter mode `params_as_argument` dans `build_step_fn()` pour support gradient | - | Mode dual ajouté, backward compatible |
| ☑ | T5 | Créer gradient.py | Créer `seapopym/optimization/gradient.py` avec `GradientRunner` utilisant `jax.value_and_grad` | T2, T3, T4 | GradientRunner + SparseObservations implémentés |
| ☑ | T6 | Créer exemple optimization | Créer `examples/optimization_0d.py` démontrant l'optimisation sur modèle simple | T5 | Exemple 0D fonctionnel, gradients calculés |
| ☑ | T7 | Tests unitaires loss | Créer `tests/optimization/test_loss.py` | T2 | 18 tests (RMSE, MSE, NRMSE + gradients) |
| ☑ | T8 | Tests unitaires optimizer | Créer `tests/optimization/test_optimizer.py` | T3 | 12 tests (init, step, run) |
| ☑ | T9 | Tests intégration gradient | Créer `tests/optimization/test_gradient.py` | T5 | 6 tests (SparseObs, step modes, gradients) |

## Historique des transitions

| De | Vers | Raison | Date |
|----|------|--------|------|
| 1. Initialisation | 2. Analyse | Besoin validé par l'utilisateur | 2026-01-28 |
| 2. Analyse | 3. Architecture | Analyse complétée | 2026-01-28 |
| 3. Architecture | 4. Planification | Architecture validée par l'utilisateur | 2026-01-28 |
| 4. Planification | 5. Execution | Todo list complétée | 2026-01-28 |
| 5. Execution | 6. Revue | Tâches T1-T6 complétées | 2026-01-28 |
| 6. Revue | 8. Test | 0 issues | 2026-01-28 |
| 8. Test | 9. Finalisation | 36 tests passent | 2026-01-28 |
| 9. Finalisation | Complété | Commit deb2dec | 2026-01-28 |

## Rapport de revue

### Vérifications automatiques

| Outil | Résultat | Erreurs | Warnings |
|-------|----------|---------|----------|
| Ruff (lint) | ✅ | 0 | 0 |
| Ruff (format) | ✅ | 0 | - |
| Tests existants | ✅ | 0 (1 préexistant) | 0 |

### Issues identifiées

| ID | Sévérité | Description | Fichier | Action |
|----|----------|-------------|---------|--------|
| - | - | Aucune issue critique | - | - |

### Analyse des tâches échouées

Aucune tâche échouée.

### Décision

0 issues → Passer directement à Test

## Tests

### Tests créés

| Fichier | Fonctionnalité testée | Nb tests | Types |
|---------|----------------------|----------|-------|
| tests/optimization/test_loss.py | rmse, mse, nrmse, masques, gradients | 18 | Unitaire |
| tests/optimization/test_optimizer.py | Optimizer init, step, run, bounds | 12 | Unitaire |
| tests/optimization/test_gradient.py | SparseObservations, step modes, gradients | 6 | Intégration |

### Résultats d'exécution

- **Date** : 2026-01-28
- **Commande** : `uv run pytest tests/optimization/`

| Statut | Nombre |
|--------|--------|
| ✅ Passés | 36 |
| ❌ Échoués | 0 |
| **Total** | 36 |
