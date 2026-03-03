# Architecture

## Principes

- **Config = QUOI** simuler (données concrètes). Identique entre simulation et optimisation.
- **Runner = COMMENT** exécuter (stratégie d'exécution). Seule pièce qui varie.
- **CompiledModel** = pivot immuable. Contient uniquement des données, pas de logique d'exécution.

## Simulation

```
Blueprint + Config → compile_model() → CompiledModel
                                            │
                                       Runner(model) → Output
```

Le Runner encapsule la stratégie d'exécution **et** le mode de sortie (disque ou mémoire).

- **Blueprint** : topologie du modèle (variables, process, tendencies). Pas de valeurs concrètes.
- **Config** : valeurs concrètes (forcings, paramètres, grille temporelle). Agnostique de l'optimisation.
- **CompiledModel** : données prêtes pour JAX. Pas de méthode d'exécution.
- **Runner** : logique d'exécution (chunking, lax.scan, I/O).
- **Output** : Writer (disque) ou Memory (xarray.Dataset). Capacité du Runner, pas un composant séparé.

## Calibration

Optimisation évolutionnaire (point estimate).

```
Blueprint + Config → compile_model() → CompiledModel (même pivot)

                    Runner + PriorSet + Objectives
                                  │
                                  ▼
                      Optimizer(strategy)
                       + metrics/weights
                                  │
                        OptimizationResult
```

### Séparation des responsabilités

| Composant | Rôle                                                                                     |
| --------- | ---------------------------------------------------------------------------------------- |
| Config    | Toutes les valeurs de paramètres (initiales pour les libres, définitives pour les fixes) |
| PriorSet  | Définit quels paramètres sont libres + leurs contraintes (bounds, distributions)         |
| Runner    | Exécute le modèle (reçoit params fixes du model + params libres proposés, merge, exécute)|
| Objective | Données d'observation + comment extraire les prédictions (target ou transform)           |
| Optimizer | Orchestre la recherche : propose des params, évalue via le Runner, décide du pas suivant |

### Paramètres : fixes vs libres

La Config contient **tous** les paramètres. La distinction fixe/libre est une préoccupation d'optimisation :

- Un paramètre est **libre** s'il a un prior dans la PriorSet.
- Pas de prior = **fixe** (valeur de la Config utilisée telle quelle).
- Le Runner reçoit les params fixes du CompiledModel et les params libres proposés par l'Optimizer. Il merge les deux et exécute le modèle.

### Frontière données utilisateur / JAX

Toute conversion vers JAX se fait aux **frontières**, jamais dans la boucle chaude :

| Frontière              | Entrée                             | Sortie (JAX)               |
| ---------------------- | ---------------------------------- | -------------------------- |
| `compile_model()`      | xarray (forcings, state initial)   | CompiledModel (arrays JAX) |
| Setup calibration      | xarray ou DataFrame (observations) | indices + valeurs JAX      |

La conversion au setup concerne les Objectives avec `target`. Pour les Objectives avec `transform`, l'utilisateur fournit directement des arrays JAX et gère lui-même la conversion.

L'utilisateur travaille avec ses formats habituels (xarray, pandas). Le système convertit. La boucle d'optimisation est 100% JAX.

### Objective — brique d'observation

Un `Objective` regroupe : observations + comment extraire les prédictions correspondantes. Pas de métrique — c'est l'Optimizer qui décide **comment** comparer.

#### Formats d'observations supportés

Les observations peuvent être fournies en **xarray** (grillées) ou **DataFrame** (éparses) :

```python
# xarray — observations grillées (avec coords T, Y, X dans les dimensions)
Objective(observations=obs_xarray, target="biomass")

# DataFrame — observations éparses (colonnes de coordonnées + valeur)
# | date       | lat   | lon   | biomass |
# | 2005-03-15 | -12.5 | 145.2 | 0.42    |
Objective(observations=obs_dataframe, target="biomass")
```

La conversion au setup est la même logique :

- **xarray** : coords des dimensions → `coords_to_indices`, valeurs → array JAX
- **DataFrame** : colonnes de coordonnées (T, Y, X) → `coords_to_indices`, colonne cible → array JAX

Le DataFrame est plus naturel pour les observations terrain (campagnes en mer, bouées, positions irrégulières). Le xarray convient aux données satellitaires re-grillées.

#### Cas simple : target (state ou dérivé direct)

Les observations sont matchées automatiquement aux coordonnées du modèle via `coords_to_indices`. Pas de code JAX à écrire :

```python
Objective(
    observations=obs_xarray,   # ou DataFrame
    target="biomass",          # variable à extraire des outputs
)
```

En interne, au setup de l'Optimizer :

1. Extrait les coordonnées des observations (T, Y, X) — depuis les dims xarray ou les colonnes DataFrame
2. Appelle `coords_to_indices(model.coords, T=..., Y=..., X=...)` automatiquement
3. Extrait les valeurs comme array JAX
4. Construit la transform : `outputs["biomass"][idx_t, idx_y, idx_x]`

#### Cas custom : transform

L'observation correspond à une quantité dérivée (agrégat, présence/absence, ratio...). L'utilisateur écrit sa transform en JAX :

```python
Objective(
    observations=obs_presence_jax,   # array JAX (déjà converti)
    transform=lambda o: (o["biomass"][idx_t, idx_y, idx_x] > 0).astype(float),
)
```

`target` et `transform` sont **mutuellement exclusifs**.

#### Vérification de shape

Pour le cas `transform`, le système vérifie au setup que la shape du résultat correspond aux observations :

```python
# Au setup (dry run avec eval_shape, pas d'exécution réelle)
pred_shape = jax.eval_shape(transform, dummy_outputs).shape
if pred_shape != observations.shape:
    raise ValueError(f"transform output {pred_shape} != observations {observations.shape}")
```

Cette vérification attrape les erreurs d'indexation avant le JIT. Pour le cas `target`, la cohérence est garantie automatiquement par `coords_to_indices`.

### Optimizer — optimisation par point estimate

L'Optimizer assemble les pièces et construit la `loss_fn` automatiquement. La **metric** (comment comparer) et le **weight** (importance relative) sont configurés ici, pas dans l'Objective :

```python
optimizer = Optimizer(
    runner=Runner.optimization(vmap=True),
    priors=PriorSet.load("priors.yaml"),
    objectives=[
        (Objective(observations=obs_biomass_xr, target="biomass"), "nrmse_std", 1.0),
        (Objective(observations=obs_presence, transform=presence_fn), "bce", 0.5),
    ],
    strategy="cma_es",
)
result: OptimizationResult = optimizer.run(model)
```

### Chaîne d'évaluation

```
free_params (Optimizer) ──┐
                          ▼
      runner(model, free_params) → merge fixed+free → outputs
                          │
                ┌─────────┼─────────┐
                ▼         ▼         ▼
          objective_1  objective_2  ...
          transform    transform
          pred_1       pred_2
                │         │
                ▼         ▼
          metric(pred, obs)
          scalar_1  scalar_2
                │         │
                ▼         ▼
          loss = Σ(w_i × s_i)
               + penalty
```

- **Runner** : reçoit params fixes (model) + libres (proposés), merge, exécute une seule fois
- **Transform** : extrait/agrège les prédictions (auto-générée pour `target`, manuelle pour `transform`)
- **Optimizer** : compare via metric (RMSE, NRMSE...) → loss à minimiser
- **PriorSet** : fournit les bounds (penalty si violation) ou log-prior (futur bayésien)

Cette composition est faite une seule fois au setup. La boucle interne est du JAX pur, JIT-compilé.

#### Transform et différentiabilité

La `transform` est le maillon faible pour l'auto-diff. `jax.grad` doit différentier toute la chaîne, mais certaines transforms ont un gradient nul :

| Transform           | Exemple                               | Differentiable         |
| ------------------- | ------------------------------------- | ---------------------- |
| Extraction spatiale | `outputs["biomass"][:, idx_y, idx_x]` | oui                    |
| Agrégation          | `.sum(axis=-1)`, `.mean()`            | oui                    |
| Log-transform       | `jnp.log(x + eps)`                    | oui                    |
| Présence/absence    | `(x > 0).astype(float)`               | **non** (gradient = 0) |
| Seuillage           | `jnp.where(x > seuil, 1, 0)`          | **non**                |

Conséquence : **le choix de la transform contraint les stratégies disponibles**.

| Transform          | Gradient (Adam, L-BFGS) | Évolutionnaire (CMA-ES) |
| ------------------ | ----------------------- | ----------------------- |
| Differentiable     | oui                     | oui                     |
| Non-differentiable | **non**                 | oui                     |

L'Optimizer devrait vérifier la compatibilité transform/stratégie au `.run()` ou au minimum avertir l'utilisateur. Les objectives avec `target` (sans transform custom) sont toujours differentiables.

### Composition Runner + Optimizer

Le Runner est **injecté** dans l'Optimizer, pas possédé. Plusieurs types de Runners sont nécessaires et la composition permet de les interchanger.

## Runner — Architecture composable

### Capacités

Le Runner combine des capacités indépendantes :

| Capacité             | Description                                                                        |
| -------------------- | ---------------------------------------------------------------------------------- |
| `closure` / `carry`  | Params en closure (simulation, pas de grad) ou carry (optimisation, grad)          |
| `scan` / `fori_loop` | Accumuler les outputs à chaque timestep (scan) ou juste le state final (fori_loop) |
| `vmap_param`         | Paralléliser sur une population de paramètres                                      |
| `vmap_forcing`       | Paralléliser sur des scénarios de forçage                                          |
| `pmap`               | Paralléliser sur plusieurs devices (multi-GPU)                                     |
| `checkpoint`         | Gradient checkpointing (re-calculer au lieu de stocker)                            |
| `chunk_T`            | Découper la dimension temporelle en morceaux                                       |
| `output_disk`        | Écriture progressive sur disque (Zarr)                                             |
| `output_memory`      | xarray.Dataset en RAM                                                              |

### Contraintes entre capacités

Certaines capacités sont incompatibles ou ordonnées :

- `closure` ⊕ `carry` — mutuellement exclusifs
- `scan` ⊕ `fori_loop` — mutuellement exclusifs
- `checkpoint` → requiert `carry` (pas de grad avec closure)
- `output_disk` → requiert `scan` (besoin des outputs intermédiaires)
- `chunk_T` + `carry` — le gradient doit traverser les chunks (complexe)

### Presets + Builder

Des presets couvrent les cas courants. Un builder est accessible pour les cas avancés.

```python
# Presets — couvrent 90% des cas
Runner.simulation(chunk_size=365, output="disk")
Runner.simulation(output="memory")
Runner.optimization(vmap=True)
Runner.optimization(vmap=True, checkpoint=True)
Runner.ensemble(n_scenarios=50)

# Builder — cas custom
Runner.builder().carry().scan().vmap_param().chunk(365).build()
```

Le `.build()` valide la cohérence des capacités et lève une erreur si combinaison invalide.

### Presets détaillés

| Preset                          | Capacités activées                     |
| ------------------------------- | -------------------------------------- |
| `simulation(output="disk")`     | closure + scan + chunk_T + output_disk |
| `simulation(output="memory")`   | closure + scan + output_memory         |
| `optimization()`                | carry + scan                           |
| `optimization(vmap=True)`       | carry + scan + vmap_param              |
| `optimization(checkpoint=True)` | carry + scan + checkpoint              |
| `ensemble()`                    | closure + scan + vmap_forcing          |

## Extensibilité

Le Runner est le point de variation pour tous les modes. Blueprint et CompiledModel ne changent jamais.

Un nouveau mode d'exécution (online/streaming, adjoint, etc.) = un nouveau preset ou une nouvelle combinaison via le builder. Pas de nouvelle classe à créer.

## Points d'attention

- `chunk_size` est un paramètre de `Runner.simulation()`, pas du modèle compilé.
