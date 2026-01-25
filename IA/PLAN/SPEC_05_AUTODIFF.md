# SPEC 05 : Auto-Diff & Optimisation

**Version** : 1.0
**Date** : 2026-01-25
**Statut** : Validé

---

## 1. Vue d'ensemble

Ce module définit l'architecture pour la calibration automatique des paramètres via gradient descent. Il s'appuie sur les capacités d'auto-différentiation de JAX.

### 1.1 Objectifs

- Permettre la déclaration de paramètres optimisables
- Fournir une interface de loss function
- Gérer la mémoire via checkpointing
- Intégrer avec les optimiseurs (Optax)

### 1.2 Dépendances

- **Amont** : Axe 1 (Config `trainable: true`), Axe 3 (GradientRunner)
- **Aval** : Aucune (composant terminal)

---

## 2. Déclaration des Paramètres

### 2.1 Syntaxe dans la Configuration

Les paramètres optimisables sont marqués dans `run.yaml` :

```yaml
parameters:
  # Paramètre fixe
  mortality_rate:
    value: 0.05
    trainable: false

  # Paramètre optimisable
  growth_rate:
    value: 0.1
    trainable: true
    bounds: [0.0, 1.0]    # Contraintes (optionnel)
    prior:                 # Prior bayésien (optionnel)
      type: "normal"
      mean: 0.1
      std: 0.02

  # Paramètre vectoriel optimisable
  temperature_sensitivity:
    value: [0.1, 0.2, 0.3]  # Par cohorte
    trainable: true
    bounds: [0.0, 0.5]
```

### 2.2 Extraction des Paramètres

Le Compilateur sépare les paramètres en deux pytrees :

```python
@dataclass
class ParameterSet:
    fixed: dict[str, Array]      # Paramètres constants
    trainable: dict[str, Array]  # Paramètres optimisables
    bounds: dict[str, tuple]     # Contraintes par paramètre
    priors: dict[str, Prior]     # Priors (optionnel)
```

---

## 3. Architecture d'Optimisation

### 3.1 Vue d'Ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│                     BOUCLE D'OPTIMISATION                       │
│                                                                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │  Paramètres │────→│   Model     │────→│   Loss      │       │
│  │  θ_t        │     │   Forward   │     │   L(θ)      │       │
│  └─────────────┘     └─────────────┘     └──────┬──────┘       │
│         ↑                                       │              │
│         │            ┌─────────────┐            │              │
│         │            │  Gradient   │←───────────┘              │
│         │            │  ∇L(θ)      │   jax.grad                │
│         │            └──────┬──────┘                           │
│         │                   │                                  │
│         │            ┌──────↓──────┐                           │
│         └────────────│  Optimizer  │                           │
│                      │  (Optax)    │                           │
│                      └─────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Composants

| Composant | Rôle | Implémentation |
|-----------|------|----------------|
| **Model Forward** | Simulation complète | `GradientRunner.run()` |
| **Loss Function** | Écart observations/simulation | Définie par l'utilisateur |
| **Gradient** | Dérivée de la loss | `jax.grad` / `jax.value_and_grad` |
| **Optimizer** | Mise à jour des paramètres | `optax` |

---

## 4. Loss Function

### 4.1 Interface

```python
from typing import Callable
from jax import Array

LossFn = Callable[[dict[str, Array]], Array]

def create_loss_fn(
    model: CompiledModel,
    observations: dict[str, Array],
    weights: dict[str, float] = None
) -> LossFn:
    """Crée une fonction de loss wrappée."""

    def loss_fn(params: dict[str, Array]) -> Array:
        # 1. Injection des paramètres
        model_with_params = model.with_trainable_params(params)

        # 2. Forward pass (simulation)
        final_state, outputs = model_with_params.run_for_gradient()

        # 3. Calcul de la loss
        loss = compute_mse(outputs, observations, weights)

        return loss

    return loss_fn
```

### 4.2 Losses Prédéfinies

```python
def mse_loss(predictions: Array, targets: Array, mask: Array = None) -> Array:
    """Mean Squared Error."""
    diff = predictions - targets
    if mask is not None:
        diff = diff * mask
        return jnp.sum(diff ** 2) / jnp.sum(mask)
    return jnp.mean(diff ** 2)

def mae_loss(predictions: Array, targets: Array, mask: Array = None) -> Array:
    """Mean Absolute Error."""
    diff = jnp.abs(predictions - targets)
    if mask is not None:
        diff = diff * mask
        return jnp.sum(diff) / jnp.sum(mask)
    return jnp.mean(diff)

def nll_loss(predictions: Array, targets: Array, sigma: Array) -> Array:
    """Negative Log-Likelihood (Gaussian)."""
    return 0.5 * jnp.mean(((predictions - targets) / sigma) ** 2 + jnp.log(sigma ** 2))
```

### 4.3 Loss Composite

```python
def composite_loss(
    outputs: dict[str, Array],
    observations: dict[str, Array],
    weights: dict[str, float]
) -> Array:
    """Combine plusieurs termes de loss."""
    total_loss = 0.0
    for var, weight in weights.items():
        pred = outputs[var]
        obs = observations[var]
        total_loss += weight * mse_loss(pred, obs)
    return total_loss
```

---

## 5. Checkpointing (jax.remat)

### 5.1 Problème Mémoire

Le backward pass stocke tous les états intermédiaires pour calculer le gradient :

```
Forward:  S_0 → S_1 → S_2 → ... → S_T → Loss
                ↓      ↓           ↓
Memory:       [save] [save]     [save]  ← Explosion mémoire !
```

Pour T = 365 jours × variables × spatial → plusieurs dizaines de GB.

### 5.2 Solution : Gradient Checkpointing

On ne stocke que des checkpoints périodiques et on recalcule le reste :

```
Forward:  S_0 → S_1 → S_2 → ... → S_50 → S_51 → ... → S_100
                                  [CP]                  [CP]

Backward: On recalcule S_51..S_99 à partir de S_50
```

### 5.3 Implémentation

```python
import jax

def checkpointed_scan(step_fn, init, xs, checkpoint_every: int = 50):
    """Scan avec checkpointing périodique."""

    @jax.checkpoint
    def step_block(carry, xs_block):
        """Un bloc de N steps, checkpointé."""
        def inner_scan(c, x):
            return step_fn(c, x)
        return jax.lax.scan(inner_scan, carry, xs_block)

    # Reshape xs en blocs
    n_steps = xs.shape[0]
    n_blocks = n_steps // checkpoint_every
    xs_blocks = xs.reshape(n_blocks, checkpoint_every, *xs.shape[1:])

    # Scan sur les blocs (checkpointé)
    final_carry, outputs_blocks = jax.lax.scan(step_block, init, xs_blocks)

    # Reshape outputs
    outputs = outputs_blocks.reshape(n_steps, *outputs_blocks.shape[2:])

    return final_carry, outputs
```

### 5.4 Configuration

```yaml
optimization:
  checkpointing:
    enabled: true
    interval: 50        # Checkpoint tous les 50 pas
    # Ou auto-ajustement basé sur la mémoire disponible
    auto: false
    target_memory_gb: 8.0
```

### 5.5 Trade-off Mémoire/Calcul

| Checkpoint Interval | Mémoire | Temps Forward | Temps Backward |
|---------------------|---------|---------------|----------------|
| 1 (tout stocker) | MAX | 1x | 1x |
| 10 | ~10% | 1x | ~1.1x |
| 50 | ~2% | 1x | ~1.5x |
| 100 | ~1% | 1x | ~2x |

---

## 6. GradientRunner

### 6.1 Contrainte vs StreamingRunner

| Aspect | StreamingRunner | GradientRunner |
|--------|-----------------|----------------|
| I/O disque | Entre chunks | Aucun |
| Chaîne gradient | Brisée | Préservée |
| Mémoire | Bornée par chunk | Tout en VRAM |
| Checkpointing | Non | Obligatoire |

### 6.2 Implémentation

```python
class GradientRunner:
    """Runner compatible auto-diff."""

    def __init__(
        self,
        model: CompiledModel,
        checkpoint_interval: int = 50
    ):
        self.model = model
        self.checkpoint_interval = checkpoint_interval

    def run_for_gradient(
        self,
        trainable_params: dict[str, Array]
    ) -> tuple[Array, dict[str, Array]]:
        """Forward pass compatible avec jax.grad."""
        # Merge paramètres
        params = {**self.model.parameters.fixed, **trainable_params}

        # Step function
        step_fn = self._make_step_fn(params)

        # Scan avec checkpointing
        final_state, outputs = checkpointed_scan(
            step_fn=step_fn,
            init=self.model.state,
            xs=self.model.forcings,
            checkpoint_every=self.checkpoint_interval
        )

        return final_state, outputs
```

---

## 7. Boucle d'Optimisation

### 7.1 Interface Haut Niveau

```python
def optimize(
    model: CompiledModel,
    observations: dict[str, Array],
    optimizer: optax.GradientTransformation,
    n_epochs: int = 100,
    loss_weights: dict[str, float] = None
) -> OptimizationResult:
    """Boucle d'optimisation complète."""

    # Création de la loss
    loss_fn = create_loss_fn(model, observations, loss_weights)

    # Initialisation
    params = model.parameters.trainable
    opt_state = optimizer.init(params)

    # Boucle
    history = []
    for epoch in range(n_epochs):
        # Gradient
        loss, grads = jax.value_and_grad(loss_fn)(params)

        # Mise à jour
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # Projection sur les contraintes
        params = project_bounds(params, model.parameters.bounds)

        # Logging
        history.append({"epoch": epoch, "loss": float(loss)})

    return OptimizationResult(
        optimal_params=params,
        final_loss=float(loss),
        history=history
    )
```

### 7.2 Avec Optax

```python
import optax

# Optimiseurs disponibles
optimizer = optax.adam(learning_rate=1e-3)
optimizer = optax.sgd(learning_rate=1e-2, momentum=0.9)
optimizer = optax.lbfgs()  # Quasi-Newton

# Schedulers
schedule = optax.exponential_decay(
    init_value=1e-3,
    transition_steps=100,
    decay_rate=0.9
)
optimizer = optax.adam(learning_rate=schedule)
```

### 7.3 Gestion des Contraintes

```python
def project_bounds(
    params: dict[str, Array],
    bounds: dict[str, tuple[float, float]]
) -> dict[str, Array]:
    """Projette les paramètres dans les bornes."""
    projected = {}
    for name, value in params.items():
        if name in bounds:
            low, high = bounds[name]
            projected[name] = jnp.clip(value, low, high)
        else:
            projected[name] = value
    return projected
```

---

## 8. Stratégies Avancées

### 8.1 Multi-Scale Optimization

Pour les séries longues, optimiser d'abord sur une version dégradée :

```python
def multiscale_optimize(model, observations, n_scales=3):
    """Optimisation multi-échelle."""
    params = model.parameters.trainable

    for scale in range(n_scales):
        # Sous-échantillonnage temporel
        subsample = 2 ** (n_scales - scale - 1)
        obs_scaled = {k: v[::subsample] for k, v in observations.items()}
        model_scaled = model.subsample_time(subsample)

        # Optimisation à cette échelle
        result = optimize(model_scaled, obs_scaled, ...)
        params = result.optimal_params

    return params
```

### 8.2 Ensemble Gradient

```python
def ensemble_loss(params, models, observations):
    """Loss sur un ensemble de simulations."""
    losses = jax.vmap(
        lambda m: compute_loss(m.run(params), observations)
    )(models)
    return jnp.mean(losses)
```

---

## 9. Configuration

### 9.1 Paramètres d'Optimisation

```yaml
optimization:
  # Algorithme
  optimizer: "adam"
  learning_rate: 0.001
  lr_schedule:
    type: "exponential_decay"
    decay_rate: 0.95
    decay_steps: 50

  # Convergence
  n_epochs: 500
  early_stopping:
    patience: 20
    min_delta: 1e-6

  # Mémoire
  checkpointing:
    enabled: true
    interval: 50

  # Loss
  loss:
    type: "mse"
    weights:
      biomass: 1.0
      production: 0.5

  # Contraintes
  constraints:
    method: "projection"  # ou "penalty"
```

---

## 10. Interfaces

### 10.1 API Haut Niveau

```python
from seapopym import compile_model
import optax

# Compilation
model = compile_model(blueprint, config, backend="jax")

# Observations
observations = load_observations("/data/obs.nc")

# Optimisation
result = model.optimize(
    observations=observations,
    optimizer=optax.adam(1e-3),
    n_epochs=200,
    checkpoint_interval=50
)

# Résultats
print(f"Loss finale: {result.final_loss}")
print(f"Paramètres optimaux: {result.optimal_params}")

# Visualisation
plot_loss_curve(result.history)
```

### 10.2 API Bas Niveau

```python
# Construction manuelle
loss_fn = create_loss_fn(model, observations)
grad_fn = jax.value_and_grad(loss_fn)

# Une itération
loss, grads = grad_fn(params)
updates, opt_state = optimizer.update(grads, opt_state)
params = optax.apply_updates(params, updates)
```

### 10.3 Résultats

```python
@dataclass
class OptimizationResult:
    optimal_params: dict[str, Array]
    final_loss: float
    history: list[dict]
    n_iterations: int
    converged: bool
    runtime_seconds: float
```

---

## 11. Limites V1

### 11.1 Ce qui N'EST PAS supporté

| Feature | Raison | Horizon |
|---------|--------|---------|
| Inférence bayésienne | Complexité | V2 |
| Gradients d'ordre 2 | Mémoire | V2 |
| Adjoint continu | Implémentation | V2+ |

### 11.2 Reproductibilité

La reproductibilité cross-device n'est pas garantie en V1 (ordre des opérations GPU).

**À documenter** : Les résultats peuvent varier légèrement entre exécutions.

---

## 12. Liens avec les Autres Axes

| Axe | Interaction |
|-----|-------------|
| Axe 1 (Blueprint) | Lit `trainable: true` dans Config |
| Axe 2 (Compiler) | Sépare params fixes/trainable |
| Axe 3 (Engine) | Utilise `GradientRunner` |
| Axe 4 (Parallelism) | `vmap(grad)` pour batch gradient |

---

## 13. Questions Ouvertes (V2+)

- Inférence bayésienne (MCMC, VI)
- Gradients d'ordre 2 (Newton, L-BFGS Hessian-free)
- Support du mode adjoint continu
- Parallelisation du gradient sur multiple GPUs
- Gestion automatique de la seed pour reproductibilité
