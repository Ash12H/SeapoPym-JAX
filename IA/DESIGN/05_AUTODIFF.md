# Axe 5 : Auto-Diff & Optimisation

## Objectifs

Architecture pour la calibration automatique des paramètres (Gradient Descent).

## Décisions Validées (2026-01-24)

### 1. Déclaration des Paramètres

Dans le fichier de configuration (`run.yaml`), les paramètres optimisables sont marqués :

```yaml
parameters:
    growth_rate:
        value: 0.1
        trainable: true
        bounds: [0.0, 1.0]
```

### 2. Le Loss Wrapper & Checkpointing

Une fonction chapeau enveloppe la simulation.
**Le défi mémoire (Backprop) :**
Stocker l'histoire complète pour le gradient est impossible.
**Solution : Checkpointing (`jax.remat`)**
On utilise `jax.checkpoint` pour échanger de la mémoire contre du calcul.

- Au lieu de stocker tous les états $S_t$, on ne stocke que des "checkpoints" tous les N pas.
- Lors du backward, JAX recalcule les états manquants à partir du dernier checkpoint.

### 3. Contrainte Chunking vs Gradient

Le "Chunked Runner" Python (Axe 3) brise la chaîne de dérivation (Gradient Tape) entre les chunks s'il écrit sur disque.
**Stratégie pour la Calibration :**

- Pour l'optimisation, on ne peut PAS utiliser le mode "Chunked I/O" pur.
- On doit exécuter la séquence temporelle cible **en un seul appel JAX** (un seul gros `scan`).
- Pour que ça tienne en mémoire, le Checkpointing (Point 2) est **obligatoire** et doit être agressif.
- _Note_ : On optimise souvent sur des séquences plus courtes ou des résolutions dégradées.

### 4. Gradient Tooling

L'optimisation utilise `jax.value_and_grad(loss_fn)` couplé à `Optax`.
