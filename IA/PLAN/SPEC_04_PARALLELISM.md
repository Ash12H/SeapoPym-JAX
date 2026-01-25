# SPEC 04 : Parallélisme

**Version** : 1.0
**Date** : 2026-01-25
**Statut** : Validé

---

## 1. Vue d'ensemble

Ce module définit la stratégie de passage à l'échelle pour les simulations massives. L'approche V1 privilégie le parallélisme "embarrassingly parallel" via `vmap`, avec le sharding spatial reporté à une version ultérieure.

### 1.1 Objectifs

- Permettre l'exploration de paramètres en parallèle
- Supporter les simulations d'ensemble
- Distribuer sur plusieurs GPUs (axes batchables uniquement)
- Paralléliser I/O et calcul

### 1.2 Dépendances

- **Amont** : Axe 3 (Engine) fournit le Runner
- **Aval** : Axe 5 (Auto-Diff) utilise `vmap` + `grad`

---

## 2. Taxonomie des Axes

### 2.1 Axes Libres (Batchables)

Ces axes traversent le graphe **sans interaction** entre éléments. Ils sont parallélisables trivialement.

| Axe | Description | Exemple d'usage |
|-----|-------------|-----------------|
| `E` (ensemble) | Membres d'ensemble | Simulations stochastiques |
| `P` (params) | Jeux de paramètres | Exploration / Calibration |

**Caractéristiques** :
- Aucune communication inter-éléments
- Parfait pour `jax.vmap`
- Sharding multi-GPU trivial

### 2.2 Axes Core (Couplés)

Ces axes nécessitent une **interaction** entre éléments (voisinage, réductions).

| Axe | Type d'interaction | Exemple |
|-----|-------------------|---------|
| `Y`, `X` (spatial) | Voisinage (stencil) | Transport, diffusion |
| `C` (cohort) | Réduction | Prédation `sum(prey)` |
| `Z` (depth) | Voisinage vertical | Mélange vertical |

**Caractéristiques** :
- Communication requise si shardé
- Identifiés via `@functional(core_dims=[...])`
- **Interdit de sharder en V1**

### 2.3 Matrice de Décision

| Axe | Batchable | Shardable (V1) | Shardable (V2+) |
|-----|-----------|----------------|-----------------|
| `E` | Oui | Oui | Oui |
| `T` | Non (séquentiel) | Non | Non |
| `C` | Dépend | Non | Possible |
| `Z` | Dépend | Non | Possible |
| `Y` | Non | Non | Oui (halo) |
| `X` | Non | Non | Oui (halo) |

---

## 3. Batch Parallelism (vmap)

### 3.1 Principe

`jax.vmap` transforme une fonction scalaire en fonction vectorisée sur un axe batch.

```python
# Fonction pour un seul jeu de paramètres
def simulate(params):
    model = compile_model(blueprint, config, params=params)
    return model.run()

# Vectorisée sur N jeux de paramètres
batch_simulate = jax.vmap(simulate, in_axes=(0,))

# Exécution parallèle
params_batch = jnp.array([[0.1, 0.2], [0.15, 0.25], ...])  # (N, 2)
results = batch_simulate(params_batch)  # (N, ...)
```

### 3.2 Scénarios d'Usage (V1)

#### 3.2.1 Exploration de Paramètres

```python
# Grille de paramètres
growth_rates = jnp.linspace(0.05, 0.2, 20)
mortality_rates = jnp.linspace(0.01, 0.1, 10)
params_grid = jnp.stack(jnp.meshgrid(growth_rates, mortality_rates), -1)
params_grid = params_grid.reshape(-1, 2)  # (200, 2)

# Simulation batch
results = jax.vmap(run_simulation)(params_grid)  # (200, T, Y, X, C)
```

#### 3.2.2 Ensemble de Simulations

```python
# N membres d'ensemble avec perturbations
n_members = 50
initial_states = perturb_state(base_state, n_members, noise_level=0.01)

# Simulation parallèle
ensemble_results = jax.vmap(simulate, in_axes=(0, None))(
    initial_states,  # (N, Y, X, C)
    forcings         # (T, Y, X) - répliqué
)
```

### 3.3 Intégration avec le Runner

```python
class BatchRunner:
    """Runner avec support vmap."""

    def __init__(self, model: CompiledModel, batch_axis: str = "E"):
        self.model = model
        self.batch_axis = batch_axis

    def run_batch(self, batch_params: dict[str, Array]) -> dict[str, Array]:
        """Exécute N simulations en parallèle."""

        def single_run(params):
            # Injection des paramètres
            model_copy = self.model.with_params(params)
            return model_copy.run()

        # vmap sur l'axe batch
        return jax.vmap(single_run)(batch_params)
```

---

## 4. Sharding Multi-GPU

### 4.1 Configuration Déclarative

Le sharding est configuré dans `run.yaml` :

```yaml
execution:
  device_layout:
    mesh: [4, 1]              # 4 GPUs en ligne
    sharding_rules:
      ensemble: 0             # Distribué sur mesh[0]
      params: 0               # Distribué sur mesh[0]
      spatial: null           # Répliqué (non shardé)
      cohort: null            # Répliqué
```

### 4.2 Implémentation JAX

```python
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils

def setup_sharding(config: dict) -> Mesh:
    """Configure le mesh de devices."""
    devices = mesh_utils.create_device_mesh(config["mesh"])
    mesh = Mesh(devices, axis_names=("batch",))
    return mesh

def shard_data(data: Array, mesh: Mesh, axis: int) -> Array:
    """Distribue les données sur le mesh."""
    spec = PartitionSpec("batch" if axis == 0 else None)
    sharding = NamedSharding(mesh, spec)
    return jax.device_put(data, sharding)
```

### 4.3 Contrainte V1

**Une simulation (tous les core_dims) doit tenir sur un seul GPU.**

```
GPU 0: [Ensemble 0-24]  → Simulation complète (Y, X, C, Z)
GPU 1: [Ensemble 25-49] → Simulation complète (Y, X, C, Z)
GPU 2: [Ensemble 50-74] → Simulation complète (Y, X, C, Z)
GPU 3: [Ensemble 75-99] → Simulation complète (Y, X, C, Z)
```

---

## 5. Identification des Core Dims

### 5.1 Via le Décorateur

Les fonctions déclarent explicitement leurs axes couplés :

```python
@functional(
    name="phys:transport",
    backend="jax",
    core_dims={"concentration": ["Y", "X"]}  # Nécessite les voisins
)
def transport(concentration, velocity_u, velocity_v, dx, dy, dt):
    """Advection avec schéma upwind."""
    # Utilise les voisins spatiaux
    dc_dx = (concentration[..., 1:] - concentration[..., :-1]) / dx
    dc_dy = (concentration[..., 1:, :] - concentration[..., :-1, :]) / dy
    return -velocity_u * dc_dx - velocity_v * dc_dy
```

### 5.2 Validation à la Compilation

Le Compilateur vérifie la cohérence :

```python
def validate_sharding(graph: ProcessGraph, sharding_rules: dict) -> None:
    """Vérifie qu'aucun core_dim n'est shardé."""
    for process in graph.processes:
        func_info = registry.get_info(process.name)
        for dim in func_info.core_dims.values():
            for d in dim:
                if sharding_rules.get(d) is not None:
                    raise ShardingError(
                        f"Dimension '{d}' est un core_dim de '{process.name}' "
                        f"et ne peut pas être shardée en V1"
                    )
```

---

## 6. I/O Asynchrone

### 6.1 Fonctionnement

JAX libère le GIL Python pendant l'exécution XLA, permettant le parallélisme I/O + calcul.

```
Timeline:
─────────────────────────────────────────────────────────────
Thread Principal │ dispatch_chunk_0 │ dispatch_chunk_1 │ ...
─────────────────────────────────────────────────────────────
XLA (GPU)        │     [===COMPUTE===]    [===COMPUTE===]
─────────────────────────────────────────────────────────────
I/O Thread       │                  [WRITE_0]    [WRITE_1]
─────────────────────────────────────────────────────────────
```

### 6.2 Implémentation

```python
import threading
from queue import Queue

class AsyncIOManager:
    """Gestionnaire I/O asynchrone."""

    def __init__(self, n_workers: int = 2):
        self.queue = Queue()
        self.workers = [
            threading.Thread(target=self._worker, daemon=True)
            for _ in range(n_workers)
        ]
        for w in self.workers:
            w.start()

    def schedule_write(self, data: Array, path: str, chunk_id: int):
        """Planifie une écriture."""
        # Transfert GPU → CPU (non-bloquant si async)
        data_cpu = jax.device_get(data)
        self.queue.put((data_cpu, path, chunk_id))

    def _worker(self):
        """Thread worker pour les écritures."""
        while True:
            data, path, chunk_id = self.queue.get()
            write_zarr_chunk(data, path, chunk_id)
            self.queue.task_done()

    def wait_all(self):
        """Attend la fin de toutes les écritures."""
        self.queue.join()
```

---

## 7. Patterns d'Usage

### 7.1 Exploration de Sensibilité

```python
# Définition de l'espace de paramètres
param_space = {
    "growth_rate": jnp.linspace(0.05, 0.3, 10),
    "mortality": jnp.linspace(0.01, 0.1, 10)
}

# Grille complète
grid = jnp.stack(jnp.meshgrid(*param_space.values()), -1)
grid = grid.reshape(-1, len(param_space))  # (100, 2)

# Simulation parallèle
runner = BatchRunner(model)
results = runner.run_batch(grid)

# Analyse
sensitivity = compute_sensitivity(results, grid)
```

### 7.2 Ensemble Forecast

```python
# Perturbation des conditions initiales
n_members = 100
rng = jax.random.PRNGKey(42)
perturbations = jax.random.normal(rng, (n_members, *state.shape)) * 0.01
ensemble_states = state[None, ...] + perturbations

# Simulation
runner = BatchRunner(model, batch_axis="E")
forecasts = runner.run_batch({"state": ensemble_states})

# Statistiques d'ensemble
mean_forecast = jnp.mean(forecasts, axis=0)
std_forecast = jnp.std(forecasts, axis=0)
```

---

## 8. Limites V1

### 8.1 Ce qui N'EST PAS supporté

| Feature | Raison | Horizon |
|---------|--------|---------|
| Sharding spatial | Nécessite halo exchange | V2 |
| Sharding cohort | Réductions inter-cohort | V2 |
| Multi-node | Communication réseau | V3 |

### 8.2 Contraintes Mémoire

La taille maximale d'une simulation est limitée par la VRAM d'un seul GPU :

```
Estimation mémoire pour une simulation :
- State: (C=100, Z=50, Y=180, X=360) × float32 = 1.2 GB
- Forcings (1 chunk): (T=365, Y=180, X=360) × float32 = 94 MB
- Outputs: ~1-2 GB

Total approximatif: 2-4 GB par simulation
```

Pour des domaines plus grands, options :
- Réduire la résolution
- Augmenter le nombre de chunks temporels
- Attendre V2 (sharding spatial)

---

## 9. Configuration

### 9.1 Paramètres

```yaml
execution:
  # Parallélisme batch
  batch:
    enabled: true
    axis: "E"
    size: 100

  # Sharding multi-GPU
  device_layout:
    mesh: [4, 1]
    sharding_rules:
      E: 0
      spatial: null

  # I/O asynchrone
  async_io:
    enabled: true
    workers: 2
    buffer_size: 3  # Chunks en mémoire
```

### 9.2 Auto-détection

```python
def auto_configure_parallelism() -> dict:
    """Configure automatiquement selon le hardware."""
    n_gpus = jax.device_count()

    return {
        "mesh": [n_gpus, 1],
        "sharding_rules": {"E": 0},
        "async_io": {"workers": min(2, n_gpus)}
    }
```

---

## 10. Interfaces

### 10.1 API Python

```python
from seapopym import compile_model, BatchRunner

# Compilation
model = compile_model(blueprint, config, backend="jax")

# Runner batch
runner = BatchRunner(
    model,
    batch_axis="E",
    sharding={"E": 0},
    async_io=True
)

# Exécution
results = runner.run_batch(
    params=param_grid,    # (N, n_params)
    output_path="/results/"
)
```

### 10.2 Méthodes Principales

| Méthode | Description |
|---------|-------------|
| `run_batch(params)` | Simulations parallèles |
| `setup_mesh(config)` | Configure le mesh GPU |
| `shard(data, axis)` | Distribue les données |

---

## 11. Liens avec les Autres Axes

| Axe | Interaction |
|-----|-------------|
| Axe 1 (Blueprint) | `core_dims` déclarées dans `@functional` |
| Axe 2 (Compiler) | Ordre canonique facilite le sharding batch |
| Axe 3 (Engine) | Runner utilise vmap/sharding |
| Axe 5 (Auto-Diff) | `vmap` + `grad` composables |

---

## 12. Questions Ouvertes (V2+)

- Sharding spatial avec halo exchange
- Communication inter-GPU pour réductions
- Support multi-node (MPI/NCCL)
- Auto-tuning de la taille de batch
