# SPEC 03 : Moteur d'Exécution (Engine)

**Version** : 1.0
**Date** : 2026-01-25
**Statut** : Validé

---

## 1. Vue d'ensemble

Le Moteur d'Exécution orchestre la boucle temporelle, gère les I/O et supporte deux backends (JAX/NumPy). Il est conçu pour concilier performance et faisabilité mémoire sur des simulations longues.

### 1.1 Objectifs

- Exécuter des simulations sur de longues durées (années/siècles)
- Gérer la mémoire via chunking temporel
- Supporter le dual backend JAX/NumPy
- Permettre deux modes : streaming (production) et gradient (optimisation)

### 1.2 Dépendances

- **Amont** : Axe 2 (Compilateur) fournit le `CompiledModel`
- **Aval** : Axe 4 (Parallelism), Axe 5 (Auto-Diff) utilisent les Runners

---

## 2. Architecture Modulaire

### 2.1 Composants

```
┌─────────────────────────────────────────────────────────────────┐
│                         RUNNER                                   │
│  ┌─────────────────┐  ┌─────────────────┐                       │
│  │ StreamingRunner │  │ GradientRunner  │                       │
│  │ (I/O par chunk) │  │ (un seul scan)  │                       │
│  └────────┬────────┘  └────────┬────────┘                       │
│           │                    │                                 │
│           └──────────┬─────────┘                                │
│                      ↓                                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    STEP KERNEL                               ││
│  │  step_fn(state, forcings, params) → (new_state, outputs)    ││
│  └─────────────────────────────────────────────────────────────┘│
│                      ↓                                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    BACKEND                                   ││
│  │  ┌─────────────┐  ┌─────────────┐                           ││
│  │  │ JAXBackend  │  │ NumpyBackend│                           ││
│  │  │ (lax.scan)  │  │ (for loop)  │                           ││
│  │  └─────────────┘  └─────────────┘                           ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Responsabilités

| Composant | Rôle |
|-----------|------|
| **Runner** | Orchestration haut niveau (chunking, I/O) |
| **Step Kernel** | Logique d'un pas de temps (agnostique backend) |
| **Backend** | Implémentation de la boucle (scan ou for) |

---

## 3. Le Step Kernel

### 3.1 Signature

```python
def step_fn(
    state: dict[str, Array],
    forcings_t: dict[str, Array],
    parameters: dict[str, Array],
    mask: Array,
    dt: float
) -> tuple[dict[str, Array], dict[str, Array]]:
    """
    Exécute un pas de temps.

    Args:
        state: Variables d'état au temps t
        forcings_t: Forçages instantanés (slice temporel)
        parameters: Constantes du modèle
        mask: Masque binaire
        dt: Pas de temps (secondes)

    Returns:
        new_state: Variables d'état au temps t+1
        outputs: Diagnostiques à sauvegarder
    """
```

### 3.2 Logique Interne

```python
def step_fn(state, forcings_t, parameters, mask, dt):
    # 1. Calcul des tendances via le graphe de processus
    tendencies = {}
    for process in graph.processes:
        func = registry.get(process.name, backend)
        tendency = func(**process.inputs)
        tendencies[process.output] = tendency

    # 2. Intégration temporelle (Euler explicite)
    new_state = {}
    for var, value in state.items():
        total_tendency = sum(tendencies.get(f"{var}_tendency", []))
        new_state[var] = value + total_tendency * dt

    # 3. Application du masque (optionnel)
    new_state = {k: v * mask for k, v in new_state.items()}

    # 4. Extraction des diagnostiques
    outputs = extract_diagnostics(new_state, tendencies)

    return new_state, outputs
```

### 3.3 Intégrateur Temporel

V1 supporte uniquement **Euler Explicite** :

$$S_{t+1} = S_t + \sum_{i} \frac{dS_i}{dt} \times \Delta t$$

**V2+** : Support de schémas plus avancés (RK4, Adams-Bashforth).

---

## 4. Runners

### 4.1 StreamingRunner (Mode Production)

Pour les simulations longues avec écriture sur disque.

```python
class StreamingRunner:
    """Runner avec chunking temporel et I/O asynchrone."""

    def __init__(self, model: CompiledModel, chunk_size: int = 365):
        self.model = model
        self.chunk_size = chunk_size
        self.backend = get_backend(model.backend)

    def run(self, output_path: str) -> None:
        """Exécute la simulation complète."""
        state = self.model.state
        n_chunks = self.model.shapes["T"] // self.chunk_size

        # Boucle Python sur les chunks
        for i in range(n_chunks):
            # 1. Charge les forçages du chunk
            forcings_chunk = self._load_chunk(i)

            # 2. Exécute le scan sur le chunk
            state, outputs = self.backend.scan(
                step_fn=self._make_step_fn(),
                init=state,
                xs=forcings_chunk
            )

            # 3. Écriture asynchrone
            self._write_async(outputs, i, output_path)

        # Attente de fin d'écriture
        self._flush_writes()
```

**Caractéristiques** :
- Chunks de taille configurable (ex: 365 jours = 1 an)
- I/O asynchrone entre chunks (thread pool)
- Mémoire bornée par taille du chunk

### 4.2 GradientRunner (Mode Optimisation)

Pour la calibration avec gradients.

```python
class GradientRunner:
    """Runner sans chunking, compatible autodiff."""

    def __init__(self, model: CompiledModel):
        self.model = model
        self.backend = get_backend("jax")  # JAX obligatoire

    def run(self) -> tuple[Array, dict[str, Array]]:
        """Exécute la simulation en un seul appel."""
        state = self.model.state
        forcings = self.model.forcings  # Toute la série

        # Un seul scan (toute la dimension T)
        final_state, outputs = self.backend.scan(
            step_fn=self._make_step_fn(),
            init=state,
            xs=forcings
        )

        return final_state, outputs

    def loss_fn(self, params: dict[str, Array]) -> Array:
        """Wrapper pour jax.value_and_grad."""
        # Injection des paramètres
        self.model.parameters.update(params)
        final_state, outputs = self.run()
        return compute_loss(outputs, self.model.observations)
```

**Caractéristiques** :
- Un seul appel `jax.lax.scan` (chaîne de gradient préservée)
- Pas d'I/O intermédiaire
- Checkpointing requis pour la mémoire (Axe 5)

### 4.3 Choix du Runner

| Critère | StreamingRunner | GradientRunner |
|---------|-----------------|----------------|
| Simulations longues | Oui | Limité (mémoire) |
| Optimisation | Non | Oui |
| Écriture disque | Oui | Non |
| Backend | JAX ou NumPy | JAX uniquement |

**API** :

```python
# Production
model.run(output_path="/results/")  # → StreamingRunner

# Optimisation
model.optimize(loss_fn, optimizer)  # → GradientRunner
```

---

## 5. Backends

### 5.1 JAXBackend

```python
class JAXBackend:
    """Backend utilisant jax.lax.scan."""

    def scan(self, step_fn, init, xs):
        """Exécute la boucle temporelle via lax.scan."""
        import jax.lax as lax

        def wrapped_step(carry, x):
            new_carry, outputs = step_fn(carry, x)
            return new_carry, outputs

        final_carry, all_outputs = lax.scan(wrapped_step, init, xs)
        return final_carry, all_outputs
```

### 5.2 NumpyBackend

```python
class NumpyBackend:
    """Backend utilisant une boucle Python."""

    def scan(self, step_fn, init, xs):
        """Exécute la boucle temporelle via for loop."""
        carry = init
        outputs_list = []

        for t in range(xs["time"].shape[0]):
            # Slice temporel
            x_t = {k: v[t] for k, v in xs.items()}
            carry, outputs = step_fn(carry, x_t)
            outputs_list.append(outputs)

        # Stack des outputs
        all_outputs = {
            k: np.stack([o[k] for o in outputs_list])
            for k in outputs_list[0]
        }
        return carry, all_outputs
```

### 5.3 Sélection du Backend

```python
def get_backend(name: str) -> Backend:
    """Retourne le backend approprié."""
    if name == "jax":
        return JAXBackend()
    elif name == "numpy":
        return NumpyBackend()
    else:
        raise ValueError(f"Backend inconnu: {name}")
```

---

## 6. I/O Asynchrone

### 6.1 Mécanisme

Le GIL Python est libéré pendant l'exécution JAX (Asynchronous Dispatch), permettant le parallélisme I/O + calcul.

```python
from concurrent.futures import ThreadPoolExecutor

class AsyncWriter:
    """Écrivain asynchrone pour les outputs."""

    def __init__(self, max_workers: int = 2):
        self.executor = ThreadPoolExecutor(max_workers)
        self.futures = []

    def write_async(self, data: dict, chunk_id: int, path: str):
        """Lance une écriture en arrière-plan."""
        future = self.executor.submit(self._write_chunk, data, chunk_id, path)
        self.futures.append(future)

    def flush(self):
        """Attend la fin de toutes les écritures."""
        for future in self.futures:
            future.result()
        self.futures.clear()

    def _write_chunk(self, data, chunk_id, path):
        """Écrit un chunk sur disque."""
        # Zarr append ou NetCDF unlimited dim
        ...
```

### 6.2 Timeline d'Exécution

```
Chunk 0: [===COMPUTE===]
                        [WRITE]
Chunk 1:                [===COMPUTE===]
                                        [WRITE]
Chunk 2:                                [===COMPUTE===]
                                                        [WRITE]
```

---

## 7. Gestion des Erreurs

### 7.1 Stratégie

En cas d'erreur pendant l'exécution :
1. Sauvegarder tout ce qui a été calculé jusqu'à présent
2. Écrire sur disque
3. Lever l'exception avec contexte

```python
def run(self, output_path: str) -> None:
    try:
        for i in range(n_chunks):
            state, outputs = self._run_chunk(i, state)
            self._write_async(outputs, i, output_path)
    except Exception as e:
        # Flush des écritures en attente
        self._flush_writes()
        # Log de l'état
        logger.error(f"Erreur au chunk {i}: {e}")
        raise RuntimeError(f"Simulation interrompue au chunk {i}") from e
```

### 7.2 Pas de Mécanisme de Reprise (V1)

La reprise depuis un checkpoint n'est pas supportée en V1. L'utilisateur doit relancer depuis le début.

---

## 8. Sémantique du Scan JAX

### 8.1 Vocabulaire

| Terme JAX | Rôle dans SeapoPym | Exemple |
|-----------|-------------------|---------|
| **Carry** | State (mutable) | `biomass`, `concentration` |
| **Inputs (xs)** | Forcings (par temps) | `temperature[t]`, `current[t]` |
| **Outputs (ys)** | Diagnostiques | `tendency`, `flux` |
| **Static** | Parameters (closure) | `growth_rate` |

### 8.2 Schéma

```python
# Signature lax.scan
final_carry, outputs = jax.lax.scan(
    f=step_fn,        # (carry, x) → (carry, y)
    init=state_0,     # État initial
    xs=forcings,      # Séquence temporelle (T, ...)
    length=n_steps    # Optionnel si xs fourni
)
```

---

## 9. Configuration

### 9.1 Paramètres du Runner

```yaml
# Dans run.yaml
execution:
  backend: "jax"              # "jax" ou "numpy"
  chunk_size: 365             # Jours par chunk (StreamingRunner)
  async_io: true              # I/O asynchrone
  io_workers: 2               # Threads d'écriture
```

### 9.2 Validation

Le système vérifie automatiquement :
- `chunk_size` divise `T` (ou ajuste le dernier chunk)
- Backend disponible (JAX installé si demandé)
- Permissions d'écriture sur `output_path`

---

## 10. Interfaces

### 10.1 API Haut Niveau

```python
from seapopym import compile_model

# Compilation
model = compile_model(blueprint, config, backend="jax")

# Mode Production (StreamingRunner)
model.run(
    output_path="/results/sim_001/",
    chunk_size=365
)

# Mode Optimisation (GradientRunner)
result = model.optimize(
    loss_fn=my_loss,
    optimizer=optax.adam(1e-3),
    n_epochs=100
)
```

### 10.2 Méthodes Principales

| Méthode | Runner | Description |
|---------|--------|-------------|
| `run(output_path, chunk_size)` | Streaming | Simulation avec écriture |
| `optimize(loss_fn, optimizer)` | Gradient | Calibration par gradient descent |
| `step(state, forcings_t)` | - | Un pas de temps (debug) |

---

## 11. Liens avec les Autres Axes

| Axe | Interaction |
|-----|-------------|
| Axe 1 (Blueprint) | Résout les fonctions via le registre |
| Axe 2 (Compiler) | Reçoit `CompiledModel` prêt |
| Axe 4 (Parallelism) | `vmap` sur le batch axis (E) |
| Axe 5 (Auto-Diff) | GradientRunner + checkpointing |

---

## 12. Questions Ouvertes (V2+)

- Support de schémas d'intégration avancés (RK4, implicit)
- Mécanisme de reprise depuis checkpoint
- Adaptive time stepping
- Streaming temps réel (online)
