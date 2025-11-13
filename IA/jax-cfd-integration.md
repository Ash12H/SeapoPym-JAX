# Intégration JAX-CFD avec l'Architecture Ray

## Question Principale

**Est-ce que JAX-CFD est compatible avec l'architecture Ray message+agent pour faire du transport océanique ?**

**Réponse : Oui, avec une architecture hybride Ray (orchestration) + JAX (calcul local)**

---

## Vue d'Ensemble

### JAX-CFD et JAX-Fluids

**JAX-CFD** (Google)
- Projet de recherche pour CFD avec ML/auto-différentiation
- Accélération GPU/TPU

**JAX-Fluids 2.0** (2024)
- Solveur CFD différentiable pour écoulements compressibles 3D
- Scalabilité HPC : testé sur 512 GPUs A100, 2048 TPU-v3
- Méthodes volumes finis haute ordre
- Intégration ML native
- Écoulements réactifs multi-composants

### Capacités Clés pour Notre Projet

1. **Schémas numériques haute fidélité** - advection, diffusion
2. **Différentiation automatique** - calibration, assimilation de données
3. **Performance GPU/TPU** - compilation JIT très efficace
4. **Multi-device** - parallélisme natif JAX

---

## Architecture Recommandée : Hybride Ray + JAX

### Principe de Séparation

```
┌─────────────────────────────────────┐
│   Ray Scheduler (orchestration)     │  ← Temps global, événements
└─────────────────────────────────────┘
              ↓ ↓ ↓
    ┌──────────┬──────────┬──────────┐
    │ Worker 1 │ Worker 2 │ Worker N │  ← Acteurs Ray
    │  (Ray)   │  (Ray)   │  (Ray)   │     Messages inter-workers
    └──────────┴──────────┴──────────┘
         ↓          ↓          ↓
    ┌──────────┬──────────┬──────────┐
    │   JAX    │   JAX    │   JAX    │  ← Calculs locaux compilés
    │ Compute  │ Compute  │ Compute  │     GPU/TPU si disponible
    │ (local)  │ (local)  │ (local)  │
    └──────────┴──────────┴──────────┘
```

### Responsabilités

| Composant | Rôle |
|-----------|------|
| **Ray** | Orchestration inter-workers, messages asynchrones, synchronisation globale, résilience |
| **JAX** | Transport local (advection/diffusion), réactions biogéochimiques, calculs intensifs |

---

## Implémentation Concrète

### Structure d'un Worker

```python
import ray
import jax
import jax.numpy as jnp
from jax_cfd.base import grids, advection

@ray.remote(num_gpus=1)  # Chaque worker = 1 GPU
class OceanBioWorker:
    def __init__(self, cell_ids, grid_shape):
        # Compilation JAX une fois au démarrage
        self.advection_step = jax.jit(self._advection_kernel)
        self.diffusion_step = jax.jit(self._diffusion_kernel)
        self.bgc_step = jax.jit(self._biogeochemistry)
        self.grid = grids.Grid(grid_shape)

    def _advection_kernel(self, concentration, velocity, dt):
        """Kernel JAX pour l'advection"""
        return advection.semi_lagrangian(
            concentration,
            velocity,
            dt,
            grid=self.grid
        )

    def _diffusion_kernel(self, field, diffusivity, dt):
        """Kernel JAX pour la diffusion"""
        # Implémentation schéma diffusion
        return diffused_field

    def _biogeochemistry(self, biomass, temp, nutrients, dt):
        """Dynamique de population SEAPOPYM"""
        growth = self._compute_growth(temp, nutrients)
        mortality = self._compute_mortality(biomass, temp)
        return biomass + (growth - mortality) * dt

    async def run_timestep(self, t, dt):
        # 1. Transport physique (JAX-CFD, local)
        biomass = self.advection_step(
            self.biomass,
            self.velocity,
            dt
        )
        biomass = self.diffusion_step(
            biomass,
            self.diffusivity,
            dt
        )

        # 2. Échange aux frontières (Ray messages)
        if self.at_boundary():
            # Message asynchrone non-bloquant
            neighbor_data = await self.request_boundary_data()
            biomass = self.merge_boundary(biomass, neighbor_data)

        # 3. Biogéochimie (JAX, local)
        biomass = self.bgc_step(
            biomass,
            self.temperature,
            self.nutrients,
            dt
        )

        self.biomass = biomass
        return self.biomass
```

### Gestion de la Sérialisation

```python
import numpy as np

# Conversion JAX → NumPy pour messages Ray
def send_to_neighbor(self, data_jax):
    # JAX array → NumPy (sérialisation Ray)
    data_numpy = np.array(data_jax)
    neighbor_ref = self.neighbors[0].receive.remote(data_numpy)
    return neighbor_ref

def receive_from_neighbor(self, data_numpy):
    # NumPy → JAX array
    return jnp.array(data_numpy)
```

---

## Points d'Attention Critiques

### 1. Éviter le Double Parallélisme

❌ **NE PAS FAIRE**
```python
# Conflit : Ray distribue + JAX pmap
@ray.remote
class Worker:
    def compute(self):
        # jax.pmap à travers workers Ray → CONFLIT
        return jax.pmap(kernel)(data)  # ❌
```

✅ **FAIRE**
```python
@ray.remote(num_gpus=1)
class Worker:
    def compute(self):
        # JAX en mono-device par worker
        return jax.jit(kernel)(data)  # ✅
```

### 2. Compilation JIT

**Bonne pratique** : compiler une seule fois au démarrage du worker

```python
class Worker:
    def __init__(self):
        # Compilation à l'initialisation
        self.kernel = jax.jit(self._compute)
        # Warm-up pour éviter overhead premier appel
        dummy = jnp.zeros((10, 10))
        _ = self.kernel(dummy, 0.1)

    def step(self, state, dt):
        # Pas de recompilation ici
        return self.kernel(state, dt)
```

### 3. Gestion Mémoire

- **SharedFieldStore** (Zarr) pour forçages volumineux
- **Messages Ray légers** : frontières/halos uniquement
- **JAX device memory** reste locale au worker

---

## Cas d'Usage Optimal

### Exemple : Simulation Océan-Biogéochimie

**Domaine** : Pacifique tropical, résolution 1/12°

**Architecture**
- 100 workers Ray (1 GPU chacun)
- Chaque worker gère un patch 100×100 cellules
- JAX-CFD pour le transport physique
- JAX pour la biogéochimie SEAPOPYM

**Pipeline par Pas de Temps**

```
1. Transport local (JAX-CFD, parallèle)
   ├─ Advection semi-lagrangienne
   └─ Diffusion horizontale

2. Échange halo (Ray messages, bloquant)
   ├─ Envoi frontières aux voisins
   └─ Réception et fusion

3. Biogéochimie (JAX, parallèle)
   ├─ Croissance (température, nutriments)
   ├─ Mortalité (densité-dépendante)
   └─ Prédation

4. Synchronisation (Scheduler Ray)
   └─ Barrier temporelle si nécessaire
```

---

## Alternatives et Extensions

### Alternative 1 : JAX-Fluids Pur

**Quand ?** Domaines très grands, parallélisme massif (>100 GPUs)

**Approche**
- JAX-Fluids 2.0 pour tout le calcul (parallélisme natif)
- Ray uniquement pour couplage entre modèles hétérogènes
  - Ex : océan (JAX-Fluids) ↔ biogéo (Python/Numba)

### Alternative 2 : Hybride JAX + Numba

**Pour**
- Transport → JAX-CFD (GPU)
- Biogéochimie → Numba (CPU, si équations complexes non-vectorisables)

---

## Avantages de l'Approche Hybride

| Aspect | Avantage | Impact |
|--------|----------|--------|
| **Performance** | JAX JIT + GPU/TPU | 10-100× vs NumPy |
| **Différentiabilité** | Auto-diff natif | Assimilation données, calibration |
| **Scalabilité** | Ray distribue, JAX optimise localement | Cluster + GPU efficace |
| **Schémas numériques** | JAX-CFD haute ordre | Meilleure précision transport |
| **Flexibilité** | Architecture modulaire | Facile d'ajouter processus |

---

## Limitations et Solutions

### Limitation 1 : Sérialisation JAX ↔ Ray

**Problème** : JAX arrays ne se sérialisent pas directement

**Solution** : Conversion explicite JAX → NumPy → Ray → NumPy → JAX

**Coût** : Modéré (uniquement aux frontières, pas tout le domaine)

### Limitation 2 : Complexité Initiale

**Problème** : Courbe d'apprentissage JAX + Ray

**Solution**
- Prototyper en NumPy d'abord
- Migrer kernel par kernel vers JAX
- Ajouter Ray en dernier

### Limitation 3 : Debugging Distribué

**Problème** : Erreurs JAX dans workers Ray difficiles à tracer

**Solution**
- Tests unitaires des kernels JAX isolés
- Ray dashboard pour monitoring
- Logging structuré (pas de print)

---

## Roadmap d'Implémentation

### Phase 1 : Prototype Mono-Worker
- [ ] Implémenter kernels JAX (advection, diffusion, biogéo)
- [ ] Valider précision numérique vs référence
- [ ] Benchmarker performance GPU vs CPU

### Phase 2 : Distribution Ray
- [ ] Créer architecture workers Ray
- [ ] Implémenter échange halo
- [ ] Tester scalabilité (1 → 10 → 100 workers)

### Phase 3 : Optimisation
- [ ] Minimiser communication inter-workers
- [ ] Profiling (Ray dashboard + JAX profiler)
- [ ] Tuning tailles de patch

### Phase 4 : Production
- [ ] Intégration forçages réels (Zarr)
- [ ] Checkpointing / reprise
- [ ] Validation scientifique

---

## Ressources

### Documentation
- **JAX-CFD** : https://github.com/google/jax-cfd
- **JAX-Fluids 2.0** : https://github.com/tumaer/JAXFLUIDS
  - Paper : https://arxiv.org/abs/2402.05193
- **Ray + JAX** : https://docs.ray.io/en/latest/train/getting-started-jax.html

### Exemples
- JAX-CFD tutorials : https://jax-cfd.readthedocs.io/
- Ray actors patterns : https://docs.ray.io/en/latest/actors.html

---

## Conclusion

**JAX-CFD est compatible et même recommandé** pour votre architecture, avec la stratégie suivante :

1. **Ray** = orchestrateur distribué (messages, synchronisation)
2. **JAX/JAX-CFD** = moteur de calcul local (transport, biogéo)
3. **Séparation claire** : éviter double parallélisme
4. **Communication minimale** : échanges halo uniquement

Cette approche combine :
- La **performance** de JAX (GPU, JIT)
- La **flexibilité** de Ray (résilience, distribution)
- La **précision** de schémas haute ordre (JAX-CFD)

**Prochaine étape recommandée** : prototype minimal d'un worker Ray avec kernel JAX-CFD pour l'advection 2D.
