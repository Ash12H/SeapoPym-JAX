# Refactoring : Transport Worker Externe

**Date** : 2025-11-15
**Objectif** : Externaliser l'advection et la diffusion dans un worker Ray dédié

---

## 🎯 Vision Architecturale

### Principe

```
┌─────────────────────────────────────────────────────────────┐
│                      EventScheduler                          │
│  • Coordonne les steps temporels                            │
│  • Orchestre biologie → transport → biologie                │
└──────────────────┬──────────────────────────────────────────┘
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
┌──────────────────┐  ┌──────────────────────────────────┐
│  CellWorker2D    │  │   TransportWorker (NEW)          │
│  (biologie)      │  │   • Reçoit état global           │
│  • Growth        │  │   • Advection (JAX-CFD)          │
│  • Recruitment   │  │   • Diffusion (JAX)              │
│  • Mortality     │  │   • Renvoie état transporté      │
│  • Physics       │  │   • GPU-optimisé                 │
└──────────────────┘  └──────────────────────────────────┘
```

### Workflow d'un step

```
1. Scheduler: "Step biologie" → CellWorkers exécutent localement
2. Scheduler: Collecte état global (biomasse de toutes les cellules)
3. Scheduler: "Step transport" → TransportWorker
   ├─ TransportWorker reçoit: biomass_global, u, v, D, mask, dt
   ├─ TransportWorker exécute: advection + diffusion (JAX)
   └─ TransportWorker renvoie: biomass_transported
4. Scheduler: Redistribue biomass_transported aux CellWorkers
5. Next step...
```

---

## ✅ Avantages

| Aspect           | Bénéfice                                                               |
| ---------------- | ---------------------------------------------------------------------- |
| **Simplicité**   | CellWorker2D ne gère plus les halos                                    |
| **Performance**  | Transport sur GPU (JAX/JAX-CFD optimisé)                               |
| **Conservation** | Vue globale → meilleure conservation de masse                          |
| **Flexibilité**  | Facile de changer de solveur (upwind → flux-limiter → semi-Lagrangien) |
| **Testabilité**  | Transport isolé, testé indépendamment                                  |
| **Scalabilité**  | Peut paralléliser transport si domaine trop grand (voir Phase 3)       |

---

## ⚠️ Points d'attention & Questions

### 1. **Communication overhead**

**Question** : Transférer tout le domaine à chaque step n'est-il pas coûteux ?

**Analyse** :

-   Domaine 1000×2000 = 2M cellules × 4 bytes (float32) = **8 MB** par champ
-   Champs à transférer : biomass (8 MB), u (8 MB), v (8 MB), D (8 MB) = **32 MB**
-   Sur réseau local : ~1 Gb/s → **0.25s** de transfert
-   Temps de calcul transport : ~0.1-1s (JAX compilé) → **Overhead acceptable**

**Réponse** : ✅ OK pour domaines ≤ 10M cellules. Au-delà, voir Phase 3 (multi-transport workers).

---

### 2. **Scalabilité : 1 ou N transport workers ?**

**Options** :

| Option         | Description                       | Avantages            | Inconvénients           |
| -------------- | --------------------------------- | -------------------- | ----------------------- |
| **A. Un seul** | 1 TransportWorker global          | Simple, pas de halos | Limite ~10-20M cellules |
| **B. Multi**   | N TransportWorkers (1 par région) | Scalable             | Halos entre régions     |

**Recommandation Phase 1** : **Option A** (1 worker)

-   Suffit pour domaines réalistes (1000×2000)
-   Simplifie architecture
-   Phase 3 (si nécessaire) : passer à Option B

---

### 3. **GPU : Dédier un GPU au TransportWorker ?**

**Question** : TransportWorker tourne sur GPU ou CPU ?

**Recommandation** :

```python
# Créer TransportWorker avec ressource GPU dédiée
transport_worker = TransportWorker.options(num_gpus=1).remote()
```

**Bénéfices** :

-   JAX compile sur GPU → **10-100× plus rapide**
-   CellWorkers restent sur CPU (biologie moins intensive)
-   Utilisation optimale des ressources

---

### 4. **Operator splitting : ordre des opérations**

**Question** : Quel ordre ? Biologie → Transport ou Transport → Biologie ?

**Recommandation** : **Strang splitting (ordre 2)**

```python
# Step n → n+1
1. Biologie(dt/2)      # CellWorkers
2. Transport(dt)       # TransportWorker (advection + diffusion)
3. Biologie(dt/2)      # CellWorkers
```

**Alternative (ordre 1, plus simple)** :

```python
1. Biologie(dt)        # CellWorkers
2. Transport(dt)       # TransportWorker
```

**Choix Phase 1** : Ordre 1 (simplicité), Ordre 2 en Phase 2 si nécessaire.

---

### 5. **Halos : toujours nécessaires ?**

**Réponse** : ❌ **Plus nécessaires pour le transport** !

-   TransportWorker a la vue **globale** → pas besoin de halos
-   CellWorkers gardent halos **uniquement pour biologie** si couplages spatiaux (ex: prédation spatiale)

**Impact** :

-   Simplifie `CellWorker2D.step()`
-   Supprime `get_boundary_north/south/east/west` pour transport
-   Garde seulement pour biologie globale (si nécessaire)

---

### 6. **Que fait CellWorker2D alors ?**

**Nouvelles responsabilités** :

```python
@ray.remote
class CellWorker2D:
    """Worker pour processus biologiques locaux uniquement."""

    async def step_biology(self, dt: float, forcings: dict):
        """Exécute uniquement la biologie (local phase)."""

        # Phases locales (pas de halos)
        self.state = self.kernel.execute_local_phase(
            self.state,
            dt=dt,
            params=self.params,
            forcings=forcings,
        )

        return self._compute_diagnostics()

    def get_biomass(self):
        """Retourne biomasse locale (pour collecte globale)."""
        return self.state["biomass"]

    def set_biomass(self, biomass: jnp.ndarray):
        """Reçoit biomasse après transport."""
        self.state["biomass"] = biomass
```

**Kernel** : Ne contient plus `compute_advection_2d`, `compute_diffusion_2d`

---

## 📋 Plan d'implémentation

### **Phase 1 : Créer TransportWorker (1-2 jours)**

#### 1.1 Créer `src/seapopym_message/transport/worker.py`

```python
@ray.remote
class TransportWorker:
    """Worker dédié au transport (advection + diffusion)."""

    def __init__(
        self,
        advection_method: str = "jax_cfd",
        diffusion_method: str = "crank_nicolson",
    ):
        self.advection_method = advection_method
        self.diffusion_method = diffusion_method

        # Initialiser solveurs
        self._init_solvers()

    def _init_solvers(self):
        """Initialise les solveurs JAX/JAX-CFD."""
        if self.advection_method == "jax_cfd":
            from jax_cfd.base import advection
            self.advection_func = advection.advect_van_leer
        elif self.advection_method == "flux_conservative":
            self.advection_func = self._advection_flux_conservative
        # ... autres méthodes

    def transport_step(
        self,
        biomass: jnp.ndarray,      # (nlat, nlon)
        u: jnp.ndarray,            # (nlat, nlon) [m/s]
        v: jnp.ndarray,            # (nlat, nlon) [m/s]
        D: jnp.ndarray,            # (nlat, nlon) [m²/s]
        dt: float,                 # [s]
        dx: float,                 # [m]
        dy: float,                 # [m]
        mask: jnp.ndarray | None = None,  # (nlat, nlon) bool
    ) -> dict:
        """Effectue advection + diffusion sur domaine global.

        Returns:
            {
                'biomass': jnp.ndarray,     # État transporté
                'diagnostics': {
                    'mass_before': float,
                    'mass_after': float,
                    'mass_error': float,
                    'advection_time': float,
                    'diffusion_time': float,
                }
            }
        """
        import time

        mass_before = jnp.sum(biomass)

        # 1. Advection
        t0 = time.time()
        biomass_advected = self._advection(biomass, u, v, dt, dx, dy, mask)
        t_advection = time.time() - t0

        # 2. Diffusion
        t0 = time.time()
        biomass_transported = self._diffusion(biomass_advected, D, dt, dx, dy, mask)
        t_diffusion = time.time() - t0

        mass_after = jnp.sum(biomass_transported)

        return {
            'biomass': biomass_transported,
            'diagnostics': {
                'mass_before': float(mass_before),
                'mass_after': float(mass_after),
                'mass_error': float(abs(mass_after - mass_before) / mass_before),
                'advection_time': t_advection,
                'diffusion_time': t_diffusion,
                'advection_method': self.advection_method,
                'diffusion_method': self.diffusion_method,
            }
        }

    def _advection(self, biomass, u, v, dt, dx, dy, mask):
        """Advection avec méthode configurée."""
        # Implémentation selon méthode choisie
        # ...
        pass

    def _diffusion(self, biomass, D, dt, dx, dy, mask):
        """Diffusion avec méthode configurée."""
        # Implémentation
        # ...
        pass
```

#### 1.2 Intégrer JAX-CFD pour advection

```python
def _advection_jax_cfd(self, biomass, u, v, dt, dx, dy, mask):
    """Advection avec JAX-CFD (van Leer, ordre 2)."""
    from jax_cfd.base import advection, grids

    # Créer grille
    nlat, nlon = biomass.shape
    grid = grids.Grid(
        shape=(nlat, nlon),
        domain=((0, nlat * dy), (0, nlon * dx)),
    )

    # Advection
    biomass_new = advection.advect_van_leer(
        c=biomass,
        velocity=(v, u),  # JAX-CFD utilise (y, x)
        dt=dt,
        grid=grid,
    )

    # Appliquer masque si fourni
    if mask is not None:
        biomass_new = jnp.where(mask, biomass_new, 0.0)

    return biomass_new
```

#### 1.3 Implémenter diffusion Crank-Nicolson

```python
def _diffusion_crank_nicolson(self, biomass, D, dt, dx, dy, mask):
    """Diffusion avec schéma Crank-Nicolson (implicite, ordre 2)."""
    from scipy.sparse import linalg

    # Construction matrice implicite
    # A * B_{n+1} = B_n + dt/2 * Laplacian(B_n)
    # ...

    # Résolution système linéaire
    # biomass_new = linalg.spsolve(A, rhs)

    # Note: Implémenter version JAX pure (jax.scipy.sparse.linalg)
    # pour compatibilité GPU
    pass
```

#### 1.4 Tests unitaires `tests/unit/test_transport_worker.py`

```python
@pytest.mark.unit
class TestTransportWorker:
    def test_mass_conservation_advection(self):
        """Vérifier conservation de masse (advection seule)."""
        # Blob gaussien, courant uniforme
        # Vérifier erreur < 1%
        pass

    def test_mass_conservation_diffusion(self):
        """Vérifier conservation de masse (diffusion seule)."""
        pass

    def test_advection_with_mask(self):
        """Vérifier que flux bloqué aux îles."""
        pass

    def test_gpu_compatibility(self):
        """Vérifier que ça tourne sur GPU si disponible."""
        pass
```

---

### **Phase 2 : Modifier EventScheduler (1 jour)**

#### 2.1 Nouveau workflow dans `EventScheduler.step()`

```python
# distributed/scheduler.py

class EventScheduler:
    def __init__(
        self,
        workers: list,
        patches: list,
        transport_worker: ray.ObjectRef,  # NEW
        dt: float,
        t_max: float,
        forcing_manager: ForcingManager,
    ):
        self.workers = workers
        self.patches = patches
        self.transport_worker = transport_worker  # NEW
        # ...

    def step(self):
        """Exécute un pas de temps : biologie → transport."""

        # 1. Préparer forcings pour ce step
        forcings = self.forcing_manager.prepare_timestep(time=self.t)
        forcings_ref = ray.put(forcings)

        # 2. PHASE BIOLOGIE (parallèle)
        bio_futures = [
            worker.step_biology.remote(dt=self.dt, forcings=forcings_ref)
            for worker in self.workers
        ]
        bio_diagnostics = ray.get(bio_futures)

        # 3. COLLECTE BIOMASSE GLOBALE
        biomass_global = self._collect_global_biomass()

        # 4. PHASE TRANSPORT (centralisée)
        transport_result = ray.get(
            self.transport_worker.transport_step.remote(
                biomass=biomass_global,
                u=forcings["u"],
                v=forcings["v"],
                D=forcings.get("D", jnp.zeros_like(biomass_global)),
                dt=self.dt,
                dx=self.dx,
                dy=self.dy,
                mask=self.mask,
            )
        )

        # 5. REDISTRIBUTION BIOMASSE AUX WORKERS
        self._distribute_biomass(transport_result['biomass'])

        # 6. UPDATE TIME
        self.t += self.dt

        # 7. DIAGNOSTICS
        self._log_diagnostics(bio_diagnostics, transport_result['diagnostics'])

    def _collect_global_biomass(self) -> jnp.ndarray:
        """Collecte biomasse de tous les workers → array global."""
        biomass_futures = [
            worker.get_biomass.remote() for worker in self.workers
        ]
        biomass_patches = ray.get(biomass_futures)

        # Reconstruire array global
        nlat_global = max(p["lat_end"] for p in self.patches)
        nlon_global = max(p["lon_end"] for p in self.patches)
        biomass_global = jnp.zeros((nlat_global, nlon_global))

        for patch, biomass_local in zip(self.patches, biomass_patches):
            biomass_global = biomass_global.at[
                patch["lat_start"]:patch["lat_end"],
                patch["lon_start"]:patch["lon_end"]
            ].set(biomass_local)

        return biomass_global

    def _distribute_biomass(self, biomass_global: jnp.ndarray):
        """Redistribue biomasse transportée aux workers."""
        futures = []
        for worker, patch in zip(self.workers, self.patches):
            biomass_local = biomass_global[
                patch["lat_start"]:patch["lat_end"],
                patch["lon_start"]:patch["lon_end"]
            ]
            future = worker.set_biomass.remote(biomass_local)
            futures.append(future)

        ray.wait(futures, num_returns=len(futures))
```

---

### **Phase 3 : Adapter CellWorker2D (0.5 jour)**

#### 3.1 Simplifier `CellWorker2D`

```python
# distributed/worker.py

@ray.remote
class CellWorker2D:
    """Worker pour biologie locale uniquement."""

    async def step_biology(
        self, dt: float, forcings_global: dict | None = None
    ) -> dict:
        """Exécute uniquement les processus biologiques locaux.

        Note: Le transport est géré par TransportWorker externe.
        """
        # Extract forcings for local patch
        forcings_local = None
        if forcings_global is not None:
            forcings_local = self._extract_local_forcings(forcings_global)

        # Execute ONLY local phase (biology)
        self.state = self.kernel.execute_local_phase(
            self.state,
            dt=dt,
            params=self.params,
            grid_shape=(self.nlat, self.nlon),
            forcings=forcings_local,
        )

        # Update time
        self.t += dt

        return self._compute_diagnostics()

    # REMOVED: step() with global phase
    # REMOVED: _request_halos(), _gather_halos()

    # NEW: Getter/setter pour biomasse
    def get_biomass(self) -> jnp.ndarray:
        """Retourne biomasse locale."""
        return self.state["biomass"]

    def set_biomass(self, biomass: jnp.ndarray):
        """Reçoit biomasse après transport."""
        self.state["biomass"] = biomass
```

#### 3.2 Retirer transport du Kernel

```python
# Kernels biologiques NE CONTIENNENT PLUS:
# - compute_advection_2d
# - compute_diffusion_2d

# Exemple kernel biologie pure:
from seapopym_message.kernels.biology import compute_growth, compute_recruitment

kernel = Kernel([
    compute_recruitment,  # Local
    compute_growth,       # Local
    # PAS de compute_advection_2d !
])
```

---

### **Phase 4 : Migration des exemples (0.5 jour)**

#### 4.1 Exemple `advection_blob.py` avec TransportWorker

```python
# examples/advection_blob.py (NEW VERSION)

# Setup
transport_worker = TransportWorker.options(num_gpus=1).remote(
    advection_method="jax_cfd",
    diffusion_method="none",  # Pas de diffusion pour cet exemple
)

# Créer workers biologiques (sans transport)
workers, patches = create_distributed_simulation(
    grid=grid,
    kernel=Kernel([]),  # Pas de biologie pour cet exemple
    params=params,
    num_workers_lat=2,
    num_workers_lon=2,
)

# Créer scheduler avec transport worker
scheduler = EventScheduler(
    workers=workers,
    patches=patches,
    transport_worker=transport_worker,  # NEW !
    dt=dt,
    t_max=t_max,
    forcing_manager=forcing_manager,
)

# Run
while scheduler.get_current_time() < t_max:
    scheduler.step()
```

#### 4.2 Exemple `island_blocking.py`

**Option 1** : Migrer vers TransportWorker distribué
**Option 2** : Garder version simple (non-distribuée) pour démo rapide

**Recommandation** : Option 2 pour Phase 1, Option 1 pour Phase 4.

---

### **Phase 5 : Optimisations avancées (optionnel)**

#### 5.1 Multi-TransportWorkers (si domaine > 10M cellules)

```python
# Créer N transport workers (1 par région)
transport_workers = [
    TransportWorker.options(num_gpus=1).remote()
    for _ in range(num_regions)
]

# Scheduler distribue aux transport workers
# (nécessite gestion halos entre régions de transport)
```

#### 5.2 Strang splitting (ordre 2)

```python
# Dans EventScheduler.step()
# Au lieu de: Bio(dt) → Transport(dt)
# Faire: Bio(dt/2) → Transport(dt) → Bio(dt/2)
```

#### 5.3 Monitoring GPU

```python
# Dans TransportWorker
def get_gpu_stats(self):
    """Retourne utilisation GPU."""
    import jax
    return {
        'device': str(jax.devices()[0]),
        'memory_allocated': ...,
    }
```

---

## 📊 Timeline

| Phase       | Durée         | Tâches                                       |
| ----------- | ------------- | -------------------------------------------- |
| **Phase 1** | 1-2 jours     | TransportWorker + JAX-CFD + tests            |
| **Phase 2** | 1 jour        | Modifier EventScheduler (collect/distribute) |
| **Phase 3** | 0.5 jour      | Simplifier CellWorker2D                      |
| **Phase 4** | 0.5 jour      | Migrer exemples                              |
| **Total**   | **3-4 jours** | Version fonctionnelle                        |
| **Phase 5** | Optionnel     | Optimisations avancées                       |

---

## 🧪 Tests de validation

### Critères de succès

1. ✅ **Conservation de masse** : Erreur < 1% sur 10 jours (advection + diffusion)
2. ✅ **Performance** : Temps de calcul comparable ou meilleur qu'avant
3. ✅ **Flux blocking** : Aucune biomasse ne traverse les îles
4. ✅ **Scalabilité** : Fonctionne jusqu'à 5M cellules
5. ✅ **GPU** : Utilise GPU si disponible (speedup > 5×)

### Scénarios de test

```python
# Test 1: Advection pure (rotation)
# - Domaine 100×100
# - Blob gaussien en rotation
# - Conservation > 99%

# Test 2: Diffusion pure
# - Blob central
# - Diffusion isotrope
# - Conservation = 100% (analytique)

# Test 3: Island blocking
# - Île centrale
# - Courant uniforme
# - Pas de flux à travers île

# Test 4: Grand domaine
# - 1000×2000 cellules
# - Temps < 1s par step
```

---

## 🔄 Migration progressive

### Stratégie de déploiement

**Option A : Big Bang** (tout d'un coup)

-   Phase 1-4 d'un coup
-   Tests intensifs
-   Déploiement

**Option B : Progressive** (recommandé)

1. Créer TransportWorker en parallèle (ne casse rien)
2. Ajouter flag `use_external_transport` dans EventScheduler
3. Tester côte à côte (ancien vs nouveau)
4. Validation → Basculer par défaut
5. Supprimer ancien code

```python
# EventScheduler avec flag de migration
class EventScheduler:
    def __init__(self, ..., use_external_transport: bool = False):
        self.use_external_transport = use_external_transport

    def step(self):
        if self.use_external_transport:
            self._step_with_external_transport()
        else:
            self._step_legacy()  # Ancien comportement
```

---

## ❓ Questions ouvertes

### Q1 : Gestion des profondeurs (3D)

**Question** : Si biomasse est 3D (lat, lon, depth), comment gérer ?

**Réponse** : Pour l'instant on se concentre sur la 2D.

```python
# TransportWorker gère 3D nativement
def transport_step(
    self,
    biomass: jnp.ndarray,  # (ndepth, nlat, nlon) ou (nlat, nlon, ndepth)
    u: jnp.ndarray,        # Idem
    v: jnp.ndarray,
    w: jnp.ndarray,        # Vitesse verticale
    D: jnp.ndarray,
    # ...
):
    # Advection 3D
    # Diffusion 3D (horizontale + verticale)
    pass
```

### Q2 : Multiples espèces/cohortes

**Question** : Si plusieurs biomasses à transporter ?

**Réponse** : On peu imaginer des simulations en parallèle qui utilisent les mêmes forçages. Est-ce que c'est contradictoire avec ce qu'on a mis en place jusqu'à maintenant ?

```python
# Option 1: Boucle sur espèces
for species in species_list:
    transport_worker.transport_step(biomass=biomass[species], ...)

# Option 2: Vectoriser (plus rapide)
biomass_all = jnp.stack([biomass[sp] for sp in species_list])  # (nspecies, nlat, nlon)
result = transport_worker.transport_step_multi(biomass_all, ...)
```

### Q3 : Forcings variables spatialement ET temporellement

**Question** : Si u, v, D changent à chaque step ?

**Réponse** : ✅ Déjà prévu ! ForcingManager fournit forcings(t), TransportWorker les utilise.

---

## 📚 Dépendances

### Nouvelles dépendances

```toml
# pyproject.toml
[project]
dependencies = [
    # ... existantes
    "jax-cfd>=0.2.0",  # NEW: Pour advection ordre 2
]

[project.optional-dependencies]
gpu = [
    "jax[cuda12]>=0.4.0",  # Pour GPU support
]
```

### Installation

```bash
# CPU
uv sync

# GPU (recommandé pour TransportWorker)
uv sync --extra gpu
```

---

## 🎯 Décision : Go/No-Go ?

### Arguments POUR

✅ **Conservation** : Vue globale → meilleure conservation
✅ **Performance** : GPU pour transport → 10-100× plus rapide
✅ **Simplicité** : CellWorker2D plus simple (pas de halos pour transport)
✅ **Flexibilité** : Facile de changer solveur
✅ **Qualité** : JAX-CFD = schémas éprouvés

### Arguments CONTRE

⚠️ **Communication** : Overhead transfert domaine global
⚠️ **Complexité initiale** : Refactoring architectural
⚠️ **Dépendance** : JAX-CFD (mais réversible)

### Recommandation

✅ **GO** - Les bénéfices l'emportent largement

**Conditions** :

1. Phase 1 = Prototype rapide (2 jours max) pour valider conservation
2. Si conservation < 99% → STOP et debug
3. Si conservation OK → Phases 2-4
4. Garder ancien code avec flag pendant migration

---

## 📝 Actions immédiates

**Si tu approuves ce plan** : J'approuve ce plan !

1. ✅ Créer branche `feature/external-transport-worker`
2. ✅ Phase 1.1 : Créer `src/seapopym_message/transport/worker.py` (squelette)
3. ✅ Phase 1.2 : Intégrer JAX-CFD (advection)
4. ✅ Phase 1.3 : Implémenter diffusion simple (explicit Euler)
5. ✅ Phase 1.4 : Tests unitaires (conservation)
6. ✅ **Validation checkpoint** : Si conservation > 99% → continuer

**Temps estimé Phase 1** : 1-2 jours
