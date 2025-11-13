# Plan d'Implémentation Détaillé - Prototype Minimal

## Objectif Global

**Créer un prototype fonctionnel minimal (Version 1 : transport distribué)**
- Modèle simple : `dB/dt = R - λB` avec diffusion 2D
- 2×2 workers sur grille 20×20
- Architecture événementielle complète
- Tests validés

---

## Phase 1 : Core (Unit + Kernel) [1-2 jours]

### 1.1 - Classe `Unit`

**Fichier** : `src/seapopym_message/core/unit.py`

**Contenu** :
```python
@dataclass
class Unit:
    name: str
    func: Callable
    inputs: list[str]
    outputs: list[str]
    scope: Literal['local', 'global']
    compiled: bool = False

    def can_execute(self, available_vars: set[str]) -> bool
    def execute(self, state: dict, **kwargs) -> dict

def unit(name, inputs, outputs, scope='local', compiled=False):
    """Décorateur pour créer des Units."""
```

**Tests** : `tests/unit/test_unit.py`
- ✓ Création d'une Unit
- ✓ can_execute avec inputs disponibles/manquants
- ✓ execute retourne les outputs corrects
- ✓ Décorateur @unit fonctionne
- ✓ Compilation JAX si compiled=True

**Estimation** : 2-3 heures

---

### 1.2 - Classe `Kernel`

**Fichier** : `src/seapopym_message/core/kernel.py`

**Contenu** :
```python
class Kernel:
    def __init__(self, units: list[Unit])

    # Propriétés
    @property
    def local_units(self) -> list[Unit]
    @property
    def global_units(self) -> list[Unit]

    # Méthodes principales
    def execute_local_phase(self, state: dict, dt: float, params: dict) -> dict
    def execute_global_phase(self, state: dict, dt: float, params: dict,
                            neighbor_data: dict | None = None) -> dict
    def has_global_units(self) -> bool

    # Méthodes internes
    def _check_dependencies(self) -> None
    def _topological_sort(self, units: list[Unit]) -> list[Unit]
```

**Tests** : `tests/unit/test_kernel.py`
- ✓ Création Kernel avec liste vide
- ✓ Séparation local_units / global_units
- ✓ execute_local_phase exécute dans l'ordre
- ✓ Détection de dépendances cycliques
- ✓ Tri topologique correct
- ✓ Erreur si input manquant

**Estimation** : 3-4 heures

---

## Phase 2 : Units de Base (Biologie Simple) [1 jour]

### 2.1 - Units Biologiques Basiques

**Fichier** : `src/seapopym_message/kernels/biology.py`

**Units à implémenter** :

```python
@unit(name='recruitment', inputs=[], outputs=['R'],
      scope='local', compiled=True)
def compute_recruitment(params: dict) -> jnp.ndarray:
    """R constant."""
    return params['R']

@unit(name='mortality', inputs=['biomass'], outputs=['mortality_rate'],
      scope='local', compiled=True)
def compute_mortality(biomass: jnp.ndarray, params: dict) -> jnp.ndarray:
    """Mortalité linéaire : λB."""
    return params['lambda'] * biomass

@unit(name='growth', inputs=['biomass', 'R', 'mortality_rate'],
      outputs=['biomass'], scope='local', compiled=True)
def compute_growth(biomass: jnp.ndarray, R: jnp.ndarray,
                  mortality_rate: jnp.ndarray,
                  dt: float, params: dict) -> jnp.ndarray:
    """B_new = B + (R - M) * dt."""
    return biomass + (R - mortality_rate) * dt
```

**Tests** : `tests/unit/test_biology.py`
- ✓ Recruitment retourne valeur constante
- ✓ Mortality proportionnelle à biomasse
- ✓ Growth intègre correctement
- ✓ Convergence vers équilibre (B_eq = R/λ)
- ✓ Conservation de la masse

**Estimation** : 2-3 heures

---

### 2.2 - Unit Diffusion 2D Basique

**Fichier** : `src/seapopym_message/kernels/transport.py`

**Unit à implémenter** :

```python
@unit(name='diffusion_2d', inputs=['biomass'], outputs=['biomass'],
      scope='global', compiled=True)
def compute_diffusion_2d(
    biomass: jnp.ndarray,
    dt: float,
    params: dict,
    halo_north: dict | None = None,
    halo_south: dict | None = None,
    halo_east: dict | None = None,
    halo_west: dict | None = None
) -> jnp.ndarray:
    """
    Diffusion 2D avec halo.

    Laplacien : ∂²B/∂x² + ∂²B/∂y²
    Conditions limites : Neumann (flux nul)
    """
    # Construire domaine étendu
    # Calculer laplacien
    # Retourner biomasse diffusée
```

**Tests** : `tests/unit/test_transport.py`
- ✓ Diffusion d'un pic gaussien s'élargit
- ✓ Conservation de la masse totale
- ✓ Halo null → condition Neumann
- ✓ Stabilité numérique (critère CFL)

**Estimation** : 3-4 heures

---

## Phase 3 : Utilitaires Grille 2D [0.5 jour]

### 3.1 - GridInfo et Domain Splitting

**Fichier** : `src/seapopym_message/utils/grid.py`

```python
@dataclass
class GridInfo:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    nlat: int
    nlon: int

    @property
    def lat_coords(self) -> jnp.ndarray
    @property
    def lon_coords(self) -> jnp.ndarray
    @property
    def dlat(self) -> float
    @property
    def dlon(self) -> float
    @property
    def dx(self) -> float  # mètres
    @property
    def dy(self) -> float  # mètres
```

**Fichier** : `src/seapopym_message/utils/domain.py`

```python
def split_domain_2d(
    nlat: int,
    nlon: int,
    num_workers_lat: int,
    num_workers_lon: int
) -> list[dict]:
    """
    Découpe domaine 2D en patches.

    Returns:
        Liste de dicts avec :
        - worker_id
        - lat_slice: (start, end)
        - lon_slice: (start, end)
        - neighbors: {'north', 'south', 'east', 'west'}
    """
```

**Tests** : `tests/unit/test_utils.py`
- ✓ GridInfo calcule coordonnées correctement
- ✓ Conversion degrés → mètres
- ✓ split_domain_2d découpe correctement (2×2)
- ✓ Voisins assignés correctement

**Estimation** : 2-3 heures

---

## Phase 4 : Workers Ray [1 jour]

### 4.1 - CellWorker2D

**Fichier** : `src/seapopym_message/distributed/worker.py`

```python
@ray.remote
class CellWorker2D:
    def __init__(
        self,
        worker_id: int,
        grid_info: GridInfo,
        lat_slice: tuple[int, int],
        lon_slice: tuple[int, int],
        kernel: Kernel,
        params: dict
    )

    # Configuration
    def set_neighbors(self, neighbors: dict[str, ray.ObjectRef]) -> None
    def set_initial_state(self, state: dict) -> None

    # Frontières
    def get_boundary_north(self) -> dict
    def get_boundary_south(self) -> dict
    def get_boundary_east(self) -> dict
    def get_boundary_west(self) -> dict

    # Exécution
    async def step(self, dt: float) -> dict

    # Diagnostics
    def get_state(self) -> dict
```

**Structure `step()` :**
```python
async def step(self, dt: float) -> dict:
    # 1. Phase locale
    self.state = self.kernel.execute_local_phase(
        self.state, dt, self.params
    )

    # 2. Échange halo (si nécessaire)
    if self.kernel.has_global_units():
        halo_futures = self._request_halos()
        halo_data = await self._gather_halos(halo_futures)

        # 3. Phase globale
        self.state = self.kernel.execute_global_phase(
            self.state, dt, self.params, halo_data
        )

    self.t += dt
    return {'t': self.t, 'biomass_mean': float(jnp.mean(self.state['biomass']))}
```

**Tests** : `tests/integration/test_worker.py`
- ✓ Worker initialise correctement
- ✓ set_neighbors fonctionne
- ✓ get_boundary_* retourne bonnes valeurs
- ✓ step() sans global units (kernel local only)
- ✓ step() avec global units et halos

**Estimation** : 4-5 heures

---

## Phase 5 : Scheduler Événementiel [1 jour]

### 5.1 - EventScheduler

**Fichier** : `src/seapopym_message/distributed/scheduler.py`

```python
@dataclass(order=True)
class Event:
    time: float
    worker_id: int = field(compare=False)
    action: str = field(compare=False)
    data: Any = field(default=None, compare=False)

@ray.remote
class EventScheduler:
    def __init__(
        self,
        workers: list[ray.ObjectRef],
        dt: float,
        t_end: float
    )

    async def run(self) -> list[dict]:
        """
        Boucle événementielle principale.

        Algorithme:
        1. Initialiser : ajouter event t=0 pour chaque worker
        2. Tant que queue non vide et t <= t_end :
            a. Extraire event minimal (heappop)
            b. Exécuter worker.step()
            c. Planifier prochain event (t + dt)
        3. Retourner états finaux
        """
```

**Tests** : `tests/integration/test_scheduler.py`
- ✓ Scheduler initialise avec workers
- ✓ Événements ordonnés par temps
- ✓ Simulation 10 pas de temps
- ✓ Tous les workers avancent au même rythme
- ✓ t_end respecté

**Estimation** : 3-4 heures

---

## Phase 6 : Intégration Complète [1 jour]

### 6.1 - Setup Simulation 2D

**Fichier** : `src/seapopym_message/utils/setup.py`

```python
def setup_simulation_2d(
    nlat_global: int,
    nlon_global: int,
    num_workers_lat: int,
    num_workers_lon: int,
    kernel: Kernel,
    params: dict,
    grid_bounds: dict
) -> tuple[list[ray.ObjectRef], GridInfo]:
    """
    Configure simulation 2D distribuée.

    Returns:
        (workers, grid_info)
    """
    # 1. Créer GridInfo
    # 2. Découper domaine
    # 3. Créer workers
    # 4. Connecter voisins
    # 5. Initialiser états
    return workers, grid_info
```

**Tests** : `tests/integration/test_full_simulation.py`

**Test Principal : Simulation Complète** :
```python
def test_full_simulation_2x2_workers():
    """
    Test intégration complète :
    - Grille 20×20
    - 2×2 workers (10×10 chacun)
    - Modèle dB/dt = R - λB + diffusion
    - 100 pas de temps
    """
    # Setup kernel
    kernel = Kernel([
        compute_recruitment,
        compute_mortality,
        compute_growth,
        compute_diffusion_2d
    ])

    # Setup simulation
    workers, grid = setup_simulation_2d(
        nlat_global=20, nlon_global=20,
        num_workers_lat=2, num_workers_lon=2,
        kernel=kernel,
        params={'R': 10.0, 'lambda': 0.1, 'D': 100.0},
        grid_bounds={...}
    )

    # Run
    scheduler = EventScheduler.remote(workers, dt=0.1, t_end=10.0)
    results = ray.get(scheduler.run.remote())

    # Validations
    assert len(results) == 4  # 4 workers
    # Vérifier convergence vers équilibre
    # Vérifier conservation masse globale
```

**Estimation** : 4-5 heures

---

### 6.2 - Visualisation

**Fichier** : `src/seapopym_message/utils/viz.py`

```python
def reconstruct_global_grid(
    workers: list[ray.ObjectRef],
    patches: list[dict],
    nlat: int,
    nlon: int
) -> dict:
    """Reconstruit grille globale depuis workers."""

def plot_global_field(
    field: np.ndarray,
    grid_info: GridInfo,
    title: str = "Biomass"
) -> None:
    """Visualise champ 2D avec matplotlib."""
```

**Estimation** : 1-2 heures

---

## Phase 7 : Documentation et Notebooks [0.5 jour]

### 7.1 - Notebook Exemple

**Fichier** : `notebooks/01_minimal_example.ipynb`

**Contenu** :
1. Import et configuration
2. Définir modèle simple (kernel)
3. Setup simulation 2×2 workers
4. Lancer simulation
5. Visualiser résultats
6. Analyser convergence vers équilibre

**Estimation** : 2-3 heures

---

## Résumé Chronologique

| Phase | Description | Fichiers | Tests | Temps |
|-------|-------------|----------|-------|-------|
| **1** | Core (Unit + Kernel) | 2 | 2 | 6h |
| **2** | Units biologie + transport | 2 | 2 | 7h |
| **3** | Utils grille 2D | 2 | 1 | 3h |
| **4** | CellWorker2D | 1 | 1 | 5h |
| **5** | EventScheduler | 1 | 1 | 4h |
| **6** | Intégration + viz | 2 | 1 | 6h |
| **7** | Documentation + notebook | 1 | - | 3h |
| **TOTAL** | | **11 fichiers** | **10 tests** | **~34h** |

---

## Ordre d'Implémentation Optimal

```
Jour 1 (8h) :
  ✓ Phase 1 : Unit + Kernel (6h)
  ✓ Début Phase 2 : biology.py (2h)

Jour 2 (8h) :
  ✓ Fin Phase 2 : transport.py (5h)
  ✓ Phase 3 : Utils grille 2D (3h)

Jour 3 (8h) :
  ✓ Phase 4 : CellWorker2D (5h)
  ✓ Début Phase 5 : EventScheduler (3h)

Jour 4 (8h) :
  ✓ Fin Phase 5 : EventScheduler (1h)
  ✓ Phase 6 : Intégration complète (6h)
  ✓ Tests intégration (1h)

Jour 5 (2h) :
  ✓ Phase 7 : Documentation + notebook (2h)
```

**Total : ~4.5 jours de développement**

---

## Critères de Validation

### Tests Unitaires (>80% coverage)
- ✓ Toutes les classes core testées
- ✓ Toutes les Units testées isolément
- ✓ Utils testés

### Tests d'Intégration
- ✓ Worker fonctionne seul
- ✓ Scheduler orchestre correctement
- ✓ Simulation complète 2×2 workers réussit

### Validation Scientifique
- ✓ Convergence vers B_eq = R/λ (erreur < 1%)
- ✓ Conservation de la masse (erreur < 0.1%)
- ✓ Diffusion d'un pic gaussien conforme théorie

### Code Quality
- ✓ Ruff : 0 erreur
- ✓ Mypy : 0 erreur
- ✓ Coverage : >80%
- ✓ Docstrings : style Google

---

## Livrables Finaux

### Code
```
src/seapopym_message/
├── core/
│   ├── unit.py             ✓
│   └── kernel.py           ✓
├── kernels/
│   ├── biology.py          ✓
│   └── transport.py        ✓
├── distributed/
│   ├── worker.py           ✓
│   └── scheduler.py        ✓
└── utils/
    ├── grid.py             ✓
    ├── domain.py           ✓
    ├── setup.py            ✓
    └── viz.py              ✓
```

### Tests
```
tests/
├── unit/
│   ├── test_unit.py        ✓
│   ├── test_kernel.py      ✓
│   ├── test_biology.py     ✓
│   ├── test_transport.py   ✓
│   └── test_utils.py       ✓
└── integration/
    ├── test_worker.py      ✓
    ├── test_scheduler.py   ✓
    └── test_full_simulation.py ✓
```

### Documentation
```
notebooks/
└── 01_minimal_example.ipynb ✓
```

---

## Questions Avant de Commencer

1. **Approche incrémentale** : On implémente phase par phase avec tests ? ✓
2. **Type hints stricts** : Utiliser partout (requis par Mypy) ? ✓
3. **JAX JIT** : Compiler toutes les Units numériques ? ✓
4. **Docstrings** : Style Google sur toutes les fonctions ? ✓

---

## Prêt à Démarrer ?

**Proposition** : Commencer par **Phase 1** (Unit + Kernel) maintenant ?

C'est la base de tout, ~6h de travail, avec tests validés.

**Voulez-vous que je commence l'implémentation ?**
