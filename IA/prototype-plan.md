# Plan de Prototype : Architecture Ray + JAX pour Dynamique de Population

## Modèle de Base

Équation simple de dynamique de population :

```
dB/dt = R - λB
```

Où :
- **B** : biomasse (g/m²)
- **R** : recrutement constant (g/m²/jour)
- **λ** : taux de mortalité constant (jour⁻¹)

### Solution Analytique

Pour validation :

```
B(t) = (R/λ) + (B₀ - R/λ) * exp(-λt)
```

État d'équilibre : `B_eq = R/λ`

---

## Vue d'Ensemble des Phases

```
Phase 1: 0D     Phase 2: 1D      Phase 3: 1D+Transport   Phase 4: Ray      Phase 5: Forçages
[Cellule]  →  [C][C][C][C]  →  [C]→[C]→[C]→[C]  →  [W1] [W2]  →  [W1+Temp]
               indépendantes      diffusion            Ray msgs      Zarr/xarray
```

---

## Phase 1 : Modèle 0D - Une Cellule Isolée

**Objectif** : Valider l'intégration temporelle de l'équation différentielle

### Fonctions Unitaires

#### 1.1 - `compute_recruitment`
```python
def compute_recruitment(params: dict) -> float:
    """
    Calcule le recrutement constant.

    Args:
        params: {'R': float}  # recrutement en g/m²/jour

    Returns:
        R: recrutement
    """
    return params['R']
```

#### 1.2 - `compute_mortality`
```python
def compute_mortality(biomass: float, params: dict) -> float:
    """
    Calcule la mortalité linéaire.

    Args:
        biomass: biomasse actuelle (g/m²)
        params: {'lambda': float}  # taux de mortalité (jour⁻¹)

    Returns:
        mortality: λB (g/m²/jour)
    """
    return params['lambda'] * biomass
```

#### 1.3 - `compute_derivative`
```python
def compute_derivative(biomass: float, params: dict) -> float:
    """
    Calcule dB/dt = R - λB.

    Args:
        biomass: biomasse actuelle
        params: {'R': float, 'lambda': float}

    Returns:
        dB_dt: dérivée temporelle
    """
    R = compute_recruitment(params)
    mortality = compute_mortality(biomass, params)
    return R - mortality
```

#### 1.4 - `euler_step`
```python
def euler_step(biomass: float, dt: float, params: dict) -> float:
    """
    Intégration d'Euler explicite : B(t+dt) = B(t) + dt * dB/dt.

    Args:
        biomass: biomasse à t
        dt: pas de temps (jours)
        params: paramètres du modèle

    Returns:
        biomass_new: biomasse à t+dt
    """
    dB_dt = compute_derivative(biomass, params)
    return biomass + dt * dB_dt
```

#### 1.5 - `runge_kutta4_step` (optionnel, pour précision)
```python
def rk4_step(biomass: float, dt: float, params: dict) -> float:
    """
    Intégration Runge-Kutta 4ème ordre (plus précise).

    Args:
        biomass: biomasse à t
        dt: pas de temps
        params: paramètres

    Returns:
        biomass_new: biomasse à t+dt
    """
    k1 = compute_derivative(biomass, params)
    k2 = compute_derivative(biomass + 0.5 * dt * k1, params)
    k3 = compute_derivative(biomass + 0.5 * dt * k2, params)
    k4 = compute_derivative(biomass + dt * k3, params)

    return biomass + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
```

### Kernel d'Exécution

```python
def run_cell_timestep(state: dict, dt: float, params: dict) -> dict:
    """
    Exécute un pas de temps pour une cellule.

    Ordre d'exécution:
    1. compute_recruitment
    2. compute_mortality
    3. compute_derivative
    4. euler_step (ou rk4_step)

    Args:
        state: {'biomass': float, 't': float}
        dt: pas de temps
        params: paramètres du modèle

    Returns:
        new_state: état mis à jour
    """
    biomass_new = euler_step(state['biomass'], dt, params)

    return {
        'biomass': biomass_new,
        't': state['t'] + dt
    }
```

### Validation

**Test 1** : Convergence vers équilibre
```python
# Paramètres
params = {'R': 10.0, 'lambda': 0.1}  # B_eq = 10/0.1 = 100
state = {'biomass': 50.0, 't': 0.0}
dt = 0.1

# Simulation 100 jours
for _ in range(1000):
    state = run_cell_timestep(state, dt, params)

# Vérifier : state['biomass'] ≈ 100.0
```

**Test 2** : Comparaison solution analytique
```python
import numpy as np

B0 = 50.0
R, lam = 10.0, 0.1
t_values = np.arange(0, 100, 0.1)

# Solution analytique
B_analytic = (R/lam) + (B0 - R/lam) * np.exp(-lam * t_values)

# Solution numérique (boucle)
# Comparer erreur relative
```

### Livrables Phase 1

- [ ] Fichier `kernels/cell_0d.py` avec les 5 fonctions
- [ ] Fichier `tests/test_cell_0d.py` avec validation
- [ ] Notebook `notebooks/phase1_validation.ipynb` avec comparaison analytique

---

## Phase 2 : Modèle 1D - Grille Sans Transport

**Objectif** : Vectoriser pour plusieurs cellules indépendantes (pas d'interaction)

### Évolution du Modèle

Maintenant `B` est un vecteur : `B[i]` pour chaque cellule `i`

```
dB[i]/dt = R[i] - λ[i] * B[i]
```

Chaque cellule évolue **indépendamment** (idéal pour parallélisation).

### Fonctions Unitaires (Vectorisées)

#### 2.1 - `compute_recruitment_grid`
```python
import jax.numpy as jnp

def compute_recruitment_grid(params: dict, grid_size: int) -> jnp.ndarray:
    """
    Recrutement pour toute la grille.

    Args:
        params: {'R': float ou array}
        grid_size: nombre de cellules

    Returns:
        R: array shape (grid_size,)
    """
    R = params['R']
    if isinstance(R, (int, float)):
        return jnp.full(grid_size, R)
    else:
        return jnp.array(R)
```

#### 2.2 - `compute_mortality_grid`
```python
def compute_mortality_grid(biomass: jnp.ndarray, params: dict) -> jnp.ndarray:
    """
    Mortalité pour toute la grille.

    Args:
        biomass: array shape (grid_size,)
        params: {'lambda': float ou array}

    Returns:
        mortality: λ[i] * B[i] pour chaque cellule
    """
    lam = params['lambda']
    if isinstance(lam, (int, float)):
        return lam * biomass
    else:
        return jnp.array(lam) * biomass
```

#### 2.3 - `euler_step_grid`
```python
def euler_step_grid(biomass: jnp.ndarray, dt: float, params: dict) -> jnp.ndarray:
    """
    Intégration Euler pour toute la grille (vectorisé).

    Args:
        biomass: array shape (grid_size,)
        dt: pas de temps
        params: paramètres

    Returns:
        biomass_new: array shape (grid_size,)
    """
    R = compute_recruitment_grid(params, len(biomass))
    mortality = compute_mortality_grid(biomass, params)
    dB_dt = R - mortality

    return biomass + dt * dB_dt
```

### Version JAX Compilée

```python
from jax import jit

@jit
def euler_step_grid_jit(biomass: jnp.ndarray, dt: float,
                        R: jnp.ndarray, lam: jnp.ndarray) -> jnp.ndarray:
    """
    Version JIT-compilée pour performance.

    Note: params décomposés en R et lam pour compatibilité JIT.
    """
    mortality = lam * biomass
    dB_dt = R - mortality
    return biomass + dt * dB_dt
```

### Kernel d'Exécution

```python
def run_grid_timestep(state: dict, dt: float, params: dict) -> dict:
    """
    Exécute un pas de temps pour toute la grille.

    Ordre d'exécution:
    1. compute_recruitment_grid
    2. compute_mortality_grid
    3. euler_step_grid

    Args:
        state: {'biomass': jnp.ndarray, 't': float}
        dt: pas de temps
        params: paramètres

    Returns:
        new_state: état mis à jour
    """
    biomass_new = euler_step_grid(state['biomass'], dt, params)

    return {
        'biomass': biomass_new,
        't': state['t'] + dt
    }
```

### Validation

**Test 1** : Chaque cellule converge vers son équilibre
```python
# Grille de 10 cellules
grid_size = 10
params = {
    'R': jnp.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50]),  # variable
    'lambda': 0.1  # constant
}

state = {
    'biomass': jnp.zeros(grid_size),  # B0 = 0 partout
    't': 0.0
}

# Simulation 100 jours
for _ in range(1000):
    state = run_grid_timestep(state, dt=0.1, params)

# Vérifier équilibres : B_eq[i] = R[i] / lambda
B_eq_expected = params['R'] / params['lambda']
np.testing.assert_allclose(state['biomass'], B_eq_expected, rtol=0.01)
```

**Test 2** : Benchmark JAX vs NumPy
```python
import time

# NumPy version
def euler_numpy(B, dt, R, lam):
    return B + dt * (R - lam * B)

# JAX version
euler_jax = jit(lambda B, dt, R, lam: B + dt * (R - lam * B))

# Warmup JAX
_ = euler_jax(jnp.zeros(100), 0.1, jnp.ones(100), 0.1)

# Benchmark
grid_size = 10000
B = np.random.rand(grid_size)
R, lam = np.ones(grid_size), 0.1

# NumPy
t0 = time.time()
for _ in range(100):
    B = euler_numpy(B, 0.1, R, lam)
t_numpy = time.time() - t0

# JAX
B_jax = jnp.array(B)
t0 = time.time()
for _ in range(100):
    B_jax = euler_jax(B_jax, 0.1, jnp.array(R), lam)
    B_jax.block_until_ready()  # Attendre compilation/exécution
t_jax = time.time() - t0

print(f"Speedup JAX: {t_numpy / t_jax:.2f}x")
```

### Livrables Phase 2

- [ ] Fichier `kernels/cell_1d.py` avec fonctions vectorisées
- [ ] Fichier `kernels/cell_1d_jax.py` avec versions JIT
- [ ] Fichier `tests/test_cell_1d.py` avec validation
- [ ] Notebook `notebooks/phase2_benchmark.ipynb` avec comparaison JAX/NumPy

---

## Phase 3 : Ajout Transport Simple (Diffusion)

**Objectif** : Introduire interactions spatiales via diffusion

### Modèle Étendu

```
dB/dt = R - λB + D * ∂²B/∂x²
```

Où :
- **D** : coefficient de diffusion (m²/jour)
- **∂²B/∂x²** : laplacien (différences finies)

### Fonctions Unitaires

#### 3.1 - `compute_laplacian_1d`
```python
def compute_laplacian_1d(biomass: jnp.ndarray, dx: float) -> jnp.ndarray:
    """
    Calcule le laplacien par différences finies centrées.

    Schéma : ∂²B/∂x²[i] ≈ (B[i-1] - 2*B[i] + B[i+1]) / dx²

    Conditions aux limites : Neumann (flux nul, ∂B/∂x = 0)

    Args:
        biomass: array shape (N,)
        dx: espacement spatial (m)

    Returns:
        laplacian: array shape (N,)
    """
    # Padding pour conditions aux limites (réflexion)
    B_padded = jnp.pad(biomass, pad_width=1, mode='edge')

    # Différences finies
    laplacian = (B_padded[:-2] - 2 * B_padded[1:-1] + B_padded[2:]) / (dx**2)

    return laplacian
```

#### 3.2 - `compute_diffusion`
```python
def compute_diffusion(biomass: jnp.ndarray, params: dict) -> jnp.ndarray:
    """
    Calcule le terme de diffusion D * ∂²B/∂x².

    Args:
        biomass: array shape (N,)
        params: {'D': float, 'dx': float}

    Returns:
        diffusion: array shape (N,)
    """
    laplacian = compute_laplacian_1d(biomass, params['dx'])
    return params['D'] * laplacian
```

#### 3.3 - `euler_step_grid_with_diffusion`
```python
@jit
def euler_step_with_diffusion(biomass: jnp.ndarray, dt: float,
                               R: jnp.ndarray, lam: float,
                               D: float, dx: float) -> jnp.ndarray:
    """
    Intégration Euler avec réaction + diffusion.

    dB/dt = R - λB + D * ∂²B/∂x²

    Args:
        biomass: array shape (N,)
        dt: pas de temps (jours)
        R: recrutement, shape (N,)
        lam: taux de mortalité
        D: coefficient de diffusion (m²/jour)
        dx: espacement spatial (m)

    Returns:
        biomass_new: array shape (N,)
    """
    # Réaction (local)
    reaction = R - lam * biomass

    # Diffusion (couplage spatial)
    B_padded = jnp.pad(biomass, pad_width=1, mode='edge')
    laplacian = (B_padded[:-2] - 2*B_padded[1:-1] + B_padded[2:]) / (dx**2)
    diffusion = D * laplacian

    # Total
    dB_dt = reaction + diffusion

    return biomass + dt * dB_dt
```

### Stabilité Numérique

**Critère CFL pour diffusion** :

```
dt ≤ dx² / (2D)
```

Si `dx = 1000 m` et `D = 100 m²/jour` :
```
dt_max = 1000² / (2 * 100) = 5000 jours  # OK, très stable
```

### Kernel d'Exécution

```python
def run_grid_timestep_diffusion(state: dict, dt: float, params: dict) -> dict:
    """
    Exécute un pas de temps avec réaction + diffusion.

    Ordre d'exécution:
    1. compute_recruitment_grid (local)
    2. compute_mortality_grid (local)
    3. compute_diffusion (couplage spatial)
    4. euler_step_with_diffusion

    Args:
        state: {'biomass': jnp.ndarray, 't': float}
        dt: pas de temps
        params: {'R', 'lambda', 'D', 'dx'}

    Returns:
        new_state: état mis à jour
    """
    R = compute_recruitment_grid(params, len(state['biomass']))

    biomass_new = euler_step_with_diffusion(
        state['biomass'],
        dt,
        R,
        params['lambda'],
        params['D'],
        params['dx']
    )

    return {
        'biomass': biomass_new,
        't': state['t'] + dt
    }
```

### Validation

**Test 1** : Diffusion d'un pic gaussien
```python
import matplotlib.pyplot as plt

# Condition initiale : pic au centre
grid_size = 100
x = jnp.linspace(0, 10000, grid_size)  # 10 km
dx = x[1] - x[0]

# Gaussienne centrée
x0 = 5000
sigma = 500
B0 = jnp.exp(-((x - x0)**2) / (2 * sigma**2))

params = {
    'R': jnp.zeros(grid_size),  # pas de recrutement
    'lambda': 0.0,  # pas de mortalité
    'D': 100.0,  # m²/jour
    'dx': dx
}

state = {'biomass': B0, 't': 0.0}

# Simulation 10 jours
biomass_history = [state['biomass']]
for _ in range(100):
    state = run_grid_timestep_diffusion(state, dt=0.1, params)
    biomass_history.append(state['biomass'])

# Visualiser élargissement du pic
plt.plot(x, B0, label='t=0')
plt.plot(x, biomass_history[50], label='t=5 jours')
plt.plot(x, biomass_history[-1], label='t=10 jours')
plt.legend()
plt.xlabel('Position (m)')
plt.ylabel('Biomasse')
plt.title('Diffusion d\'un pic gaussien')
```

**Test 2** : Conservation de la masse
```python
# Conditions aux limites Neumann → masse totale conservée
mass_initial = jnp.sum(B0) * dx
mass_final = jnp.sum(state['biomass']) * dx

np.testing.assert_allclose(mass_final, mass_initial, rtol=0.01)
```

### Livrables Phase 3

- [ ] Fichier `kernels/diffusion.py` avec fonctions diffusion
- [ ] Fichier `tests/test_diffusion.py` avec validation
- [ ] Notebook `notebooks/phase3_diffusion.ipynb` avec visualisations

---

## Phase 4 : Distribution Ray (Workers + Messages)

**Objectif** : Distribuer la grille sur plusieurs workers Ray avec échange de halo

### Architecture

```
Grid: [0 1 2 3 4 5 6 7 8 9]
       └──W1───┘ └──W2───┘
       cells 0-4  cells 5-9
```

Chaque worker gère un sous-domaine et échange les valeurs aux frontières.

### Fonctions Unitaires

#### 4.1 - `split_domain`
```python
def split_domain(grid_size: int, num_workers: int) -> list[tuple[int, int]]:
    """
    Découpe le domaine en sous-domaines pour workers.

    Args:
        grid_size: nombre total de cellules
        num_workers: nombre de workers

    Returns:
        subdomains: [(start, end), ...] indices pour chaque worker
    """
    cells_per_worker = grid_size // num_workers
    subdomains = []

    for i in range(num_workers):
        start = i * cells_per_worker
        end = start + cells_per_worker if i < num_workers - 1 else grid_size
        subdomains.append((start, end))

    return subdomains
```

#### 4.2 - `extract_subdomain`
```python
def extract_subdomain(biomass_full: jnp.ndarray,
                      subdomain: tuple[int, int]) -> jnp.ndarray:
    """
    Extrait un sous-domaine de la grille globale.

    Args:
        biomass_full: grille complète
        subdomain: (start, end) indices

    Returns:
        biomass_sub: sous-grille
    """
    start, end = subdomain
    return biomass_full[start:end]
```

#### 4.3 - Ray Worker Actor

```python
import ray

@ray.remote(num_gpus=0.25)  # Partage GPU entre workers
class GridWorker:
    def __init__(self, worker_id: int, subdomain: tuple[int, int],
                 params: dict, neighbors: dict):
        """
        Worker gérant un sous-domaine.

        Args:
            worker_id: identifiant unique
            subdomain: (start, end) indices du sous-domaine
            params: paramètres du modèle
            neighbors: {'left': worker_id ou None, 'right': worker_id ou None}
        """
        self.id = worker_id
        self.subdomain = subdomain
        self.params = params
        self.neighbors = neighbors

        # État local
        size = subdomain[1] - subdomain[0]
        self.biomass = jnp.zeros(size)
        self.t = 0.0

        # Buffers pour halo
        self.halo_left = None
        self.halo_right = None

        # Compiler kernels JAX
        self.step_kernel = jit(euler_step_with_diffusion)

    def set_biomass(self, biomass: jnp.ndarray):
        """Initialise la biomasse locale."""
        self.biomass = biomass

    def get_boundary_left(self) -> float:
        """Retourne la valeur de la frontière gauche."""
        return float(self.biomass[0])

    def get_boundary_right(self) -> float:
        """Retourne la valeur de la frontière droite."""
        return float(self.biomass[-1])

    def set_halo_left(self, value: float):
        """Reçoit la valeur du halo gauche."""
        self.halo_left = value

    def set_halo_right(self, value: float):
        """Reçoit la valeur du halo droite."""
        self.halo_right = value

    async def step(self, dt: float):
        """
        Exécute un pas de temps avec échange de halo.

        Ordre d'exécution:
        1. Demander valeurs frontières voisins (messages)
        2. Attendre réception (await)
        3. Calculer diffusion avec halo
        4. Mettre à jour biomasse locale
        """
        # 1. Demander frontières aux voisins (non-bloquant)
        left_future = None
        right_future = None

        if self.neighbors['left'] is not None:
            left_future = self.neighbors['left'].get_boundary_right.remote()

        if self.neighbors['right'] is not None:
            right_future = self.neighbors['right'].get_boundary_left.remote()

        # 2. Attendre réception (bloquant)
        if left_future is not None:
            self.halo_left = await left_future
        else:
            self.halo_left = self.biomass[0]  # Neumann BC

        if right_future is not None:
            self.halo_right = await right_future
        else:
            self.halo_right = self.biomass[-1]  # Neumann BC

        # 3. Calcul avec halo
        biomass_with_halo = jnp.concatenate([
            jnp.array([self.halo_left]),
            self.biomass,
            jnp.array([self.halo_right])
        ])

        # Diffusion sur domaine étendu
        R = jnp.full(len(self.biomass), self.params['R'])
        biomass_new = self.step_kernel(
            biomass_with_halo[1:-1],  # domaine sans halo
            dt,
            R,
            self.params['lambda'],
            self.params['D'],
            self.params['dx']
        )

        # 4. Mise à jour
        self.biomass = biomass_new
        self.t += dt

        return self.biomass
```

#### 4.4 - Scheduler

```python
@ray.remote
class Scheduler:
    def __init__(self, workers: list, params: dict):
        """
        Orchestrateur des workers.

        Args:
            workers: liste des WorkerActors Ray
            params: paramètres de simulation
        """
        self.workers = workers
        self.params = params
        self.t = 0.0

    async def run_simulation(self, t_end: float, dt: float):
        """
        Exécute la simulation jusqu'à t_end.

        Args:
            t_end: temps final (jours)
            dt: pas de temps (jours)
        """
        num_steps = int(t_end / dt)

        for step in range(num_steps):
            # Lancer tous les workers en parallèle
            futures = [w.step.remote(dt) for w in self.workers]

            # Attendre que tous finissent (synchronisation)
            results = await asyncio.gather(*futures)

            self.t += dt

            if step % 10 == 0:
                print(f"Step {step}/{num_steps}, t={self.t:.1f}")

        return results
```

### Initialisation

```python
def setup_distributed_simulation(grid_size: int, num_workers: int,
                                  params: dict) -> tuple:
    """
    Configure la simulation distribuée.

    Returns:
        workers: liste d'acteurs Ray
        scheduler: acteur Scheduler
    """
    # Découper domaine
    subdomains = split_domain(grid_size, num_workers)

    # Créer workers
    workers = []
    for i, subdomain in enumerate(subdomains):
        # Définir voisins
        neighbors = {
            'left': None if i == 0 else i - 1,
            'right': None if i == num_workers - 1 else i + 1
        }

        worker = GridWorker.remote(i, subdomain, params, neighbors)
        workers.append(worker)

    # Résoudre références voisins
    for i, worker in enumerate(workers):
        neighbors_refs = {}
        if i > 0:
            neighbors_refs['left'] = workers[i-1]
        if i < len(workers) - 1:
            neighbors_refs['right'] = workers[i+1]

        # Mettre à jour (via méthode Ray)
        ray.get(worker.update_neighbors.remote(neighbors_refs))

    # Créer scheduler
    scheduler = Scheduler.remote(workers, params)

    return workers, scheduler
```

### Validation

**Test 1** : Cohérence avec version non-distribuée
```python
# Version séquentielle
state_seq = run_sequential_simulation(grid_size=100, t_end=10, dt=0.1)

# Version distribuée (4 workers)
workers, scheduler = setup_distributed_simulation(100, 4, params)
state_dist = ray.get(scheduler.run_simulation.remote(t_end=10, dt=0.1))

# Reconstruire grille globale
biomass_dist = jnp.concatenate([ray.get(w.biomass) for w in workers])

# Comparer
np.testing.assert_allclose(state_seq['biomass'], biomass_dist, rtol=0.01)
```

**Test 2** : Scalabilité
```python
import time

for num_workers in [1, 2, 4, 8]:
    workers, scheduler = setup_distributed_simulation(1000, num_workers, params)

    t0 = time.time()
    ray.get(scheduler.run_simulation.remote(t_end=10, dt=0.1))
    elapsed = time.time() - t0

    print(f"Workers: {num_workers}, Time: {elapsed:.2f}s")
```

### Livrables Phase 4

- [ ] Fichier `distributed/worker.py` avec GridWorker
- [ ] Fichier `distributed/scheduler.py` avec Scheduler
- [ ] Fichier `distributed/setup.py` avec fonctions initialisation
- [ ] Fichier `tests/test_distributed.py` avec validation
- [ ] Notebook `notebooks/phase4_scalability.ipynb` avec benchmarks

---

## Phase 5 : Forçages Environnementaux

**Objectif** : Rendre les paramètres dépendants de l'environnement (température)

### Modèle Étendu

```
dB/dt = R(T) - λ(T) * B + D * ∂²B/∂x²
```

Où `T` = température (°C)

#### Exemple de Dépendances

**Recrutement** (augmente avec température) :
```
R(T) = R_max * exp(-(T - T_opt)² / (2σ²))
```

**Mortalité** (augmente avec température) :
```
λ(T) = λ_0 * exp(k * T)
```

### Fonctions Unitaires

#### 5.1 - `load_forcing_data`
```python
import xarray as xr

def load_forcing_data(filepath: str, variable: str) -> xr.DataArray:
    """
    Charge les données de forçage depuis Zarr.

    Args:
        filepath: chemin vers le store Zarr
        variable: nom de la variable ('temperature', 'current_u', etc.)

    Returns:
        data: xarray.DataArray avec dimensions (time, lat, lon)
    """
    ds = xr.open_zarr(filepath)
    return ds[variable]
```

#### 5.2 - `interpolate_forcing`
```python
def interpolate_forcing(data: xr.DataArray, t: float,
                        positions: jnp.ndarray) -> jnp.ndarray:
    """
    Interpole les forçages spatio-temporels.

    Args:
        data: xarray.DataArray (time, lat, lon)
        t: temps actuel (jours depuis début)
        positions: array shape (N, 2) de coordonnées (lat, lon)

    Returns:
        values: array shape (N,) valeurs interpolées
    """
    # Interpolation temporelle
    data_t = data.interp(time=t, method='linear')

    # Interpolation spatiale
    lats, lons = positions[:, 0], positions[:, 1]
    values = data_t.interp(lat=lats, lon=lons, method='linear')

    return jnp.array(values.values)
```

#### 5.3 - `compute_recruitment_temperature`
```python
@jit
def compute_recruitment_temperature(temperature: jnp.ndarray,
                                    params: dict) -> jnp.ndarray:
    """
    Recrutement dépendant de la température.

    R(T) = R_max * exp(-(T - T_opt)² / (2σ²))

    Args:
        temperature: array shape (N,) en °C
        params: {'R_max', 'T_opt', 'sigma'}

    Returns:
        R: array shape (N,)
    """
    T = temperature
    T_opt = params['T_opt']
    sigma = params['sigma']
    R_max = params['R_max']

    return R_max * jnp.exp(-((T - T_opt)**2) / (2 * sigma**2))
```

#### 5.4 - `compute_mortality_temperature`
```python
@jit
def compute_mortality_temperature(biomass: jnp.ndarray,
                                  temperature: jnp.ndarray,
                                  params: dict) -> jnp.ndarray:
    """
    Mortalité dépendant de la température.

    λ(T) = λ_0 * exp(k * T)
    mortality = λ(T) * B

    Args:
        biomass: array shape (N,)
        temperature: array shape (N,) en °C
        params: {'lambda_0', 'k'}

    Returns:
        mortality: array shape (N,)
    """
    lambda_T = params['lambda_0'] * jnp.exp(params['k'] * temperature)
    return lambda_T * biomass
```

### Worker Modifié

```python
@ray.remote(num_gpus=0.25)
class GridWorkerWithForcing:
    def __init__(self, worker_id: int, subdomain: tuple[int, int],
                 params: dict, forcing_data: xr.DataArray,
                 positions: jnp.ndarray, neighbors: dict):
        """
        Worker avec forçages environnementaux.

        Args:
            forcing_data: xarray.DataArray de température
            positions: coordonnées (lat, lon) des cellules
        """
        self.id = worker_id
        self.subdomain = subdomain
        self.params = params
        self.forcing_data = forcing_data
        self.positions = positions
        self.neighbors = neighbors

        # État
        size = subdomain[1] - subdomain[0]
        self.biomass = jnp.zeros(size)
        self.temperature = jnp.zeros(size)
        self.t = 0.0

        # Kernels
        self.recruitment_fn = jit(compute_recruitment_temperature)
        self.mortality_fn = jit(compute_mortality_temperature)
        self.diffusion_fn = jit(euler_step_with_diffusion)

    def update_temperature(self, t: float):
        """Met à jour la température locale depuis les forçages."""
        self.temperature = interpolate_forcing(
            self.forcing_data,
            t,
            self.positions
        )

    async def step(self, dt: float):
        """
        Pas de temps avec forçages.

        Ordre d'exécution:
        1. update_temperature (lecture forçages)
        2. compute_recruitment_temperature
        3. compute_mortality_temperature
        4. Échange halo (messages)
        5. Diffusion
        """
        # 1. Forçages
        self.update_temperature(self.t)

        # 2-3. Réaction locale
        R = self.recruitment_fn(self.temperature, self.params)
        mortality = self.mortality_fn(self.biomass, self.temperature, self.params)

        # 4. Halo (comme Phase 4)
        if self.neighbors['left'] is not None:
            halo_left = await self.neighbors['left'].get_boundary_right.remote()
        else:
            halo_left = self.biomass[0]

        if self.neighbors['right'] is not None:
            halo_right = await self.neighbors['right'].get_boundary_left.remote()
        else:
            halo_right = self.biomass[-1]

        # 5. Diffusion (à implémenter avec R et mortality modifiés)
        # ... (similaire Phase 4 mais avec R(T) et λ(T))

        self.t += dt
        return self.biomass
```

### Validation

**Test 1** : Gradient thermique spatial
```python
# Température : 20°C à gauche, 28°C à droite
# Recrutement maximal au centre (T_opt = 24°C)

T_grid = jnp.linspace(20, 28, 100)
params_forcing = {
    'R_max': 10.0,
    'T_opt': 24.0,
    'sigma': 2.0,
    'lambda_0': 0.01,
    'k': 0.05
}

# Simulation 50 jours
# Vérifier : pic de biomasse vers x=50 (T=24°C)
```

**Test 2** : Cycle saisonnier
```python
# Température oscillante : T(t) = 24 + 4 * sin(2π * t / 365)
# Biomasse doit osciller aussi avec décalage de phase
```

### Livrables Phase 5

- [ ] Fichier `forcing/loader.py` avec chargement Zarr
- [ ] Fichier `forcing/interpolation.py` avec interpolation spatio-temporelle
- [ ] Fichier `kernels/temperature_dependent.py` avec R(T) et λ(T)
- [ ] Fichier `distributed/worker_forcing.py` avec GridWorkerWithForcing
- [ ] Fichier `tests/test_forcing.py` avec validation
- [ ] Notebook `notebooks/phase5_forcing.ipynb` avec visualisations

---

## Structure du Projet

```
seapopym-message/
├── src/
│   ├── kernels/
│   │   ├── __init__.py
│   │   ├── cell_0d.py          # Phase 1
│   │   ├── cell_1d.py          # Phase 2
│   │   ├── cell_1d_jax.py      # Phase 2
│   │   ├── diffusion.py        # Phase 3
│   │   └── temperature_dependent.py  # Phase 5
│   ├── distributed/
│   │   ├── __init__.py
│   │   ├── worker.py           # Phase 4
│   │   ├── scheduler.py        # Phase 4
│   │   └── worker_forcing.py   # Phase 5
│   ├── forcing/
│   │   ├── __init__.py
│   │   ├── loader.py           # Phase 5
│   │   └── interpolation.py    # Phase 5
│   └── utils/
│       ├── __init__.py
│       └── domain.py           # split_domain, etc.
├── tests/
│   ├── test_cell_0d.py
│   ├── test_cell_1d.py
│   ├── test_diffusion.py
│   ├── test_distributed.py
│   └── test_forcing.py
├── notebooks/
│   ├── phase1_validation.ipynb
│   ├── phase2_benchmark.ipynb
│   ├── phase3_diffusion.ipynb
│   ├── phase4_scalability.ipynb
│   └── phase5_forcing.ipynb
├── IA/
│   ├── jax-cfd-integration.md
│   └── prototype-plan.md       # ce fichier
├── pyproject.toml
├── README.md
└── requirements.txt
```

---

## Ordre d'Exécution Récapitulatif

### Phase 1 (0D)
```
1. compute_recruitment()
2. compute_mortality()
3. compute_derivative()
4. euler_step()
```

### Phase 2 (1D sans transport)
```
1. compute_recruitment_grid()
2. compute_mortality_grid()
3. euler_step_grid()  # vectorisé JAX
```

### Phase 3 (1D + diffusion)
```
1. compute_recruitment_grid()
2. compute_mortality_grid()
3. compute_laplacian_1d()
4. compute_diffusion()
5. euler_step_with_diffusion()
```

### Phase 4 (Distribution Ray)
```
Pour chaque worker en parallèle:
  1. Demander frontières voisins (messages Ray)
  2. Attendre réception (await)
  3. compute_recruitment_grid()
  4. compute_mortality_grid()
  5. compute_diffusion() avec halo
  6. euler_step_with_diffusion()

Scheduler synchronise tous les workers
```

### Phase 5 (Forçages)
```
Pour chaque worker en parallèle:
  1. update_temperature() depuis Zarr
  2. compute_recruitment_temperature(T)
  3. compute_mortality_temperature(T)
  4. Échange halo (messages)
  5. compute_diffusion()
  6. euler_step_with_diffusion()
```

---

## Métriques de Validation

### Phase 1
- [ ] Convergence vers B_eq = R/λ (erreur < 1%)
- [ ] Comparaison solution analytique (erreur < 0.1%)

### Phase 2
- [ ] Chaque cellule atteint son équilibre
- [ ] Speedup JAX > 5x vs NumPy (grille 10000)

### Phase 3
- [ ] Conservation de la masse (erreur < 1%)
- [ ] Élargissement gaussien conforme à théorie diffusion

### Phase 4
- [ ] Cohérence distribuée vs séquentielle (erreur < 1%)
- [ ] Scalabilité quasi-linéaire (efficacité > 80% jusqu'à 8 workers)

### Phase 5
- [ ] Pic de biomasse aligné avec T_opt
- [ ] Oscillation saisonnière de biomasse détectée

---

## Prochaines Étapes Suggérées

Après Phase 5, extensions possibles :

1. **Advection** (JAX-CFD)
   - Ajouter champ de vitesse U(x,t)
   - Schéma upwind ou semi-lagrangien

2. **Prédation** (couplage multi-espèces)
   - 2 compartiments : proie B₁, prédateur B₂
   - Terme de Holling type II

3. **Transport Lagrangien**
   - Particules individuelles au lieu de grille
   - Advection par forçages de courants

4. **Optimisation**
   - Calibration automatique (λ, R) via gradient JAX
   - Assimilation de données (observations satellite)

5. **GPU Massif**
   - Passer de CPU à GPU pour tous les workers
   - Tester sur cluster avec 100+ GPUs

---

## Estimation Temps de Développement

| Phase | Durée estimée | Priorité |
|-------|---------------|----------|
| Phase 1 | 1-2 jours | Critique |
| Phase 2 | 2-3 jours | Critique |
| Phase 3 | 3-4 jours | Haute |
| Phase 4 | 5-7 jours | Haute |
| Phase 5 | 4-5 jours | Moyenne |
| **Total** | **15-21 jours** | |

---

## Dépendances Logicielles

```python
# requirements.txt
jax[cpu]==0.4.20  # ou jax[cuda] pour GPU
ray[default]==2.9.0
xarray==2023.12.0
zarr==2.16.1
numpy==1.26.2
matplotlib==3.8.2
pytest==7.4.3
jupyter==1.0.0
```

---

## Conclusion

Ce plan propose une progression **incrémentale et testable** :

1. ✅ **Simplicité d'abord** : équation dB/dt = R - λB
2. ✅ **Vectorisation** : JAX pour performance
3. ✅ **Couplage spatial** : diffusion puis advection
4. ✅ **Distribution** : Ray pour scalabilité
5. ✅ **Réalisme** : forçages environnementaux

Chaque phase est **validable indépendamment** avec des tests unitaires et des notebooks de visualisation.

**Prêt à démarrer Phase 1 ?**
