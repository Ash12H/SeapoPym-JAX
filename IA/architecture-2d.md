# Architecture 2D : Grille Latitude × Longitude

## Adaptation pour la Géométrie 2D

### Changements Principaux par Rapport à 1D

| Aspect | 1D | 2D |
|--------|----|----|
| **État** | `biomass: (N,)` | `biomass: (nlat, nlon)` |
| **Voisins** | 2 (left, right) | 4 ou 8 (N, S, E, W + diagonales) |
| **Halo** | 2 valeurs scalaires | 4 arrays 1D (ou 8 si diagonales) |
| **Transport** | `∂²/∂x²` | `∂²/∂x² + ∂²/∂y²` |
| **Découpage** | Linéaire | Rectangulaire (patches) |

---

## 1. Structure des Données 2D

### État d'un Worker

```python
# Au lieu de :
state = {
    'biomass': jnp.array([...])  # shape (N,)
}

# On a maintenant :
state = {
    'biomass': jnp.array([[...], [...], ...])  # shape (nlat, nlon)
    'temperature': jnp.array([[...], [...], ...])  # shape (nlat, nlon)
}
```

### Coordonnées Géographiques

```python
@dataclass
class GridInfo:
    """Information sur la grille 2D."""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    nlat: int
    nlon: int

    @property
    def lat_coords(self) -> jnp.ndarray:
        """Coordonnées latitude (centres de cellules)."""
        return jnp.linspace(self.lat_min, self.lat_max, self.nlat)

    @property
    def lon_coords(self) -> jnp.ndarray:
        """Coordonnées longitude (centres de cellules)."""
        return jnp.linspace(self.lon_min, self.lon_max, self.nlon)

    @property
    def dlat(self) -> float:
        """Résolution latitudinale (degrés)."""
        return (self.lat_max - self.lat_min) / (self.nlat - 1)

    @property
    def dlon(self) -> float:
        """Résolution longitudinale (degrés)."""
        return (self.lon_max - self.lon_min) / (self.nlon - 1)

    @property
    def meshgrid(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Grille 2D des coordonnées."""
        return jnp.meshgrid(self.lon_coords, self.lat_coords)
```

---

## 2. Découpage du Domaine 2D en Patches

### Stratégie de Partitionnement

```
Grille globale : 120 lat × 180 lon
Nombre de workers : 6 (2×3)

Découpage en patches rectangulaires :

┌─────────┬─────────┬─────────┐
│ W0      │ W1      │ W2      │  ← 60 lat
│ 60×60   │ 60×60   │ 60×60   │
├─────────┼─────────┼─────────┤
│ W3      │ W4      │ W5      │  ← 60 lat
│ 60×60   │ 60×60   │ 60×60   │
└─────────┴─────────┴─────────┘
  60 lon   60 lon   60 lon
```

### Fonction de Découpage

```python
from typing import List, Tuple
import numpy as np

def split_domain_2d(nlat: int, nlon: int,
                    num_workers_lat: int, num_workers_lon: int) -> List[dict]:
    """
    Découpe le domaine 2D en patches rectangulaires.

    Args:
        nlat: nombre total de cellules en latitude
        nlon: nombre total de cellules en longitude
        num_workers_lat: nombre de divisions en latitude
        num_workers_lon: nombre de divisions en longitude

    Returns:
        patches: liste de dictionnaires avec info de chaque patch
    """
    # Taille de chaque patch
    patch_nlat = nlat // num_workers_lat
    patch_nlon = nlon // num_workers_lon

    patches = []
    worker_id = 0

    for i_lat in range(num_workers_lat):
        for i_lon in range(num_workers_lon):
            # Indices de début/fin
            lat_start = i_lat * patch_nlat
            lat_end = lat_start + patch_nlat if i_lat < num_workers_lat - 1 else nlat

            lon_start = i_lon * patch_nlon
            lon_end = lon_start + patch_nlon if i_lon < num_workers_lon - 1 else nlon

            # Identifiants des voisins
            neighbors = {
                'north': worker_id - num_workers_lon if i_lat > 0 else None,
                'south': worker_id + num_workers_lon if i_lat < num_workers_lat - 1 else None,
                'west': worker_id - 1 if i_lon > 0 else None,
                'east': worker_id + 1 if i_lon < num_workers_lon - 1 else None,
            }

            patches.append({
                'worker_id': worker_id,
                'lat_slice': (lat_start, lat_end),
                'lon_slice': (lon_start, lon_end),
                'i_lat': i_lat,
                'i_lon': i_lon,
                'neighbors': neighbors
            })

            worker_id += 1

    return patches
```

---

## 3. Worker 2D avec Gestion des Voisins

### Worker Ray Actor Adapté

```python
import ray
import jax.numpy as jnp
from typing import Dict, Optional

@ray.remote
class CellWorker2D:
    """
    Worker gérant un patch 2D de la grille.

    Architecture:
    - Chaque worker gère une sous-grille (nlat_local, nlon_local)
    - 4 voisins potentiels : N, S, E, W
    - Halo exchange pour les Units globales
    """

    def __init__(self,
                 worker_id: int,
                 grid_info: GridInfo,
                 lat_slice: Tuple[int, int],
                 lon_slice: Tuple[int, int],
                 kernel: 'Kernel',
                 params: dict):
        """
        Args:
            worker_id: identifiant unique
            grid_info: information sur la grille globale
            lat_slice: (start, end) indices latitude
            lon_slice: (start, end) indices longitude
            kernel: Kernel à exécuter
            params: paramètres du modèle
        """
        self.id = worker_id
        self.grid_info = grid_info
        self.lat_slice = lat_slice
        self.lon_slice = lon_slice
        self.kernel = kernel
        self.params = params

        # Taille locale
        self.nlat_local = lat_slice[1] - lat_slice[0]
        self.nlon_local = lon_slice[1] - lon_slice[0]

        # État local (grilles 2D)
        self.state = self._initialize_state()
        self.t = 0.0

        # Voisins (références Ray)
        self.neighbors = {
            'north': None,
            'south': None,
            'east': None,
            'west': None
        }

    def _initialize_state(self) -> dict:
        """Initialise l'état local avec des grilles 2D."""
        return {
            'biomass': jnp.zeros((self.nlat_local, self.nlon_local)),
            'temperature': jnp.ones((self.nlat_local, self.nlon_local)) * 24.0,
            # Coordonnées du patch
            'lat': self.grid_info.lat_coords[self.lat_slice[0]:self.lat_slice[1]],
            'lon': self.grid_info.lon_coords[self.lon_slice[0]:self.lon_slice[1]],
        }

    def set_neighbors(self, neighbors: Dict[str, 'CellWorker2D']):
        """Configure les voisins du worker."""
        self.neighbors = neighbors

    def get_boundary_north(self) -> dict:
        """Retourne la frontière nord (première ligne)."""
        return {
            'biomass': self.state['biomass'][0, :],  # shape (nlon_local,)
            'temperature': self.state['temperature'][0, :],
        }

    def get_boundary_south(self) -> dict:
        """Retourne la frontière sud (dernière ligne)."""
        return {
            'biomass': self.state['biomass'][-1, :],  # shape (nlon_local,)
            'temperature': self.state['temperature'][-1, :],
        }

    def get_boundary_east(self) -> dict:
        """Retourne la frontière est (dernière colonne)."""
        return {
            'biomass': self.state['biomass'][:, -1],  # shape (nlat_local,)
            'temperature': self.state['temperature'][:, -1],
        }

    def get_boundary_west(self) -> dict:
        """Retourne la frontière ouest (première colonne)."""
        return {
            'biomass': self.state['biomass'][:, 0],  # shape (nlat_local,)
            'temperature': self.state['temperature'][:, 0],
        }

    async def step(self, dt: float) -> dict:
        """
        Exécute un pas de temps complet.

        Étapes:
        1. Phase locale (calculs indépendants sur la grille 2D)
        2. Si Units globales: échange halo avec les 4 voisins
        3. Phase globale (transport 2D avec halo)
        """
        # ═══════════════════════════════════════════════════
        # PHASE 1: CALCULS LOCAUX (parallélisables)
        # ═══════════════════════════════════════════════════
        self.state = self.kernel.execute_local_phase(
            self.state,
            dt,
            self.params
        )

        # ═══════════════════════════════════════════════════
        # PHASE 2: ÉCHANGE HALO (si nécessaire)
        # ═══════════════════════════════════════════════════
        if self.kernel.has_global_units():
            # Demander frontières aux 4 voisins (non-bloquant)
            halo_futures = {}

            if self.neighbors['north'] is not None:
                halo_futures['north'] = self.neighbors['north'].get_boundary_south.remote()

            if self.neighbors['south'] is not None:
                halo_futures['south'] = self.neighbors['south'].get_boundary_north.remote()

            if self.neighbors['east'] is not None:
                halo_futures['east'] = self.neighbors['east'].get_boundary_west.remote()

            if self.neighbors['west'] is not None:
                halo_futures['west'] = self.neighbors['west'].get_boundary_east.remote()

            # Attendre réception (bloquant)
            halo_data = {}
            for direction, future in halo_futures.items():
                halo_data[f'halo_{direction}'] = await future

            # ═══════════════════════════════════════════════════
            # PHASE 3: CALCULS GLOBAUX (avec halo)
            # ═══════════════════════════════════════════════════
            self.state = self.kernel.execute_global_phase(
                self.state,
                dt,
                self.params,
                halo_data
            )

        self.t += dt
        return {'state': self.state, 't': self.t}
```

---

## 4. Units 2D (Exemples)

### Unit Locale : Mortalité

```python
from src.kernels.base import unit
import jax.numpy as jnp
from jax import jit

@unit(name='mortality_2d',
      inputs=['biomass', 'temperature'],
      outputs=['mortality_rate'],
      scope='local',
      compiled=True)
def compute_mortality_2d(biomass, temperature, params):
    """
    Mortalité dépendant de la température (2D).

    Args:
        biomass: shape (nlat, nlon)
        temperature: shape (nlat, nlon)
        params: {'lambda_0': float, 'k': float}

    Returns:
        mortality_rate: shape (nlat, nlon)
    """
    lambda_T = params['lambda_0'] * jnp.exp(params['k'] * temperature)
    return lambda_T * biomass
```

### Unit Globale : Diffusion 2D

```python
@unit(name='diffusion_2d',
      inputs=['biomass'],
      outputs=['biomass'],
      scope='global',
      compiled=True)
def compute_diffusion_2d(biomass, dt, params,
                         halo_north=None, halo_south=None,
                         halo_east=None, halo_west=None):
    """
    Diffusion 2D avec halo.

    Laplacien : ∂²B/∂x² + ∂²B/∂y²

    Args:
        biomass: shape (nlat, nlon) du patch
        halo_north: dict avec 'biomass' shape (nlon,)
        halo_south: dict avec 'biomass' shape (nlon,)
        halo_east: dict avec 'biomass' shape (nlat,)
        halo_west: dict avec 'biomass' shape (nlat,)

    Returns:
        biomass_new: shape (nlat, nlon)
    """
    nlat, nlon = biomass.shape

    # ═══════════════════════════════════════════════════
    # Construction du domaine étendu avec halo
    # ═══════════════════════════════════════════════════

    # Extraire biomass des halos (ou conditions aux limites)
    if halo_north is not None and 'biomass' in halo_north:
        north_row = halo_north['biomass']  # shape (nlon,)
    else:
        north_row = biomass[0, :]  # Neumann BC

    if halo_south is not None and 'biomass' in halo_south:
        south_row = halo_south['biomass']
    else:
        south_row = biomass[-1, :]

    if halo_west is not None and 'biomass' in halo_west:
        west_col = halo_west['biomass']  # shape (nlat,)
    else:
        west_col = biomass[:, 0]

    if halo_east is not None and 'biomass' in halo_east:
        east_col = halo_east['biomass']
    else:
        east_col = biomass[:, -1]

    # Construire grille étendue (nlat+2, nlon+2)
    # Coins : on peut les négliger ou interpoler

    # Version simplifiée sans coins :
    # On étend seulement les bords N, S, E, W

    # Ajouter lignes N/S
    biomass_extended_lat = jnp.vstack([
        north_row.reshape(1, -1),  # (1, nlon)
        biomass,                    # (nlat, nlon)
        south_row.reshape(1, -1)    # (1, nlon)
    ])  # shape (nlat+2, nlon)

    # Ajouter colonnes W/E
    # On doit étendre west_col et east_col pour inclure les coins
    # Pour simplification, on utilise des valeurs aux bords
    west_col_extended = jnp.concatenate([
        jnp.array([north_row[0]]),   # coin NW
        west_col,                     # bord W
        jnp.array([south_row[0]])     # coin SW
    ])  # shape (nlat+2,)

    east_col_extended = jnp.concatenate([
        jnp.array([north_row[-1]]),  # coin NE
        east_col,                     # bord E
        jnp.array([south_row[-1]])    # coin SE
    ])  # shape (nlat+2,)

    biomass_extended = jnp.column_stack([
        west_col_extended.reshape(-1, 1),  # (nlat+2, 1)
        biomass_extended_lat,               # (nlat+2, nlon)
        east_col_extended.reshape(-1, 1)   # (nlat+2, 1)
    ])  # shape (nlat+2, nlon+2)

    # ═══════════════════════════════════════════════════
    # Calcul du Laplacien 2D
    # ═══════════════════════════════════════════════════

    dlat = params['dlat']  # degrés
    dlon = params['dlon']  # degrés
    D = params['D']        # coefficient de diffusion

    # Convertir degrés en mètres (approximation)
    # À latitude φ : dx ≈ R * cos(φ) * dlon * π/180
    #               dy ≈ R * dlat * π/180
    R_earth = 6371000  # mètres

    # Simplification : on utilise latitude moyenne du patch
    # (pour une version plus précise, il faudrait un facteur par ligne)
    lat_mean = params['lat_mean']
    dx = R_earth * jnp.cos(jnp.radians(lat_mean)) * jnp.radians(dlon)
    dy = R_earth * jnp.radians(dlat)

    # Laplacien par différences finies
    # ∂²B/∂x² ≈ (B[i,j-1] - 2*B[i,j] + B[i,j+1]) / dx²
    # ∂²B/∂y² ≈ (B[i-1,j] - 2*B[i,j] + B[i+1,j]) / dy²

    laplacian_x = (
        biomass_extended[1:-1, :-2] - 2*biomass_extended[1:-1, 1:-1] + biomass_extended[1:-1, 2:]
    ) / (dx**2)

    laplacian_y = (
        biomass_extended[:-2, 1:-1] - 2*biomass_extended[1:-1, 1:-1] + biomass_extended[2:, 1:-1]
    ) / (dy**2)

    laplacian = laplacian_x + laplacian_y  # shape (nlat, nlon)

    # ═══════════════════════════════════════════════════
    # Mise à jour
    # ═══════════════════════════════════════════════════

    biomass_new = biomass + D * laplacian * dt

    return biomass_new
```

### Unit Globale : Advection 2D (avec JAX-CFD)

```python
from jax_cfd.base import advection, grids

@unit(name='advection_2d',
      inputs=['biomass', 'velocity_u', 'velocity_v'],
      outputs=['biomass'],
      scope='global',
      compiled=False)  # JAX-CFD gère sa propre compilation
def compute_advection_2d(biomass, velocity_u, velocity_v, dt, params,
                         halo_north=None, halo_south=None,
                         halo_east=None, halo_west=None):
    """
    Advection 2D avec JAX-CFD.

    Args:
        biomass: shape (nlat, nlon)
        velocity_u: vitesse zonale (m/s), shape (nlat, nlon)
        velocity_v: vitesse méridionale (m/s), shape (nlat, nlon)
        halos: données des voisins

    Returns:
        biomass_new: shape (nlat, nlon)
    """
    # Créer grille JAX-CFD
    nlat, nlon = biomass.shape
    grid = grids.Grid(shape=(nlat, nlon), domain=((0, nlat), (0, nlon)))

    # Construire domaine étendu (comme diffusion)
    biomass_extended = _build_extended_domain(
        biomass, halo_north, halo_south, halo_east, halo_west
    )

    # Champ de vitesse
    velocity = (velocity_u, velocity_v)

    # Advection semi-lagrangienne (JAX-CFD)
    biomass_advected = advection.advect_van_leer(
        c=biomass_extended[1:-1, 1:-1],  # retirer halo pour calcul
        v=velocity,
        dt=dt
    )

    return biomass_advected
```

---

## 5. Initialisation d'une Simulation 2D

### Setup Complet

```python
import ray

def setup_simulation_2d(
    nlat_global: int,
    nlon_global: int,
    num_workers_lat: int,
    num_workers_lon: int,
    kernel: 'Kernel',
    params: dict,
    grid_bounds: dict
) -> tuple:
    """
    Configure une simulation 2D distribuée.

    Args:
        nlat_global: nombre total de cellules en latitude
        nlon_global: nombre total de cellules en longitude
        num_workers_lat: nombre de workers en latitude
        num_workers_lon: nombre de workers en longitude
        kernel: Kernel à exécuter
        params: paramètres du modèle
        grid_bounds: {'lat_min', 'lat_max', 'lon_min', 'lon_max'}

    Returns:
        workers: liste d'acteurs Ray
        grid_info: information sur la grille globale
    """
    # Information sur la grille globale
    grid_info = GridInfo(
        lat_min=grid_bounds['lat_min'],
        lat_max=grid_bounds['lat_max'],
        lon_min=grid_bounds['lon_min'],
        lon_max=grid_bounds['lon_max'],
        nlat=nlat_global,
        nlon=nlon_global
    )

    # Découper le domaine
    patches = split_domain_2d(nlat_global, nlon_global,
                              num_workers_lat, num_workers_lon)

    # Créer les workers
    workers = []
    for patch in patches:
        worker = CellWorker2D.remote(
            worker_id=patch['worker_id'],
            grid_info=grid_info,
            lat_slice=patch['lat_slice'],
            lon_slice=patch['lon_slice'],
            kernel=kernel,
            params=params
        )
        workers.append(worker)

    # Connecter les voisins
    for patch in patches:
        wid = patch['worker_id']
        neighbor_refs = {}

        for direction, neighbor_id in patch['neighbors'].items():
            if neighbor_id is not None:
                neighbor_refs[direction] = workers[neighbor_id]

        ray.get(workers[wid].set_neighbors.remote(neighbor_refs))

    return workers, grid_info


# ════════════════════════════════════════════════════════
# EXEMPLE D'UTILISATION
# ════════════════════════════════════════════════════════

ray.init()

# Définir le modèle
kernel = Kernel([
    compute_recruitment_2d,
    compute_mortality_2d,
    compute_growth_2d,
    compute_diffusion_2d  # Global
])

# Paramètres
params = {
    'R': 10.0,
    'lambda_0': 0.01,
    'k': 0.05,
    'D': 1000.0,  # m²/jour
    'dlat': 0.25,  # degrés
    'dlon': 0.25,
    'lat_mean': 0.0  # équateur
}

# Grille globale : Pacifique équatorial
# 120 lat × 180 lon (résolution 0.25°)
# Workers : 4×6 = 24 workers (30 lat × 30 lon chacun)

workers, grid_info = setup_simulation_2d(
    nlat_global=120,
    nlon_global=180,
    num_workers_lat=4,
    num_workers_lon=6,
    kernel=kernel,
    params=params,
    grid_bounds={
        'lat_min': -15.0,  # 15°S
        'lat_max': 15.0,   # 15°N
        'lon_min': 120.0,  # 120°E
        'lon_max': -75.0   # 75°W (285°E)
    }
)

# Lancer la simulation (événementielle)
scheduler = EventScheduler.remote(
    workers=workers,
    dt=0.1,  # jours
    t_end=365.0  # 1 an
)

results = ray.get(scheduler.run.remote())
```

---

## 6. Visualisation et Diagnostics 2D

### Reconstruction de la Grille Globale

```python
async def reconstruct_global_grid(workers: List, patches: List[dict],
                                   nlat_global: int, nlon_global: int) -> dict:
    """
    Reconstruit la grille globale depuis les workers distribués.

    Args:
        workers: liste d'acteurs Ray
        patches: info de découpage
        nlat_global: taille globale en latitude
        nlon_global: taille globale en longitude

    Returns:
        global_state: dictionnaire avec grilles globales
    """
    # Récupérer états locaux
    local_states = await asyncio.gather(*[w.get_state.remote() for w in workers])

    # Initialiser grilles globales
    global_biomass = np.zeros((nlat_global, nlon_global))
    global_temp = np.zeros((nlat_global, nlon_global))

    # Remplir depuis chaque patch
    for patch, state in zip(patches, local_states):
        lat_slice = patch['lat_slice']
        lon_slice = patch['lon_slice']

        global_biomass[lat_slice[0]:lat_slice[1],
                      lon_slice[0]:lon_slice[1]] = np.array(state['biomass'])

        global_temp[lat_slice[0]:lat_slice[1],
                   lon_slice[0]:lon_slice[1]] = np.array(state['temperature'])

    return {
        'biomass': global_biomass,
        'temperature': global_temp
    }


# Visualisation
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def plot_global_field(field: np.ndarray, grid_info: GridInfo,
                      title: str = 'Biomass'):
    """Visualise un champ 2D sur une carte."""
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Contours
    im = ax.contourf(
        grid_info.lon_coords,
        grid_info.lat_coords,
        field,
        levels=20,
        cmap='viridis',
        transform=ccrs.PlateCarree()
    )

    ax.coastlines()
    ax.gridlines(draw_labels=True)
    plt.colorbar(im, ax=ax, label=title)
    plt.title(title)
    plt.show()


# Utilisation
global_state = ray.get(reconstruct_global_grid(workers, patches, 120, 180))
plot_global_field(global_state['biomass'], grid_info, title='Biomass (g/m²)')
```

---

## 7. Intégration avec JAX-CFD pour Transport 2D

### Configuration JAX-CFD

```python
from jax_cfd.base import grids, equations

def create_jax_cfd_grid(nlat: int, nlon: int,
                        lat_bounds: tuple, lon_bounds: tuple) -> grids.Grid:
    """
    Crée une grille JAX-CFD pour un patch.

    Args:
        nlat, nlon: taille du patch
        lat_bounds: (lat_min, lat_max) en degrés
        lon_bounds: (lon_min, lon_max) en degrés

    Returns:
        grid: grille JAX-CFD
    """
    # Convertir en mètres (approximation plane)
    R = 6371000
    lat_mean = (lat_bounds[0] + lat_bounds[1]) / 2

    x_range = R * jnp.cos(jnp.radians(lat_mean)) * jnp.radians(lon_bounds[1] - lon_bounds[0])
    y_range = R * jnp.radians(lat_bounds[1] - lat_bounds[0])

    grid = grids.Grid(
        shape=(nlat, nlon),
        domain=((0, y_range), (0, x_range))  # (lat, lon) en mètres
    )

    return grid
```

---

## 8. Optimisations pour la 2D

### Communication Overlap (Avancé)

```python
@ray.remote
class OptimizedCellWorker2D(CellWorker2D):
    """
    Version optimisée avec recouvrement calcul/communication.
    """

    async def step_optimized(self, dt: float):
        """
        Optimisation : commencer la phase locale pendant l'échange halo.
        """
        # Lancer échange halo en arrière-plan
        if self.kernel.has_global_units():
            halo_futures = self._request_all_halos()

        # Pendant l'échange, exécuter calculs locaux
        self.state = self.kernel.execute_local_phase(self.state, dt, self.params)

        # Maintenant attendre les halos (si pas déjà arrivés)
        if self.kernel.has_global_units():
            halo_data = await self._gather_halos(halo_futures)
            self.state = self.kernel.execute_global_phase(
                self.state, dt, self.params, halo_data
            )

        self.t += dt
        return self.state
```

### Chunking pour Grandes Grilles

```python
def split_large_domain(nlat: int, nlon: int,
                       max_chunk_size: int = 100) -> List[dict]:
    """
    Découpe automatiquement pour éviter patches trop gros.

    Args:
        nlat, nlon: taille globale
        max_chunk_size: taille maximale par dimension

    Returns:
        patches adaptés
    """
    num_workers_lat = int(np.ceil(nlat / max_chunk_size))
    num_workers_lon = int(np.ceil(nlon / max_chunk_size))

    return split_domain_2d(nlat, nlon, num_workers_lat, num_workers_lon)
```

---

## 9. Gestion des Conditions aux Limites Sphériques

### Périodicité en Longitude

```python
def split_domain_2d_periodic_lon(nlat: int, nlon: int,
                                 num_workers_lat: int,
                                 num_workers_lon: int) -> List[dict]:
    """
    Découpage avec périodicité en longitude (Terre sphérique).

    Le voisin est du dernier worker en longitude est le premier.
    """
    patches = split_domain_2d(nlat, nlon, num_workers_lat, num_workers_lon)

    # Connecter périodicité
    for i in range(num_workers_lat):
        # Premier worker de la ligne
        first_in_row = i * num_workers_lon
        # Dernier worker de la ligne
        last_in_row = first_in_row + num_workers_lon - 1

        # Connecter est-ouest
        patches[first_in_row]['neighbors']['west'] = last_in_row
        patches[last_in_row]['neighbors']['neighbors']['east'] = first_in_row

    return patches
```

---

## 10. Résumé des Adaptations 2D

### Changements par Rapport à 1D

| Composant | Adaptation pour 2D |
|-----------|-------------------|
| **État** | `(N,)` → `(nlat, nlon)` |
| **GridInfo** | Ajout lat/lon, dlat/dlon, métriques sphériques |
| **Découpage** | `split_domain()` → `split_domain_2d()` |
| **Voisins** | 2 (L/R) → 4 (N/S/E/W) |
| **Halo** | 2 scalaires → 4 arrays 1D |
| **Transport** | Laplacien 1D → Laplacien 2D |
| **JAX-CFD** | `Grid(shape=(N,))` → `Grid(shape=(nlat, nlon))` |
| **Visualisation** | Plot 1D → Cartopy maps |

### Architecture Globale Inchangée

✅ **Scheduler événementiel** : fonctionne identique en 2D

✅ **Kernel composable** : Units adaptées mais principe identique

✅ **Découplage local/global** : même séparation, juste avec 4 voisins

✅ **Ray Actors** : même architecture, état 2D au lieu de 1D

---

## Prochaines Étapes

Maintenant avec la 2D en place, voulez-vous que je :

1. **Implémente un exemple minimal 2D** (2×2 workers, grille 20×20)
2. **Détaille l'intégration JAX-CFD** pour advection 2D réaliste
3. **Crée les fonctions de forçage 2D** (interpolation depuis NetCDF/Zarr)
4. **Autre chose** ?

Quelle direction préférez-vous ?
