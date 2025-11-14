Parfait ! Votre cas d'usage est très clair. Analysons les options pour gérer ces forçages multi-dimensionnels de
manière efficace.

Vos besoins identifiés

Forçages :

-   Température : (time, depth, lat, lon) avec depth = couches (0-150m, 150-400m, etc.)
-   Production primaire : (time, lat, lon)
-   Courants u, v : (time, depth, lat, lon)

Opérations dérivées :

-   Temps passé par couche selon jour de l'année
-   Moyenne temporelle de température sur dt
-   Coefficient de transfert sur production primaire → recrutement R

Contraintes :

-   Format source : Zarr/NetCDF (lazy loading)
-   Nombreux workers (2x2, 4x4...)
-   Interpolations spatiales + temporelles
-   Calculs dérivés (moyennes, coefficients)

Architecture proposée : 3 options

Option 1 : ForcingManager centralisé (recommandé)

Concept : Un Ray actor unique qui gère tous les forçages et répond aux requêtes des workers.

@ray.remote
class ForcingManager:
def **init**(self, zarr_paths: dict[str, str]):
"""Charge les datasets Zarr/NetCDF en lazy mode."""
import xarray as xr
self.datasets = {
"temperature": xr.open_zarr(zarr_paths["temperature"]),
"primary_prod": xr.open_zarr(zarr_paths["primary_prod"]),
"currents_u": xr.open_zarr(zarr_paths["currents_u"]),
"currents_v": xr.open_zarr(zarr_paths["currents_v"]),
} # Cache pour interpolations récentes
self.cache = {}

      def get_forcing_slice(
          self,
          variable: str,
          time: float,
          lat_slice: slice,
          lon_slice: slice,
          depth_idx: int | None = None,
      ) -> jnp.ndarray:
          """Récupère et interpole un sous-domaine spatial à un instant t."""
          # Interpolation temporelle
          data = self._interpolate_time(variable, time)

          # Sélection spatiale
          if depth_idx is not None:
              result = data[depth_idx, lat_slice, lon_slice]
          else:
              result = data[lat_slice, lon_slice]

          return jnp.array(result.values)

      def get_derived_forcing(
          self,
          forcing_type: str,  # "avg_temperature", "recruitment", etc.
          time: float,
          lat_slice: slice,
          lon_slice: slice,
          **kwargs,
      ) -> jnp.ndarray:
          """Calcule des forçages dérivés."""
          if forcing_type == "avg_temperature":
              return self._compute_avg_temperature(time, lat_slice, lon_slice, **kwargs)
          elif forcing_type == "recruitment":
              return self._compute_recruitment(time, lat_slice, lon_slice, **kwargs)
          # ...

Utilisation dans CellWorker2D :
class CellWorker2D:
def **init**(self, ..., forcing_manager_ref):
self.forcing_manager = forcing_manager_ref

      async def step(self, dt):
          # Récupère température pour la couche actuelle
          temp = await self.forcing_manager.get_forcing_slice.remote(
              variable="temperature",
              time=self.t_current,
              lat_slice=slice(self.lat_start, self.lat_end),
              lon_slice=slice(self.lon_start, self.lon_end),
              depth_idx=self.current_depth_layer,
          )

          # Calcule recrutement dérivé
          recruitment = await self.forcing_manager.get_derived_forcing.remote(
              forcing_type="recruitment",
              time=self.t_current,
              lat_slice=slice(self.lat_start, self.lat_end),
              lon_slice=slice(self.lon_start, self.lon_end),
              transfer_coef=0.15,
          )

          # Utilise dans le kernel
          params = {"R": recruitment, "temperature": temp, ...}
          self.state = self.kernel.apply(self.state, dt, params)

Avantages :
✅ Un seul chargement des datasets Zarr (mémoire)
✅ Cache centralisé des interpolations
✅ Logic métier (calculs dérivés) centralisée
✅ Workers restent légers

Inconvénients :
❌ Goulot d'étranglement si nombreux workers (sérialization)
❌ Communication réseau pour chaque requête

---

Option 2 : Forçages pré-chargés dans object store Ray

Concept : Charger les forçages interpolés temporellement dans Ray object store, partagés entre workers.

def prepare_forcings_for_timestep(t: float, zarr_paths: dict) -> dict:
"""Charge et interpole tous les forçages pour un timestep."""
import xarray as xr

      # Interpolation temporelle
      temp = interpolate_zarr_at_time(zarr_paths["temperature"], t)
      pp = interpolate_zarr_at_time(zarr_paths["primary_prod"], t)
      u = interpolate_zarr_at_time(zarr_paths["currents_u"], t)
      v = interpolate_zarr_at_time(zarr_paths["currents_v"], t)

      return {
          "temperature": jnp.array(temp.values),  # Shape: (depth, lat, lon)
          "primary_prod": jnp.array(pp.values),   # Shape: (lat, lon)
          "u": jnp.array(u.values),               # Shape: (depth, lat, lon)
          "v": jnp.array(v.values),
      }

# Dans EventScheduler

class EventScheduler:
def step(self): # Prépare forçages pour ce timestep
forcings = prepare_forcings_for_timestep(self.t_current, self.zarr_paths)

          # Met dans object store Ray (zéro-copy)
          forcings_ref = ray.put(forcings)

          # Passe la référence aux workers
          futures = [worker.step.remote(self.dt, forcings_ref) for worker in self.workers]
          ray.get(futures)

# Dans CellWorker2D

async def step(self, dt, forcings_ref): # Récupère depuis object store (zero-copy, partagé)
forcings = ray.get(forcings_ref)

      # Sélectionne sous-domaine spatial
      temp_local = forcings["temperature"][:, self.lat_start:self.lat_end, self.lon_start:self.lon_end]
      u_local = forcings["u"][:, self.lat_start:self.lat_end, self.lon_start:self.lon_end]

      # Calcule dérivés localement
      recruitment = self._compute_recruitment(forcings["primary_prod"][self.lat_start:self.lat_end, ...])

      params = {"R": recruitment, "temperature": temp_local, "u": u_local, ...}
      self.state = self.kernel.apply(self.state, dt, params)

Avantages :
✅ Zéro-copy (mémoire partagée Ray)
✅ Pas de sérialization répétée
✅ Parallélisation parfaite

Inconvénients :
❌ Charge TOUTES les grilles en mémoire à chaque timestep
❌ Mémoire importante si grilles larges (globales)
❌ Calculs dérivés dupliqués sur chaque worker

---

Option 3 : Hybrid - ForcingCache avec chunking spatial

Concept : Combiner Option 1 + 2 avec un cache intelligent par chunk spatial.

@ray.remote
class ForcingCache:
def **init**(self, zarr_paths: dict, chunk_size_lat=50, chunk_size_lon=50):
self.datasets = {k: xr.open_zarr(v) for k, v in zarr_paths.items()}
self.chunk_size = (chunk_size_lat, chunk_size_lon)

          # Cache : clé = (variable, time_idx, lat_chunk_idx, lon_chunk_idx)
          self.cache = {}

      def get_chunk(
          self,
          variable: str,
          time: float,
          lat_chunk_idx: int,
          lon_chunk_idx: int,
          depth_idx: int | None = None,
      ):
          """Récupère un chunk spatial interpolé temporellement."""
          cache_key = (variable, time, lat_chunk_idx, lon_chunk_idx, depth_idx)

          if cache_key in self.cache:
              return self.cache[cache_key]

          # Calcule indices spatiaux
          lat_start = lat_chunk_idx * self.chunk_size[0]
          lat_end = (lat_chunk_idx + 1) * self.chunk_size[0]
          lon_start = lon_chunk_idx * self.chunk_size[1]
          lon_end = (lon_chunk_idx + 1) * self.chunk_size[1]

          # Charge depuis Zarr avec interpolation temporelle
          data = self._load_and_interpolate(
              variable, time,
              slice(lat_start, lat_end),
              slice(lon_start, lon_end),
              depth_idx,
          )

          # Met en cache et dans object store
          data_ref = ray.put(jnp.array(data))
          self.cache[cache_key] = data_ref

          return data_ref

# Création

forcing_cache = ForcingCache.remote(zarr_paths, chunk_size_lat=25, chunk_size_lon=25)

# Dans CellWorker2D

async def step(self, dt): # Identifie le chunk spatial correspondant au worker
lat_chunk_idx = self.lat_start // 25
lon_chunk_idx = self.lon_start // 25

      # Requête au cache (peut retourner depuis cache ou charger)
      temp_chunk_ref = await self.forcing_cache.get_chunk.remote(
          "temperature", self.t_current, lat_chunk_idx, lon_chunk_idx, depth_idx=0
      )
      temp_chunk = ray.get(temp_chunk_ref)

      # Extrait sous-domaine local du chunk
      local_offset_lat = self.lat_start % 25
      local_offset_lon = self.lon_start % 25
      temp_local = temp_chunk[
          local_offset_lat:local_offset_lat + self.nlat_local,
          local_offset_lon:local_offset_lon + self.nlon_local,
      ]

Avantages :
✅ Compromis mémoire/performance
✅ Cache intelligent (pas de rechargement)
✅ Chunks alignés avec décomposition domaine
✅ Supporte très grandes grilles (globales)

Inconvénients :
❌ Plus complexe à implémenter
❌ Gestion cache (invalidation, taille max)

---

Comparaison et recommandation

| Critère          | Option 1: Manager           | Option 2: Object Store | Option 3: Hybrid Cache   |
| ---------------- | --------------------------- | ---------------------- | ------------------------ |
| Mémoire          | ⭐⭐⭐ Lazy loading         | ⭐ Tout en RAM         | ⭐⭐ Chunks cachés       |
| Performance      | ⭐⭐ Requêtes séquentielles | ⭐⭐⭐ Zero-copy       | ⭐⭐⭐ Zero-copy + cache |
| Scalabilité      | ⭐⭐ Goulot manager         | ⭐⭐⭐ Parfaite        | ⭐⭐⭐ Parfaite          |
| Simplicité       | ⭐⭐⭐ Simple               | ⭐⭐ Moyen             | ⭐ Complexe              |
| Grilles globales | ⭐⭐⭐ OK                   | ⭐ Problématique       | ⭐⭐⭐ OK                |

Ma recommandation : Approche progressive

Phase 1 : Commencer avec Option 2 (Object Store) simplifié

-   Idéal pour prototyper rapidement
-   Fonctionne bien pour domaines régionaux (< 1000x1000)
-   Simple à intégrer dans architecture actuelle

Phase 2 : Si problèmes mémoire → Option 3 (Hybrid Cache)

-   Pour grilles globales ou très haute résolution
-   Quand vous avez beaucoup de workers (>16)

Éviter Option 1 pour l'instant (goulot d'étranglement)

Proposition de structure de code

# src/seapopym_message/forcing/manager.py

from typing import Protocol
import jax.numpy as jnp

class ForcingProvider(Protocol):
"""Interface pour fournir des forçages aux workers."""

      def get_forcing(
          self,
          variable: str,
          time: float,
          lat_slice: slice,
          lon_slice: slice,
          depth_idx: int | None = None,
      ) -> jnp.ndarray:
          """Récupère un forçage interpolé."""
          ...

class SimpleObjectStoreForcingProvider:
"""Implémentation simple avec Ray object store."""

      def __init__(self, zarr_paths: dict[str, str]):
          self.zarr_paths = zarr_paths
          self._cache = {}

      def prepare_timestep(self, time: float):
          """Charge et interpole tous les forçages pour un timestep."""
          # Interpolation temporelle + mise en object store
          ...

      def get_forcing(self, variable, time, lat_slice, lon_slice, depth_idx=None):
          # Récupère depuis cache/object store
          ...

Voulez-vous que je :

1. Commence par implémenter Option 2 (SimpleObjectStoreForcingProvider) ?
2. Ou d'abord créer l'interface abstraite et plusieurs implémentations ?
3. Ou discuter plus en détail d'un cas spécifique (ex: comment gérer le "temps par couche selon jour de l'année") ?

---

# Suite

Excellente réflexion ! Vous avez raison : centraliser les forçages dans un ForcingManager suit le même pattern
architectural que EventScheduler pour le temps ou un futur TransportWorker. C'est cohérent et réutilisable.

Analysons comment rendre cela extensible pour qu'un utilisateur puisse facilement ajouter aragonite, oxygène, ou
tout autre forçage.

Architecture proposée : ForcingManager extensible

Principe de design

Objectif : L'utilisateur déclare ses besoins en forçages, le ForcingManager s'occupe du reste (chargement,
interpolation, cache, distribution aux workers).

Pattern similaire à Unit/Kernel :

-   Unit = fonction transformant des données
-   Forcing = source de données externes
-   ForcingManager = orchestrateur (comme EventScheduler)

Solution 1 : Configuration déclarative (le plus simple)

L'utilisateur déclare ses forçages dans un fichier de config ou un dictionnaire :

# config/forcings.yaml (ou dict Python)

forcings:
temperature:
source: "gs://ocean-data/temperature.zarr"
dims: ["time", "depth", "lat", "lon"]
units: "°C"
interpolation:
time: "linear"
space: "bilinear"

    primary_production:
      source: "gs://ocean-data/pp.zarr"
      dims: ["time", "lat", "lon"]
      units: "mg C/m³/day"
      interpolation:
        time: "linear"

    currents_u:
      source: "gs://ocean-data/currents_u.zarr"
      dims: ["time", "depth", "lat", "lon"]
      interpolation:
        time: "linear"

    currents_v:
      source: "gs://ocean-data/currents_v.zarr"
      dims: ["time", "depth", "lat", "lon"]
      interpolation:
        time: "linear"

    # Nouvel utilisateur ajoute aragonite
    aragonite:
      source: "/my/data/aragonite.zarr"
      dims: ["time", "lat", "lon"]
      units: "µmol/kg"
      interpolation:
        time: "linear"

    # Et oxygène
    oxygen:
      source: "/my/data/oxygen.zarr"
      dims: ["time", "depth", "lat", "lon"]
      units: "ml/L"
      interpolation:
        time: "nearest"  # Peut choisir différente méthode

Utilisation :
from seapopym_message.forcing import ForcingManager

# L'utilisateur charge sa config

forcing_manager = ForcingManager.from_config("config/forcings.yaml")

# Ou directement en Python

forcing_config = {
"temperature": {"source": "...", "dims": [...], ...},
"aragonite": {"source": "...", "dims": [...], ...},
}
forcing_manager = ForcingManager(forcing_config)

# Le ForcingManager gère automatiquement tous les forçages

workers, patches = create_distributed_simulation(
grid=grid,
kernel=kernel,
forcing_manager=forcing_manager, # Passe le manager
...
)

Avantages :
✅ Zéro code pour ajouter un forçage standard
✅ Déclaratif et lisible
✅ Facile de partager des configs entre projets
✅ Validation automatique (dims, unités, etc.)

Limites :
❌ Ne gère que des forçages "simples" (chargement direct)
❌ Pas adapté pour forçages dérivés (ex: recrutement = f(pp))

---

Solution 2 : Forçages dérivés avec décorateur @derived_forcing

Pour les transformations personnalisées, un pattern similaire à @unit :

from seapopym_message.forcing import derived_forcing

@derived_forcing(
name="recruitment",
inputs=["primary_production"], # Dépend d'un autre forçage
params=["transfer_coefficient", "day_of_year"],
)
def compute_recruitment(
primary_production: jnp.ndarray,
transfer_coefficient: float,
day_of_year: int,
) -> jnp.ndarray:
"""Calcule le recrutement à partir de la production primaire.

      Args:
          primary_production: PP en surface (lat, lon)
          transfer_coefficient: Coefficient de transfert trophique
          day_of_year: Jour de l'année (pour saisonnalité)

      Returns:
          Recrutement (lat, lon)
      """
      # Modulation saisonnière
      seasonal_factor = 1.0 + 0.3 * jnp.sin(2 * jnp.pi * day_of_year / 365)

      # Transfert trophique
      recruitment = primary_production * transfer_coefficient * seasonal_factor

      return recruitment

@derived_forcing(
name="mean_temperature_weighted",
inputs=["temperature"], # Dépend de température 4D
params=["time_in_layer", "depth_layers"],
)
def compute_weighted_temperature(
temperature: jnp.ndarray, # Shape: (depth, lat, lon)
time_in_layer: jnp.ndarray, # Shape: (depth,) - temps passé par couche
depth_layers: list[tuple[float, float]],
) -> jnp.ndarray:
"""Moyenne pondérée de température par temps passé dans chaque couche.

      Args:
          temperature: Température par couche (depth, lat, lon)
          time_in_layer: Fraction de temps dans chaque couche (depth,)
          depth_layers: Définition des couches [(0, 150), (150, 400), ...]

      Returns:
          Température moyenne pondérée (lat, lon)
      """
      # Pondération par temps passé
      weights = time_in_layer / jnp.sum(time_in_layer)  # Normalise

      # Moyenne pondérée sur depth
      weighted_temp = jnp.sum(
          temperature * weights[:, None, None],  # Broadcast weights
          axis=0,  # Sum over depth
      )

      return weighted_temp

# Utilisateur avec aragonite custom

@derived\*forcing(
name="aragonite_saturation",
inputs=["aragonite", "temperature"],
params=["salinity"],
)
def compute_aragonite_saturation(
aragonite: jnp.ndarray,
temperature: jnp.ndarray,
salinity: float,
) -> jnp.ndarray:
"""Calcule l'état de saturation en aragonite (Omega_arag).""" # Formule chimique personnalisée de l'utilisateur
K_sp = 10 \*\* (-171.945 - 0.077993 \_ temperature + 2903.293 / temperature)
omega = aragonite / (K_sp \* salinity)
return omega

Enregistrement dans ForcingManager :

# Le ForcingManager détecte automatiquement les dépendances

forcing_manager = ForcingManager(forcing_config)

# Enregistre les forçages dérivés

forcing_manager.register_derived(compute_recruitment)
forcing_manager.register_derived(compute_weighted_temperature)
forcing_manager.register_derived(compute_aragonite_saturation)

# Le manager résout automatiquement le graphe de dépendances:

# 1. Charge temperature, primary_production, aragonite depuis Zarr

# 2. Calcule recruitment depuis primary_production

# 3. Calcule aragonite_saturation depuis aragonite + temperature

Utilisation dans un Unit :
from seapopym_message.core.unit import unit

@unit(
name="growth_with_aragonite",
inputs=["biomass"],
outputs=["biomass"],
scope="local",
forcings=["recruitment", "aragonite_saturation"], # Déclare besoins
)
def compute_growth_with_aragonite(
biomass: jnp.ndarray,
dt: float,
params: dict,
forcings: dict, # ForcingManager injecte automatiquement
) -> jnp.ndarray:
"""Croissance influencée par recrutement et saturation aragonite."""
R = forcings["recruitment"]
omega_arag = forcings["aragonite_saturation"]

      # Modulation par aragonite (ex: larves sensibles à acidification)
      aragonite_stress = jnp.where(
          omega_arag < 1.0,  # Sous-saturation
          0.5,  # Stress fort
          1.0,  # Pas de stress
      )

      growth_rate = R * aragonite_stress

      return biomass + growth_rate * dt

Avantages :
✅ Extensible : l'utilisateur crée ses propres @derived_forcing
✅ Déclaratif : les Units déclarent leurs besoins en forcings=[...]
✅ Résolution automatique des dépendances (graphe)
✅ Cohérent avec pattern @unit
✅ Testable indépendamment

---

Solution 3 : Plugin system (pour utilisateurs avancés)

Pour des forçages très personnalisés (ex: lecture format propriétaire, API web, etc.) :

from seapopym_message.forcing import ForcingPlugin

class AragoniteFromAPI(ForcingPlugin):
"""Charge aragonite depuis une API REST au lieu de Zarr."""

      name = "aragonite"
      dims = ["time", "lat", "lon"]

      def __init__(self, api_url: str, api_key: str):
          self.api_url = api_url
          self.api_key = api_key

      def load_at_time(
          self,
          time: float,
          lat_slice: slice,
          lon_slice: slice,
      ) -> jnp.ndarray:
          """Charge depuis l'API."""
          response = requests.get(
              f"{self.api_url}/aragonite",
              params={
                  "time": time,
                  "lat_min": lat_slice.start,
                  "lat_max": lat_slice.stop,
                  "lon_min": lon_slice.start,
                  "lon_max": lon_slice.stop,
              },
              headers={"Authorization": f"Bearer {self.api_key}"},
          )
          data = jnp.array(response.json()["data"])
          return data

# Enregistrement

forcing_manager.register_plugin(
AragoniteFromAPI(api_url="https://my-ocean-api.com", api_key="...")
)

Avantages :
✅ Flexibilité maximale
✅ Supporte sources de données non-standard

Inconvénients :
❌ Plus complexe
❌ Pour utilisateurs avancés uniquement

---

Architecture recommandée : Système à 3 niveaux

┌─────────────────────────────────────────────────────────┐
│ ForcingManager │
│ (Orchestrateur central - comme EventScheduler) │
│ │
│ ┌────────────────────────────────────────────────┐ │
│ │ Niveau 1: Forçages de base (Zarr/NetCDF) │ │
│ │ - Config déclarative │ │
│ │ - Chargement automatique │ │
│ │ - Interpolation temporelle/spatiale │ │
│ │ - Cache │ │
│ └────────────────────────────────────────────────┘ │
│ │
│ ┌────────────────────────────────────────────────┐ │
│ │ Niveau 2: Forçages dérivés (@derived_forcing)│ │
│ │ - Graphe de dépendances │ │
│ │ - Calculs personnalisés utilisateur │ │
│ │ - Injection automatique dans Units │ │
│ └────────────────────────────────────────────────┘ │
│ │
│ ┌────────────────────────────────────────────────┐ │
│ │ Niveau 3: Plugins (ForcingPlugin) │ │
│ │ - Sources de données custom │ │
│ │ - API, formats propriétaires, etc. │ │
│ └────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
│
▼
┌────────────────┐
│ CellWorker2D │
│ │
│ kernel.apply( │
│ forcings={} │
│ ) │
└────────────────┘

Exemple complet d'utilisation

# 1. Config des forçages de base (Zarr)

forcing_config = {
"temperature": {"source": "data/temp.zarr", "dims": ["time", "depth", "lat", "lon"]},
"primary_production": {"source": "data/pp.zarr", "dims": ["time", "lat", "lon"]},
"aragonite": {"source": "data/aragonite.zarr", "dims": ["time", "lat", "lon"]},
}

# 2. Créer le manager

forcing_manager = ForcingManager(forcing_config)

# 3. Enregistrer forçages dérivés personnalisés

forcing_manager.register_derived(compute_recruitment)
forcing_manager.register_derived(compute_aragonite_saturation)

# 4. Créer Units qui utilisent les forçages

@unit(
name="my_growth",
inputs=["biomass"],
outputs=["biomass"],
forcings=["recruitment", "aragonite_saturation"], # Déclare besoins
)
def my_custom_growth(biomass, dt, params, forcings):
R = forcings["recruitment"]
omega = forcings["aragonite_saturation"] # ... logique métier
return biomass_new

# 5. Créer simulation

kernel = Kernel([my_custom_growth, compute_diffusion_2d])

workers, patches = create_distributed_simulation(
grid=grid,
kernel=kernel,
forcing_manager=forcing_manager, # Passe le manager
num_workers_lat=2,
num_workers_lon=2,
)

# 6. Le ForcingManager s'occupe de tout:

# - Charger temperature, pp, aragonite depuis Zarr

# - Interpoler au temps courant

# - Calculer recruitment et aragonite_saturation

# - Distribuer aux workers (object store Ray)

# - Injecter dans le kernel

Roadmap d'implémentation

Phase "Forcing System" (~8h)

1. Créer ForcingManager de base (~2h)

    - Chargement config
    - Interface avec Zarr/NetCDF
    - Interpolation temporelle linéaire

2. Décorateur @derived_forcing (~2h)

    - Pattern similaire à @unit
    - Résolution graphe de dépendances

3. Intégration avec CellWorker2D (~2h)

    - Passage forcing_manager_ref aux workers
    - Injection automatique dans kernel.apply()

4. Tests + exemple (~2h)

    - Test avec temperature + derived recruitment
    - Exemple montrant extensibilité

---

# Suite : Questions

Questions de clarification

Q1 : Dépendances externes

-   Faut-il ajouter xarray et zarr dans les dépendances du projet ? Ou sont-ils déjà installés ?
-   Version préférée de xarray (>= 2023.x pour meilleures perfs) ?

Oui on peut commencer avec xarray et zarr pour les forçages. Corriges moi si je me trompe mais l'idée est d'utiliser numpy/jax ? Donc nativement pris en charge par xarray.

Q2 : Format config

-   Pour commencer : dict Python uniquement, ou supporter YAML dès le début ?
-   Je suggère dict Python pour Phase 1, YAML optionnel après. OK ?

Est-ce qu'utiliser une bibliothèque comme dataclass/attrs/pydantic etc... te parrait intéressant pour pouvoir vérifier l'intégrité des paramètres ? Si c'est trop complexe pour commencer on peut faire plus simple.

Q3 : Validation des forçages

-   Doit-on valider que les dimensions des forçages matchent la grille (lat/lon) ?
-   Ou on fait confiance à l'utilisateur pour Phase 1 ?

La validation se fera en amont de la simulation. Je souhaite conserver uniquement ce qui est important dans le modèle. Pour tous ce qui est vérification j'imagine avoir un autre module en amont qui permet ensuite de générer ma simulation.

Q4 : Interpolation temporelle

-   Linéaire suffit pour commencer ?
-   Ou besoin de "nearest", "cubic" dès Phase 1 ?

Linéaire suffit mais de toute manière il me semble que xarray gère ça nativement. Qu'en penses tu, on se base sur cette dépendance ? Est-ce que ça change les plans ?

Q5 : Gestion erreurs

-   Si un forçage n'a pas de données pour un timestep (t en dehors de la plage) :
    -   Erreur stricte ?
    -   Extrapolation (dernier/premier timestep disponible) ?
    -   Warning + valeur par défaut ?

On fait uniquement de l'interpolation. Pour l'extrapolation c'est une erreur et c'est à l'utilisateur de gérer lui même. L'interpolation nous permet de réduire le pas de temps si besoin ce qui est essentiel pour nos calculs de transport.

Q6 : Structure fichiers

-   Je crée src/seapopym_message/forcing/ avec :
    -   manager.py (ForcingManager)
    -   derived.py (décorateur @derived_forcing)
    -   **init**.py
-   OK pour vous ?

Oui

Q7 : Pour les tests

-   Voulez-vous que je crée de faux fichiers Zarr en mémoire (pour tests unitaires) ?
-   Ou utiliser des fixtures avec des arrays NumPy directement ?

On peut commencer directement avec les dataset/dataarray de xarray comme ça on laisse la dépendance aux fichier à cette dépendance.
