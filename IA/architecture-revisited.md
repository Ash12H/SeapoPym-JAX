# Architecture Révisée : Ray + Kernel Composable + Événementiel

## Retour aux Principes Fondamentaux

Vous avez raison, reprenons votre vision initiale des discussions :

### 1️⃣ Scheduler Événementiel
✅ **Pas de boucle `for t in range(steps)`**
✅ EventLoop avec PriorityQueue d'événements
✅ Workers avancent à leur propre rythme
✅ Synchronisation uniquement quand nécessaire

### 2️⃣ Kernel = Liste de Units Composables
✅ Un modèle = définir simplement : `[compute_daylength, mortality, growth, transport]`
✅ Chaque Unit déclare inputs/outputs
✅ Ordre d'exécution automatique (topologique)
✅ Ajout/retrait de Units facile

### 3️⃣ Découplage Local vs Global
✅ **Calculs locaux** (par cellule) : indépendants, parallélisables
✅ **Calculs globaux** (transport) : nécessitent synchronisation
✅ Barrière explicite entre les deux

---

## Architecture Complète

```
┌──────────────────────────────────────────────────────────┐
│                    EventScheduler                         │
│  - PriorityQueue[(time, event)]                          │
│  - Gère quand chaque worker doit travailler              │
│  - Pas de boucle for sur le temps !                      │
└──────────────────────────────────────────────────────────┘
                        ↓ ↓ ↓
    ┌─────────────────┬─────────────────┬─────────────────┐
    │   Worker 1      │   Worker 2      │   Worker N      │
    │  Ray Actor      │  Ray Actor      │  Ray Actor      │
    └─────────────────┴─────────────────┴─────────────────┘
            ↓                 ↓                 ↓
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │  Cell[0-24] │   │  Cell[25-49]│   │  Cell[50-74]│
    │             │   │             │   │             │
    │  Kernel:    │   │  Kernel:    │   │  Kernel:    │
    │  [Unit1,    │   │  [Unit1,    │   │  [Unit1,    │
    │   Unit2,    │   │   Unit2,    │   │   Unit2,    │
    │   Unit3]    │   │   Unit3]    │   │   Unit3]    │
    └─────────────┘   └─────────────┘   └─────────────┘

Exemple de Kernel:
  Unit1 = compute_temperature     (local)
  Unit2 = compute_mortality        (local)
  Unit3 = compute_growth           (local)
  ─── Barrière de synchronisation ───
  Unit4 = transport_diffusion      (global, nécessite voisins)
```

---

## 1. Système de Units Composables

### Classe de Base : Unit

```python
from dataclasses import dataclass
from typing import Callable, List, Any
import jax.numpy as jnp

@dataclass
class Unit:
    """
    Unité d'exécution élémentaire.

    Attributs:
        name: identifiant unique
        func: fonction à exécuter
        inputs: noms des variables requises en entrée
        outputs: noms des variables produites
        scope: 'local' (par cellule) ou 'global' (nécessite voisins)
        compiled: si True, func est JIT-compilée
    """
    name: str
    func: Callable
    inputs: List[str]
    outputs: List[str]
    scope: str = 'local'  # 'local' ou 'global'
    compiled: bool = False

    def __post_init__(self):
        if self.compiled:
            from jax import jit
            self.func = jit(self.func)

    def can_execute(self, available_vars: set) -> bool:
        """Vérifie si toutes les inputs sont disponibles."""
        return set(self.inputs).issubset(available_vars)

    def execute(self, state: dict, **kwargs) -> dict:
        """
        Exécute l'unité.

        Args:
            state: dictionnaire des variables disponibles
            **kwargs: paramètres additionnels (dt, t, params, etc.)

        Returns:
            updates: dictionnaire des variables à mettre à jour
        """
        # Extraire les inputs
        args = {name: state[name] for name in self.inputs}

        # Exécuter
        result = self.func(**args, **kwargs)

        # Emballer les outputs
        if len(self.outputs) == 1:
            return {self.outputs[0]: result}
        else:
            return dict(zip(self.outputs, result))
```

### Décorateur pour Créer des Units

```python
def unit(name: str, inputs: List[str], outputs: List[str],
         scope: str = 'local', compiled: bool = False):
    """
    Décorateur pour transformer une fonction en Unit.

    Exemple:
        @unit(name='mortality', inputs=['biomass', 'temperature'],
              outputs=['mortality_rate'], scope='local', compiled=True)
        def compute_mortality(biomass, temperature, params):
            lambda_t = params['lambda_0'] * jnp.exp(params['k'] * temperature)
            return lambda_t * biomass
    """
    def decorator(func):
        return Unit(
            name=name,
            func=func,
            inputs=inputs,
            outputs=outputs,
            scope=scope,
            compiled=compiled
        )
    return decorator
```

---

## 2. Le Kernel : Orchestrateur de Units

### Classe Kernel

```python
from typing import List, Dict

class Kernel:
    """
    Noyau d'exécution : liste ordonnée de Units.

    Le Kernel gère:
    - L'ordre d'exécution (topologique selon dépendances)
    - La séparation local/global
    - L'exécution des Units
    """

    def __init__(self, units: List[Unit]):
        self.units = units
        self.local_units = [u for u in units if u.scope == 'local']
        self.global_units = [u for u in units if u.scope == 'global']

        # Vérifier dépendances et tri topologique
        self._check_dependencies()
        self.local_units = self._topological_sort(self.local_units)
        self.global_units = self._topological_sort(self.global_units)

    def _check_dependencies(self):
        """Vérifie que toutes les dépendances peuvent être satisfaites."""
        all_outputs = set()
        for unit in self.units:
            all_outputs.update(unit.outputs)

        for unit in self.units:
            for inp in unit.inputs:
                # inp doit être soit dans initial_state, soit produit par une Unit
                # (on vérifiera à l'exécution)
                pass

    def _topological_sort(self, units: List[Unit]) -> List[Unit]:
        """
        Trie les Units selon leurs dépendances (inputs/outputs).

        Utilise l'algorithme de Kahn.
        """
        # Construire graphe de dépendances
        from collections import defaultdict, deque

        graph = defaultdict(list)
        in_degree = defaultdict(int)

        # Outputs disponibles initialement (avant toute Unit)
        available = set()

        for unit in units:
            in_degree[unit.name] = 0

        # Pour chaque Unit, trouver ses dépendances
        for unit in units:
            for inp in unit.inputs:
                # Trouver quelle Unit produit cet input
                for other in units:
                    if inp in other.outputs and other != unit:
                        graph[other.name].append(unit.name)
                        in_degree[unit.name] += 1

        # Tri topologique
        queue = deque([u.name for u in units if in_degree[u.name] == 0])
        sorted_names = []

        while queue:
            name = queue.popleft()
            sorted_names.append(name)

            for neighbor in graph[name]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_names) != len(units):
            raise ValueError("Dépendances cycliques détectées dans le Kernel!")

        # Reconstruire liste ordonnée
        name_to_unit = {u.name: u for u in units}
        return [name_to_unit[name] for name in sorted_names]

    def execute_local_phase(self, state: dict, dt: float, params: dict) -> dict:
        """
        Exécute toutes les Units locales (pas d'interaction spatiale).

        Ces Units peuvent s'exécuter indépendamment pour chaque cellule.

        Args:
            state: état actuel des cellules
            dt: pas de temps
            params: paramètres du modèle

        Returns:
            new_state: état mis à jour après phase locale
        """
        current_state = state.copy()

        for unit in self.local_units:
            # Vérifier que les inputs sont disponibles
            if not unit.can_execute(set(current_state.keys())):
                missing = set(unit.inputs) - set(current_state.keys())
                raise RuntimeError(
                    f"Unit '{unit.name}' ne peut s'exécuter: "
                    f"variables manquantes {missing}"
                )

            # Exécuter l'unité
            updates = unit.execute(current_state, dt=dt, params=params)

            # Mettre à jour l'état
            current_state.update(updates)

        return current_state

    def execute_global_phase(self, state: dict, dt: float, params: dict,
                            neighbor_data: dict = None) -> dict:
        """
        Exécute toutes les Units globales (nécessitent synchronisation).

        Ces Units peuvent avoir besoin de données des voisins.

        Args:
            state: état actuel
            dt: pas de temps
            params: paramètres
            neighbor_data: données reçues des voisins

        Returns:
            new_state: état mis à jour après phase globale
        """
        current_state = state.copy()

        # Ajouter neighbor_data à l'état si fourni
        if neighbor_data:
            current_state.update(neighbor_data)

        for unit in self.global_units:
            if not unit.can_execute(set(current_state.keys())):
                missing = set(unit.inputs) - set(current_state.keys())
                raise RuntimeError(
                    f"Unit '{unit.name}' (global) ne peut s'exécuter: "
                    f"variables manquantes {missing}"
                )

            updates = unit.execute(current_state, dt=dt, params=params)
            current_state.update(updates)

        return current_state

    def has_global_units(self) -> bool:
        """Retourne True si le Kernel contient des Units globales."""
        return len(self.global_units) > 0
```

---

## 3. Worker avec Kernel Composable

### Worker Ray Actor

```python
import ray
from typing import Optional

@ray.remote
class CellWorker:
    """
    Worker gérant un ensemble de cellules.

    Architecture:
    - Chaque worker a N cellules
    - Toutes les cellules partagent le même Kernel
    - Calculs locaux: exécutés pour toutes les cellules en parallèle (vectorisé JAX)
    - Calculs globaux: nécessitent communication avec voisins
    """

    def __init__(self, worker_id: int, cell_ids: List[int],
                 kernel: Kernel, initial_state: dict, params: dict):
        """
        Args:
            worker_id: identifiant unique du worker
            cell_ids: liste des IDs des cellules gérées
            kernel: Kernel (liste de Units) à exécuter
            initial_state: état initial des cellules
            params: paramètres du modèle
        """
        self.id = worker_id
        self.cell_ids = cell_ids
        self.num_cells = len(cell_ids)
        self.kernel = kernel
        self.params = params

        # État actuel (vectorisé sur toutes les cellules du worker)
        # Format: {'biomass': jnp.array([...]), 'temperature': jnp.array([...]), ...}
        self.state = initial_state
        self.t = 0.0

        # Références aux voisins (workers adjacents)
        self.neighbors = {}  # {'left': WorkerActor, 'right': WorkerActor}

        # File d'événements locale
        self.pending_events = []

    def set_neighbors(self, neighbors: dict):
        """Configure les voisins du worker."""
        self.neighbors = neighbors

    async def step(self, dt: float) -> dict:
        """
        Exécute un pas de temps complet.

        Étapes:
        1. Phase locale (calculs indépendants)
        2. Si Units globales: échange avec voisins
        3. Phase globale (transport, diffusion)

        Args:
            dt: pas de temps

        Returns:
            état mis à jour
        """
        # ═══════════════════════════════════════════════════
        # PHASE 1: CALCULS LOCAUX (pas d'interaction spatiale)
        # ═══════════════════════════════════════════════════
        self.state = self.kernel.execute_local_phase(
            self.state,
            dt,
            self.params
        )

        # ═══════════════════════════════════════════════════
        # PHASE 2: SYNCHRONISATION (si nécessaire)
        # ═══════════════════════════════════════════════════
        if self.kernel.has_global_units():
            # Demander données aux voisins (non-bloquant)
            neighbor_futures = {}

            if 'left' in self.neighbors:
                neighbor_futures['left'] = self.neighbors['left'].get_boundary_data.remote('right')

            if 'right' in self.neighbors:
                neighbor_futures['right'] = self.neighbors['right'].get_boundary_data.remote('left')

            # Pendant qu'on attend, on pourrait faire d'autres calculs indépendants
            # (c'est ici qu'on peut passer à une autre cellule dans une version plus avancée)

            # Attendre réception (bloquant)
            neighbor_data = {}
            for side, future in neighbor_futures.items():
                neighbor_data[f'boundary_{side}'] = await future

            # ═══════════════════════════════════════════════════
            # PHASE 3: CALCULS GLOBAUX (avec données voisins)
            # ═══════════════════════════════════════════════════
            self.state = self.kernel.execute_global_phase(
                self.state,
                dt,
                self.params,
                neighbor_data
            )

        self.t += dt
        return {'state': self.state, 't': self.t}

    def get_boundary_data(self, side: str) -> jnp.ndarray:
        """
        Retourne les données de frontière pour un voisin.

        Args:
            side: 'left' ou 'right'

        Returns:
            données de frontière (ex: biomass[0] ou biomass[-1])
        """
        if side == 'left':
            # Le voisin de gauche veut notre frontière gauche
            return {k: v[0] for k, v in self.state.items() if isinstance(v, jnp.ndarray)}
        else:  # 'right'
            return {k: v[-1] for k, v in self.state.items() if isinstance(v, jnp.ndarray)}
```

---

## 4. Event Scheduler (Sans Boucle For)

### EventLoop avec PriorityQueue

```python
import asyncio
from dataclasses import dataclass, field
from typing import Any
import heapq

@dataclass(order=True)
class Event:
    """Événement temporel."""
    time: float
    worker_id: int = field(compare=False)
    action: str = field(compare=False)
    data: Any = field(default=None, compare=False)

@ray.remote
class EventScheduler:
    """
    Scheduler événementiel : pas de boucle for sur le temps !

    Principe:
    - Chaque worker planifie son prochain événement
    - Le scheduler traite toujours l'événement le plus proche dans le temps
    - Les workers avancent à leur propre rythme
    """

    def __init__(self, workers: List, dt: float, t_end: float):
        self.workers = workers
        self.dt = dt
        self.t_end = t_end

        # File de priorité des événements (min-heap sur time)
        self.event_queue = []

        # Initialiser : chaque worker planifie son premier événement à t=0
        for i, worker in enumerate(workers):
            heapq.heappush(self.event_queue, Event(time=0.0, worker_id=i, action='step'))

    async def run(self):
        """
        Boucle événementielle principale.

        PAS de for t in range(...) !
        On traite les événements dans l'ordre temporel.
        """
        step_count = 0

        while self.event_queue:
            # Récupérer le prochain événement (temps minimal)
            event = heapq.heappop(self.event_queue)

            # Vérifier si on a dépassé t_end
            if event.time > self.t_end:
                break

            # Traiter l'événement
            if event.action == 'step':
                worker = self.workers[event.worker_id]

                # Exécuter le pas de temps du worker (asynchrone)
                result = await worker.step.remote(self.dt)

                # Planifier le prochain événement pour ce worker
                next_time = event.time + self.dt
                if next_time <= self.t_end:
                    heapq.heappush(
                        self.event_queue,
                        Event(time=next_time, worker_id=event.worker_id, action='step')
                    )

                step_count += 1

                if step_count % 100 == 0:
                    print(f"Step {step_count}, t={event.time:.2f}, worker={event.worker_id}")

        print(f"Simulation terminée: {step_count} steps exécutés")
        return await self._collect_final_states()

    async def _collect_final_states(self):
        """Récupère l'état final de tous les workers."""
        futures = [w.get_state.remote() for w in self.workers]
        return await asyncio.gather(*futures)
```

---

## 5. Exemple Complet : Modèle Simple

### Définition des Units

```python
# ════════════════════════════════════════════════════════
# UNITS LOCALES (indépendantes par cellule)
# ════════════════════════════════════════════════════════

@unit(name='recruitment',
      inputs=[],
      outputs=['recruitment_rate'],
      scope='local',
      compiled=True)
def compute_recruitment(params):
    """Recrutement constant."""
    R = params['R']
    # Si R est un scalaire, on le broadcast sur toutes les cellules
    # (géré par le Worker)
    return R

@unit(name='mortality',
      inputs=['biomass', 'temperature'],
      outputs=['mortality_rate'],
      scope='local',
      compiled=True)
def compute_mortality(biomass, temperature, params):
    """Mortalité dépendant de la température."""
    lambda_T = params['lambda_0'] * jnp.exp(params['k'] * temperature)
    return lambda_T * biomass

@unit(name='growth',
      inputs=['biomass', 'recruitment_rate', 'mortality_rate'],
      outputs=['biomass'],
      scope='local',
      compiled=True)
def compute_growth(biomass, recruitment_rate, mortality_rate, dt, params):
    """Mise à jour biomasse : B_new = B + (R - M) * dt."""
    return biomass + (recruitment_rate - mortality_rate) * dt

# ════════════════════════════════════════════════════════
# UNITS GLOBALES (nécessitent voisins)
# ════════════════════════════════════════════════════════

@unit(name='diffusion',
      inputs=['biomass', 'boundary_left', 'boundary_right'],
      outputs=['biomass'],
      scope='global',
      compiled=True)
def compute_diffusion(biomass, boundary_left, boundary_right, dt, params):
    """
    Diffusion avec données des voisins.

    boundary_left: dict avec biomass du voisin gauche
    boundary_right: dict avec biomass du voisin droit
    """
    # Construire tableau étendu avec halo
    biomass_left = boundary_left.get('biomass', biomass[0])
    biomass_right = boundary_right.get('biomass', biomass[-1])

    biomass_extended = jnp.concatenate([
        jnp.array([biomass_left]),
        biomass,
        jnp.array([biomass_right])
    ])

    # Laplacien
    dx = params['dx']
    D = params['D']

    laplacian = (biomass_extended[:-2] - 2*biomass_extended[1:-1] + biomass_extended[2:]) / (dx**2)

    # Mise à jour
    return biomass + D * laplacian * dt
```

### Construction du Modèle

```python
# ════════════════════════════════════════════════════════
# DÉFINITION DU MODÈLE = LISTE DE UNITS
# ════════════════════════════════════════════════════════

# Modèle simple : croissance + mortalité (pas de transport)
simple_kernel = Kernel([
    compute_recruitment,
    compute_mortality,
    compute_growth
])

# Modèle avec transport
full_kernel = Kernel([
    compute_recruitment,
    compute_mortality,
    compute_growth,
    compute_diffusion  # Unit globale → nécessite synchronisation
])

# Modèle complexe (ajout facile de nouvelles Units)
@unit(name='temperature_update',
      inputs=['position'],
      outputs=['temperature'],
      scope='local',
      compiled=False)
def update_temperature(position, t, forcing_data):
    """Interpoler température depuis forçages."""
    # ... interpolation xarray
    return interpolated_temp

complex_kernel = Kernel([
    update_temperature,      # Nouveau : mise à jour température
    compute_recruitment,
    compute_mortality,
    compute_growth,
    compute_diffusion
])
```

### Lancement de la Simulation

```python
import ray

ray.init()

# ════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════

num_workers = 4
cells_per_worker = 25
total_cells = num_workers * cells_per_worker

params = {
    'R': 10.0,
    'lambda_0': 0.01,
    'k': 0.05,
    'D': 100.0,
    'dx': 1000.0
}

# ════════════════════════════════════════════════════════
# INITIALISATION DES WORKERS
# ════════════════════════════════════════════════════════

workers = []
for i in range(num_workers):
    # État initial pour ce worker
    initial_state = {
        'biomass': jnp.ones(cells_per_worker) * 50.0,
        'temperature': jnp.ones(cells_per_worker) * 24.0,
        'position': jnp.linspace(i*25, (i+1)*25, cells_per_worker)
    }

    cell_ids = list(range(i * cells_per_worker, (i+1) * cells_per_worker))

    worker = CellWorker.remote(
        worker_id=i,
        cell_ids=cell_ids,
        kernel=full_kernel,  # Modèle avec diffusion
        initial_state=initial_state,
        params=params
    )
    workers.append(worker)

# Connecter les voisins
for i, worker in enumerate(workers):
    neighbors = {}
    if i > 0:
        neighbors['left'] = workers[i-1]
    if i < num_workers - 1:
        neighbors['right'] = workers[i+1]

    ray.get(worker.set_neighbors.remote(neighbors))

# ════════════════════════════════════════════════════════
# LANCER LA SIMULATION (ÉVÉNEMENTIELLE)
# ════════════════════════════════════════════════════════

scheduler = EventScheduler.remote(
    workers=workers,
    dt=0.1,
    t_end=100.0
)

# PAS DE BOUCLE FOR !
# Le scheduler gère les événements de manière asynchrone
final_states = ray.get(scheduler.run.remote())

print("Simulation terminée!")
print(f"États finaux: {len(final_states)} workers")
```

---

## 6. Réponses à Vos Questions

### ✅ Question 1: Scheduler par Pas de Temps ?

**Oui**, via `EventScheduler` :
- Utilise une `PriorityQueue` d'événements
- Chaque worker planifie son prochain événement
- Pas de boucle `for t in range(...)` globale
- Workers avancent à leur propre rythme

### ✅ Question 2: Découplage Local vs Global ?

**Oui**, via `scope` des Units :
```python
# Local : exécuté indépendamment par cellule
scope='local'  → kernel.execute_local_phase()

# Global : nécessite synchronisation
scope='global' → kernel.execute_global_phase()
```

**Séparation claire** :
1. Phase locale : tous les workers en parallèle
2. Barrière de synchronisation (échange messages)
3. Phase globale : transport avec données voisins

### ✅ Question 3: Kernel Composable ?

**Oui**, via `Kernel([Unit1, Unit2, ...])` :

```python
# Modèle simple
kernel = Kernel([mortality, growth])

# Modèle complexe
kernel = Kernel([
    compute_daylength,
    compute_temp_avg,
    mortality,
    growth,
    transport
])
```

**Avantages** :
- Ajouter/retirer Units facilement
- Ordre d'exécution automatique (topologique)
- Déclaration inputs/outputs explicite
- Compilation JAX par Unit

---

## 7. Architecture Événementielle Avancée

### Worker Avec Événements Locaux Multiples

```python
@ray.remote
class AdvancedCellWorker:
    """
    Worker avec file d'événements locale.

    Permet de passer à un autre calcul quand bloqué.
    """

    def __init__(self, worker_id, cell_ids, kernel, initial_state, params):
        # ... (comme avant)

        # File d'événements locale
        self.local_queue = []

    async def step_with_events(self, dt: float):
        """
        Version avancée : si bloqué, traiter d'autres événements.
        """
        # Ajouter événement principal : exécuter kernel
        self.local_queue.append(('execute_local', dt))

        while self.local_queue:
            event_type, event_data = self.local_queue.pop(0)

            if event_type == 'execute_local':
                # Phase locale (non-bloquante)
                self.state = self.kernel.execute_local_phase(
                    self.state, dt, self.params
                )

                # Si Units globales, demander voisins
                if self.kernel.has_global_units():
                    # Envoyer requêtes (non-bloquant)
                    self.local_queue.append(('wait_neighbors', dt))

            elif event_type == 'wait_neighbors':
                # Lancer requêtes
                futures = self._request_neighbor_data()

                # Ajouter événement pour traiter réponses
                self.local_queue.append(('process_neighbors', (dt, futures)))

                # EN ATTENDANT : traiter d'autres événements indépendants
                # (exemple : calcul de statistiques, diagnostics, etc.)
                if self._has_independent_work():
                    self.local_queue.insert(0, ('independent_work', None))

            elif event_type == 'independent_work':
                # Faire du travail qui ne dépend pas des voisins
                self._compute_diagnostics()

            elif event_type == 'process_neighbors':
                dt, futures = event_data

                # Maintenant on attend vraiment (bloquant)
                neighbor_data = await self._gather_neighbor_data(futures)

                # Phase globale
                self.state = self.kernel.execute_global_phase(
                    self.state, dt, self.params, neighbor_data
                )

        self.t += dt
        return {'state': self.state, 't': self.t}
```

---

## 8. Comparaison Avant/Après

### ❌ Approche Classique (Mon Erreur Initiale)

```python
# Boucle for globale
for t in range(num_steps):
    # Tous les workers font la même chose en même temps
    for worker in workers:
        state = worker.compute_growth(state, dt)
        state = worker.compute_mortality(state, dt)

    # Transport séparé
    for worker in workers:
        state = worker.transport(state, dt)
```

**Problèmes** :
- Boucle séquentielle sur le temps
- Pas de composabilité (functions hardcodées)
- Pas de découplage local/global clair

### ✅ Votre Architecture (Correcte)

```python
# Définition du modèle = liste de Units
kernel = Kernel([
    unit1_local,   # Indépendant
    unit2_local,   # Indépendant
    unit3_global   # Nécessite voisins
])

# Workers avec kernel
workers = [CellWorker(kernel=kernel) for _ in range(N)]

# Scheduler événementiel (PAS DE FOR)
scheduler = EventScheduler(workers)
scheduler.run()  # Gère les événements automatiquement
```

**Avantages** :
- ✅ Scheduler événementiel
- ✅ Kernel composable
- ✅ Découplage local/global explicite
- ✅ Extensibilité facile

---

## Prochaines Étapes

Voulez-vous que je :

1. **Implémente Phase 1** avec cette architecture correcte ?
2. **Crée un exemple minimal** exécutable (2-3 Units, 2 workers) ?
3. **Détaille un aspect spécifique** (EventScheduler, tri topologique, etc.) ?

Quelle direction préférez-vous ?
