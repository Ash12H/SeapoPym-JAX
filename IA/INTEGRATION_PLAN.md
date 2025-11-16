# Plan d'Intégration TransportWorker avec EventScheduler

## Statut actuel

✅ **Phase 2A complète** : TransportWorker avec physique (advection + diffusion)
- Grilles sphériques/planes
- Conservation > 99% sur 10 jours
- 52 tests unitaires

🎯 **Objectif Phase 3** : Intégrer TransportWorker dans le workflow distribué

---

## Architecture actuelle (sans transport)

```
EventScheduler.step():
  1. Prépare forcings (ForcingManager)
  2. Lance CellWorker.step() en parallèle (biologie)
  3. Attend completion
  4. Agrège diagnostics
```

**Problème** : Chaque CellWorker gère sa propre patch locale, mais le transport nécessite la grille globale pour garantir la conservation.

---

## Architecture cible (avec transport)

```
EventScheduler.step():
  ┌─────────────────────────────────────────┐
  │ 1. PHASE BIOLOGIE (parallèle)          │
  │    - CellWorker[i].biology_step()      │
  │    - Compute growth, mortality, etc.   │
  │    - Pas de transport                  │
  └─────────────────────────────────────────┘
               ↓
  ┌─────────────────────────────────────────┐
  │ 2. COLLECTE BIOMASSE                    │
  │    - Récupérer state['biomass'] de      │
  │      tous les CellWorkers               │
  │    - Assembler grille globale (nlat×n)  │
  └─────────────────────────────────────────┘
               ↓
  ┌─────────────────────────────────────────┐
  │ 3. PHASE TRANSPORT (centralisé)         │
  │    - TransportWorker.transport_step()   │
  │    - Advection + Diffusion globale      │
  │    - Forcings : u, v, D, mask           │
  │    - Conservation garantie              │
  └─────────────────────────────────────────┘
               ↓
  ┌─────────────────────────────────────────┐
  │ 4. REDISTRIBUTION BIOMASSE              │
  │    - Découper grille globale en patches │
  │    - CellWorker[i].set_biomass(patch)   │
  └─────────────────────────────────────────┘
               ↓
  ┌─────────────────────────────────────────┐
  │ 5. AGRÉGATION DIAGNOSTICS               │
  │    - Diagnostics biologie + transport   │
  │    - Conservation globale               │
  └─────────────────────────────────────────┘
```

---

## Modifications nécessaires

### 1. **EventScheduler** (scheduler.py)

**Ajouts** :
- `transport_worker: TransportWorker` (Ray actor)
- `transport_enabled: bool` (flag activation)
- Méthode `_collect_global_biomass()` → assemble grille depuis workers
- Méthode `_redistribute_biomass()` → distribue grille vers workers
- Modifier `step()` pour workflow 5 phases

**Workflow modifié** :
```python
def step(self):
    # 1. Phase biologie (existant)
    forcings_ref = self.forcing_manager.prepare_timestep(...)
    futures = [w.biology_step.remote(dt, forcings_ref) for w in self.workers]
    bio_diagnostics = ray.get(futures)

    # 2. Collecte biomasse (nouveau)
    biomass_global = self._collect_global_biomass()

    # 3. Phase transport (nouveau, si activé)
    if self.transport_enabled:
        u, v, D, mask = self._get_transport_forcings()
        transport_result = ray.get(
            self.transport_worker.transport_step.remote(
                biomass=biomass_global,
                u=u, v=v, D=D, dt=self.dt, mask=mask
            )
        )
        biomass_global = transport_result['biomass']
        transport_diag = transport_result['diagnostics']

    # 4. Redistribution (nouveau)
    self._redistribute_biomass(biomass_global)

    # 5. Agrégation diagnostics (modifié)
    return self._aggregate_all_diagnostics(bio_diagnostics, transport_diag)
```

---

### 2. **CellWorker2D** (worker.py)

**Modifications** :

**Option A** : Séparer biologie et transport
```python
def biology_step(self, dt, forcings_ref):
    """Exécute seulement la biologie (growth, mortality, recruitment)."""
    # Ancien step() mais sans transport
    forcings = ray.get(forcings_ref) if forcings_ref else {}
    self.state = self.kernel.step(self.state, self.params, dt, forcings)
    return self._compute_diagnostics()

def set_biomass(self, biomass_patch):
    """Reçoit biomasse après transport."""
    self.state['biomass'] = biomass_patch

def get_biomass(self):
    """Retourne biomass patch."""
    return self.state.get('biomass', jnp.zeros((self.nlat, self.nlon)))
```

**Option B** : Garder step() et ajouter flag
```python
def step(self, dt, forcings_ref, transport_enabled=False):
    """Exécute biologie, optionnellement transport."""
    # Si transport désactivé localement, skip transport dans kernel
    ...
```

**Recommandation** : **Option A** (plus clair, séparation explicite)

---

### 3. **ForcingManager** (forcing/manager.py)

**Ajouts** :
- Méthodes pour préparer forcings transport :
  - `get_velocity_fields(time)` → retourne (u, v)
  - `get_diffusivity(time)` → retourne D
  - `get_ocean_mask()` → retourne mask

**Exemple** :
```python
def get_transport_forcings(self, time):
    """Prépare forcings pour TransportWorker."""
    u = self.get_forcing('u', time)  # Velocity zonal
    v = self.get_forcing('v', time)  # Velocity meridional
    D = self.params.get('horizontal_diffusivity', 1000.0)
    mask = self.get_forcing('ocean_mask', time)

    return {'u': u, 'v': v, 'D': D, 'mask': mask}
```

---

### 4. **TransportWorker** (transport/worker.py)

**Modifications** : ✅ Déjà compatible (aucune modification nécessaire)

Interface actuelle parfaite :
```python
result = transport_worker.transport_step.remote(
    biomass=biomass_global,  # (nlat, nlon)
    u=u, v=v, D=D, dt=dt, mask=mask
)
# Returns: {'biomass': ..., 'diagnostics': {...}}
```

---

## Stratégie de collecte/redistribution

### Collecte : Workers → Grille globale

```python
def _collect_global_biomass(self):
    """Assemble biomasse globale depuis tous les workers."""
    # 1. Récupérer biomass de chaque worker
    futures = [w.get_biomass.remote() for w in self.workers]
    patches = ray.get(futures)

    # 2. Assembler selon la topologie des patches
    biomass_global = jnp.zeros((self.global_nlat, self.global_nlon))
    for i, worker_info in enumerate(self.worker_topology):
        lat_slice = slice(worker_info['lat_start'], worker_info['lat_end'])
        lon_slice = slice(worker_info['lon_start'], worker_info['lon_end'])
        biomass_global = biomass_global.at[lat_slice, lon_slice].set(patches[i])

    return biomass_global
```

### Redistribution : Grille globale → Workers

```python
def _redistribute_biomass(self, biomass_global):
    """Redistribue biomasse transportée vers workers."""
    futures = []
    for i, worker_info in enumerate(self.worker_topology):
        lat_slice = slice(worker_info['lat_start'], worker_info['lat_end'])
        lon_slice = slice(worker_info['lon_start'], worker_info['lon_end'])
        patch = biomass_global[lat_slice, lon_slice]

        futures.append(self.workers[i].set_biomass.remote(patch))

    # Attendre que tous les workers aient reçu la biomasse
    ray.get(futures)
```

---

## Étapes d'implémentation

### Phase 3.1-3.2 : Analyse et Architecture (1-2h)
- ✅ Analyser workflow actuel
- ✅ Définir architecture intégration (ce document)
- Décider : Option A vs B pour CellWorker

### Phase 3.3 : Modification EventScheduler (2-3h)
- Ajouter `transport_worker` parameter
- Implémenter `_collect_global_biomass()`
- Implémenter `_redistribute_biomass()`
- Modifier `step()` avec workflow 5 phases
- Sauvegarder topologie workers (lat/lon ranges)

### Phase 3.4-3.5 : Collecte/Redistribution (1-2h)
- Tester collecte sur grille simple (2×2 workers)
- Tester redistribution avec conservation
- Vérifier pas de perte aux frontières patches

### Phase 3.6 : Gestion forcings (1-2h)
- Ajouter méthodes transport à ForcingManager
- Préparer u, v, D, mask pour TransportWorker
- Gérer cas temporel variable (si nécessaire)

### Phase 3.7 : Tests d'intégration (2-3h)
- Test 1 : Biologie seule (transport disabled)
- Test 2 : Transport seul (pas de biologie)
- Test 3 : Biologie + Transport couplé
- Test 4 : Conservation sur simulation longue (10 jours)
- Test 5 : 4 workers × 10 cohorts

### Phase 3.8 : Benchmarks (1h)
- Temps biologie vs temps transport
- Scalabilité : 1, 4, 16 workers
- Overhead collecte/redistribution

### Phase 3.9 : Exemple complet (1-2h)
- Script ou notebook simulation réaliste
- Domaine SEAPOPYM typique (60°S-60°N, 360°)
- Forcings réalistes (currents, diffusion)
- Visualisations

### Phase 3.10 : Documentation (1h)
- Mettre à jour README transport module
- Docstrings nouvelles méthodes
- Guide utilisateur : activer/désactiver transport

---

## Tests de validation

### Test 1 : Conservation patch boundaries
```python
# Vérifier que collecte + redistribution conserve la masse
biomass_before = sum(worker.get_total_mass() for all workers)
biomass_global = _collect_global_biomass()
_redistribute_biomass(biomass_global)
biomass_after = sum(worker.get_total_mass() for all workers)

assert abs(biomass_after - biomass_before) < 1e-6
```

### Test 2 : Transport disabled = no change
```python
# Transport désactivé → biomasse inchangée
scheduler = EventScheduler(..., transport_enabled=False)
biomass_0 = _collect_global_biomass()
scheduler.step()
biomass_1 = _collect_global_biomass()

assert jnp.allclose(biomass_0, biomass_1)
```

### Test 3 : Conservation globale 10 jours
```python
# Simulation 240 timesteps avec biologie + transport
diagnostics = scheduler.run()
conservation = [d['conservation_fraction'] for d in diagnostics]

assert all(c > 0.99 for c in conservation)  # >99% à chaque step
```

---

## Points d'attention

### ⚠️ Performance
- **Collecte/redistribution** : Peut être coûteuse si workers nombreux
- **Solution** : Utiliser `ray.put()` pour objets volumineux
- **Optimisation** : Collecter seulement variables transportées (pas tout le state)

### ⚠️ Synchronisation
- Transport doit attendre **tous** les workers (biologie complète)
- Pas de pipeline possible (biologie step N+1 doit attendre transport step N)

### ⚠️ Masques cohérents
- Mask ocean doit être cohérent entre workers et TransportWorker
- Vérifier alignement grilles lors de l'initialisation

### ⚠️ Conditions limites workers
- Frontières entre patches : gérer halo exchange ? Non, transport global résout ça
- Mais vérifier cohérence indices (pas de gaps, pas d'overlap)

---

## Critères de succès Phase 3

| Critère | Cible | Validation |
|---------|-------|------------|
| Conservation collecte/redistribution | 100% | Test unitaire |
| Conservation simulation 10 jours | >99% | Test intégration |
| Overhead collecte/redistribution | <10% | Benchmark |
| Tests intégration | 5 tests passent | pytest |
| Documentation | API complète | Revue |

---

## Prochaine étape

**Commencer Phase 3.3** : Modifier EventScheduler
- Implémenter collecte/redistribution
- Modifier workflow step()
- Tester sur cas simple (2×2 workers, grille 20×20)
