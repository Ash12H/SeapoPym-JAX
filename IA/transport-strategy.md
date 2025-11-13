# Stratégie Transport : Deux Versions Possibles

## Réponse Courte

✅ **Oui, les deux approches sont possibles et compatibles avec votre architecture !**

Grâce au système de **Kernel composable** et **Units modulaires**, vous pouvez avoir :

1. **Version Simple** : transport distribué (local dans chaque worker)
2. **Version Précise** : transport centralisé (worker spécialisé externe)

Et **passer de l'une à l'autre facilement** sans réécrire toute l'architecture.

---

## Vue d'Ensemble

```
                    VOTRE ARCHITECTURE
                    (Kernel composable)
                           │
                ┌──────────┴──────────┐
                │                     │
         Version Simple        Version Précise
         (Prototype)           (Production)
                │                     │
         Transport dans         Transport externe
         chaque worker          (TransportWorker)
                │                     │
         ┌──────┴──────┐       ┌─────┴─────┐
         │             │       │           │
    Diffusion   Advection   JAX-CFD    Schémas
    manuelle    basique     complet    publiés
```

---

## Version 1 : Transport Distribué (Simple)

### Architecture

```
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  CellWorker 0   │   │  CellWorker 1   │   │  CellWorker 2   │
│                 │   │                 │   │                 │
│  Kernel:        │   │  Kernel:        │   │  Kernel:        │
│  ├─ biogeo_1    │   │  ├─ biogeo_1    │   │  ├─ biogeo_1    │
│  ├─ biogeo_2    │   │  ├─ biogeo_2    │   │  ├─ biogeo_2    │
│  ├─ diffusion ◄─┼───┼─►├─ diffusion ◄─┼───┼─►├─ diffusion   │
│  └─ advection   │   │  └─ advection   │   │  └─ advection   │
└─────────────────┘   └─────────────────┘   └─────────────────┘
   Halo: ~400 vals     Halo: ~400 vals      Halo: ~400 vals
```

### Code

```python
# Version Simple : tout dans le worker

# Units de transport (scope='global')
@unit(name='diffusion_simple',
      inputs=['biomass'],
      outputs=['biomass'],
      scope='global')
def diffusion_manual(biomass, dt, params, halo_north, halo_south,
                     halo_east, halo_west):
    """Diffusion par différences finies simples."""
    # Construire domaine étendu avec halo
    # Laplacien 2D
    # Retourner biomasse diffusée
    return biomass_new

# Kernel complet
kernel_simple = Kernel([
    compute_recruitment,      # local
    compute_mortality,        # local
    compute_growth,           # local
    diffusion_manual,         # global (dans le worker)
    advection_simple          # global (dans le worker)
])

# Workers identiques
workers = [
    CellWorker2D.remote(id=i, kernel=kernel_simple, ...)
    for i in range(24)
]
```

### Caractéristiques

| Aspect | Valeur |
|--------|--------|
| **Complexité code** | Simple (un seul type de worker) |
| **Communication** | Minimale (~10 KB/timestep) |
| **Scalabilité** | Excellente (linéaire) |
| **Précision numérique** | Moyenne (schémas ordre 2) |
| **Performance** | ~17 ms/timestep |
| **Usage** | ✅ Prototype, validation architecture |

### Avantages ✅

- Architecture la plus simple
- Maximum de parallélisme
- Communication minimale (halos uniquement)
- Scalabilité prouvée
- Cohérent avec vision initiale (workers autonomes)

### Limitations ⚠️

- Schémas numériques basiques (ordre 2)
- Duplication de logique transport dans chaque worker
- Moins adapté pour couplage multi-modèles

---

## Version 2 : Transport Externe (Précis)

### Architecture

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│CellWorker 0 │  │CellWorker 1 │  │CellWorker 2 │
│             │  │             │  │             │
│ Kernel:     │  │ Kernel:     │  │ Kernel:     │
│ ├─ biogeo_1 │  │ ├─ biogeo_1 │  │ ├─ biogeo_1 │
│ ├─ biogeo_2 │  │ ├─ biogeo_2 │  │ ├─ biogeo_2 │
│ └─ biogeo_3 │  │ └─ biogeo_3 │  │ └─ biogeo_3 │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        ↓ Envoyer état (240 KB)
              ┌─────────────────────┐
              │  TransportWorker    │
              │  (Ray Actor)        │
              │                     │
              │  Kernel:            │
              │  ├─ jax_cfd_adv     │
              │  └─ jax_cfd_diff    │
              │                     │
              │  GPU A100           │
              └─────────────────────┘
                        ↓ Redistribuer (240 KB)
       ┌────────────────┼────────────────┐
       ↓                ↓                ↓
```

### Code

```python
# Version Précise : transport externe

# Kernel simplifié pour CellWorkers (biogéo seulement)
kernel_bio_only = Kernel([
    compute_recruitment,      # local
    compute_mortality,        # local
    compute_growth            # local
])

# Kernel pour TransportWorker
from jax_cfd.base import advection, diffusion

@unit(name='transport_jax_cfd',
      inputs=['biomass', 'velocity_u', 'velocity_v'],
      outputs=['biomass'],
      scope='global')
def transport_precise(biomass, velocity_u, velocity_v, dt, params):
    """Transport JAX-CFD haute précision."""
    grid = params['jax_grid']

    # Advection (schéma Van Leer ou WENO)
    biomass = advection.advect_van_leer(
        c=biomass,
        v=(velocity_u, velocity_v),
        dt=dt,
        grid=grid
    )

    # Diffusion (Crank-Nicolson implicite)
    biomass = diffusion.solve_crank_nicolson(
        c=biomass,
        D=params['D'],
        dt=dt,
        grid=grid
    )

    return biomass

kernel_transport = Kernel([
    transport_precise
])

# Deux types de workers
cell_workers = [
    CellWorker2D.remote(id=i, kernel=kernel_bio_only, ...)
    for i in range(24)
]

transport_worker = TransportWorker.remote(
    kernel=kernel_transport,
    num_gpus=1,
    grid_info=grid_info
)

# Scheduler coordonné
scheduler = CoordinatedScheduler.remote(
    cell_workers=cell_workers,
    transport_worker=transport_worker,
    dt=0.1,
    t_end=365.0
)
```

### Flux d'Exécution

```python
class CoordinatedScheduler:
    async def run(self):
        while t < t_end:
            # 1. Phase biogéochimie (parallèle)
            bio_futures = [w.step_bio.remote(dt) for w in cell_workers]
            await asyncio.gather(*bio_futures)

            # 2. Collecte états
            states = await self._collect_states(cell_workers)

            # 3. Transport centralisé
            global_state = self._reconstruct_global(states)
            transported = await transport_worker.step.remote(
                global_state, dt
            )

            # 4. Redistribution
            await self._distribute_states(transported, cell_workers)

            t += dt
```

### Caractéristiques

| Aspect | Valeur |
|--------|--------|
| **Complexité code** | Moyenne (deux types workers, coordination) |
| **Communication** | Élevée (~480 KB/timestep) |
| **Scalabilité** | Limitée (goulot TransportWorker) |
| **Précision numérique** | Excellente (schémas publiés) |
| **Performance** | ~80 ms/timestep |
| **Usage** | ✅ Production, publications scientifiques |

### Avantages ✅

- Schémas numériques publiés (Van Leer, WENO, Crank-Nicolson)
- JAX-CFD natif (validation scientifique)
- Séparation claire biogéo/physique
- GPU dédié pour transport
- Facilite couplage multi-modèles

### Limitations ⚠️

- Communication 48× plus élevée
- Goulot d'étranglement (TransportWorker)
- Scalabilité plafonnée
- Complexité accrue (coordination)

---

## Transition Entre les Deux Versions

### Clé : Architecture Modulaire

Les **Units** permettent de changer de stratégie **sans réécrire le reste** !

```python
# Même base de code, juste changement de configuration

# ═══════════════════════════════════════════════
# CONFIG 1 : Transport distribué
# ═══════════════════════════════════════════════

config = {
    'transport_strategy': 'distributed',
    'kernel': Kernel([
        bio_units...,
        transport_distributed_units...  # Dans chaque worker
    ]),
    'worker_type': CellWorker2D
}

# ═══════════════════════════════════════════════
# CONFIG 2 : Transport externe (juste changer config !)
# ═══════════════════════════════════════════════

config = {
    'transport_strategy': 'centralized',
    'kernel_bio': Kernel([bio_units...]),
    'kernel_transport': Kernel([transport_jax_cfd...]),
    'worker_types': {
        'cell': CellWorker2D,
        'transport': TransportWorker
    }
}
```

### Étapes de Migration

```
Phase 1 : Prototype (Version 1)
  ├─ Implémenter transport distribué manuel
  ├─ Valider architecture Ray + Kernel
  └─ Tests scalabilité (1 → 10 → 100 workers)
        ↓
  Fonctionne bien ? ✓
        ↓
Phase 2 : Raffinage (Version 1 optimisée)
  ├─ Optimiser schémas numériques
  ├─ Ajouter forçages de courants
  └─ Validation scientifique basique
        ↓
  Besoin précision publiable ?
        ↓
Phase 3 : Production (Version 2)
  ├─ Créer TransportWorker
  ├─ Intégrer JAX-CFD complet
  ├─ Benchmark vs Version 1
  └─ Choisir selon cas d'usage
```

### Comparaison Directe Possible

```python
# Vous pourrez comparer les deux !

# Simulation A : transport distribué
results_v1 = run_simulation(config_distributed)

# Simulation B : transport externe
results_v2 = run_simulation(config_centralized)

# Comparaison
compare_results(results_v1, results_v2)
# - Précision numérique
# - Temps d'exécution
# - Scalabilité
```

---

## Cas d'Usage Recommandés

### Version 1 (Distribué) ✅

**Utiliser pour :**
- ✅ Prototypage et développement initial
- ✅ Études de sensibilité (nombreuses simulations)
- ✅ Grilles très grandes (>500×500)
- ✅ Cluster HPC classique (CPU)
- ✅ Transport simple suffit

**Exemple :**
> "Je teste 1000 scénarios de recrutement différents sur le Pacifique équatorial (240×360), j'ai besoin de vitesse."

### Version 2 (Externe) ✅

**Utiliser pour :**
- ✅ Publications scientifiques (précision requise)
- ✅ Couplage multi-modèles (plusieurs biogéos)
- ✅ GPU massif disponible (A100, H100)
- ✅ Validation avec modèles publiés
- ✅ Forçages océaniques haute résolution

**Exemple :**
> "Je veux publier des résultats de dynamique de population avec transport réaliste issu de NEMO, schémas validés."

### Hybride (Avancé) 🔄

**Combiner les deux :**
- Transport distribué (Version 1)
- + Service centralisé pour forçages (Version 2 partielle)

```
Workers font transport localement (scalable)
     ↓
ForcingService (centralisé) : lit NetCDF U/V, cache, interpole
```

---

## Feuille de Route Recommandée

### Étape 1 : Prototype (Version 1) [2-3 semaines]

```
✓ Architecture Ray + Kernel + EventScheduler
✓ CellWorker2D avec transport distribué
✓ Diffusion manuelle (différences finies)
✓ Advection simple (upwind)
✓ Tests : 2×2 workers, grille 20×20
✓ Validation : équilibre, conservation masse
```

**Livrable :** Architecture fonctionnelle prouvée

### Étape 2 : Raffinement (Version 1+) [2-3 semaines]

```
✓ Forçages température depuis NetCDF
✓ R(T), λ(T) dépendants température
✓ Scalabilité : 4×6 = 24 workers
✓ Grille réaliste : 120×180 (Pacifique équatorial)
✓ Validation scientifique basique
```

**Livrable :** Modèle SEAPOPYM simplifié opérationnel

### Étape 3 : Production (Version 2) [3-4 semaines]

```
✓ Créer TransportWorker
✓ Intégrer JAX-CFD (advection Van Leer, diffusion CN)
✓ Forçages courants U, V depuis NEMO
✓ Benchmark Version 1 vs Version 2
✓ Validation vs publications
✓ Documentation complète
```

**Livrable :** Deux versions disponibles, publiables

---

## Décision Maintenant

### Question pour Vous

**Pour démarrer, quelle version vous semble la plus logique ?**

**Option A : Version 1 directement** ⭐ *Recommandé*
- On implémente transport distribué
- Valide l'architecture complète rapidement
- Peut évoluer vers Version 2 plus tard

**Option B : Version 2 directement**
- On va directement vers TransportWorker
- Plus complexe dès le début
- Risque de ne pas valider scalabilité

**Option C : Deux prototypes en parallèle**
- On code les deux versions minimal
- Comparaison directe dès le début
- Plus de travail initial

---

## Résumé Exécutif

### Les Deux Versions Sont Possibles ✅

| Version | Phase | Usage | Scalabilité | Précision |
|---------|-------|-------|-------------|-----------|
| **V1 : Distribué** | Prototype → Production | Études sensibilité | ★★★★★ | ★★★☆☆ |
| **V2 : Externe** | Production scientifique | Publications | ★★★☆☆ | ★★★★★ |

### Architecture Permet les Deux ✅

Grâce à :
- **Kernel composable** (Units modulaires)
- **Scope local/global** (séparation claire)
- **Ray Actors flexibles** (plusieurs types possibles)

### Recommandation ⭐

1. **Démarrer avec Version 1** (transport distribué)
   - Valide architecture rapidement
   - Maximum scalabilité
   - Simple

2. **Évoluer vers Version 2** (si besoin)
   - Quand précision publiable requise
   - Quand couplage multi-modèles
   - Architecture permet migration facile

### Vous Aurez les Deux ✅

À terme, votre bibliothèque offrira :

```python
# Simple API avec deux modes

# Mode développement/études
model = SEAPOPYMModel(transport='distributed')

# Mode production/publication
model = SEAPOPYMModel(transport='centralized')
```

**Flexibilité maximale !**

---

## Prochaine Étape

**Prêt à implémenter le prototype avec Version 1 (transport distribué) ?**

Si oui, je peux :
1. **Créer la structure de projet** (src/, tests/, config/)
2. **Implémenter exemple minimal** (2×2 workers, grille simple)
3. **Autre chose** ?

**Qu'en dites-vous ?**
