# Comparaison : Transport Distribué vs Worker Spécialisé

## Question

**Est-ce que l'advection et la diffusion peuvent être gérés à l'extérieur des workers de cellules ?**
**Par exemple : un worker spécialisé TransportWorker ?**

---

## Option 1 : Transport Distribué (Architecture Actuelle)

### Principe

Chaque `CellWorker` gère **à la fois** :
- La biogéochimie locale (croissance, mortalité)
- Le transport local (advection, diffusion) avec échange de halo

```
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  CellWorker 0   │   │  CellWorker 1   │   │  CellWorker 2   │
│                 │   │                 │   │                 │
│  - Biogéo       │   │  - Biogéo       │   │  - Biogéo       │
│  - Diffusion    │◄──┼──► Halo        │◄──┼──► Diffusion    │
│  - Advection    │   │  - Advection    │   │  - Advection    │
└─────────────────┘   └─────────────────┘   └─────────────────┘
     Patch 0               Patch 1               Patch 2
```

### Flux d'Exécution

```
Pour chaque timestep :
  1. Phase locale (biogéo) : parallèle
     - Worker 0, 1, 2 calculent simultanément

  2. Échange halo : peer-to-peer
     - Worker 0 ↔ Worker 1
     - Worker 1 ↔ Worker 2

  3. Phase globale (transport) : parallèle
     - Chaque worker fait sa diffusion/advection locale
```

### Avantages ✅

| Aspect | Détail |
|--------|--------|
| **Scalabilité** | Parfaite : chaque worker indépendant |
| **Localité des données** | Excellente : pas de transfert massif |
| **Communication** | Minimale : uniquement halos (frontières) |
| **Parallélisme** | Maximal : N workers calculent en même temps |
| **Latence** | Faible : pas de goulot d'étranglement |

### Inconvénients ❌

| Aspect | Détail |
|--------|--------|
| **Complexité du code** | Chaque worker doit gérer le transport |
| **Duplication de logique** | Schémas numériques dans chaque worker |
| **Halo management** | Manuel, risque d'erreurs |

### Coût de Communication

```python
# Par timestep, par worker :
# - Envoyer 4 frontières (N, S, E, W) : O(√n) éléments (où n = taille patch)
# - Recevoir 4 frontières : O(√n)

# Exemple : patch 100×100
# - Frontière nord/sud : 100 valeurs
# - Frontière est/ouest : 100 valeurs
# Total par worker : ~400 valeurs échangées

# Pour 24 workers : 24 × 400 = ~10 000 valeurs/timestep
```

**Communication distribuée, faible latence**

---

## Option 2 : Worker Spécialisé TransportWorker

### Principe

Séparation des responsabilités :
- **CellWorkers** : biogéochimie uniquement (local)
- **TransportWorker(s)** : advection + diffusion pour tout le domaine

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ CellWorker0 │  │ CellWorker1 │  │ CellWorker2 │
│             │  │             │  │             │
│ - Biogéo    │  │ - Biogéo    │  │ - Biogéo    │
│   only      │  │   only      │  │   only      │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        ↓ Envoyer état complet
              ┌─────────────────────┐
              │  TransportWorker    │
              │                     │
              │  - Reçoit grille    │
              │  - Advection globale│
              │  - Diffusion globale│
              │  - Renvoie résultat │
              └─────────────────────┘
                        ↓ Redistribuer
       ┌────────────────┼────────────────┐
       ↓                ↓                ↓
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ CellWorker0 │  │ CellWorker1 │  │ CellWorker2 │
│ (mis à jour)│  │ (mis à jour)│  │ (mis à jour)│
└─────────────┘  └─────────────┘  └─────────────┘
```

### Flux d'Exécution

```
Pour chaque timestep :
  1. Phase locale (biogéo) : parallèle
     - Workers 0, 1, 2 calculent simultanément

  2. Collecte : centralisée
     - Tous les workers envoient leur état → TransportWorker
     - TransportWorker reconstruit la grille globale

  3. Transport : centralisé (ou semi-distribué)
     - TransportWorker calcule advection/diffusion
     - Sur toute la grille (ou gros morceaux)

  4. Redistribution : centralisée
     - TransportWorker découpe et renvoie à chaque CellWorker
```

### Avantages ✅

| Aspect | Détail |
|--------|--------|
| **Séparation des responsabilités** | Biogéo vs physique claire |
| **Simplicité CellWorker** | Ne gère que la biogéo locale |
| **Schémas globaux** | Possibilité d'utiliser JAX-CFD natif sur grille complète |
| **Couplage multiple** | Plusieurs modèles de biogéo peuvent partager le même transport |
| **Expertise numérique** | TransportWorker peut être ultra-optimisé (GPU, schémas avancés) |

### Inconvénients ❌

| Aspect | Détail |
|--------|--------|
| **Goulot d'étranglement** | TransportWorker devient un point central |
| **Communication massive** | Transfert de grilles complètes (ou gros morceaux) |
| **Latence** | Attendre collecte → calcul → redistribution |
| **Scalabilité limitée** | TransportWorker ne scale pas linéairement |
| **Mémoire** | TransportWorker doit stocker beaucoup de données |

### Coût de Communication

```python
# Par timestep :

# 1. Collecte (tous → TransportWorker)
# Chaque worker envoie son patch complet : n valeurs (ex: 100×100 = 10 000)
# 24 workers : 24 × 10 000 = 240 000 valeurs

# 2. Redistribution (TransportWorker → tous)
# TransportWorker renvoie les patches mis à jour : 240 000 valeurs

# Total : ~480 000 valeurs/timestep

# Comparaison avec Option 1 : 10 000 valeurs/timestep
# Ratio : 48× plus de communication !
```

**Communication centralisée, latence élevée**

---

## Variante : TransportWorker Semi-Distribué

### Principe

Compromis : plusieurs TransportWorkers spécialisés (pas qu'un seul).

```
┌─────┐ ┌─────┐     ┌─────┐ ┌─────┐     ┌─────┐ ┌─────┐
│ CW0 │ │ CW1 │     │ CW2 │ │ CW3 │     │ CW4 │ │ CW5 │
└──┬──┘ └──┬──┘     └──┬──┘ └──┬──┘     └──┬──┘ └──┬──┘
   └───────┼───────────┘       │             │       │
           ↓                   ↓             ↓
    ┌──────────┐        ┌──────────┐   ┌──────────┐
    │Transport │        │Transport │   │Transport │
    │ Worker 0 │        │ Worker 1 │   │ Worker 2 │
    └──────────┘        └──────────┘   └──────────┘
```

Chaque TransportWorker gère un sous-domaine (ex: 2 patches).

### Avantages vs Option 2 Pure

- ✅ Réduit le goulot d'étranglement
- ✅ Communication moins massive
- ✅ Meilleure scalabilité

### Inconvénients vs Option 2 Pure

- ⚠️ Complexité accrue (coordination entre TransportWorkers)
- ⚠️ Toujours plus de communication que Option 1

---

## Comparaison Quantitative

### Scénario : Grille 240 lat × 360 lon, 24 CellWorkers (4×6)

| Métrique | Option 1 (Distribué) | Option 2 (Centralisé) | Ratio |
|----------|---------------------|---------------------|-------|
| **Communication/timestep** | ~10 KB | ~480 KB | 48× |
| **Latence transport** | Faible (parallèle) | Élevée (séquentiel) | 5-10× |
| **Scalabilité workers** | Linéaire | Plafonnée | - |
| **Complexité CellWorker** | Moyenne | Faible | - |
| **Complexité système** | Faible | Moyenne | - |

### Temps d'Exécution Estimé (par timestep)

```
Option 1 (Distribué) :
  - Biogéo locale : 10 ms (parallèle)
  - Échange halo : 2 ms (peer-to-peer)
  - Transport local : 5 ms (parallèle)
  Total : ~17 ms

Option 2 (Centralisé) :
  - Biogéo locale : 10 ms (parallèle)
  - Collecte données : 10 ms (réseau)
  - Transport global : 50 ms (séquentiel sur TransportWorker)
  - Redistribution : 10 ms (réseau)
  Total : ~80 ms

Rapport : 4.7× plus lent
```

*Note : valeurs indicatives, dépendent fortement du réseau, taille grille, etc.*

---

## Quand Utiliser Chaque Option ?

### ✅ Option 1 (Distribué) est meilleure si :

- **Scalabilité prioritaire** : vous voulez passer à 100+ workers
- **Faible latence** : simulations longues (>1000 timesteps)
- **Transport simple** : diffusion basique suffit
- **Cluster HPC classique** : bonne bande passante peer-to-peer

### ✅ Option 2 (Centralisé) est meilleure si :

- **Transport complexe** : schémas avancés, JAX-CFD complet
- **Couplage multiple** : plusieurs modèles de biogéo partagent le même transport
- **GPU massif dédié** : TransportWorker a un A100 pour lui seul
- **Grille petite/moyenne** : <100×100 cellules totales
- **Validation scientifique** : besoin de schémas publiés exactement

### 🤔 Option 2.5 (Semi-distribué) si :

- **Compromis** : besoin de scalabilité mais transport complexe
- **Domaines hétérogènes** : résolution variable (haute près des côtes)

---

## Impact sur l'Architecture Kernel

### Option 1 : Units dans le Kernel

```python
# Le transport fait partie du Kernel de chaque worker
kernel = Kernel([
    compute_recruitment,    # local
    compute_mortality,      # local
    compute_growth,         # local
    compute_diffusion,      # global (dans le worker)
    compute_advection       # global (dans le worker)
])
```

**Le worker exécute tout**

### Option 2 : Transport Externe

```python
# Kernel de CellWorker : uniquement biogéo
bio_kernel = Kernel([
    compute_recruitment,    # local
    compute_mortality,      # local
    compute_growth          # local
])

# Kernel séparé pour TransportWorker
transport_kernel = Kernel([
    compute_diffusion,      # global
    compute_advection       # global
])
```

**Deux types de workers, deux kernels**

---

## Impact sur le Scheduler

### Option 1 : Scheduler Simple

```python
# Tous les workers sont identiques
scheduler = EventScheduler(workers=[w1, w2, w3, ...])
```

### Option 2 : Scheduler Coordonné

```python
# Scheduler doit gérer deux types de workers
scheduler = CoordinatedScheduler(
    cell_workers=[cw1, cw2, cw3, ...],
    transport_workers=[tw1]
)

# Flux modifié :
# 1. Event: CellWorkers step (biogéo)
# 2. Event: Collecte → TransportWorker
# 3. Event: TransportWorker step (transport)
# 4. Event: Redistribution → CellWorkers
```

**Plus complexe**

---

## Recommandation

### Pour Votre Prototype Initial : **Option 1 (Distribué)**

**Raisons :**

1. ✅ **Architecture plus simple** : un seul type de worker
2. ✅ **Scalabilité maximale** : valider que Ray fonctionne bien
3. ✅ **Communication minimale** : halos uniquement
4. ✅ **Cohérent avec votre vision initiale** : workers autonomes

**Diffusion "manuelle" suffit** pour valider l'architecture.

### Évolution Future : **Envisager Option 2 si**

- Vous intégrez **forçages de courants complexes** (U, V depuis modèle océanique)
- Vous voulez **JAX-CFD complet** avec schémas haute précision
- Vous couplez **plusieurs modèles** (micronecton, phytoplancton, etc.) avec même physique

**Mais gardez Option 1 comme baseline** pour comparaison.

---

## Exemple Hybride (Avancé)

**Idée :** Transport dans les workers, mais avec un service centralisé pour forçages.

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ CellWorker 0 │  │ CellWorker 1 │  │ CellWorker 2 │
│              │  │              │  │              │
│ - Biogéo     │  │ - Biogéo     │  │ - Biogéo     │
│ - Transport  │  │ - Transport  │  │ - Transport  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         ↓ Query velocities (U, V)
               ┌───────────────────┐
               │  ForcingService   │
               │  (Ray Actor)      │
               │                   │
               │  - Lit NetCDF     │
               │  - Interpole U, V │
               │  - Cache données  │
               └───────────────────┘
```

**Avantages :**
- Workers font le transport localement (scalable)
- Service centralisé gère les données volumineuses (forçages)
- Compromis raisonnable

---

## Conclusion

| Critère | Option 1 (Distribué) | Option 2 (Centralisé) |
|---------|---------------------|---------------------|
| **Simplicité architecture** | ★★★★★ | ★★★☆☆ |
| **Scalabilité** | ★★★★★ | ★★☆☆☆ |
| **Performance** | ★★★★★ | ★★☆☆☆ |
| **Communication** | ★★★★★ | ★★☆☆☆ |
| **Précision numérique** | ★★★☆☆ | ★★★★★ |
| **Séparation responsabilités** | ★★★☆☆ | ★★★★★ |

**Mon avis : Démarrer avec Option 1 (distribué), c'est l'approche la plus cohérente avec votre vision initiale et la plus scalable.**

**TransportWorker spécialisé = optimisation prématurée pour le prototype.**

---

## Question pour Vous

**Quelle est votre intuition ?**

- Préférez-vous **Option 1** (transport distribué dans chaque worker) ?
- Ou **Option 2** vous semble plus élégante (séparation biogéo/physique) ?
- Ou voulez-vous **prototyper les deux** pour comparer ?

**Qu'en pensez-vous ?**
