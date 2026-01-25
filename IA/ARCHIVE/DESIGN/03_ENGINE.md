# Axe 3 : Moteur & Mémoire

## Objectifs

Définir l'architecture d'exécution (Boucle temporelle, I/O, Dual Backend) pour concilier performance JAX et faisabilité mémoire.

## État des Discussions

### 1. Stratégie Time Loop & I/O

_Objectif : Simuler des durées longues (siècles) sans saturer la VRAM._

**Problématique :**
`jax.lax.scan` garde en mémoire (ou tente de le faire) tous les outputs intermédiaires pour le gradient.
Impossible de tout tenir en VRAM sur 100 ans.

**Proposition (Architecte) : Le Chunked Runner**
Le moteur n'exécute pas toute la simulation d'un coup.
Il découpe le temps en "Chunks" (ex: 1 an ou 1 mois).

1.  **Compilation** : On compile une fonction `step_chunk(state, forcings_chunk)` qui exécute N pas de temps.
2.  **Orchestration** : Une boucle Python charge les forçages pour le Chunk N, exécute `step_chunk`, récupère les résultats, les écrit sur disque (Zarr/NetCDF), et passe l'état final au Chunk N+1.

### 2. Le Kernel "Step"

_Objectif : Définir la fonction atomique d'un pas de temps._

Cette fonction doit :

- Prendre l'état courant `S_t`.
- Prendre les forçages instantanés `F_t` (extraits par le scan).
- Exécuter les processus du Blueprint pour obtenir les **Tendances** ($dU/dt$).
- Appliquer le **TimeIntegrator** (ex: Euler Explicit) : $S_{t+1} = S_t + \sum(Tendances) \times dt$.
- Retourner `S_t+1` et les diagnostiques `D_t`.

### 3. Dual Backend (Numpy/JAX)

_Objectif : Exécuter le même modèle sur CPU sans JAX._

**Approche :**
Si JAX n'est pas dispo, le Chunked Runner appelle une implémentation `SequentialBackend` qui :

- Remplace `scan` par une boucle `for` Python.
- Utilise les versions Numpy des fonctions (via le registre).

**Décisions Validées (2026-01-24)**

### 1. Architecture Scan & Chunking (Réponse Q3.1 & Q3.2)

L'exécution repose sur `jax.lax.scan` découpé en chunks temporels.
**Sémantique Scan :**

- **Carry (State)** : Variables d'état évolutives (Biomasse).
- **Inputs (Forcings)** : Données streamées (Température).
- **Static (Parameters)** : Constantes capturées par la closure (Taux).
- **Outputs** : Données à sauvegarder.

### 2. Le Chunked Runner

Une boucle Python externe gère :

1.  Le chargement des `Forcings` pour le Chunk N.
2.  L'appel au kernel JAX compilé (`scan` sur N pas).
3.  L'écriture asynchrone des `Outputs` sur disque.
4.  Le passage de l'état final (`Carry`) au Chunk suivant.

### 3. Dual Backend (Réponse Q3.3)

Pour garantir la portabilité :

- Le Blueprint génère une fonction `step_fn` pure (agnostique).
- Le Moteur choisit l'implémentation de boucle :
    - Mode JAX : `jax.lax.scan(step_fn, ...)`
    - Mode Numpy : `def python_scan(...)`: boucle `for` simple.
