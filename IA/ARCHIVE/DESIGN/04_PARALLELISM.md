# Axe 4 : Parallélisme

## Objectifs

Définir la stratégie de passage à l'échelle pour les simulations massives.

## Décisions Validées (2026-01-24)

### 1. Priorité au Batch Parallelism (vmap)

L'approche initiale favorise le parallélisme "embarrassingly parallel".

**Scénarios d'usage (V1) :**

- **Exploration de Paramètres** : Exécuter le même modèle avec N jeux de paramètres différents (ex: taux de croissance `[0.1, 0.2, ...]`). `vmap` gère cela nativement (Single Program Multiple Data).
- **Membres d'Ensemble** : Simulations stochastiques multiples.

### 2. Stratégie de Sharding (Partitionnement)

Le découpage se décide dans le fichier de config (`run.yaml`).

**Interface Déclarative :**

```yaml
execution:
    device_layout:
        mesh: [4, 1] # 4 GPUs
        sharding_rules:
            ensemble: 0 # Axe 'ensemble' -> Mesh axe 0 (Parallèle)
            spatial: null # Axe 'spatial' -> Replicated (Non shardé)
```

**Distinction Axes Libres vs Core :**

- **Axes Libres (Batchable)** : Traversent le graphe sans interaction. Ex: `Ensemble`, `Time_Chunk`. -> Sharding trivial (V1).
- **Axes Core (Couplés)** :
    - **Spatial** : Nécessite des voisins (Transport).
    - **Espèces/Taille** : Nécessite des réductions (Mortalité par prédation `sum(Prey)`).
    - Ces axes sont identifiés par le décorateur `@functional(core_dims=["spatial"])`.
    - **Décision** : Interdiction de sharder ces axes en V1 (nécessite communication complexe). Une simulation (Core dims) doit tenir sur un GPU.

### 3. I/O Asynchrone

Le Chunked Runner permet intrinsèquement de paralléliser Écriture Disque (CPU) et Calcul (GPU) via `threading` sur la boucle Python.
