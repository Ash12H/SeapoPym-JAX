# Workflow State

## Informations générales

- **Projet** : SeapoPym-JAX Design
- **Étape courante** : 3. Architecture (Validée)
- **Rôle actif** : Architecte
- **Dernière mise à jour** : 2026-01-25 09:55

## Résumé du besoin

Conception de l'architecture de migration Seapopym vers JAX.
L'architecture est définie à travers 5 axes stratégiques, documentés dans `IA/DESIGN/`.
Le périmètre fonctionnel et technique est validé.

## Décisions d'architecture (Validées)

- **Axe 1 (Blueprint)** : Format YAML Split (Modèle/Config), Registre via décorateur `@functional`.
- **Axe 2 (Compilateur)** : Layout Canonique `(T,C,Z,Y,X)`, Transpose Xarray, Inférence par lecture laz, Masques explicites.
- **Axe 3 (Moteur)** : `jax.lax.scan` sur Chunks temporels, I/O asynchrone Python, Step Kernel avec TimeIntegrator explicite.
- **Axe 4 (Parallélisme)** : Focus initial sur Batch Parallelism (`vmap`), Spatial Sharding proscrit pour V1.
- **Axe 5 (Auto-Diff)** : Paramètres `trainable=True` dans Config, Loss Function Wrapper avec Gradient Checkpointing.

## Todo List & Questions Ouvertes (Design Phase)

### Axe 1 : Blueprint & Data

| État | ID   | Sujet      | Question Clé                                                                                                   |
| ---- | ---- | ---------- | -------------------------------------------------------------------------------------------------------------- |
| ☑    | Q1.1 | Format     | Comment structurer un Blueprint purement déclaratif (YAML/JSON) qui remplace les appels `register_unit` ?      |
| ☑    | Q1.2 | Registre   | Quelle mécanique pour mapper une String (`"biol:growth"`) vers une implémentation spécifique (Python ou JAX) ? |
| ☑    | Q1.3 | Validation | Comment garantir la cohérence des Unités et Dimensions AVANT la compilation ?                                  |

### Axe 2 : Compilateur

| État | ID   | Sujet     | Question Clé                                                  |
| ---- | ---- | --------- | ------------------------------------------------------------- |
| ☑    | Q2.1 | Inférence | Lecture des métadonnées Xarray pour figer les static_argnums. |
| ☑    | Q2.2 | Linker    | Transposition Canonique des inputs `(T,C,Z,Y,X)`.             |

### Axe 3 : Moteur & Mémoire

| État | ID   | Sujet     | Question Clé                                                        |
| ---- | ---- | --------- | ------------------------------------------------------------------- |
| ☑    | Q3.1 | Time Loop | Utilisation de `jax.lax.scan` avec Chunking temporel (Python loop). |
| ☑    | Q3.2 | Chunking  | Streaming Zarr/NetCDF entre les chunks.                             |
| ☑    | Q3.3 | Dualité   | `step_fn` pure compatible Numpy et JAX.                             |

### Axe 4 : Parallélisme

| État | ID   | Sujet     | Question Clé                                          |
| ---- | ---- | --------- | ----------------------------------------------------- |
| ☑    | Q4.1 | Partition | Batch Parallelism (`vmap`) privilégié pour l'instant. |
| ☑    | Q4.2 | Halo      | Sharding spatial reporté à une phase ultérieure.      |

### Axe 5 : Auto-Diff

| État | ID   | Sujet      | Question Clé                                               |
| ---- | ---- | ---------- | ---------------------------------------------------------- |
| ☑    | Q5.1 | Paramètres | Flag `trainable=True` dans le fichier de config.           |
| ☑    | Q5.2 | Loss       | Fonction wrapper `loss_fn(params)` + `jax.value_and_grad`. |

## Historique des transitions

| De                | Vers              | Raison                          | Date       |
| ----------------- | ----------------- | ------------------------------- | ---------- |
| -                 | 1. Initialisation | Démarrage du projet             | 2026-01-23 |
| 1. Initialisation | 2. Analyse        | Identification des Axes         | 2026-01-23 |
| 2. Analyse        | 3. Architecture   | Validation des 5 Axes de Design | 2026-01-25 |
