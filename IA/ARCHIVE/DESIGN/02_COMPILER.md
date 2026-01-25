# Axe 2 : Compilateur

## Objectifs

Concevoir le système de transformation (Linker) qui convertit le graphe "Modèle + Données" en un code exécutable JAX, en gérant l'alignement mémoire et dimensionnel.

## État des Discussions

### 1. Inférence de Shape (Inference Engine)

_Objectif : Transformer les dimensions sémantiques ("cohort") en dimensions entières ("100") pour préparer les allocations statiques JAX._

**Problématique :**
Le Blueprint ne contient que des noms. Le fichier de config contient des données avec des tailles.
Le Compilateur doit lire (lazy load) les fichiers de config pour "résoudre" la taille de chaque dimension sémantique.
_Question_ : Comment gérer les dimensions dynamiques (ex: dimension de temps infinie en streaming) vs dimensions statiques (grille spatiale) ?

### 2. Le Layout Canonique

_Objectif : Standardiser l'ordre des axes pour JAX._

**Context (Axe 1) :**
Nous avons décidé que les inputs utilisateurs peuvent avoir n'importe quel ordre.
Le Compilateur doit donc imposer un ordre interne unique.
**Proposition :**
Adopter un format "Channel-Last" ou "Channel-First" ?
Exemple : `(Time, Z, Y, X, Cohort)` ?
Cela impacte fortement la performance (C-contiguous vs F-contiguous).

### 3. Stripping & Masques

_Objectif : Préparer les tableaux numpy nus pour JAX._

**Problématique :**

- `xarray.Dataset` -> `dict[str, jnp.array]`.
- Les `NaN` (terres) doivent être convertis en 0.0 + un Masque Binaire.
- Ce masque doit-il être passé explicitement à chaque fonction JAX ?

**Décisions Validées (2026-01-24)**

### 1. Layout Canonique & Transposition (Réponse Q2.2)

Le Compilateur définit un ordre optimal des dimensions (ex: `Time, Cohort, Depth, Lat, Lon`).
Pour garantir cet ordre indépendamment de l'input utilisateur :

1.  Il utilise `xarray.Dataset.transpose(*target_dims)` lors de la préparation.
2.  Cela aligne la mémoire pour une performance JAX optimale (C-contiguous).

### 2. Gestion des Masques & NaNs (Réponse Q3 - Stripping)

JAX ne tolère pas les NaNs.
**Stratégie :**

1.  **Prétraitement** : Remplacement de tous les `NaN` par `0.0` dans les données brutes avant conversion JAX.
2.  **Masque Explicite** : L'utilisateur fournit un champ Masque (booléen ou 0/1) comme un input standard.
3.  **Flexibilité** : Ce masque peut être statique `(Y,X)` ou dynamique `(T,Y,X)`, le broadcasting gère la différence.

### 3. Inférence de Shape (Réponse Q2.1)

L'inférence n'est pas "devinée" mais **lue**.
Le Compilateur charge les métadonnées des DataArrays alignés (`ds.sizes`) pour figer les dimensions statiques du graphe JAX (`jax.jit` static_argnums).
