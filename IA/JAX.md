# Spécifications Architecturales : Moteur de Simulation "Blueprint-to-JAX"

## 1. Vue d'ensemble

L'objectif est de développer un simulateur biogéochimique/physique haute performance en séparant strictement la **définition du modèle** (flexible, riche, validée) de son **exécution** (statique, vectorisée, compilée sur GPU/CPU).

L'architecture repose sur un paradigme de **compilation** : un Blueprint Python est transformé en un plan d'exécution optimisé pour JAX.

---

## 2. Le Blueprint : Définition du Modèle

Le Blueprint est un graphe orienté acyclique (DAG) déclaratif. Il ne contient pas de boucles de temps, mais définit les dépendances fonctionnelles.

### Composants du Blueprint

- **États (States) :** Les variables qui évoluent dans le temps (ex: `Biomasse`, `Nutriments`).
- **Forçages (Forcings) :** Données exogènes (ex: `Température`, `Courants`). Ils sont définis par des fichiers (NetCDF) ou des constantes.
- **Paramètres :** Valeurs scalaires ou champs statiques (ex: `Taux de mortalité`, `Masque Terre/Mer`).
- **Processus (Functions) :** Les lois d'évolution (ex: `calc_growth`, `calc_transport`). Elles prennent des entrées et retournent des **Tendances**.

### Responsabilité Utilisateur

L'utilisateur nomme les variables et relie les nœuds (ex: "La fonction `Growth` prend `Biomasse` et `Température`"). Il n'a pas à se soucier des indices de tableaux.

---

## 3. Le Compilateur (Python/Xarray/Pint)

C'est le cœur de l'intelligence du système. Il s'exécute **une seule fois** avant le lancement de la boucle temporelle.

### A. Gestion des Unités (Pint)

- **Règle :** Le moteur JAX est "unit-agnostic" (il ne manipule que des `float`).
- **Action :** Le compilateur vérifie la cohérence dimensionnelle de toutes les entrées via `Pint`.
- **Normalisation :** Il convertit automatiquement toutes les données en unités SI standards (ou celles du modèle) avant de générer les tableaux bruts.

### B. Inférence de Forme et Alignement (Xarray)

- **Problème :** JAX exige des dimensions statiques et cohérentes (ex: `(Batch, Time, Z, Y, X)`).
- **Action :**

1. Utilisation de `Xarray` pour charger les données.
2. **Inférence :** Propagation des "shapes" dans le graphe pour valider que la sortie d'une fonction correspond à l'entrée de la suivante.
3. **Transposition Canonique :** Le compilateur réordonne les axes de tous les tableaux d'entrée pour respecter un layout mémoire strict (ex: `C-Contiguous`).
4. **Stripping :** Extraction des `.values` (Numpy arrays) pour les passer au Backend.

### C. Linker (Génération du Bytecode)

- Transformation des noms (Strings) en indices (Integers).
- Création d'une liste ordonnée d'opérations (Tri Topologique du graphe).
- Génération des masques statiques (ex: `land_mask`) sous forme de matrices de 0 et 1.

---

## 4. Le Backend d'Exécution (JAX)

Cette partie est purement fonctionnelle. Elle reçoit l'état initial, les paramètres et les fonctions compilées.

### A. Formulation des Fonctions (Compatibilité JAX)

Pour être compilables (`jax.jit`) et dérivables (`jax.grad`), les fonctions physiques doivent respecter ces règles :

1. **Pureté Fonctionnelle :**

- Pas d'effets de bord (pas de `print`, pas d'écriture fichier, pas de modification de variable globale).
- Signature stricte : `Sortie = f(Entrées, Paramètres)`.
- Pas de mutation "In-Place" (on ne fait pas `x[i] += 1`, on retourne `x + 1`).

2. **Vectorisation (Pas de `if/else`) :**

- Remplacement de la logique conditionnelle par du **Masquage (Masking)**.
- Utilisation de `jnp.where(condition, val_true, val_false)` pour les branchements complexes.
- _Note :_ Le calcul est effectué sur toute la grille, puis filtré.

3. **Voisinage (Stencils / Volumes Finis) :**

- Interdiction des boucles `for` sur les indices spatiaux (`i, j`).
- Utilisation de `jnp.roll` pour décaler les grilles et aligner les voisins.
- _Exemple Transport :_ `Flux = (State - jnp.roll(State, shift=1)) * Vitesse`.

### B. Gestion des fonctions incompatibles

Si une fonction utilisateur ne peut pas être écrite en JAX pur (ex: appel à une librairie C externe, code legacy complexe) :

1. **Option 1 : Pré-calcul (Si statique)**

- Si la fonction ne dépend pas de l'état (ex: forçage solaire complexe), elle est exécutée en Python pur lors de la compilation, et le résultat est passé comme un tableau de données (Forçage).

2. **Option 2 : `jax.pure_callback` (Si dynamique)**

- Permet d'appeler du code Python arbitraire depuis JAX.
- _Coût :_ Brise la fusion des opérations sur GPU et oblige un aller-retour vers le CPU (goulot d'étranglement potentiel). À utiliser en dernier recours.

3. **Option 3 : Approximation**

- Réimplémenter une version simplifiée de la fonction incompatible directement avec les primitives `jax.numpy`.

---

## 5. Points Complémentaires (Ne pas oublier !)

### A. Le Schéma d'Intégration Temporelle

Tu as mentionné l'addition des tendances.

- Le backend doit implémenter un "Stepper" (Euler, Runge-Kutta 4, etc.).
- Avec JAX, on utilise `jax.lax.scan` pour la boucle temporelle. C'est ce qui permet de compiler **toute la simulation** (du temps t=0 à t=fin) en un seul kernel GPU ultra-rapide, au lieu de lancer 10 000 petits kernels (un par pas de temps).

### B. Stratégie I/O (Entrées/Sorties)

JAX est si rapide que l'écriture sur disque peut devenir le goulot d'étranglement.

- Ne pas sauvegarder à chaque pas de temps.
- Implémenter un buffer en mémoire GPU/RAM qui sauvegarde tous les `N` pas de temps, puis écrit en bloc (chunk) vers NetCDF/Zarr à la fin (ou périodiquement).

### C. Mode "Debug"

JAX est difficile à débugger (les erreurs sont cryptiques une fois compilées).

- Prévoir un flag `DEBUG_MODE = True` dans le Compilateur.
- Ce mode désactive `jax.jit` et exécute le graphe en mode "Eager" (pas à pas, comme Numpy standard), permettant l'utilisation de `print()` et de debuggers Python classiques.
