# Rapport de Comparaison : Méthodes de Transport

## Introduction

Ce document compare deux approches numériques pour la résolution du transport (advection-diffusion) dans le modèle SEAPODYM :
1.  **L'approche originale (History)** : Basée sur un schéma implicite à directions alternées (ADI).
2.  **L'approche actuelle (Seapopym)** : Basée sur une approche explicite de calcul de tendances (Method of Lines).

## 1. Description des Méthodes

### A. Modèle Original (ADI - Alternating Direction Implicit)

Le modèle original utilise un schéma de **splitting** (décomposition) qui résout le problème 2D en deux étapes 1D successives à chaque pas de temps :

1.  **Étape Zonale (Longitude)** : Résolution implicite en λ, explicite en φ.
    *   Système tridiagonal résolu par l'algorithme de Thomas.
    *   La mortalité est appliquée durant cette étape.
2.  **Étape Méridionale (Latitude)** : Résolution implicite en φ, utilisant le résultat intermédiaire.
    *   Système tridiagonal résolu par l'algorithme de Thomas.

**Caractéristiques Clés :**
*   **Implicite** : Permet de grands pas de temps sans instabilité (inconditionnellement stable pour la diffusion).
*   **Upwind** : Advection décentrée selon le signe de la vitesse.
*   **Séquentiel** : Chaque ligne/colonne dépend de la solution précédente dans l'algorithme tridiagonal, ce qui rend la parallélisation massive difficile sur GPU.

### B. Modèle Actuel (Explicite / Volumes Finis)

Le nouveau modèle (Seapopym) adopte une approche modulaire où le calcul des **tendances** (`dC/dt`) est séparé de l'intégration temporelle.

1.  **Advection** : Schéma de volumes finis "Upwind" explicite.
    *   Calcul des flux aux interfaces des cellules (`Flux = Vitesse * Concentration_Amont * Aire`).
    *   Divergence des flux pour obtenir la tendance : `dC/dt = -(Flux_Sortant - Flux_Entrant) / Volume`.
2.  **Diffusion** : Schéma centré explicite.
    *   Approximation du Laplacien par différences finies.
    *   `dC/dt = D * Laplacien(C)`.

**Caractéristiques Clés :**
*   **Explicite** : Le futur état dépend uniquement de l'état présent.
*   **Contrainte CFL** : Le pas de temps (`dt`) est strictement limité par la condition de stabilité CFL (`Courant-Friedrichs-Lewy`). Si `dt` est trop grand (vitesse rapide ou maille fine), le modèle diverge (explose).
*   **Parallélisable** : Chaque cellule peut être calculée indépendamment, idéal pour l'accélération GPU/Numba.

## 2. Comparaison et Différences Attendues

### 1. Stabilité et Pas de Temps
*   **Original** : Très robuste. Peut fonctionner avec des pas de temps longs (ex: 1 semaine) sans exploser, même si la précision diminue.
*   **Actuel** : Fragile si le pas de temps est mal choisi. Nécessite un pas de temps adaptatif ou très court (ex: quelques minutes ou heures) pour respecter la condition CFL (`v * dt / dx < 1`).
    *   *Conséquence* : Le nouveau modèle nécessitera beaucoup plus de pas de temps (itérations) pour simuler la même durée.

### 2. Diffusion Numérique
Les deux méthodes utilisent un schéma "Upwind" (décentré) pour l'advection, qui est connu pour être naturellement "diffusif" (il lisse les gradients).
*   Cependant, le splitting ADI introduit une erreur de splitting (erreur en `O(dt²)`) qui peut créer des artefacts anisotropes (ex: diffusion différente en X et en Y selon l'ordre de résolution).
*   L'approche explicite est plus "isotrope" géométriquement, mais sa diffusion numérique dépend fortement du nombre de pas de temps effectués.

### 3. Conservation de la Masse
*   **Original** : Conserve la masse globalement (aux erreurs d'arrondi près), mais la conservation locale exacte dépend de la qualité de la résolution tridiagonale aux bords.
*   **Actuel** : L'approche par flux (Volumes Finis) garantit par construction une conservation de la masse **parfaite** (ce qui sort d'une cellule entre forcément dans la voisine), tant que les conditions aux limites sont correctes.

### 4. Précision aux Pôles et Métrique
*   Les deux modèles intègrent les termes métriques (`cos(latitude)`).
*   L'approche actuelle gère explicitement les surfaces des cellules (`cell_areas`) et des faces (`face_areas`), ce qui peut offrir une meilleure cohérence géométrique sur des grilles complexes par rapport aux coefficients linéarisés du modèle original.

### 5. Interaction avec la Biologie
*   **Original** : La mortalité est couplée mathématiquement dans la matrice de résolution de la première étape (X). Cela lie fortement transport et biologie.
*   **Actuel** : Le transport calcule une tendance `dC/dt |_transport`. La biologie calcule une tendance `dC/dt |_bio`. Les deux sont sommées par le solveur (ex: Euler ou Runge-Kutta). Cela permet une séparation totale des processus (Modularité).

## Conclusion

Le passage à la méthode explicite offre une **modularité** et une **capacité de parallélisation** supérieures, essentielles pour l'optimisation moderne (GPU). Cependant, cela se paie par une contrainte forte sur le pas de temps (stabilité conditionnelle).

Il faut s'attendre à devoir utiliser un pas de temps beaucoup plus fin (`dt` petit) avec le nouveau modèle. Si les résultats diffèrent, la cause la plus probable (hors bugs) sera la différence entre la résolution implicite vs explicite des gradients temporels rapides.
