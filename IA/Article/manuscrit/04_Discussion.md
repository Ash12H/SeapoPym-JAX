# Discussion

## Points Clés à Développer

### Limites Identifiées

-   Le speedup Strong Scaling est limité à **~1.25×** par la dominance du transport de production (80% du temps de calcul)
-   Conformément à la Loi d'Amdahl, la parallélisation inter-tâches ne peut dépasser cette borne tant que la tâche dominante reste séquentielle
-   Le schéma Upwind (premier ordre) introduit une diffusion numérique ; des schémas d'ordre supérieur (flux-limiters) pourraient réduire cette erreur
-   La contrainte CFL impose un pas de temps petit pour les courants rapides ou les grilles fines

### Ce qui Fonctionne

-   L'architecture DAG parallélise correctement les tâches indépendantes (validé par le test sleep : 10.34× avec 12 workers)
-   La complexité O(N) garantit une scalabilité linéaire en fonction de la taille du problème
-   Le couplage transport-biologie converge correctement (O(Δx)) sans biais de Time Splitting
-   Les processus biologiques reproduisent les solutions analytiques avec une erreur < 0.2%

### Recommandations Techniques

-   **Parallélisation intra-tâche** : Pour dépasser la limite d'Amdahl, paralléliser le transport lui-même via chunking spatial (Dask Arrays)
-   **Distribution des données** : Utiliser `dask.scatter` ou `client.scatter` pour pré-distribuer les forçages sur les workers (évite la sérialisation répétée)
-   **Équilibrage de charge** : Concevoir des groupes fonctionnels avec des charges de travail comparables
-   **Réduction de l'overhead** : Minimiser les wrappers Python/xarray autour des kernels Numba

### Perspectives de Recherche

-   Intégration de schémas de transport d'ordre supérieur (TVD, WENO)
-   Décomposition de domaine pour les très grandes grilles (global 1/12°)
-   Couplage avec des modèles biogéochimiques externes via le DAG
-   Extension à la dimension verticale (migration nycthémérale du micronecton)
-   Utilisation de GPU (Numba CUDA) pour accélérer le transport

### Avantages de l'Architecture DAG

-   Modularité : ajout d'un processus (prédation, migration) = ajout d'un nœud, sans modifier le contrôleur
-   Reproductibilité : le graphe est explicite et auditable
-   Flexibilité : changement de backend (séquentiel → Dask → GPU) sans modifier le modèle
-   Testabilité : chaque nœud peut être testé unitairement

### Comparaison avec les Approches Existantes

**Vs SeapoPym v0.3 (configuration 0D)** : La nouvelle architecture reproduit fidèlement les résultats de la version précédente (corrélation > 0.999, biais négligeable de 0.0005 g/m²). L'erreur L2 de ~3% s'explique par les différences d'implémentation numérique (ordre des opérations, précision flottante) sans impact sur la dynamique du modèle. L'avantage majeur est la réintégration du transport spatial, impossible dans v0.3.

**Vs Seapodym-LMTL (C++, avec transport)** : La comparaison sur le Pacifique (2002-2004) montre une MAPE de 31% avec transport activé, contre 52% sans transport. Cette amélioration de 20 points confirme que l'architecture DAG capture correctement les flux physiques de biomasse. Les différences résiduelles (~30%) sont attendues compte tenu des divergences fondamentales :

-   Schéma numérique : Volumes Finis (DAG) vs Différences Finies sur grille Arakawa C (LMTL)
-   Gestion des masques et conditions aux limites
-   Interpolation des forçages sur les interfaces

**Avantages architecturaux** :

-   Vs C++ monolithique : modularité, lisibilité, écosystème Python riche (xarray, Dask, numba)
-   Vs SeapoPym v0.3 : réintégration complète du transport spatial
-   Testabilité : chaque nœud du DAG peut être validé unitairement
