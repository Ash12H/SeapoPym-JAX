# Discussion

## Points Clés à Développer

### Limites Identifiées

-   **Schéma Upwind** : Le schéma d'advection au premier ordre introduit une diffusion numérique qui réduit l'ordre de convergence effectif (O(Δx^0.22) sur le test de Zalesak). Des schémas d'ordre supérieur (flux-limiters, TVD, WENO) pourraient réduire cette erreur mais augmenteraient le coût de calcul et la complexité d'implémentation.
-   **Contrainte CFL** : Le pas de temps reste contraint par la stabilité numérique explicite, imposant des itérations fréquentes pour les grilles fines. Un schéma implicite permettrait des pas de temps plus grands mais au prix d'une résolution matricielle coûteuse.

### Ce qui Fonctionne

-   **Architecture DAG** : L'approche par graphe unifie avec succès la biologie et la physique, garantissant la conservation de la masse et la reproductibilité numérique. L'orchestration par DAG n'introduit pas de surcoût significatif : le temps de calcul est dominé par les noyaux numériques (transport, mortalité) plutôt que par la gestion du graphe.
-   **Scalabilité Linéaire** : La complexité O(N^1.01) est validée, assurant que le modèle peut traiter des domaines globaux à haute résolution moyennant des ressources proportionnelles. Le temps de calcul double lorsque la taille du domaine double, sans dégradation algorithmique.
-   **Modularité et Testabilité** : Chaque processus (transport, mortalité, production) est un nœud indépendant du graphe, facilitant les tests unitaires, le débogage et l'ajout de nouveaux processus sans modifier l'architecture globale.

---

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
