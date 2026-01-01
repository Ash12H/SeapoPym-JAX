# Discussion

## Points Clés à Développer

### Limites Identifiées

- Le speedup Strong Scaling est limité à **~1.25×** par la dominance du transport de production (80% du temps de calcul)
- Conformément à la Loi d'Amdahl, la parallélisation inter-tâches ne peut dépasser cette borne tant que la tâche dominante reste séquentielle
- Le schéma Upwind (premier ordre) introduit une diffusion numérique ; des schémas d'ordre supérieur (flux-limiters) pourraient réduire cette erreur
- La contrainte CFL impose un pas de temps petit pour les courants rapides ou les grilles fines

### Ce qui Fonctionne

- L'architecture DAG parallélise correctement les tâches indépendantes (validé par le test sleep : 11.5× avec 12 workers)
- La complexité O(N) garantit une scalabilité linéaire en fonction de la taille du problème
- Le couplage transport-biologie converge correctement (O(Δx)) sans biais de Time Splitting
- Les processus biologiques reproduisent les solutions analytiques avec une erreur < 0.2%

### Recommandations Techniques

- **Parallélisation intra-tâche** : Pour dépasser la limite d'Amdahl, paralléliser le transport lui-même via chunking spatial (Dask Arrays)
- **Distribution des données** : Utiliser `dask.scatter` ou `client.scatter` pour pré-distribuer les forçages sur les workers (évite la sérialisation répétée)
- **Équilibrage de charge** : Concevoir des groupes fonctionnels avec des charges de travail comparables
- **Réduction de l'overhead** : Minimiser les wrappers Python/xarray autour des kernels Numba

### Perspectives de Recherche

- Intégration de schémas de transport d'ordre supérieur (TVD, WENO)
- Décomposition de domaine pour les très grandes grilles (global 1/12°)
- Couplage avec des modèles biogéochimiques externes via le DAG
- Extension à la dimension verticale (migration nycthémérale du micronecton)
- Utilisation de GPU (Numba CUDA) pour accélérer le transport

### Avantages de l'Architecture DAG

- Modularité : ajout d'un processus (prédation, migration) = ajout d'un nœud, sans modifier le contrôleur
- Reproductibilité : le graphe est explicite et auditable
- Flexibilité : changement de backend (séquentiel → Dask → GPU) sans modifier le modèle
- Testabilité : chaque nœud peut être testé unitairement

### Comparaison avec les Approches Existantes

- [ À compléter après résultats SeapoPym v0.3 et Seapodym-LMTL ]
- Avantage vs C++ monolithique : modularité, lisibilité, écosystème Python
- Avantage vs SeapoPym v0.3 (0D) : réintégration du transport spatial
