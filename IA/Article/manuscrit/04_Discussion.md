# Discussion

## Points Clés à Développer

### Limites Identifiées

-   **Overhead de l'Ordonnanceur** : Le parallélisme de données induit un surcoût lié à la gestion du graphe Dask, composé d'un coût fixe de setup (~2s) et d'un coût variable proportionnel au nombre de chunks (~0.02-0.2s par chunk). Pour les simulations légères (ex: Zooplancton, 12 cohortes, temps séquentiel 1.0s), cet overhead (~2.4s) dépasse le gain de calcul, rendant l'exécution séquentielle 3× plus rapide. Pour les simulations complexes (ex: Micronecton, 527 cohortes, temps séquentiel 102s), l'overhead (~10s) devient négligeable (~10% du temps total), permettant un speedup de 2.4× avec 8-12 workers.
-   **Schéma Upwind** : Le schéma d'advection au premier ordre introduit une diffusion numérique qui réduit l'ordre de convergence effectif (O(Δx^0.66)). Des schémas d'ordre supérieur (flux-limiters) pourraient réduire cette erreur mais augmenteraient le coût de calcul.
-   **Contrainte CFL** : Le pas de temps reste contraint par la stabilité numérique explicite, imposant des itérations fréquentes pour les grilles fines.

### Ce qui Fonctionne

-   **Dépassement de la Loi d'Amdahl via Data Parallelism** : Grâce au Data Parallelism sur la dimension `cohort`, nous avons brisé la limite de speedup de 1.25× inhérente au Task Parallelism (imposée par la fraction séquentielle de 80% du transport). Le système atteint un speedup de **2.41×** pour le cas complexe Micronecton (527 cohortes, 8-12 workers), soit **2.2× supérieur au Task Parallelism** (1.08×). Note : Pour le cas Zooplankton (12 cohortes), le Task Parallelism atteint 1.30×, dépassant légèrement la limite théorique de 1.25×. Cette anomalie apparente s'explique par les effets de cache CPU sur les très petits problèmes (temps séquentiel 1.0s) et l'overhead Dask relativement faible du Task Parallel comparé au Data Parallel.
-   **Architecture DAG** : L'approche par graphe unifie avec succès la biologie et la physique, garantissant la conservation de la masse et la reproductibilité numérique (RMSE < 10⁻¹²) tout en offrant une flexibilité totale de déploiement (Séquentiel ou Parallèle).
-   **Scalabilité Linéaire** : La complexité O(N) est validée, assurant que le modèle peut traiter des domaines globaux à haute résolution moyennant des ressources proportionnelles.

### Modèle de Rentabilité du Data Parallelism

![Figure 4D: Cohort Scaling Comparison](../../data/article/figures/fig_04d_cohort_scaling_comparison_scaling.png)

**Figure 4D. Strong scaling comparison: Impact of cohort number on Data Parallelism profitability.** (**Left**) Zooplankton (12 cohorts, T_seq = 1.0s): Data Parallelism (blue squares) remains slower than Sequential baseline (grey) due to Dask overhead (~2.4s) dominating computation time (max speedup 0.65×). Task Parallelism (orange circles) achieves modest gains (1.30×). (**Right**) Micronecton (527 cohorts, T_seq = 102.6s): Data Parallelism outperforms both Sequential and Task Parallelism, reaching **2.41× speedup** with 8 workers—a **2.2× improvement** over Task Parallelism (1.08×). By parallelizing transport across cohorts, Data Parallelism circumvents Amdahl's law (dotted line at 1.25×). Speedup plateau at 8-12 workers indicates saturation. Dashed line: ideal linear scaling. Grid: 500×500 cells, 10 time steps, RMSE < 10⁻¹² g/m² vs Sequential.

---

La figure ci-dessus illustre la transition entre ces deux régimes. Pour le Zooplankton (panel gauche), le Data Parallelism reste systématiquement sous la ligne Sequential (speedup < 1.0), tandis que pour le Micronecton (panel droit), il dépasse rapidement le Sequential et le Task Parallelism, atteignant un plateau à 2.4×.

Le speedup observé du Data Parallelism peut être modélisé par :

$$S = \frac{T_{\text{seq}}}{T_{\text{seq}}/N_w + O_{\text{Dask}}}$$

où $T_{\text{seq}}$ est le temps séquentiel, $N_w$ le nombre de workers, et $O_{\text{Dask}}$ l'overhead constant (~10s pour 500×500 grille). Le Data Parallelism est rentable (S > 1) si :

$$T_{\text{seq}} > \frac{O_{\text{Dask}} \times N_w}{N_w - 1}$$

**Validation empirique** :
- **Zooplankton** (12 cohortes) : $T_{\text{seq}} = 1.0\text{s} < 11.4\text{s}$ (seuil pour $N_w=8$) → Data Parallel NON rentable (speedup = 0.65×)
- **Micronecton** (527 cohortes) : $T_{\text{seq}} = 102\text{s} > 11.4\text{s}$ → Data Parallel rentable (speedup = 2.41×)

Le plateau de speedup à 8-12 workers (~2.4×) indique une saturation, probablement due à la contention mémoire ou aux synchronisations inter-threads sur les architectures avec cœurs performance/efficience (Apple M4).

### Recommandations Techniques

-   **Stratégie Adaptative** : Le choix du backend dépend du ratio temps_calcul/overhead_dask :
    -   Pour **N ≲ 20 cohortes** (Zooplancton) : Privilégier le backend **Séquentiel** pour éviter l'overhead Dask (~2.4s fixe + 0.2s/cohorte). Observation empirique : speedup Data Parallel max = 0.65× (plus lent que Sequential).
    -   Pour **N ≳ 100 cohortes** (Micronecton, Multi-espèces) : Activer le **Data Parallelism** (`chunks={'cohort': 1}`) pour maximiser le débit. Observation empirique : speedup max = 2.41× avec 8 workers (plateau de saturation).
    -   **Zone intermédiaire (20-100 cohortes)** : Le choix dépend du temps de calcul par cohorte et du nombre de workers disponibles. Règle empirique : Data Parallelism rentable si `T_seq > 10s` pour 8 workers sur grille 500×500.
-   **Distribution des données** : Utiliser `dask.scatter` ou `client.scatter` pour pré-distribuer les forçages sur les workers (évite la sérialisation répétée).
-   **Parallélisation intra-tâche** : Pour aller au-delà du parallélisme par cohorte (ex: très haute résolution spatiale), envisager le chunking spatial (Dask Arrays), bien que cela nécessite la gestion complexe des "halos" pour le transport.

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
