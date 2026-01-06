# Résultats

Cette section présente les résultats des expériences de validation de l'architecture DAG, organisés autour de trois axes : (1) la validation des composants individuels, (2) la validation du couplage, et (3) l'analyse des performances.

---

## 1. Validation des Composants Biologiques (0D)

### 1.1. Convergence Asymptotique

L'expérience valide la stabilité et la précision des processus biologiques en configuration 0D sur une large gamme de températures (0°C à 35°C). Pour garantir la précision tout en optimisant le temps de calcul, nous utilisons une approche adaptative où la durée de simulation et le pas de temps ($\Delta t$) sont ajustés dynamiquement en fonction de l'échelle de temps caractéristique de la mortalité $\tau = 1/\lambda(T)$ :

$$ \Delta t = \frac{\tau}{N\_{steps}} = \frac{1}{\lambda(T) \times 100} $$

Cela assure une résolution temporelle constante (100 points par période caractéristique) quelle que soit la cinétique de la réaction.

**Configuration** :

-   Modèle 0D (sans transport)
-   Scan en température : 0°C à 35°C ($\Delta T = 1^{\circ}C$)
-   Pas de temps et durée adaptatifs

La **Figure 1A** compare la biomasse finale simulée avec l'asymptote théorique $B_{eq}(T) = R/\lambda(T)$ pour chaque température.

![Figure 1A : Convergence asymptotique de la biomasse](../../../data/article/figures/fig_01a_temperature_scan.png)

> Figure 1A : A. Biomasse d'équilibre simulée (points) vs théorique (ligne) en fonction de la température. B. Erreur relative.

Les résultats démontrent une précision excellente sur toute la plage thermique :

-   **Erreur moyenne** : ~0.0001%
-   **Erreur maximale** : < 0.001%

La validation est donc **réussie** : l'architecture DAG reproduit fidèlement la dynamique asymptotique attendue, indépendamment des conditions thermiques et du pas de temps, confirmant la robustesse de l'intégration numérique adaptative.

### 1.2. Comparaison avec SeapoPym v0.3 (Sans Transport)

Nous comparons ici les résultats d'une simulation globale (2000-2019) réalisée avec la nouvelle architecture DAG ('SeapoPym v1.0') contre ceux de l'implémentation Python précédente ('SeapoPym v0.3') en configuration sans transport (0D). Les deux modèles utilisent les mêmes forçages (température, production primaire) et les mêmes paramètres LMTL.

**Configuration** :

-   Modèle 0D (sans transport)
-   Période de comparaison : 2000-2019 (après spin-up de 2 ans)
-   Forçages : Données globales réelles

La **Figure 1B** présente la distribution spatiale des différences entre les deux modèles.

![Figure 1B : Comparaison v0.3 vs DAG](../../../data/article/figures/fig_02a_comparison_v03_diff.png)

> Figure 1B : Différences entre SeapoPym DAG et SeapoPym v0.3. Gauche : Biais moyen (g/m²). Droite : NRMSE moyenne.

L'analyse quantitative sur la période 2000-2019 montre une reproduction extrêmement fidèle des résultats :

-   **Corrélation spatio-temporelle** : **0.9994**
-   **Biais moyen** : **0.0005 g/m²** (négligeable)
-   TODO : **NRMSE** : **0.035**

La faible valeur de la NRMSE (0.035), associée à une corrélation quasi-parfaite (> 0.999) et à l'absence de biais systématique, confirme que la dynamique et les patrons spatiaux sont correctement reproduits. Les écarts mineurs observés sont attribuables au changement d'architecture : le passage d'une organisation séquentielle par processus à une résolution par Graphe de Flux (DAG) modifie l'ordre d'application des opérateurs.

La validation de non-régression est donc **réussie** : l'architecture DAG reproduit fidèlement le comportement du modèle de référence en absence de transport.

---

## 2. Validation du Module de Transport

### 2.1. Transport 1D vs Solution Analytique

Le schéma de transport (Advection + Diffusion) est validé contre une solution analytique dans un domaine 1D.

**Configuration** : Advection d'une distribution gaussienne, schéma Upwind + diffusion centrée, conditions aux limites fermées.

La **Figure 2A** compare le profil de concentration simulé avec la solution analytique à différents instants. L'accord est excellent, avec une légère diffusion numérique inhérente au schéma Upwind du premier ordre.

![Figure 2A : Profil de transport vs solution analytique](../../../data/article/figures/fig_02a_transport_profile.png)

La **Figure 2B** présente l'évolution de la masse totale normalisée. La masse reste constante à **100.000%** (aux erreurs d'arrondi près) pendant toute la simulation, confirmant la **conservation stricte** du schéma Volumes Finis (voir Figure 2B).

![Figure 2B : Conservation de la masse](../../../data/article/figures/fig_02b_transport_mass.png)

### 2.2. Stabilité et Condition CFL

La sensibilité du schéma numérique à la condition CFL est analysée pour déterminer le pas de temps optimal (voir **Figure 2C**).

![Figure 2C : Stabilité CFL](../../../data/article/figures/fig_02c_transport_cfl.png)

Le schéma est stable pour CFL < 1, avec un **optimum de précision observé autour de CFL ≈ 0.5**, où la **NRMSE atteint son minimum de 0.0703**.

> **Note sur la Diffusion Numérique** :
> L'erreur minimale à CFL=0.5 s'explique par le compromis inhérent au schéma Upwind du premier ordre. Sa diffusion numérique étant proportionnelle à $(1-\text{CFL})\frac{u\Delta x}{2}$, réduire excessivement le pas de temps (CFL $\to$ 0) augmente paradoxalement la diffusion artificielle, dégradant la solution [LeVeque, 2002].

Le schéma valide donc les critères de stabilité et de précision suffisants pour l'application cible.

---

## 3. Validation du Couplage Transport-Biologie

### 3.1. Test de Convergence en Grille

> TODO(Jules): Soit -> garder un Dt unique ; soit -> espace 2D ou je fais varier a la fois Dt et Dx

Cette expérience valide le couplage entre transport et biologie en configuration 2D. Un patch de biomasse initial est advecté tout en subissant des réactions biologiques (mortalité). Nous testons 50 résolutions de grille réparties logarithmiquement pour démontrer la convergence.

**Configuration** : Domaine 2D, température constante, courant uniforme, 50 résolutions de 1° à 1/24° (Δx : 111 km → 4.6 km).

La **Figure 3B** superpose les profils de concentration à une latitude fixe pour 5 résolutions représentatives. Les profils convergent vers une solution commune lorsque Δx diminue, l'effet de diffusion numérique étant clairement visible pour les résolutions grossières.

![Figure 3B : Profils de concentration](../../../data/article/figures/fig_03b_coupling_profile.png)

La **Figure 3D** présente la NRMSE (Normalized Root Mean Square Error) en fonction de Δx pour les 50 résolutions testées :

![Figure 3D : Convergence en grille](../../../data/article/figures/fig_03d_grid_convergence.png)

La **pente mesurée est de 1.10** (R² > 0.94), proche de la valeur attendue de 1.0 pour un schéma Upwind du premier ordre. L'écart avec la théorie (pente entre 1.0 et 1.2 attendue) s'explique par la superposition de la diffusion physique (D = 1000 m²/s) et de la diffusion numérique du schéma.

On observe des **oscillations légères autour de la droite de convergence**. Ces fluctuations sont principalement dues aux **variations du nombre de Courant effectif** entre résolutions : le pas de temps est ajusté pour maintenir la stabilité (CFL < 1) selon `dt = min(dt_adv, dt_diff)`, mais le basculement entre limitation par advection (dt ∝ Δx) et par diffusion (dt ∝ Δx²) crée un changement non-monotone de la diffusion numérique :

$$ D*{num} \approx \frac{u \cdot \Delta x}{2} \times (1 - \text{CFL}*{eff}) $$

Malgré ces oscillations, la **tendance O(Δx) est clairement établie** (R² > 0.94), confirmant que l'architecture DAG couple correctement transport et biologie **sans biais de Time Splitting**. L'erreur observée est d'origine purement numérique et décroît bien proportionnellement au raffinement de la grille.

### 3.2. Comparaison avec Seapodym-LMTL (Avec Transport)

Le modèle complet (couplage transport-biologie) est validé en le comparant au modèle de référence Seapodym-LMTL sur une simulation réaliste dans le Pacifique. L'objectif est de vérifier que la nouvelle architecture DAG reproduit correctement la dynamique spatio-temporelle complexe issue de l'interaction entre les courants et les processus biologiques.

**Configuration** :

-   **Domaine** : Pacifique (110°E - 290°E, Latitudes -60° à +60°).
-   **Période** : 2002-2004 (après 2 ans de spin-up).
-   **Forçages** : Données physiques (GLORYS) et biogéochimiques (PISCES) journalières réelles.
-   **Référence** : Sortie officielle du modèle Seapodym-LMTL exécuté avec les mêmes forçages.

La **Figure 3** illustre la distribution spatiale de l'erreur (NRMSE) entre le modèle DAG et la référence.

![Figure 3 : Erreurs spatiales](../../../data/article/figures/fig_05e_spatial_nrmse.png)

> Figure 3 : Cartes de NRMSE (Erreur quadratique moyenne normalisée) entre la simulation DAG et la référence Seapodym-LMTL. Gauche : DAG avec transport. Droite : DAG sans transport.

**Analyse Quantitative** :

L'activation du module de transport dans l'architecture DAG permet une réduction significative de l'erreur par rapport à une simulation statique (sans transport) :

-   **NRMSE Global** : **0.44** (contre 0.67 sans transport).

L'amélioration apportée par le transport est substantielle, avec une **réduction de 34% de l'erreur normalisée**. Bien que des différences subsistent (NRMSE = 0.44), elles sont attendues compte tenu des différences fondamentales entre les schémas numériques (Volumes Finis en Python vs Différences Finies en C++ sur grille Arakawa C) et la gestion des masques/bords. Cette amélioration confirme que le modèle capture bien les flux physiques de biomasse.

La **Figure 3-bis** (Séries temporelles par zone) démontre que la dynamique saisonnière et interannuelle est correctement synchronisée entre les deux modèles dans les différentes zones climatiques (Tempérée Nord, Tropicale, Tempérée Sud).

![Figure 3-bis : Séries temporelles par zone](../../../data/article/figures/fig_05e_pacific_timeseries_zones.png)

Conclusion : L'architecture DAG reproduit de manière satisfaisante la dynamique du modèle de référence en conditions réelles.

---

## 4. Analyse des Performances

### 4.1. Complexité Algorithmique (Weak Scaling)

La complexité algorithmique est évaluée en mesurant le temps de calcul pour des grilles de taille croissante.

**Configuration** : Grilles 500×500, 1000×1000, 2000×2000, 50 cohortes, backend séquentiel.

La **Figure 4A** présente le temps de calcul par pas de temps en fonction du nombre de cellules, sur un graphe log-log.

![Figure 4A : Weak Scaling](../../../data/article/figures/fig_04a_weak_scaling.png)

| Grille    | Cellules  | Temps/Step (ms) |
| --------- | --------- | --------------- |
| 500×500   | 250,000   | 124             |
| 1000×1000 | 1,000,000 | 517             |
| 2000×2000 | 4,000,000 | 2021            |

**Pente mesurée : 1.006**, correspondant à une complexité **O(N^1.01) ≈ O(N)**.

L'architecture DAG a une complexité **linéaire** en fonction de la taille du problème, sans surcoût algorithmique caché. Le temps de calcul double approximativement lorsque la grille double.

### 4.2. Décomposition du Temps de Calcul

Pour comprendre les contraintes de parallélisation, nous analysons la répartition du temps de calcul par type de tâche (voir Figure 4B).

![Figure 4B : Décomposition du temps de calcul](../../../data/article/figures/fig_04f_time_decomposition.png)

**Configuration** : Grille 500×500, 10 cohortes, 20 pas de temps, profilage par décorateur.

| Catégorie                | Temps (s) | % du temps |
| ------------------------ | --------- | ---------- |
| **Transport Production** | 0.994     | **80.2%**  |
| Mortalité                | 0.123     | 9.9%       |
| Transport Biomasse       | 0.102     | 8.3%       |
| Production               | 0.021     | 1.7%       |

**Le transport de la production représente 80% du temps de calcul.**

Cette dominance d'une seule tâche a des implications directes pour la parallélisation. Selon la Loi d'Amdahl, avec une fraction séquentielle de 80%, le speedup maximal théorique est borné :

$$S_{max} = \frac{1}{f_{seq}} = \frac{1}{0.80} = 1.25\times$$

Même avec un nombre infini de workers, le speedup ne peut dépasser **1.25×** tant que le transport de production n'est pas lui-même parallélisé (par chunking spatial, par exemple).

### 4.3. Validation du Système Complet (Blueprint + Controller + DaskBackend)

Pour confirmer que le système complet parallélise correctement les tâches indépendantes, nous testons avec 12 groupes fonctionnels indépendants, chacun contenant une fonction synthétique (`time.sleep`) qui libère explicitement le GIL. Cette architecture simule un modèle multi-espèces réaliste.

**Configuration** : 12 groupes fonctionnels indépendants, 1 tâche sleep (100ms) par groupe, système complet Blueprint → SimulationController → DaskBackend.

La **Figure 4C** présente le speedup et l'efficacité en fonction du nombre de workers :

![Figure 4C : Validation du système complet](../../../data/article/figures/fig_04e_sleep_parallelism_blueprint.png)

| Workers | Temps (s) | Speedup    | Efficacité |
| ------- | --------- | ---------- | ---------- |
| 1       | 1.266     | 1.00×      | 100%       |
| 4       | 0.337     | 3.78×      | 95%        |
| 6       | 0.233     | 5.45×      | 91%        |
| 12      | 0.123     | **10.34×** | **86%**    |

Le speedup est **quasi-linéaire** (10.34× avec 12 workers, efficacité 86%), confirmant que le système complet parallélise efficacement les groupes fonctionnels lorsque ceux-ci sont :

1. **Indépendants** (pas de dépendances entre groupes dans le DAG)
2. **Libèrent le GIL** (condition nécessaire pour le ThreadPoolScheduler)

L'overhead du système complet (Blueprint + Controller + DaskBackend) est estimé à **~23ms** (18.7%), ce qui reste acceptable. On note également que le speedup optimal est atteint lorsque le nombre de workers divise exactement le nombre de tâches (ici 12), car les tâches s'exécutent alors en "vagues" complètes sans workers inactifs.

Ce test valide l'infrastructure de parallélisation. Le speedup limité observé dans le modèle réel (~1.25×) n'est **pas** dû à un défaut du système, mais à la **structure du modèle** : le transport de production, tâche dominante (80%), ne peut être parallélisé au niveau inter-tâches.

---

## Résumé des Validations

| Expérience                | Métrique           | Résultat   | Validation |
| ------------------------- | ------------------ | ---------- | ---------- |
| Bio 0D                    | Erreur vs théorie  | < 0.01%    | ✓          |
| Transport 1D              | Conservation masse | 100.00%    | ✓          |
| Couplage 2D               | Convergence        | O(Δx^1.25) | ✓          |
| Weak Scaling              | Complexité         | O(N^1.01)  | ✓          |
| Décomposition             | Transport dominant | 80%        | —          |
| Validation Système        | Speedup (sleep)    | 10.34×     | ✓          |
| Comparaison SeapoPym v0.3 | Corrélation        | > 0.999    | ✓          |
| Comparaison Seapodym-LMTL | NRMSE 2D           | 0.44       | ✓          |
