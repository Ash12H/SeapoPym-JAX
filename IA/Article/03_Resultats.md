# Résultats

Cette section présente les résultats des expériences de validation de l'architecture DAG, organisés autour de trois axes : (1) la validation des composants individuels, (2) la validation du couplage, et (3) l'analyse des performances.

---

## 1. Validation des Composants Biologiques (0D)

### 1.1. Convergence Asymptotique

L'expérience valide la stabilité et la précision des processus biologiques en configuration 0D sur une large gamme de températures (0°C à 35°C). Le modèle LMTL présente une forte variabilité de la dynamique temporelle selon la température :

**Variabilité du système** :

-   Taux de mortalité $\lambda(T)$ : de 0.0067 day⁻¹ à 0°C à 0.70 day⁻¹ à 35°C (ratio 105×)
-   Échelle de temps caractéristique $1/\lambda(T)$ : de 1.4 jours (35°C) à 150 jours (0°C)
-   Biomasse d'équilibre $B_{eq}(T)$ : de 0.07 g/m² (35°C) à 7.5 g/m² (0°C)

Pour garantir la précision tout en optimisant le temps de calcul, nous utilisons une **approche adaptative** où la durée de simulation et le pas de temps ($\Delta t$) sont ajustés dynamiquement en fonction de l'échelle de temps caractéristique $\tau = 1/\lambda(T)$ :

$$ \text{Durée} = 15 \times \tau \quad \text{et} \quad \Delta t = \frac{\tau}{100} $$

Cette stratégie assure une convergence > 99.9% (15 périodes caractéristiques) avec une résolution temporelle fine (100 points par période) quelle que soit la cinétique. Concrètement :

-   **Durée de simulation** : de 21 jours (35°C) à 2250 jours (0°C)
-   **Pas de temps** : de 1235 s / 0.34 h (35°C) à 43200 s / 12 h (0°C)

**Configuration** :

-   Modèle 0D (sans transport)
-   36 simulations : 0°C à 35°C par pas de 1°C
-   12 cohortes, NPP fixe à 300 mg/m²/day
-   Paramètres LMTL standards (λ₀ = 1/150 day⁻¹, γ_λ = 0.15 °C⁻¹)

La **Figure 1A** compare la biomasse finale simulée avec l'asymptote théorique $B_{eq}(T) = R/\lambda(T)$ pour chaque température.

![Figure 1A : Convergence asymptotique de la biomasse](../../../data/article/figures/fig_01a_temperature_scan.png)

> Figure 1A : Biomasse d'équilibre simulée (points) vs théorique (ligne) en fonction de la température, avec erreur relative associée.

Les résultats démontrent une précision excellente sur toute la plage thermique :

-   **NRMSE** : 0.000001
-   **Erreur moyenne** : 0.0001%
-   **Erreur maximale** : 0.0003% (à 35°C)

La validation est donc **réussie** : l'architecture DAG reproduit fidèlement la dynamique asymptotique attendue sur plus de 5 ordres de grandeur de variation (ratio 105× entre biomasses extrêmes), indépendamment des conditions thermiques et du pas de temps, confirmant la robustesse de l'intégration numérique adaptative.

### 1.2. Comparaison avec SeapoPym v0.3 (Sans Transport)

Nous comparons ici les résultats d'une simulation globale (1998-2019) réalisée avec la nouvelle architecture DAG ('SeapoPym v1.0') contre ceux de l'implémentation Python précédente ('SeapoPym v0.3') en configuration sans transport (0D). Les deux modèles utilisent les mêmes forçages (température, production primaire) et les mêmes paramètres LMTL.

**Configuration** :

-   Modèle 0D (sans transport)
-   Grille globale : 170 × 360 points (~1° × 1°)
-   Période totale : 1998-2019 (spin-up 1998-1999, comparaison 2000-2019)
-   12 cohortes, pas de temps journalier (7304 pas de temps comparés)
-   Forçages : NPP et température 3D réels

La **Figure 1B** présente la distribution spatiale des différences entre les deux modèles.

![Figure 1B : Comparaison v0.3 vs DAG](../../../data/article/figures/fig_02a_comparison_v03_nrmse.png)

> Figure 1B : NRMSE spatiale entre SeapoPym v0.3 et SeapoPym v1.0 (architecture DAG).

L'analyse quantitative sur la période 2000-2019 montre une reproduction extrêmement fidèle des résultats :

| Métrique             | Valeur          | Seuil validation |
| -------------------- | --------------- | ---------------- |
| Corrélation          | **0.9994**      | > 0.99           |
| Biais moyen          | **0.0005 g/m²** | ≈ 0              |
| Erreur L2 normalisée | **3.0%**        | < 5%             |
| NRMSE                | **0.035**       | < 0.5            |
| RMSE                 | **0.051 g/m²**  | —                |

La faible valeur de la NRMSE (0.035), associée à une corrélation quasi-parfaite (> 0.999) et à l'absence de biais systématique (< 0.001 g/m²), confirme que la dynamique et les patrons spatio-temporels sont correctement reproduits. Tous les critères de validation sont satisfaits avec une large marge.

Les écarts mineurs observés sont attribuables au changement d'architecture : le passage d'une organisation séquentielle par processus à une résolution par Graphe de Flux (DAG) modifie l'ordre d'application des opérateurs introduisant des différences au fil du temps.

La validation de **non-régression** est donc **réussie** : l'architecture DAG reproduit fidèlement le comportement du modèle de référence en absence de transport.

---

## 2. Validation du Module de Transport 2D

Le schéma de transport (advection + diffusion) est validé par le **test de Zalesak**, une référence classique dans la littérature des méthodes numériques [Zalesak, 1979]. Ce test consiste à faire tourner un disque dans un champ de vitesse en rotation solide. Après une révolution complète, le disque devrait revenir identique à son état initial. L'écart mesuré quantifie la **diffusion numérique** introduite par le schéma.

### 2.1. Configuration

Le test de Zalesak utilise un **disque avec fente rectangulaire** ("slotted disk") qui tourne dans un champ de vitesse en rotation solide. Après une révolution complète, le disque devrait revenir identique à son état initial. L'écart mesuré quantifie la diffusion numérique.

**Domaine et slotted disk** (paramètres originaux de Zalesak) :

-   Domaine carré : [0, 1] × [0, 1] (équivalent à 1000 km × 1000 km)
-   Centre du disque : (0.5, 0.75)
-   Rayon du disque : 0.15
-   Fente rectangulaire : largeur = 0.05, hauteur = 0.25 (s'étend vers le bas)
-   Centre de rotation : (0.5, 0.5)
-   Période de rotation : 1 révolution
-   3 résolutions : 50×50, 100×100, 200×200

**Schéma numérique** :

-   Méthode des Volumes Finis (conservation de masse garantie)
-   Advection : schéma Upwind (1er ordre)
-   Intégration temporelle : Euler explicite
-   CFL = 0.5

### 2.2. Résultats

La **Figure 2A** compare l'état initial et final du disque pour différentes résolutions.

![Figure 2A : Test de Zalesak - Comparaison](../../../data/article/figures/fig_03d_zalesak_comparison.png)

> Figure 2A : État du disque avant (haut) et après (bas) une révolution. La diffusion numérique est visible à basse résolution (gauche) et diminue à haute résolution (droite).

La **Figure 2B** présente l'analyse de convergence spatiale.

![Figure 2B : Test de Zalesak - Convergence](../../../data/article/figures/fig_03d_zalesak_convergence.png)

> Figure 2B : (A) NRMSE vs espacement de grille en échelle log-log. La pente mesurée (0.22) indique une convergence positive. (B) Préservation du maximum, indicateur de la diffusion numérique.

**Métriques clés** :

| Résolution | dx [km] | NRMSE | Max Preservation |
| ---------- | ------- | ----- | ---------------- |
| 50×50      | 20      | 0.75  | 66%              |
| 100×100    | 10      | 0.66  | 68%              |
| 200×200    | 5       | 0.55  | 82%              |

-   **Conservation de masse** : parfaite (erreur < 10⁻¹⁴%)
-   **Ordre de convergence** : 0.22 (R² = 0.98)
-   **Convergence monotone** : l'erreur décroît avec la résolution

### 2.3. Discussion

L'ordre de convergence mesuré (0.22) est inférieur à l'ordre théorique (1.0) du schéma Upwind. Cet écart s'explique par la **diffusion numérique** inhérente au schéma Upwind :

$$ D\_{num} = \frac{u \Delta x}{2} (1 - \text{CFL}) $$

Cette diffusion numérique, bien documentée dans la littérature [LeVeque, 2002], domine l'erreur de troncature et réduit l'ordre effectif. Néanmoins, les points essentiels sont validés :

1. ✅ **Conservation de masse parfaite** : Le schéma Volumes Finis garantit la conservation stricte
2. ✅ **Convergence vers la solution** : L'erreur décroît de manière monotone avec la résolution
3. ✅ **Comportement documenté** : La diffusion numérique est conforme aux attentes pour un schéma Upwind

**Positionnement pour SeapoPym** : Le choix du schéma Upwind est un compromis pragmatique. Pour les applications dominées par la biologie (LMTL), la diffusion numérique est acceptable car :

-   Elle est du même ordre de grandeur que la diffusion physique réelle
-   Elle n'affecte pas la conservation de masse (essentielle pour un modèle de population)
-   L'architecture DAG permet de remplacer ce module par des schémas d'ordre supérieur (TVD, MUSCL) si nécessaire

---

## 3. Validation sur Simulation Réaliste (Pacifique)

Le modèle complet (couplage transport-biologie) est validé en le comparant au modèle de référence **Seapodym-LMTL** sur une simulation réaliste dans le Pacifique. Cette validation intégrée confirme que l'architecture DAG reproduit correctement la dynamique spatio-temporelle complexe issue de l'interaction entre les courants océaniques et les processus biologiques.

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

### 4.3. Limites du Parallélisme de Tâches (Loi d'Amdahl)

Bien que l'infrastructure logicielle soit capable de paralléliser efficacement des tâches indépendantes (speedup de 10.34× sur un test synthétique `sleep` avec 12 workers), son application au modèle réel se heurte à la structure même des calculs.

Comme montré en section 4.2, le transport de la production est une tâche monolithique représentant 80% du temps de calcul. Dans une stratégie de **Task Parallelism** pur (parallélisation inter-processus), cette tâche agit comme un goulot d'étranglement séquentiel. Selon la loi d'Amdahl, le speedup maximal est théoriquement borné :

$$S_{max} = \frac{1}{f_{seq}} = \frac{1}{0.80} = 1.25\times$$

Nos mesures confirment cette limite théorique : quel que soit le nombre de workers (jusqu'à 12), le speedup du Task Parallelism plafonne à **~1.28×**. L'ajout de ressources de calcul supplémentaires est inutile sans modifier la stratégie de décomposition.

### 4.4. Passage à l'Échelle via Parallélisme de Données (Strong Scaling)

Pour briser la limite d'Amdahl, nous adoptons une stratégie de **Data Parallelism** en divisant la tâche dominante (transport de production) selon la dimension `cohort`. Chaque cohorte étant physiquement indépendante, elles peuvent être transportées en parallèle.

Nous évaluons cette approche sur deux scénarios contrastés (Figure 4) :

1.  **Scénario "Zooplancton"** (12 cohortes) : Représente des organismes à vie courte.
2.  **Scénario "Micronecton"** (527 cohortes) : Représente des organismes à vie longue nécessitant un suivi fin.

**Résultats de Strong Scaling :**

-   **Micronecton (Complexité élevée)** : Le Data Parallelism délivre un speedup de **2.41×** avec 12 workers (contre 1.06× pour le Task Parallelism). L'accélération est significative car le temps de calcul par tâche domine largement le surcoût de gestion du graphe Dask.
-   **Zooplancton (Complexité faible)** : Le Data Parallelism est **moins performant** que l'exécution séquentielle (Speedup 0.61×). L'overhead constant de l'ordonnanceur Dask (~1-2s) devient prépondérant face à la rapidité du calcul physique pour un petit nombre de cohortes.

**Validation Numérique :**
La justesse des calculs ("correctness") est validée pour toutes les configurations parallèles. L'écart quadratique moyen (RMSE) par rapport à la référence séquentielle reste inférieur à $10^{-10}$ g/m², confirmant que le découpage des données n'introduit aucun biais numérique.

En conclusion, le Data Parallelism est la clé pour le passage à l'échelle des simulations complexes (Micronecton), permettant de dépasser la barrière d'Amdahl, tandis que l'exécution séquentielle reste optimale pour les modèles légers.

## Résumé des Validations

| Expérience                         | Métrique                  | Résultat     | Validation |
| ---------------------------------- | ------------------------- | ------------ | ---------- |
| **1. Composants Biologiques (0D)** |                           |              |            |
| 1.1 Convergence asymptotique       | Erreur vs théorie (36 T°) | NRMSE < 10⁻⁶ | ✓          |
| 1.2 Comparaison SeapoPym v0.3      | Corrélation               | 0.9994       | ✓          |
|                                    | NRMSE                     | 0.035        | ✓          |
| **2. Transport 2D (Zalesak)**      |                           |              |            |
| Test de rotation                   | Conservation masse        | < 10⁻¹⁴%     | ✓          |
|                                    | Convergence spatiale      | O(Δx^{0.22}) | ✓\*        |
| **3. Simulation Pacifique**        |                           |              |            |
| Comparaison Seapodym-LMTL          | NRMSE (avec transport)    | 0.44         | ✓          |
|                                    | Amélioration vs no-trans  | -34%         | ✓          |
| **4. Performances**                |                           |              |            |
| 4.1 Weak Scaling                   | Complexité                | O(N^{1.01})  | ✓          |
| 4.2 Décomposition                  | Transport production      | 80%          | —          |
| 4.3 Task Parallelism               | Speedup max (Amdahl)      | ~1.28×       | ✓          |
| 4.4 Data Parallelism               | Speedup (Micronecton)     | **2.41×**    | ✓          |

**Note** : \*L'ordre de convergence réduit (0.22) pour le schéma Upwind est dû à la diffusion numérique, comportement attendu pour ce type de schéma [LeVeque, 2002; Zalesak, 1979]. La conservation de masse parfaite confirme la cohérence du schéma. L'architecture DAG permet de remplacer ce module par des schémas d'ordre supérieur si nécessaire.
