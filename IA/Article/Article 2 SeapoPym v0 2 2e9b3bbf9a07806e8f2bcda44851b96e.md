# Article 2 : SeapoPym v0.2

# Contexte

Cet article est le deuxième d'une série de trois. L'article 1 (soumis à GMD) présentait SeapoPym v0.1, une réimplémentation Python de Seapodym-LMTL sans transport (0D). Il mettait en évidence le problème d'équifinalité : de nombreux jeux de paramètres très différents produisent des résultats quasi identiques en 0D.

Le présent article introduit le transport advectif-diffusif et une méthode complète d'optimisation et d'échantillonnage des paramètres. L'article 3 s'appuiera sur ces outils pour estimer, à partir d'observations réelles, les flux de carbone liés au zooplancton.

# Question scientifique

**Question principale :** L'ajout du transport advectif dans un modèle de dynamique du zooplancton réduit-il l'équifinalité observée en 0D, et les paramètres du modèle deviennent-ils mieux identifiables ?

**Hypothèse :** En 0D, seule la dynamique temporelle contraint les paramètres, ce qui laisse un large espace de solutions acceptables (équifinalité démontrée dans l'article 1). L'ajout du transport introduit des contraintes spatiales supplémentaires — gradients de biomasse, fronts océanographiques, afflux dans les zones à fort courant — qui réduisent l'espace des solutions et permettent de mieux distinguer les jeux de paramètres. On s'attend à ce que CMA-ES identifie moins de modes en 2D qu'en 0D, et que NUTS révèle des distributions postérieures plus étroites.

**Contribution méthodologique :** L'article démontre également la faisabilité d'une chaîne d'estimation complète pour les modèles écologiques différentiables : CMA-ES pour la recherche globale des modes, puis NUTS pour la caractérisation de l'incertitude paramétrique. Cette chaîne est rendue possible par la différentiabilité du modèle (JAX) et son exécution rapide sur GPU. CMA-ES est choisi pour sa capacité à adapter sa matrice de covariance au cours de l'optimisation, capturant ainsi les interactions entre paramètres mises en évidence par l'analyse de Sobol de l'article 1. NUTS exploite le gradient exact fourni par la différentiation automatique pour explorer efficacement la distribution postérieure autour de chaque mode identifié.

# 1. Introduction

## 1.1 Contexte scientifique

La modélisation de la dynamique du zooplancton à l'échelle régionale ou globale nécessite de coupler des processus de dynamique de population (croissance, mortalité, reproduction) avec le transport physique (advection, diffusion). Le modèle Seapodym-LMTL (C++) réalise ce couplage en intégrant l'ensemble des processus dans une équation d'advection-diffusion-réaction résolue par un schéma numérique dédié. Cependant, cette formulation couplée rend difficile l'analyse des composants individuels du modèle — par exemple, il n'est pas possible d'isoler l'effet du transport sur l'identifiabilité des paramètres.

## 1.2 Limites de l'article 1 et motivation

L'article 1 présentait SeapoPym v0.1, une réimplémentation Python de Seapodym-LMTL sans transport (0D), et démontrait le problème d'équifinalité : de multiples jeux de paramètres très différents produisent des résultats quasi identiques. L'analyse de sensibilité de Sobol révélait de fortes interactions entre paramètres, suggérant que l'équifinalité est en partie liée à des compensations paramétriques. De plus, l'architecture de v0.1 (classes Python utilisant xarray/dask/numba) était inadaptée à l'intégration du transport et au calcul de gradients, deux prérequis pour tester si la spatialisation du modèle réduit l'équifinalité.

## 1.3 Ce que cet article propose

Nous présentons SeapoPym v0.2, une refonte fondée sur JAX, qui apporte trois capacités :

- **Le transport advectif-diffusif**, intégré via un schéma de volumes finis au sein d'une architecture centrée sur le pas de temps, où chaque processus est formulé comme un flux net par cellule.
- **La différentiation automatique** du modèle complet, permettant le calcul exact du gradient de la fonction de coût par rapport aux paramètres.
- **Le calcul sur GPU et la parallélisation** (vmap), nécessaires pour le coût computationnel de l'estimation de paramètres.

Ces capacités permettent de mettre en place une stratégie d'estimation en deux temps. D'abord, CMA-ES (Hansen, 2016) effectue une recherche globale pour identifier les modes de la fonction de coût. CMA-ES est un algorithme d'optimisation évolutionnaire qui adapte sa matrice de covariance au fil des itérations, capturant ainsi les interactions entre paramètres — propriété particulièrement pertinente étant donné les interactions révélées par l'analyse de Sobol de l'article 1. CMA-ES est devenu un outil standard pour la calibration de modèles écologiques complexes (Van der Meersch et al., 2023 ; Sauerland et al., 2023 ; Schartau et al., 2017). Ensuite, NUTS (Hoffman & Gelman, 2014), une extension de Hamiltonian Monte Carlo (Neal, 2011) qui supprime le réglage manuel de la longueur de trajectoire, échantillonne la distribution postérieure autour de chaque mode identifié. NUTS exploite le gradient exact fourni par la différentiation automatique pour explorer efficacement l'espace des paramètres et quantifier l'incertitude paramétrique.

Cette chaîne CMA-ES → NUTS permet de répondre à la question scientifique : le transport réduit-il l'équifinalité observée en 0D ?

# 2. Méthodes

## 2.1 Architecture de SeapoPym v0.2

SeapoPym v0.2 est une refonte complète de l'architecture, implémentée en JAX (compilation JIT, parallélisation via vmap, différentiation automatique native).

Le cœur de l'architecture repose sur une formulation où chaque processus — qu'il soit biogéochimique ou physique — est exprimé comme un flux net par cellule et par traceur. L'état du modèle est un ensemble de traceurs, mis à jour à chaque pas de temps par une simple somme de flux :

$$
X(t + dt) = \max(0, X(t) + dt \times \sum_i F_i(t))
$$

où $X$ est un traceur quelconque et $F_i$ les flux produits par chaque processus, organisés en un graphe acyclique directionnel (DAG). Le seuil à zéro garantit la positivité des concentrations. Cette formulation a quatre propriétés : (1) **modularité** — chaque processus est un nœud indépendant du DAG, activable ou remplaçable sans modifier le reste ; (2) **différentiabilité** — la somme de flux est trivialement différentiable, permettant la propagation du gradient à travers le DAG ; (3) **extensibilité** — ajouter un processus revient à ajouter un nœud ; (4) **indépendance à l'ordre** — tous les flux sont évalués à partir de l'état courant puis sommés, contrairement à un splitting d'opérateurs.

Le transport advectif-diffusif est traité comme n'importe quel autre processus : il produit un flux net par cellule calculé par un schéma de volumes finis (upwind ordre 1). Les détails du schéma numérique et des conditions aux limites sont donnés en Annexes B. Le fonctionnement détaillé du modèle appliqué au zooplancton (traceurs, cohortes, recrutement, mortalité) est décrit en Annexe A.

## 2.2 Stratégie d'estimation des paramètres

La stratégie d'estimation combine une recherche globale (CMA-ES) et un échantillonnage bayésien (NUTS), comme introduit en section 1.3. On détaille ici les choix opérationnels.

**CMA-ES.** On effectue plusieurs restarts avec populations croissantes (IPOP-CMA-ES ; Auger & Hansen, 2005) : à chaque restart, la taille de la population est augmentée, ce qui élargit progressivement l'exploration de l'espace des paramètres et améliore la détection de modes distincts. La fonction de coût est une moyenne pondérée de NRMSE par jeu de données (poids 1/N), conformément à la méthodologie de l'article 1.

> [...**todo**... Nombre de restarts, taille de population, budget total d'évaluations, bornes des paramètres.]
> 

**NUTS.** L'échantillonnage bayésien est réalisé via NUTS (implémentation BlackJAX), initialisé au meilleur point de chaque mode identifié par CMA-ES. NUTS nécessite la log-postérieure et son gradient, fournis par la différentiation automatique de JAX.

> [...**todo**... **Construction de la log-vraisemblance.** Pour passer de la NRMSE (CMA-ES) à NUTS, il faut définir la log-postérieure = log-vraisemblance + log-prior. Sous hypothèse d'erreurs gaussiennes : log L(θ) = −(1/2σ²) × Σ(obs − sim)². Le paramètre σ contrôle la largeur de l'incertitude estimée. Options : (1) σ fixé a priori, (2) σ estimé comme paramètre libre par NUTS avec un prior half-normal, (3) σ dérivé de la normalisation NRMSE. Si plusieurs jeux de données avec pondération 1/N, définir un σ par jeu de données.]
> 

> [...**todo**... **Priors.** Tableau complet des priors pour chaque paramètre : distribution (normale, uniforme), bornes, valeur centrale. Exemple : τ_r0 borné par la mesure maximale obtenue sur des espèces d'eau froide (~100/200 jours pour certains ptéropodes d’après certaines sources, 20 d’après bednarsek).]
> 

> [...**todo**... Nombre de chaînes, warmup, nombre d'échantillons, diagnostics de convergence.]
> 

---

# 3. Expériences et résultats

Les expériences sont organisées en quatre étapes progressives, de la validation technique à la démonstration scientifique.

## 3.1 Validation du schéma de transport

**Objectif :** vérifier que le schéma de volumes finis implémenté dans SeapoPym reproduit correctement l'advection-diffusion.

- Test de Zalesak (Zalesak, 1973).
- Convergence numérique en fonction de la résolution.

*Nature du résultat : prérequis technique. On vérifie que le code est correct.*

## 3.2 Comparaison avec LMTL (C++) et SeapoPym v0.1

**Objectif :** quantifier l'impact du transport sur les champs simulés de zooplancton, et identifier les différences avec LMTL.

- Comparaison des champs spatiaux v0.2 (avec transport) vs. v0.1 (sans transport) vs. LMTL (C++).
- Mise en évidence d'un biais en eau chaude (10–20%) entre SeapoPym (v0.1 et v0.2) et LMTL. Ce biais, déjà présent sans transport, n'est pas lié au schéma de transport. Il constitue un résultat secondaire dont les hypothèses sont discutées en section 4.3.

*Nature du résultat : résultat secondaire. Le schéma FV d'ordre 1 suffit à répliquer le comportement qualitatif de LMTL, malgré un biais quantitatif en eaux chaudes.*

## 3.3 Performance computationnelle

**Objectif :** démontrer que l'implémentation est utilisable en pratique pour l'estimation de paramètres, qui nécessite des milliers d'évaluations du modèle.

- Benchmark CPU vs. GPU.
- Comparaison des temps de calcul avec LMTL (C++).
- Gain apporté par vmap (parallélisation d'ensembles de simulations).

*Nature du résultat : résultat technique. Le modèle est assez rapide pour ce qu'on veut en faire.*

## 3.4 Expérience jumelle : CMA-ES + NUTS

**Objectif :** tester si le transport réduit l'équifinalité, et caractériser l'incertitude paramétrique.

**Protocole :** expérience jumelle (twin experiment) sur des grilles ~20×20. On génère des données synthétiques avec un jeu de paramètres « vrai » connu, puis on applique la chaîne CMA-ES → NUTS pour retrouver ces paramètres. Chaque cas est réalisé en 0D (sans transport) et en 2D (avec transport) pour permettre une comparaison directe.

**Trois cas d'étude :**

- **Cas 1 — Eau chaude, faible courant**. Peu de contraintes spatiales ; sert de baseline.
- **Cas 2 — Eau chaude, fort gradient** (ex. bordure de gyre, upwelling). Contrainte spatiale forte ; on s'attend à la plus grande réduction d'équifinalité.
- **Cas 3 — Eau froide, fort courant** (ex. Pacifique Nord subarctique). Teste la robustesse en régime de température différent.

> [...**todo**... Définir les zones géographiques précises pour chaque cas. Définir τ_r0 pour chaque cas (faible en eau chaude, plus élevé en eau froide). Contrainte mémoire VRAM : τ_r0 élevé implique une dimension cohorte importante.]
> 

**Résultats attendus :**

1. Nombre de modes identifiés par CMA-ES : 0D vs. 2D, par cas.
2. Distributions postérieures (corner plots) issues de NUTS : largeur et corrélations, 0D vs. 2D, par cas.
3. L'amplitude de la réduction d'équifinalité devrait dépendre de la dynamique spatiale locale : plus forte en zone à forts gradients, plus faible en zone homogène.

*Nature du résultat : résultat scientifique principal. Répond directement à la question de recherche.*

---

# 4. Discussion

## 4.1 Le transport comme contrainte sur l'identifiabilité

Interprétation des résultats de l'expérience jumelle selon les trois cas. Le transport introduit-il suffisamment de contraintes spatiales pour lever l'équifinalité ? L'effet varie-t-il entre les cas (baseline vs. fort gradient vs. eau froide) ? Quels paramètres deviennent identifiables et pourquoi ? Si l'équifinalité persiste dans certains cas, quelles contraintes supplémentaires seraient nécessaires (observations multi-sites, données supplémentaires) ?

## 4.2 Complémentarité optimisation / échantillonnage

Retour d'expérience sur la chaîne CMA-ES → NUTS. CMA-ES identifie les modes, NUTS caractérise chaque mode individuellement — cette séparation contourne la difficulté de NUTS face aux postérieures multimodales. Coût computationnel : chaque pas leapfrog de NUTS nécessite une évaluation du modèle et de son gradient, rendu praticable par JAX + GPU (cf. section 3.3). Applicabilité à d'autres modèles écologiques différentiables.

## 4.3 Différences avec LMTL (C++)

Un biais de 10–20% est observé en eaux chaudes entre SeapoPym et LMTL. Ce biais est présent dès v0.1 (sans transport) et persiste en v0.2, ce qui exclut le schéma de transport comme cause.

> [...**todo**... Lister les hypothèses possibles : différence de schéma numérique pour les processus biogéochimiques ? Gestion des conditions initiales ? Paramétrisations internes de LMTL non documentées ?]
> 

Ce biais ne compromet pas les conclusions de l'article car l'analyse d'équifinalité compare les expériences 0D et 2D au sein de SeapoPym, et non avec LMTL.

## 4.4 Limites et perspectives

- Expérience jumelle seulement (données synthétiques) — l'application aux données réelles d'observation est l'objet de l'article 3.
- Le coût computationnel de NUTS limite la taille de grille et le nombre de cohortes (contrainte VRAM).
- Le choix de σ dans la log-vraisemblance influence directement la largeur des postérieures estimées.
- La différentiabilité du modèle dépend de la formulation des processus ; certaines paramétrisations pourraient ne pas être différentiables.
- Perspectives : estimation des flux de carbone (article 3), calibration multi-sites avec davantage de stations d'observation, extension à d'autres groupes fonctionnels.

---

# Annexes

## Annexe A : Architecture détaillée de SeapoPym Zooplancton

> [...todo... Présenter précisément comment fonctionne le modèle SeapoPym appliqué au zooplancton : traceurs, cohortes, vieillissement, recrutement, mortalité dépendante de la température, transfert d'énergie. Inclure la figure du DAG montrant l'enchaînement des processus à chaque pas de temps.]
> 

## Annexe B : Schéma numérique du transport

> [...todo... Détail du schéma de volumes finis : discrétisation spatiale, schéma d'advection upwind ordre 1, conditions aux limites (OPEN, CLOSED, PÉRIODIQUES), masque terre/mer, condition CFL et vérification. Expliquer pourquoi la diffusion numérique peut être utilisée comme coefficient de diffusion physique. Mentionner les alternatives (schémas d'ordre supérieur) et justifier pourquoi upwind O1 suffit pour cette application.]
> 

---

# Références clés

**HMC / NUTS**

- Duane, S., Kennedy, A. D., Pendleton, B. J., & Roweth, D. (1987). Hybrid Monte Carlo. *Physics Letters B*, 195(2), 216–222.
- Neal, R. M. (2011). MCMC using Hamiltonian dynamics. In *Handbook of Markov Chain Monte Carlo* (Chapter 5). CRC Press.
- Hoffman, M. D. & Gelman, A. (2014). The No-U-Turn Sampler. *JMLR*, 15, 1593–1623.

**CMA-ES**

- Hansen, N. (2016). The CMA Evolution Strategy: A Tutorial. arXiv:1604.00772.
- Auger, A. & Hansen, N. (2005). A restart CMA evolution strategy with increasing population size. In *IEEE Congress on Evolutionary Computation*, pp. 1769–1776.

**CMA-ES en modélisation écologique / biogéochimique**

- Van der Meersch, V. et al. (2023). Estimating process-based model parameters from species distribution data using the evolutionary algorithm CMA-ES. *Methods in Ecology and Evolution*, 14, 855–868.
- Sauerland, V. et al. (2023). A CMA-ES Algorithm Allowing for Random Parameters in Model Calibration. *JAMES*, 15, e2022MS003390.
- Schartau, M. et al. (2017). Reviews and syntheses: Parameter identification in marine planktonic ecosystem modelling. *Biogeosciences*, 14, 1647–1701.

> [...todo... ajouter : références LMTL, article 1 (SeapoPym v0.1), JAX, références volumes finis (Zalesak), BlackJAX]
>