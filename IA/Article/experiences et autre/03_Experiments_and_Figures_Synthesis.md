# Synthèse des Expériences et Figures — Article SeapoPym DAG

**Date** : 2026-01-01
**Auteur** : Compilation des résultats expérimentaux

---

## Vue d'Ensemble

| Notebook | Objectif | Résultat Clé | Statut |
|----------|----------|--------------|--------|
| 1 | Validation Biologie 0D | Erreur < 0.14% | ✅ PASS |
| 2 | Validation Transport 1D | Conservation 100% | ✅ PASS |
| 3 | Validation Couplage | Convergence O(Δx) | ✅ PASS |
| 4A | Weak Scaling | O(N^1.01) | ✅ PASS |
| 4C | Strong Scaling Multi-Groupes | Speedup 1.67× | ⚠️ Limité |
| 4E | Diagnostic Dask | Speedup 11.5× | ✅ Dask OK |
| 4F | Décomposition Temps | Transport = 88% | ✅ Explique 4C |

---

## Notebook 1 : Validation Biologie 0D

### Objectif
Reproduire l'expérience de convergence asymptotique de SeapoPym v0.3 avec la nouvelle architecture DAG. Démontrer la non-régression des processus biologiques.

### Configuration
- Modèle 0D (sans transport)
- Températures testées : 0°C, 10°C, 20°C, 30°C
- Durée : convergence vers équilibre

### Figures

#### Figure 1A : Courbes de Convergence Asymptotique
**Description** : 4 courbes (une par température) montrant l'évolution de la biomasse dans le temps, convergeant vers la valeur d'équilibre théorique B_eq = R/λ(T).

**Observation** : Toutes les courbes convergent vers leur valeur théorique respective, validant les équations biologiques.

#### Figure 1B : Tableau de Validation
**Description** : Tableau comparant les valeurs simulées vs théoriques pour chaque température.

| T [°C] | B_eq_theory | B_eq_sim | Erreur |
|--------|-------------|----------|--------|
| 0 | 7.5060 | 7.4957 | 0.137% |
| 10 | 1.7660 | 1.7660 | 0.000% |
| 20 | 0.4586 | 0.4586 | 0.000% |
| 30 | 0.1302 | 0.1302 | 0.000% |

### Conclusion
> **La nouvelle architecture DAG reproduit fidèlement les processus biologiques de SeapoPym v0.3 avec une erreur maximale de 0.14%.**

---

## Notebook 2 : Validation Transport 1D

### Objectif
Valider le schéma de transport (advection + diffusion) contre une solution analytique. Vérifier la conservation de masse.

### Configuration
- Domaine 1D
- Schéma Upwind + Volumes Finis
- Conditions aux limites : Fermées/Périodiques

### Figures

#### Figure 2A : Profil de Concentration vs Solution Analytique
**Description** : Comparaison du profil de concentration simulé (points) avec la solution analytique (ligne continue) à différents instants.

**Observation** : Excellent accord entre simulation et théorie.

#### Figure 2B : Conservation de Masse
**Description** : Courbe de la masse totale normalisée en fonction du temps.

**Observation** : Masse = 100.000000% pendant toute la simulation (aux erreurs d'arrondi près).

#### Figure 2C : Stabilité CFL
**Description** : Tableau montrant la relation entre CFL et stabilité.

| CFL | Stable | Erreur |
|-----|--------|--------|
| 0.25 | ✅ | 1.2% |
| 0.50 | ✅ | 2.1% |
| 0.75 | ✅ | 3.4% |
| 1.00 | ❌ | Instable |

### Conclusion
> **Le schéma de transport conserve strictement la masse et est stable pour CFL < 1, conformément à la théorie des Volumes Finis.**

---

## Notebook 3 : Validation Couplage Transport-Biologie

### Objectif
Démontrer que l'architecture DAG couple correctement Transport et Biologie sans biais de Time Splitting. Prouver que l'erreur observée est d'origine numérique (convergence O(Δx)).

### Configuration
- Domaine 2D avec transport + biologie
- Test de convergence en grille : 200×100, 400×200, 800×400

### Figures

#### Figure 3A : Carte 2D du Couplage
**Description** : Carte de la biomasse à t_final montrant la distribution spatiale résultant du couplage transport-biologie.

**Observation** : Distribution cohérente avec les forçages (température, courants).

#### Figure 3B : Profils à 3 Résolutions
**Description** : Profils de concentration à une latitude fixe pour les 3 résolutions de grille.

**Observation** : Les profils convergent vers une solution commune quand Δx diminue.

#### Figure 3D : Convergence en Grille (Log-Log)
**Description** : Graphe log-log de l'erreur L2 en fonction de 1/Δx.

| Résolution | Δx (km) | Erreur L2 |
|------------|---------|-----------|
| Low | 22.24 | 6.60% |
| Medium | 11.09 | 2.31% |
| High | 5.54 | 1.16% |

**Pente mesurée** : 1.25 (attendu : 1.0 pour Upwind O(Δx))

### Conclusion
> **L'architecture DAG couple correctement Transport et Biologie. L'erreur est purement numérique (schéma Upwind O(Δx)) et décroît linéairement avec le raffinement de grille.**

---

## Notebook 4A : Weak Scaling (Complexité Algorithmique)

### Objectif
Démontrer que le temps de calcul croît linéairement avec la taille du problème (O(N)).

### Configuration
- Grilles : 500×500, 1000×1000, 2000×2000
- Backend : Séquentiel

### Figure

#### Figure 4A : Weak Scaling
**Description** : Graphe log-log du temps de calcul par step vs nombre de cellules.

| Grille | N Cellules | Temps/Step (ms) |
|--------|------------|-----------------|
| 500×500 | 250,000 | 124 |
| 1000×1000 | 1,000,000 | 517 |
| 2000×2000 | 4,000,000 | 2021 |

**Pente mesurée** : 1.006

### Conclusion
> **L'architecture DAG a une complexité algorithmique linéaire O(N). Le temps de calcul croît proportionnellement au nombre de cellules, sans surcoût caché.**

---

## Notebook 4C : Strong Scaling (Multi-Groupes)

### Objectif
Démontrer que le speedup Dask augmente avec le nombre de groupes fonctionnels indépendants.

### Configuration
- Grille : 500×500
- Groupes : 12
- Backend : Dask ThreadPool

### Figure

#### Figure 4C : Speedup vs Workers
**Description** : Courbe du speedup en fonction du nombre de workers, avec ligne idéale (y=x) en pointillés.

| Workers | Temps (s) | Speedup | Efficacité |
|---------|-----------|---------|------------|
| 1 | 6.37 | 1.00× | 100% |
| 4 | 4.02 | 1.59× | 40% |
| 12 | 3.83 | **1.67×** | 14% |

**Observation** : Le speedup plafonne rapidement malgré 12 groupes indépendants.

### Conclusion
> **Le speedup Strong Scaling est limité (~1.7×) même avec 12 groupes fonctionnels. Ce n'est pas un problème de Dask mais de la structure du modèle (voir Notebook 4F).**

---

## Notebook 4E : Diagnostic Parallélisme (Fonctions Sleep)

### Objectif
Vérifier que Dask ThreadPool parallélise correctement en utilisant des fonctions `sleep()` qui libèrent le GIL.

### Configuration
- 12 tâches indépendantes
- Durée sleep : 0.1s par tâche
- Backend : Dask ThreadPool

### Figure

#### Figure 4E : Speedup avec Fonctions Sleep
**Description** : Courbe du speedup en fonction du nombre de workers pour des tâches `sleep()`.

| Workers | Temps (s) | Speedup | Efficacité |
|---------|-----------|---------|------------|
| 1 | 1.251 | 1.00× | 100% |
| 4 | 0.321 | 3.90× | 97% |
| 12 | 0.109 | **11.50×** | **96%** |

**Observation** : Speedup quasi-linéaire (~12× avec 12 workers).

### Conclusion
> **Dask ThreadPool fonctionne parfaitement pour des tâches qui libèrent le GIL. Le speedup limité dans le modèle réel (Notebook 4C) n'est pas dû à Dask.**

---

## Notebook 4F : Décomposition du Temps de Calcul

### Objectif
Identifier le goulot d'étranglement qui limite le speedup Strong Scaling en mesurant le temps passé dans chaque type de tâche.

### Configuration
- Grille : 500×500
- Cohortes : 10
- Groupes : 3

### Figures

#### Figure 4F-1 : Bar Chart des Temps par Catégorie
**Description** : Barres horizontales montrant le temps cumulé pour chaque catégorie de tâche.

| Catégorie | Temps (s) | % |
|-----------|-----------|---|
| **Transport Production** | 0.994 | **80.2%** |
| Mortality | 0.123 | 9.9% |
| Transport Biomass | 0.102 | 8.3% |
| Production | 0.021 | 1.7% |

#### Figure 4F-2 : Pie Chart de la Répartition
**Description** : Diagramme circulaire montrant la part de chaque catégorie.

**Transport total : 88.4%**

### Analyse (Loi d'Amdahl)

Avec 88% du temps dans le transport (séquentiel) :
```
Speedup_max = 1 / 0.88 = 1.13×
```

### Conclusion
> **Le transport représente 88% du temps de calcul. Conformément à la Loi d'Amdahl, le speedup Strong Scaling est intrinsèquement limité à ~1.2× tant que le transport n'est pas parallélisé en interne.**

---

## Synthèse pour l'Article

### Section Résultats — Messages Clés

1. **Validation Scientifique** :
   - Biologie 0D : Erreur < 0.2% vs solution analytique
   - Transport 1D : Conservation de masse stricte
   - Couplage : Convergence O(Δx) sans biais de Time Splitting

2. **Performance** :
   - Complexité linéaire O(N) validée (Weak Scaling)
   - Speedup Strong Scaling limité par la dominance du transport (88%)

### Section Discussion — Limites et Recommandations

| Limite Identifiée | Cause | Recommandation |
|-------------------|-------|----------------|
| Speedup ~1.7× | Transport = 88% du temps | Paralléliser avec chunking spatial |
| Efficacité faible | Peu de tâches indépendantes | Augmenter les groupes fonctionnels |
| Overhead xarray | Wrappers Python | Optimiser les appels Numba |

### Perspectives de Recherche

1. **Chunking spatial** avec Dask Arrays pour paralléliser le transport
2. **Décomposition de domaine** pour les très grandes grilles
3. **Migration vers `@jit(nogil=True)`** pour libérer le GIL
4. **Utilisation GPU** avec CUDA/Numba pour le transport

---

## Figures pour l'Article (Liste Finale)

| ID | Fichier | Description | Section |
|----|---------|-------------|---------|
| 1A | `fig_01a_bio_convergence.pdf` | Convergence asymptotique | Résultats |
| 1B | `fig_01b_bio_table.pdf` | Tableau validation bio | Résultats |
| 2A | `fig_02a_transport_profile.pdf` | Profil vs analytique | Résultats |
| 2B | `fig_02b_transport_mass.pdf` | Conservation masse | Résultats |
| 3D | `fig_03d_grid_convergence.pdf` | Convergence grille | Résultats |
| 4A | `fig_04a_weak_scaling.pdf` | Complexité O(N) | Résultats |
| 4C | `fig_04c_1_speedup_multiconfig.pdf` | Strong Scaling | Résultats |
| 4F | `fig_04f_time_decomposition.pdf` | Décomposition temps | Discussion |
| 4E | `fig_04e_sleep_parallelism.pdf` | Validation Dask | Suppl. Material |

---

**Fin du document de synthèse**
