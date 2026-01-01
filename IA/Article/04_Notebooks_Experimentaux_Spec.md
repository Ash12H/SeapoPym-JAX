# Spécifications des Notebooks Expérimentaux

Ce document décrit les notebooks à créer pour produire les résultats de l'article sur l'architecture DAG.

---

## Standards pour les Figures de Publication

Toutes les figures doivent respecter les conventions suivantes pour être directement intégrables dans un article scientifique (format GMD/Copernicus, Nature, etc.).

### Dimensions et Résolution

| Type de Figure | Largeur | Hauteur | DPI |
|----------------|---------|---------|-----|
| Colonne simple | 8.5 cm (3.35 in) | Variable | 300 |
| Double colonne | 17.5 cm (6.9 in) | Variable | 300 |
| Carré (Carte)  | 8.5 cm | 8.5 cm | 300 |

### Typographie

```python
import matplotlib.pyplot as plt

# Style recommandé pour articles
plt.rcParams.update({
    # Police
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,           # Texte général
    "axes.titlesize": 9,      # Titre des axes
    "axes.labelsize": 8,      # Labels des axes
    "xtick.labelsize": 7,     # Ticks X
    "ytick.labelsize": 7,     # Ticks Y
    "legend.fontsize": 7,     # Légende

    # Lignes et marqueurs
    "lines.linewidth": 1.0,
    "lines.markersize": 4,

    # Axes
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,

    # Sauvegarde
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})
```

### Palette de Couleurs

Utiliser une palette accessible (colorblind-safe). Recommandations :

```python
# Palette qualitative (lignes distinctes)
COLORS = {
    "blue": "#0077BB",
    "orange": "#EE7733",
    "green": "#009988",
    "red": "#CC3311",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
}

# Pour les simulations vs théorie
COLOR_SIM = "#0077BB"      # Bleu pour les données simulées
COLOR_THEORY = "#CC3311"   # Rouge pour la référence théorique
LINESTYLE_THEORY = "--"    # Pointillés pour la théorie
```

### Format de Fichier

*   **Format principal** : PDF vectoriel (pour LaTeX/Overleaf).
*   **Format alternatif** : PNG 300 DPI (pour Word/Google Docs).
*   **Nommage** : `fig_01a_bio_convergence.pdf`, `fig_02b_transport_mass.pdf`, etc.

### Légendes et Labels

*   **Unités** : Toujours spécifier les unités entre crochets. Ex: `Biomass [g/m²]`, `Time [days]`.
*   **Notation scientifique** : Utiliser `$10^{-6}$` (LaTeX) pour les ordres de grandeur.
*   **Légendes** : Concises. Ex: `Simulation`, `Analytic (Eq. 5)`.

### Code de Sauvegarde Standard

```python
def save_figure(fig, name, formats=["pdf", "png"]):
    """Sauvegarde une figure dans les formats requis."""
    output_dir = Path("../figures")
    output_dir.mkdir(exist_ok=True)

    for fmt in formats:
        filepath = output_dir / f"{name}.{fmt}"
        fig.savefig(filepath, format=fmt, dpi=300, bbox_inches="tight")
        print(f"✅ Saved: {filepath}")
```

---

## Notebook 1 : Validation Biologie 0D (Réplication v0.3)

**Fichier** : `article_01_bio_0d_asymptotic.ipynb`

**Objectif** : Reproduire l'expérience de convergence asymptotique de SeapoPym v0.3 avec la nouvelle architecture DAG. Prouver la non-régression.

### Paramètres (issus de votre article v0.3)

```python
PARAMS = {
    "E": 0.1668,           # Energy transfer coefficient
    "lambda_0": 1 / 150,   # Mortality rate at T_ref [s^-1] - ATTENTION: Vérifier unité (day^-1 dans v0.3 ?)
    "gamma_lambda": 0.15,  # Thermal sensitivity mortality [1/°C]
    "tau_r_0": 10.38,      # Recruitment age at T_ref [days] - ATTENTION: convertir en secondes
    "gamma_tau_r": 0.11,   # Thermal sensitivity recruitment [1/°C]
    "T_ref": 0.0,          # Reference temperature [°C]
}

FORCING = {
    "NPP": 300.0,          # mg/m²/day - ATTENTION: convertir en g/m²/s pour le modèle
    "T": [0, 10, 20, 30],  # Températures de test [°C]
}
```

### Protocole

1.  **Configuration Blueprint** : Blueprint minimal sans transport. Enregistrer :
    *   `compute_thresholded_temperature`
    *   `compute_gillooly_temperature`
    *   `compute_recruitment_age`
    *   `compute_production_initialization`
    *   `compute_production_dynamics`
    *   `compute_mortality_tendency`
    *   Variables d'état : `biomass`, `production`

2.  **Domaine** : Une seule cellule spatiale (0D). Pas de coordonnées Y, X. Juste `cohort` et `time`.

3.  **Intégration** : Simuler jusqu'à convergence (~1000 jours pour T=0°C, ~100 jours pour T=30°C).

4.  **Mesures** :
    *   Courbe `Biomass(t)` pour chaque température.
    *   Asymptote théorique : $B_{eq} = R / \lambda$ (à calculer manuellement ou via une fonction auxiliaire).
    *   Erreur relative entre simulation et asymptote.

### Figures attendues

*   **Figure 1A** : 4 courbes de convergence (une par température), superposées avec les asymptotes théoriques (lignes pointillées).
*   **Figure 1B** : Tableau récapitulatif (T, $B_{eq}$ théorique, $B_{eq}$ simulé, Erreur %).

---

## Notebook 2 : Validation Transport 1D (Analytique)

**Fichier** : `article_02_transport_1d_analytic.ipynb`

**Objectif** : Valider le schéma de transport (Advection + Diffusion) par comparaison avec la solution analytique exacte. Ce notebook est une version *article-ready* de `03_b_transport_validation_1d_analytic.ipynb`.

### Protocole

Le notebook existant est déjà complet. Adaptation nécessaire :

1.  **Ajouter la vérification de conservation de masse** : Tracer `Masse_totale(t)` au cours de la simulation.
2.  **Ajouter différentes valeurs de CFL** : Montrer que la précision diminue quand CFL augmente (table de sensibilité).
3.  **Nettoyer les outputs** : Produire des figures publication-quality avec Matplotlib (labels, fontes, legends).

### Figures attendues

*   **Figure 2A** : Profil initial vs Profil final (Simulation vs Analytique).
*   **Figure 2B** : Conservation de la masse au cours du temps ($\int B \, dA$ vs $t$).
*   **Figure 2C** : Erreur L2 en fonction du CFL (courbe de convergence).

---

## Notebook 3 : Validation Couplage Transport-Réaction (Patch Test)

**Fichier** : `article_03_coupling_patch_test.ipynb`

**Objectif** : Démontrer que le DAG unifie correctement Transport et Biologie sans biais de "Time Splitting".

### Configuration de Base

*   **Domaine** : 2D périodique (simplification).
*   **Température** : $T = 0°C$ partout (donc $\lambda = \lambda_0$ constant).
*   **Courant** : $u = 0.1 m/s$ (zonal), $v = 0$.
*   **Diffusion Physique** : $D = 1000 m^2/s$.
*   **Condition initiale** : Gaussienne de biomasse centrée.
*   **CFL cible** : 0.5 (constant pour tous les tests).

### Protocole Principal

1.  **Configuration Blueprint** : Inclure Transport ET Mortalité.
    *   Enregistrer `compute_transport_numba` (ou wrapper).
    *   Enregistrer `compute_mortality_tendency`.
    *   Variable d'état : `biomass`.

2.  **Simulation DAG** : Exécuter N pas de temps.

3.  **Solution Théorique** : Calculer $B_{theo}(x, y, t) = G(x - ut, y, t) \times e^{-\lambda_0 t}$
    *   Où $G$ est la gaussienne advectée-diffusée (solution analytique du transport seul).

4.  **Comparaison** : Superposer les profils 1D (coupe en Y central).

### Protocole Additionnel : Test de Convergence en Grille

**Objectif** : Prouver que l'erreur observée est d'origine numérique (schéma Upwind O(Δx)) et non architecturale.

**Théorie** : La diffusion numérique du schéma Upwind est :
$$D_{num} = \frac{u \Delta x}{2} (1 - \sigma) \propto \Delta x$$

En raffinant la grille tout en maintenant le CFL constant, $D_{num}$ diminue proportionnellement.

**Protocole** :

| Résolution | nx × ny | Δx estimé | Δt (CFL=0.5) | D_num attendu |
|------------|---------|-----------|--------------|---------------|
| Basse | 200 × 100 | 22 km | ~31h | ~550 m²/s |
| Moyenne | 400 × 200 | 11 km | ~15h | ~275 m²/s |
| Haute | 800 × 400 | 5.5 km | ~7.5h | ~138 m²/s |

1.  Exécuter la simulation pour chaque résolution.
2.  Calculer l'erreur L2 relative pour chaque cas.
3.  Tracer log(Erreur) vs log(Δx) : la pente doit être ~1 (convergence 1er ordre).

### Figures attendues

*   **Figure 3A** : Carte 2D de la biomasse simulée (à t_final, résolution moyenne).
*   **Figure 3B** : Coupe 1D : Simulation vs Théorie (3 résolutions superposées).
*   **Figure 3C** : Erreur relative en fonction du temps (3 courbes).
*   **Figure 3D** : Courbe de convergence log(Erreur) vs log(Δx) avec pente annotée.

### Critère de Succès

*   Erreur < 5% pour la résolution moyenne (400×200).
*   Pente de convergence entre 0.8 et 1.2 (confirmant le comportement O(Δx)).

---

## Notebook 4A : Weak Scaling (Complexité Algorithmique)

**Fichier** : `article_04a_weak_scaling.ipynb`

**Objectif** : Démontrer que le temps de calcul croît linéairement avec la taille du problème (O(N)).

### Question posée
*"Si je double la taille de ma grille, le temps de calcul double-t-il ?"*

### Configuration

```python
CONFIG_WEAK = {
    "grid_sizes": [(500, 500), (1000, 1000), (2000, 2000)],
    "n_cohorts": 50,
    "n_steps": 20,
    "backend": "sequential",  # Pas de parallélisation ici
}
```

### Protocole

1.  **Warmup** : 3 pas de temps pour compilation JIT.
2.  **Boucle sur les tailles de grille** : Mesurer temps/step pour chaque taille.
3.  **Régression log-log** : Calculer la pente (exposant de complexité).

### Figures attendues

*   **Figure 4A** : log(Temps/step) vs log(N cellules) avec pente annotée.
*   **Tableau** : Récapitulatif (Grille, N, Temps, Complexité).

### Critère de Succès

*   Pente ~1.0 (complexité O(N) linéaire).
*   Votre résultat actuel : **O(N^1.02)** ✅

---

## Notebook 4B : Benchmark de Scalabilité (Version B - Charge Lourde)

**Fichier** : `article_04b_strong_scaling.ipynb`

**Objectif** : Démontrer que Dask accélère le calcul en parallélisant les tâches indépendantes du DAG.

### Question posée
*"Si j'ajoute des cœurs, mon calcul va-t-il plus vite ?"*

### Configuration

```python
CONFIG_STRONG = {
    "grid_size": (1000, 1000),  # Grille FIXE (1 million de cellules)
    "n_cohorts": 50,
    "n_steps": 20,
    "workers": [1, 2, 4, 8, 12],
    "backend": "dask",
}
```

### Protocole

1.  **Warmup** : 3 pas de temps pour compilation JIT.
2.  **Baseline** : Exécuter avec backend séquentiel (référence).
3.  **Boucle sur le nombre de workers** :
    *   Configurer Dask avec N workers.
    *   Mesurer temps total et temps/step.
4.  **Calculer** :
    *   Speedup = Temps(baseline) / Temps(N workers)
    *   Efficacité = Speedup / N × 100%

### Figures attendues

*   **Figure 4B** : Speedup vs Nombre de workers (avec ligne "idéal" en pointillés).
*   **Tableau** : Récapitulatif (Workers, Temps, Speedup, Efficacité).

### Critère de Succès

| Métrique | Seuil Minimum | Objectif |
|----------|---------------|----------|
| Speedup (4 workers) | > 1.5× | > 2.0× |
| Efficacité (4 workers) | > 40% | > 50% |

### Note sur l'Efficacité Attendue

L'efficacité parallèle dépend de :
1.  **Nombre de tâches indépendantes** : Transport et Mortalité sont indépendants dans le DAG.
2.  **Ratio calcul/overhead** : Charge lourde (50 cohortes, 1M cellules) favorable.
3.  **Loi d'Amdahl** : La partie séquentielle (TimeIntegrator) limite le speedup max.

---

## Note sur l'Expérience 3 : Amélioration par Raffinement de Grille

### Pourquoi l'erreur est élevée (6-10%)

Le schéma **Upwind du premier ordre** introduit une diffusion numérique :

$$D_{numérique} \approx \frac{u \cdot \Delta x}{2}$$

Avec la configuration actuelle :
*   $\Delta x = 22$ km, $u = 0.1$ m/s → $D_{num} \approx 1100$ m²/s
*   Comparable à la diffusion physique ($D = 1000$ m²/s)
*   La gaussienne s'étale **2× plus vite** que prévu

### Solution : Doubler la Résolution

| Paramètre | Version Actuelle | Version Améliorée |
|-----------|------------------|-------------------|
| Grille | 200×100 | 400×200 |
| $\Delta x$ | 22 km | 11 km |
| $D_{num}$ | 1100 m²/s | 550 m²/s |
| Erreur attendue | ~6-10% | ~3-5% |

### Protocole d'Amélioration

1.  **Test de convergence en grille** : Exécuter avec 200×100, 400×200, 800×400
2.  **Tracer erreur vs 1/Δx** : Doit montrer une pente ~1 (convergence 1er ordre)
3.  **Choisir la résolution finale** : Celle qui donne erreur < 5%

---

## Résumé des Notebooks

| ID  | Fichier                                  | Objectif Principal                     | Métrique Clé |
|-----|------------------------------------------|----------------------------------------|--------------|
| 1   | `article_01_bio_0d_asymptotic.ipynb`     | Valider Bio (non-régression v0.3)      | Erreur < 1% |
| 2   | `article_02_transport_1d_analytic.ipynb` | Valider Transport vs Analytique        | Conservation 100% |
| 3   | `article_03_coupling_patch_test.ipynb`   | Valider Couplage + Convergence grille  | Pente ~1, Erreur < 5% |
| **4A** | **`article_04a_weak_scaling.ipynb`** | **Weak Scaling (Complexité O(N))**     | **Pente ~1** |
| **4B** | **`article_04b_strong_scaling.ipynb`** | **Strong Scaling (Speedup Dask)**      | **Speedup > 1.5×** |
