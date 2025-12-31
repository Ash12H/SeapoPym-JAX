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

### Configuration

*   **Domaine** : 2D périodique (simplification).
*   **Température** : $T = 0°C$ partout (donc $\lambda = \lambda_0$ constant).
*   **Courant** : $u = 0.1 m/s$ (zonal), $v = 0$.
*   **Diffusion** : $D = 1000 m^2/s$.
*   **Condition initiale** : Gaussienne de biomasse centrée.

### Protocole

1.  **Configuration Blueprint** : Inclure Transport ET Mortalité.
    *   Enregistrer `compute_transport_numba` (ou wrapper).
    *   Enregistrer `compute_mortality_tendency`.
    *   Variable d'état : `biomass`.

2.  **Simulation DAG** : Exécuter N pas de temps.

3.  **Solution Théorique** : Calculer $B_{theo}(x, y, t) = G(x - ut, y, t) \times e^{-\lambda_0 t}$
    *   Où $G$ est la gaussienne advectée-diffusée (solution analytique du transport seul).

4.  **Comparaison** : Superposer les profils 1D (coupe en Y central).

### Figures attendues

*   **Figure 3A** : Carte 2D de la biomasse simulée (à t_final).
*   **Figure 3B** : Coupe 1D : Simulation vs Théorie.
*   **Figure 3C** : Erreur relative en fonction du temps.

---

## Notebook 4 : Benchmark de Scalabilité

**Fichier** : `article_04_performance_scaling.ipynb`

**Objectif** : Quantifier les gains de performance offerts par le parallélisme de tâches du DAG.

### Configuration

*   **Données Synthétiques** (générées dans le notebook) :
    *   Grilles de différentes tailles : 100x100, 250x250, 500x500, 1000x1000.
    *   Forçages constants : T=15°C, NPP=300 mg/m²/day, u=0.1 m/s, D=1000 m²/s.

*   **Modèle** : Blueprint LMTL complet (Production + Mortalité + Transport) avec 10 cohortes.

### Protocole

1.  **Boucle sur les tailles de grille** : Pour chaque taille, créer le Blueprint, les forçages, et exécuter N pas de temps.

2.  **Boucle sur le nombre de threads** : Exécuter avec 1, 2, 4, 8, 12 threads (via `dask.config.set(num_workers=n)`).

3.  **Mesurer** :
    *   Temps total de simulation.
    *   Temps moyen par pas de temps.

### Figures attendues

*   **Figure 4A** : Courbe de "Speedup" (Temps_1_thread / Temps_N_threads) vs N.
*   **Figure 4B** : Temps par pas de temps vs Taille de grille (pour montrer la scalabilité "Weak").
*   **Tableau** : Récapitulatif des temps de calcul.

---

## Résumé des Notebooks

| ID  | Fichier                                  | Objectif Principal                     | Dépendance Code                |
|-----|------------------------------------------|----------------------------------------|--------------------------------|
| 1   | `article_01_bio_0d_asymptotic.ipynb`     | Valider la Bio (non-régression v0.3)   | `seapopym.lmtl`, `Blueprint`   |
| 2   | `article_02_transport_1d_analytic.ipynb` | Valider le Transport vs Analytique     | `seapopym.transport`           |
| 3   | `article_03_coupling_patch_test.ipynb`   | Valider le Couplage Bio+Transport      | `seapopym.lmtl`, `transport`   |
| 4   | `article_04_performance_scaling.ipynb`   | Quantifier la Scalabilité Dask         | `DaskBackend`, Full Model      |
