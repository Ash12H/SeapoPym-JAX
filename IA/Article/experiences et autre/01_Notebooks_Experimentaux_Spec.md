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

## Notebook 4C : Strong Scaling avec Charge Équilibrée (Multi-Groupes)

**Fichier** : `article_04c_strong_scaling_multigroup.ipynb`

**Objectif** : Démontrer que le speedup Dask augmente significativement lorsque la charge est répartie sur plusieurs groupes fonctionnels indépendants.

### Hypothèse Testée

Dans le Notebook 4B, **90% du temps** est concentré dans `transport_production` (50 cohortes). Avec un seul groupe dominant, le parallélisme est limité par la Loi d'Amdahl.

**Hypothèse** : En créant 7 groupes fonctionnels indépendants (avec 10 cohortes chacun), on obtient 7 tâches de transport parallélisables → speedup ~7× théorique.

### Configuration Multi-Groupes

```python
CONFIG_MULTIGROUP = {
    "grid_size": (500, 500),  # Grille moyenne pour temps raisonnable
    "n_cohorts": 10,          # 10 cohortes par groupe (vs 50 dans 4B)
    "n_groups": 7,            # 7 groupes fonctionnels indépendants
    "n_steps": 20,
    "workers": [1, 2, 4, 7, 12],
    "backend": "dask-threads",
}

# Paramètres par groupe (seul E varie)
GROUPS = {
    "group_1": {"E": 0.10},
    "group_2": {"E": 0.12},
    "group_3": {"E": 0.14},
    "group_4": {"E": 0.16},
    "group_5": {"E": 0.18},
    "group_6": {"E": 0.20},
    "group_7": {"E": 0.22},
}
# Tous les autres paramètres (lambda, tau_r, gamma, etc.) sont identiques
```

### Architecture DAG Attendue

```
                    ┌──▶ [transport_prod_g1] ──┐
                    ├──▶ [transport_prod_g2] ──┤
[forcings] ─────────┼──▶ [transport_prod_g3] ──┼──▶ [time_integrator]
(T, u, v, D, NPP)   ├──▶ [transport_prod_g4] ──┤
                    ├──▶ [transport_prod_g5] ──┤
                    ├──▶ [transport_prod_g6] ──┤
                    └──▶ [transport_prod_g7] ──┘

                    └── 7 tâches INDÉPENDANTES ──┘
```

### Protocole

1.  **Configuration Blueprint Multi-Groupes** :
    *   Créer 7 groupes fonctionnels avec `bp.register_group()`.
    *   Chaque groupe a ses propres : `production_gX`, `biomass_gX`, `transport_prod_gX`, `mortality_gX`.
    *   Seul le paramètre `E` diffère entre les groupes.

2.  **Warmup** : 3 pas de temps pour compilation JIT.

3.  **Baseline** : Exécuter avec backend séquentiel.

4.  **Boucle sur le nombre de workers** :
    *   Config Dask ThreadPool avec 1, 2, 4, 7, 12 workers.
    *   Mesurer temps total et calculer speedup.

5.  **Comparer avec 4B** : Montrer l'amélioration du speedup grâce à l'équilibrage.

### Figures attendues

*   **Figure 4C-1** : Speedup vs Workers (comparaison 4B vs 4C).
*   **Figure 4C-2** : Efficacité vs Workers.
*   **Tableau** : Récapitulatif des résultats.

### Métriques de Succès

| Métrique | 4B (1 groupe, 50 coh.) | 4C (7 groupes, 10 coh.) | Amélioration |
|----------|------------------------|-------------------------|--------------|
| Speedup (4 workers) | ~1.2× | **> 2.5×** | **+2×** |
| Speedup (7 workers) | ~1.2× | **> 4.0×** | **+3×** |
| Efficacité (7 workers) | ~17% | **> 60%** | **+3.5×** |

### Ce que cette expérience prouve

1.  **L'architecture DAG exploite le parallélisme** quand la charge est équilibrée.
2.  **Le speedup est proportionnel** au nombre de tâches indépendantes.
3.  **Le goulot d'étranglement du 4B** n'est pas Dask, mais le modèle (1 tâche dominante).
4.  **Pour les modèles multi-espèces/multi-groupes** (cas réel LMTL), des speedups significatifs sont attendus.

---

## Notebook 4D : Diagnostic du Parallélisme (Fonctions Synthétiques)

**Fichier** : `article_04d_parallelism_diagnostic.ipynb`

**Objectif** : Identifier précisément la cause du speedup limité (~1.7×) en testant avec des fonctions contrôlées.

### Contexte : Découverte sur le GIL

Les kernels Numba dans `seapopym/transport/numba_kernels.py` utilisent :
```python
@guvectorize([...], "...", nopython=True, cache=True)
```

**Problème identifié** : `guvectorize` avec `target='cpu'` (défaut) **ne libère pas le GIL**.
Les threads Dask se bloquent mutuellement pendant l'exécution des kernels.

### Hypothèses à Tester

| # | Hypothèse | Si Vraie → Speedup | Test |
|---|-----------|-------------------|------|
| H1 | Dask ThreadPool fonctionne | ~N× avec sleep | Exp 1 |
| H2 | GIL bloque les threads Python | ~1× avec Python pur | Exp 2 |
| H3 | @jit(nogil=True) libère le GIL | ~N× avec Numba jit | Exp 3 |
| H4 | @guvectorize ne libère pas le GIL | ~1× avec guvectorize | Exp 4 |

---

### Expérience 1 : Fonctions `time.sleep()` (Baseline Dask)

**But** : Vérifier que Dask ThreadPool parallélise correctement.

```python
import time
import dask

# 12 fonctions indépendantes, chacune dort 100ms
def sleep_task(data, task_id):
    time.sleep(0.1)  # 100ms - libère le GIL
    return {"output": data}

# Créer 12 tâches delayed
tasks = [dask.delayed(sleep_task)(dummy_data, i) for i in range(12)]

# Baseline (séquentiel)
with dask.config.set(scheduler='synchronous'):
    t_seq = time.time()
    dask.compute(*tasks)
    t_seq = time.time() - t_seq  # Attendu: ~1.2s

# Parallèle (4 threads)
with dask.config.set(scheduler='threads', num_workers=4):
    t_par = time.time()
    dask.compute(*tasks)
    t_par = time.time() - t_par  # Attendu: ~0.3s

speedup = t_seq / t_par  # Attendu: ~4×
```

**Résultat attendu** : Speedup ~4× avec 4 workers. Si oui → Dask fonctionne ✅

---

### Expérience 2 : Fonctions Python CPU-bound (Test du GIL)

**But** : Confirmer que le GIL bloque les threads Python.

```python
def cpu_bound_python(data, n_iters=5_000_000):
    """Boucle Python pure - bloqué par GIL."""
    total = 0.0
    for i in range(n_iters):
        total += i * 0.0001
    return {"output": data}

# Même protocole que Exp 1
```

**Résultat attendu** : Speedup ~1× (pas de parallélisme réel).

---

### Expérience 3 : Fonctions Numba `@jit(nogil=True)` (Test du GIL Release)

**But** : Vérifier que `nogil=True` permet le parallélisme.

```python
import numba
import numpy as np

@numba.jit(nopython=True, nogil=True)  # ← clé !
def numba_cpu_bound(arr, n_iters):
    """Calcul Numba avec GIL relâché."""
    total = 0.0
    for _ in range(n_iters):
        for i in range(len(arr)):
            total += arr[i] * 0.0001
    return total

def numba_wrapper(data, n_iters=100):
    arr = data.values.ravel()
    _ = numba_cpu_bound(arr, n_iters)
    return {"output": data}

# Même protocole que Exp 1
```

**Résultat attendu** : Speedup ~N× avec N workers. Si oui → solution = `nogil=True` ✅

---

### Expérience 4 : Fonctions `@guvectorize` (Simulation du Transport)

**But** : Confirmer que `guvectorize(target='cpu')` ne libère pas le GIL.

```python
from numba import guvectorize, float64

@guvectorize(
    [(float64[:], float64[:])],
    "(n)->(n)",
    nopython=True,
    target='cpu',  # ← Le problème potentiel
)
def guvec_compute(arr_in, arr_out):
    """Simule le kernel de transport."""
    n = len(arr_in)
    for i in range(n):
        arr_out[i] = arr_in[i] * 2.0

def guvec_wrapper(data):
    arr = data.values.copy()
    result = np.empty_like(arr)
    guvec_compute(arr, result)
    return {"output": xr.DataArray(result, dims=data.dims)}

# Même protocole que Exp 1
```

**Résultat attendu** : Speedup ~1× (confirmant que guvectorize ne libère pas le GIL).

---

### Expérience 5 : `@guvectorize(target='parallel')` (Solution potentielle)

**But** : Tester si `target='parallel'` améliore le speedup.

```python
@guvectorize(
    [(float64[:], float64[:])],
    "(n)->(n)",
    nopython=True,
    target='parallel',  # ← Test de la solution
)
def guvec_parallel(arr_in, arr_out):
    n = len(arr_in)
    for i in range(n):
        arr_out[i] = arr_in[i] * 2.0
```

**Note** : `target='parallel'` utilise les threads internes de Numba. Cela peut créer un conflit avec les threads Dask (over-subscription).

---

### Protocole Complet

1.  **Setup** : Grille 500×500, données synthétiques, 12 tâches indépendantes.
2.  **Warmup** : 1 exécution pour compilation JIT.
3.  **Pour chaque expérience** :
    *   Mesurer temps séquentiel (scheduler='synchronous')
    *   Mesurer temps parallèle (scheduler='threads', 4 workers)
    *   Calculer speedup et efficacité
4.  **Comparer** les résultats.

### Tableau des Résultats Attendus

| Exp | Type de Fonction | Libère GIL ? | Speedup Attendu |
|-----|------------------|--------------|-----------------|
| 1 | `time.sleep()` | ✅ Oui | **~4×** |
| 2 | Python CPU-bound | ❌ Non | **~1×** |
| 3 | `@jit(nogil=True)` | ✅ Oui | **~4×** |
| 4 | `@guvectorize(target='cpu')` | ❌ Non | **~1×** |
| 5 | `@guvectorize(target='parallel')` | ⚠️ Interne | **~1-2×** (conflict) |

### Conclusions et Actions

| Résultat | Conclusion | Action |
|----------|------------|--------|
| Exp 1 = 4×, Exp 4 = 1× | `guvectorize` est le goulot | Refactorer les kernels avec `@jit(nogil=True)` |
| Exp 1 = 1× | Problème Dask | Investiguer la config Dask |
| Exp 3 = 4×, Exp 4 = 1× | Confirme que `nogil` est la clé | Migrer de `guvectorize` vers `jit` |

### Impact pour l'Article

Si l'expérience confirme que `guvectorize` bloque le GIL :

1.  **Documenter** la limitation dans la section Discussion.
2.  **Mentionner** que le refactoring vers `@jit(nogil=True)` est une optimisation future.
3.  **Souligner** que le Weak Scaling O(N) reste valide (indépendant du GIL).
4.  **Affirmer** que pour des tâches I/O-bound ou hybrides, le speedup serait meilleur.

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
| **4B** | **`article_04b_strong_scaling.ipynb`** | **Strong Scaling (1 groupe)**          | **Baseline parallélisme** |
| **4C** | **`article_04c_strong_scaling_multigroup.ipynb`** | **Strong Scaling (7 groupes équilibrés)** | **Speedup > 4×** |
