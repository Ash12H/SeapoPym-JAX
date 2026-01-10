# Plan d'Intégration du Parallélisme de Données (Data Parallelism)

Ce document propose une feuille de route pour intégrer la nouvelle approche de parallélisme de données (validée par le script `benchmark_full_lmtl.py`) dans le manuscrit et les expériences associées.

## 1. Contexte et Motivation

Les analyses de performances précédentes (Notebooks 04a/b/c) ont établi le diagnostic suivant :
1.  **Complexité $O(N)$ validée** : L'architecture est saine algorithmiquement.
2.  **Goulot d'étranglement identifié** : Le transport de la production représente ~80% du temps de calcul total.
3.  **Impasse du Task Parallelism** : L'approche naïve (paralléliser biologie vs transport) est mathématiquement limitée par la loi d'Amdahl à un speedup théorique maximal de $\approx 1.25\times$.

**Solution proposée** : Le **Data Parallelism** sur la dimension `cohort`.
Le transport de chaque cohorte de production étant indépendant, nous pouvons diviser le calcul selon cet axe (`chunks={'cohort': 1}`). Cela permet de paralléliser la tâche dominante elle-même, brisant ainsi la limite d'Amdahl.

### 1.1. Résultats Préliminaires (benchmark_full_lmtl.py)

Les premiers tests confirment la validité de l'approche :
- **Grille** : 1000×1000 (1M cellules), 51 cohortes, 5 jours, 4 threads
- **Sequential** : 15.61s (baseline)
- **Task Parallel** : 14.05s → **1.11× speedup** (limité par la loi d'Amdahl)
- **Data Parallel** : 6.46s → **2.42× speedup** (brise la limite d'Amdahl)

Ces résultats valident la solution proposée. La suite du plan vise à :
1. Étendre les mesures (sweep sur le nombre de workers)
2. Valider la correctness numérique
3. Intégrer formellement dans le manuscrit

## 2. Impact sur la Structure du Manuscrit

L'intégration de cette approche transforme la section "Performances" d'un constat d'échec partiel (limite de scaling) en une démonstration de solution.

### Section 2: Matériel et Méthodes
*   **Mise à jour de 2.3. Implémentation Logicielle** :
    *   Ajouter un paragraphe explicitant la stratégie hybride : Task Parallelism (pour les tâches hétérogènes indépendantes) et Data Parallelism (pour les tâches homogènes coûteuses comme le transport).
    *   Justifier le choix de la dimension `cohort` pour le chunking (indépendance physique).

### Section 3: Résultats - 4. Analyse des Performances
Cette section doit être restructurée pour suivre une logique "Problème -> Analyse -> Solution" :

*   **4.1. Complexité Algorithmique (Weak Scaling)** : *(Inchangé)* Le code est efficace en séquentiel.
*   **4.2. Analyse du Temps de Calcul** : *(Inchangé)* Le transport de la production domine (80%).
*   **4.3. Limites du Parallélisme de Tâches** : *(Modifié)*
    *   Montrer que le parallélisme de tâches seul ne suffit pas (résultats du notebook 04b/c).
    *   Illustrer la limite théorique d'Amdahl.
*   **4.4. Passage à l'Échelle via Parallélisme de Données (Strong Scaling)** : *(Nouveau)*
    *   Présenter les résultats du Data Parallelism.
    *   Démontrer un speedup quasi-linéaire sur un nœud multicœur.

### Section 4: Discussion
*   Remplacer le point "Limites Identifiées / Speedup limité" par une discussion sur l'efficacité du chunking.
*   Discuter brièvement des implications mémoire (le chunking spatial aurait moins d'overhead mémoire que le chunking par cohorte si le nombre de cohortes est grand, mais le chunking par cohorte est plus simple à implémenter sans halo exchange).

## 3. Figures Attendues

Pour soutenir ces conclusions, nous devons produire une **Figure 4 composée (C & D)** ou deux nouvelles figures :

1.  **Figure 4C : Comparaison des Stratégies (Bar Chart)**
    *   Comparaison du temps d'exécution pour une simulation fixe (ex: 500x500, 12 cohortes).
    *   Trois barres :
        1.  Séquentiel (Baseline).
        2.  Task Parallel (Max workers) -> Montre le gain minime.
        3.  Data Parallel (Max workers) -> Montre le gain significatif.

2.  **Figure 4D : Courbes de Strong Scaling**
    *   Axe X : Nombre de Workers (ex: 1, 2, 4, 6, 8, 12).
    *   Axe Y : Speedup ($T_{seq} / T_{par}$).
    *   Séries :
        *   Task Parallelism (sature vite).
        *   Data Parallelism (monte linéairement).
        *   Idéal (Linéaire).

## 4. Contenu du Notebook (`article_04d_strong_scaling_data_parallel.ipynb`)

Un nouveau notebook doit être créé pour générer ces résultats de manière reproductible. Il **complètera** les notebooks de performance actuels (04a/b/c), formant une suite logique.

### 4.0. Structure du Notebook (cohérente avec 04a/b/c)

Le notebook doit suivre la structure standard observée dans les notebooks existants:

```
1.  # Titre (markdown) - Objectif, Question, Théorie, Configuration
2.  Imports + Setup Paths (BASE_DIR, FIGURES_DIR, SUMMARY_DIR)
3.  ## Configuration Matplotlib pour Publications
4.  ## 1. Configuration du Benchmark (dict CONFIG + constantes)
5.  ## 2. Configuration du Blueprint LMTL Complet
6.  ## 3. Fonction de Benchmark Unifiée
7.  ## 4. Warmup : Compilation JIT
8.  ## 5. Expérience 1 : Baseline & Task Parallelism
9.  ## 6. Expérience 2 : Data Parallelism (Strong Scaling)
10. ## 7. Validation de Correctness
11. ## 8. Analyse et Visualisation
12. ## 9. Figure 4C : Comparaison des Backends
13. ## 10. Figure 4D : Courbes de Strong Scaling
14. ## 11. Tableau Récapitulatif
15. ## 12. Génération du Fichier Résumé
16. ## Conclusion (markdown)
```

**Conventions importantes :**
- Utiliser `print("✅ Message")` pour les confirmations
- Utiliser des séparateurs `"=" * 80` pour les tableaux
- Créer SUMMARY_DIR automatiquement : `SUMMARY_DIR.mkdir(exist_ok=True)`
- Fonction helper `save_figure(fig, name, formats=FIGURE_FORMATS)`
- Constantes en MAJUSCULES
- Config dans dict `CONFIG` ou `CONFIG_XXX`

### 4.1. Configuration du Modèle Complet

**Paramètres de Simulation :**
*   **Grille** : 500×500 (cohérent avec notebooks 04a/c)
*   **Cohortes** : 12 (divisible par 1, 2, 3, 4, 6, 12 → facilite l'analyse de scaling)
*   **Durée** : 20 pas de temps (cohérent avec 04a/c)
*   **Timestep** : CFL adaptatif (comme 04a)
*   **Forçages** : Synthétiques avec constantes standard (voir section Configuration)

**Constantes Physiques (cohérentes avec 04a/c) :**
```python
U_MAGNITUDE = 0.1        # Vitesse d'advection [m/s]
D_COEFF = 1000.0         # Coefficient de diffusion [m²/s] (04a standard)
TEMPERATURE_CONSTANT = 20.0  # Température [°C]
NPP_CONSTANT = 300.0     # Production primaire [mg/m²/day]
```

**Justification :**
- 12 cohortes (vs 51 dans le benchmark préliminaire) : réduit le temps de calcul tout en permettant d'observer le scaling
- 500×500 : cohérent avec 04c (time decomposition) pour comparaison directe
- 20 steps : même durée que 04a, suffisant pour mesures robustes
- Constantes physiques : **identiques à 04a** pour reproductibilité

### 4.2. Fonction de Benchmark Unifiée

```python
def run_benchmark(
    backend: str,
    num_workers: int = None,
    chunks: dict = None,
    num_runs: int = 3  # Répétitions pour robustesse statistique
) -> dict:
    """
    Lance une simulation et mesure le temps.

    Returns:
        {
            'backend': str,
            'workers': int,
            'setup_time': float,
            'run_time': float,
            'total_time': float,
            'speedup': float,
            'efficiency': float,
            'final_biomass_mean': float,  # Pour validation correctness
            'final_biomass_std': float
        }
    """
```

**Validation de Correctness :**
- À chaque run, sauvegarder `final_biomass_mean` et `final_biomass_std`
- Comparer avec le run séquentiel : l'erreur doit être < 1e-10 (précision numérique)
- Si erreur > seuil → ÉCHEC, le parallélisme introduit un biais

### 4.3. Expérience 1 : Baseline & Task Parallelism

**Objectif :** Reproduire la limite d'Amdahl

**Configuration :**
- `backend='sequential'` → baseline
- `backend='task_parallel'`, workers = [2, 4, 6, 8, 12]
- Pas de chunks (tâches biologiques vs transport)

**Métriques attendues :**
- Speedup plafonné à ~1.2× (limite d'Amdahl avec 80% séquentiel)
- Efficacité décroissante avec le nombre de workers

### 4.4. Expérience 2 : Data Parallelism (Strong Scaling)

**Objectif :** Démontrer le scaling au-delà de la limite d'Amdahl

**Configuration :**
- `backend='data_parallel'`
- `chunks = {'latitude': -1, 'longitude': -1, 'cohort': 1}`
- workers = [1, 2, 3, 4, 6, 12]

**Métriques attendues :**
- Speedup quasi-linéaire jusqu'à 12 workers (12 cohortes)
- Efficacité > 70% pour workers ≤ 12
- Au-delà de 12 workers : saturation (plus de cohortes que de workers)

### 4.5. Expérience 3 : Analyse de Sensibilité (Optionnel, si temps)

**Variables testées :**
1. **Nombre de cohortes** : [6, 12, 24, 51] avec 12 workers
   - Hypothèse : Plus de cohortes → meilleur load balancing
2. **Taille de grille** : [250×250, 500×500, 1000×1000] avec 12 workers
   - Hypothèse : Plus grosse grille → overhead Dask relativement plus faible
3. **Chunk size cohort** : [1, 2, 3] avec 12 workers et 12 cohortes
   - Hypothèse : chunk=1 optimal pour le load balancing

### 4.6. Analyse et Plotting

**Configuration Matplotlib (cohérente avec 04a/b/c) :**
```python
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "blue": "#0077BB",      # Data Parallel
    "orange": "#EE7733",    # Task Parallel
    "green": "#009988",
    "red": "#CC3311",       # Theory/Ideal
    "purple": "#AA3377",
    "grey": "#BBBBBB",      # Sequential baseline
}
```

**Figure 4C : Bar Chart Comparatif (speedup avec 12 workers)**
```python
fig, ax = plt.subplots(figsize=(6.9, 4))

backends = ['Sequential', 'Task Parallel\n(12 workers)', 'Data Parallel\n(12 workers)']
speedups = [1.0, speedup_task_12, speedup_data_12]
colors_bar = [COLORS["grey"], COLORS["orange"], COLORS["blue"]]

bars = ax.bar(backends, speedups, color=colors_bar, alpha=0.8)

# Limite d'Amdahl
ax.axhline(1.25, linestyle='--', color=COLORS["red"], linewidth=1.5,
           label='Amdahl Limit (80% sequential)')

ax.set_ylabel("Speedup")
ax.set_title("Strong Scaling: Backend Comparison")
ax.legend(loc="best")
ax.grid(True, axis='y', alpha=0.3)

# Annotations
for bar, speedup in zip(bars, speedups):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{speedup:.2f}×', ha='center', va='bottom', fontsize=8)

save_figure(fig, f"{FIGURE_PREFIX}_comparison")
```

**Figure 4D : Courbes de Strong Scaling**
```python
fig, ax = plt.subplots(figsize=(6.9, 4))

# Ligne idéale
ax.plot(workers, workers, '--', color=COLORS["red"], linewidth=1.5,
        label='Ideal (Linear)', alpha=0.7)

# Task Parallelism
ax.plot(workers_task, speedup_task, 'o-', color=COLORS["orange"],
        markersize=6, label='Task Parallelism')

# Data Parallelism
ax.plot(workers_data, speedup_data, 's-', color=COLORS["blue"],
        markersize=6, label='Data Parallelism')

ax.set_xlabel("Number of Workers")
ax.set_ylabel("Speedup")
ax.set_title("Strong Scaling: Data Parallelism vs Task Parallelism")
ax.legend(loc="best")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max(workers) + 1)
ax.set_ylim(0, max(workers) + 1)

save_figure(fig, f"{FIGURE_PREFIX}_scaling")
```

**Calcul des métriques :**
```python
metrics = {
    'speedup': T_seq / T_par,
    'efficiency': (T_seq / T_par) / num_workers * 100,  # en %
    'overhead': T_setup_par - T_setup_seq,
    'correctness_rmse': rmse(state_par, state_seq),
    'time_per_step': T_run / num_timesteps
}
```

**Tableau de Métriques :**
| Backend        | Workers | Time (s) | Speedup | Efficacité | Correctness (vs Seq) |
|----------------|---------|----------|---------|------------|----------------------|
| Sequential     | 1       | X.XX     | 1.00    | 100%       | Baseline             |
| Task Parallel  | 4       | X.XX     | X.XX    | XX%        | ✓ (< 1e-10)          |
| Data Parallel  | 4       | X.XX     | X.XX    | XX%        | ✓ (< 1e-10)          |
| Data Parallel  | 12      | X.XX     | X.XX    | XX%        | ✓ (< 1e-10)          |

### 4.7. Validation de la Correctness Numérique

**Protocole :**
1. Exécuter simulation séquentielle → sauvegarder état final `S_ref`
2. Pour chaque configuration parallèle → sauvegarder état final `S_par`
3. Calculer erreur : `RMSE = sqrt(mean((S_par - S_ref)^2))`
4. Critère de validation : **RMSE < 1e-10 g/m²** (précision machine)

**Justification :**
- Le parallélisme de données ne doit **pas** modifier les résultats numériques
- Dask garantit que les opérations sur les chunks sont identiques aux opérations globales
- Toute différence > 1e-10 indique un bug dans l'implémentation

**Section dans le Notebook :**
```python
# Section: Correctness Validation
correctness_check = pd.DataFrame({
    'Backend': [...],
    'Workers': [...],
    'RMSE_vs_Sequential': [...],
    'Max_Abs_Error': [...],
    'Status': ['✓ PASS' if rmse < 1e-10 else '✗ FAIL']
})
```

## 5. Paramètres Techniques Détaillés

### 5.1. Configuration Dask

**ThreadPoolScheduler** (recommandé pour ce benchmark) :
```python
import dask
dask.config.set(scheduler='threads', num_workers=N)
```

**Avantages :**
- Mémoire partagée (pas de sérialisation)
- Overhead minimal pour le Task Graph
- Adapté aux workstations (1 machine)

**Alternative Distributed** (si on veut tester le multi-nœuds) :
```python
from dask.distributed import Client
client = Client(n_workers=N, threads_per_worker=1)
```

### 5.2. Chunking Strategy

**Configuration retenue :**
```python
chunks = {
    'latitude': -1,      # Pas de chunking spatial (évite halo exchange)
    'longitude': -1,     # Pas de chunking spatial
    'cohort': 1          # 1 cohorte par chunk → max parallélisme
}
```

**Justification :**
- Le transport nécessite l'intégrité spatiale (advection/diffusion = opérations de voisinage)
- Chunker `latitude` ou `longitude` nécessiterait un "halo exchange" complexe
- Chunker `cohort` est trivial car les cohortes sont **indépendantes physiquement**

**Overhead Mémoire :**
- Avec 12 cohortes, 12 workers, chaque worker traite 1 cohorte à la fois
- Mémoire par worker : `(500×500×1) * 8 bytes ≈ 2 MB` (négligeable)

### 5.3. Limitations et Considérations

**Limite du Scaling :**
- **Optimal** : `num_workers ≤ num_cohorts`
- **Sur-allocation** : Si `num_workers > num_cohorts` → certains workers inactifs
- **Sous-allocation** : Si `num_workers < num_cohorts` → exécution en "vagues"

**Cas d'usage réalistes :**
- Modèle standard LMTL : 50-100 cohortes → scaling potentiel jusqu'à 50-100 cœurs
- Workstation typique : 8-16 cœurs → efficacité excellente
- Cluster : 100+ cœurs → nécessite plus de cohortes ou chunking spatial additionnel

## 6. Roadmap d'Implémentation

### Phase 1 : Préparation (1 jour)
- [ ] Créer `article_04d_strong_scaling_data_parallel.ipynb`
- [ ] Adapter le code de `benchmark_full_lmtl.py` en fonctions réutilisables
- [ ] Implémenter la fonction `run_benchmark()` avec validation de correctness
- [ ] Tester sur un cas simple (2 workers) pour valider la pipeline

### Phase 2 : Expériences (2-3 heures de calcul)
- [ ] Expérience 1 : Sequential + Task Parallelism (workers: 2, 4, 6, 8, 12)
- [ ] Expérience 2 : Data Parallelism (workers: 1, 2, 3, 4, 6, 12)
- [ ] Validation de correctness pour chaque configuration
- [ ] Sauvegarde des résultats en CSV/JSON

### Phase 3 : Analyse et Figures (1 jour)
- [ ] Génération Figure 4C (Bar Chart)
- [ ] Génération Figure 4D (Strong Scaling Curves)
- [ ] Calcul des métriques (Speedup, Efficacité, Overhead)
- [ ] Tableau récapitulatif avec validation de correctness

### Phase 4 : Intégration Manuscrit (1 jour)
- [ ] Mettre à jour Section 2.3 (Implémentation Logicielle) avec paragraphe Data Parallelism
- [ ] Ajouter Section 3.4.4 (Passage à l'Échelle via Parallélisme de Données)
- [ ] Mettre à jour Section 4 (Discussion) avec considérations sur le chunking
- [ ] Intégrer les Figures 4C et 4D dans le manuscrit

### Phase 5 : Expériences Optionnelles (si temps)
- [ ] Analyse de sensibilité (nombre de cohortes, taille de grille)
- [ ] Comparaison ThreadPool vs Distributed scheduler
- [ ] Benchmark avec grille réaliste 1000×1000 pour estimation production

## 7. Métriques et Critères de Validation

### 7.1. Critères de Succès

**Performance :**
- ✓ Data Parallelism speedup > 2× avec 4 workers (vs Task Parallelism ~1.1×)
- ✓ Data Parallelism speedup > 8× avec 12 workers (efficacité > 66%)
- ✓ Task Parallelism plafonné < 1.3× (confirmation limite Amdahl)

**Correctness :**
- ✓ RMSE vs Sequential < 1e-10 g/m² pour toutes les configurations
- ✓ Conservation de masse parfaite (< 1e-14%)
- ✓ Aucune instabilité numérique introduite par le chunking

**Reproductibilité :**
- ✓ Coefficient de variation < 5% sur 3 runs identiques
- ✓ Résultats identiques entre `ThreadPoolScheduler` et `sequential` (à précision machine près)

### 7.2. Métriques Calculées

Pour chaque configuration :
```python
metrics = {
    'speedup': T_seq / T_par,
    'efficiency': (T_seq / T_par) / num_workers * 100,  # en %
    'overhead': T_setup_par - T_setup_seq,  # overhead Dask
    'correctness_rmse': rmse(state_par, state_seq),
    'time_per_step': T_run / num_timesteps
}
```

## 8. Intégration avec les Résultats Existants

### 8.1. Lien avec Section 4.3 (Validation Système Complet)

La Section 4.3 actuelle du manuscrit valide le système avec **12 groupes fonctionnels indépendants** (test `sleep`) :
- Speedup : 10.34× avec 12 workers
- Efficacité : 86%

**Articulation logique :**
1. **Section 4.3** : Le système *peut* paralléliser efficacement (validé avec test synthétique)
2. **Section 4.4 (nouvelle)** : Application au modèle réel LMTL → Data Parallelism brise la limite d'Amdahl

**Message clé :**
> "L'infrastructure de parallélisation est validée (Section 4.3). La limite de speedup observée précédemment (~1.25×) n'est pas un défaut du système, mais résulte de la structure du modèle. Le Data Parallelism sur la dimension `cohort` permet de paralléliser la tâche dominante (transport de production), dépassant ainsi la limite théorique."

### 8.2. Cohérence avec les Autres Expériences

| Notebook | Focus                  | Backend          | Grille    | Résultat clé                   |
|----------|------------------------|------------------|-----------|--------------------------------|
| 04a      | Weak Scaling           | Sequential       | Variable  | Complexité O(N)                |
| 04b      | Task Parallelism       | task_parallel    | 500×500   | Speedup ~1.2× (limite Amdahl)  |
| 04c      | Time Decomposition     | Sequential       | 500×500   | Transport production = 80%     |
| **04d**  | **Data Parallelism**   | **data_parallel**| **500×500**| **Speedup > 2× (brise limite)**|

**Grille cohérente (500×500)** pour 04b, 04c, 04d → permet comparaison directe.

## 9. Considérations pour la Publication

### 9.1. Narratif Scientifique

**Arc narratif :**
1. **Problème** (Sections 4.1-4.2) : Le modèle est O(N) mais dominé par une tâche (80%)
2. **Diagnostic** (Section 4.3) : Task Parallelism limité par Amdahl (~1.25×)
3. **Solution** (Section 4.4) : Data Parallelism → paralléliser la tâche dominante elle-même
4. **Validation** : Speedup 2-8× démontré, correctness préservée

**Message fort :**
> "L'architecture DAG n'est pas seulement flexible, elle est également **évolutive** : en identifiant le bon axe de parallélisation (cohort), nous dépassons les limites théoriques du Task Parallelism."

### 9.2. Limitations à Discuter (Section Discussion)

**Transparence scientifique :**
1. **Chunking spatial non implémenté** :
   - Pour des grilles > mémoire, il faudrait chunker `latitude`/`longitude`
   - Nécessiterait un halo exchange pour le transport
   - Complexité d'implémentation significative

2. **Overhead mémoire du chunking** :
   - Avec 100 cohortes, 100 workers → overhead mémoire négligeable
   - Mais : création de 100 tasks Dask → overhead de scheduling

3. **Limitation par le nombre de cohortes** :
   - Scaling limité à `num_workers ≤ num_cohorts`
   - Pour modèle standard (50 cohortes) : max 50 cœurs utiles
   - Au-delà : nécessite chunking spatial ou autres dimensions

4. **ThreadPoolScheduler vs Distributed** :
   - ThreadPool : mémoire partagée, optimal pour 1 machine
   - Distributed : nécessaire pour cluster, mais overhead de sérialisation
   - Résultats présentés : ThreadPool (cas d'usage typique workstation)

### 9.3. Perspectives

**Évolutions futures :**
- Chunking hybride : `cohort` + spatial pour scaling > 100 cœurs
- GPU acceleration via CuPy/Dask-CUDA
- Chunking sur dimension temporelle pour simulations très longues
- Application à d'autres modèles avec structures similaires (dimension âge, taille, espèce)

## 10. Checklist Finale

Avant de considérer l'intégration complète :

**Expériences :**
- [ ] Notebook 04d créé et exécuté
- [ ] Figures 4C et 4D générées (qualité publication)
- [ ] Validation de correctness : RMSE < 1e-10 pour toutes les configs
- [ ] Résultats sauvegardés (CSV + figures)

**Manuscrit :**
- [ ] Section 2.3 mise à jour (paragraphe Data Parallelism)
- [ ] Section 3.4.4 ajoutée avec résultats
- [ ] Section 4 (Discussion) mise à jour
- [ ] Figures 4C et 4D intégrées avec légendes
- [ ] Références croisées cohérentes

**Validation Scientifique :**
- [ ] Revue par co-auteurs
- [ ] Vérification cohérence métriques (speedup, efficacité)
- [ ] Relecture narrative "Problème → Solution"
- [ ] Vérification orthographe/grammaire
