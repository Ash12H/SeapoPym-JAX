# Plan de Comparaison SeapoPym DAG vs Seapodym-LMTL (Pacifique)

## Objectif

Valider l'implémentation du transport couplé (Advection-Diffusion-Réaction) dans l'architecture DAG en comparant ses résultats avec le modèle de référence Seapodym-LMTL sur le domaine Pacifique.

Cette expérience correspond à la section **3.2** des Résultats de l'article :

> "Comparaison avec Seapodym-LMTL (Avec Transport)"

---

## 1. Flux de Travail

### Étape 1 : Préparation des Données (Fait)

**Notebook** : `data/article/article_05_prepare_data_pacific.ipynb`

-   Entrée : Zarr global Seapodym (`post_processed_light...`)
-   Traitement :
    -   Sélection Pacifique (Lon 110°E-290°E, Lat ±60°)
    -   Sélection Période (2000-2011)
    -   Correction Z=1
-   Sortie : `data/article/data/seapodym_lmtl_forcings_pacific.zarr`

### Étape 2 : Simulation (À faire)

**Notebook** : `data/notebook/article_03_simulation_pacific.ipynb`
**Objectif** : Produire les résultats du modèle DAG dans deux configurations.

#### Configuration A : Avec Transport (Cible)

-   **Modèle** : SeapoPym DAG v1.0
-   **Physique** : Advection + Diffusion ($D=500 m^2/s$)
-   **Frontières** : OPEN (ou masquées)
-   **Pas de temps** : Calculé pour CFL ≈ 0.5 (max 1 jour)
-   **Sortie** : `seapopym_pacific_transport.zarr`

#### Configuration B : Sans Transport (Contrôle)

-   **Modèle** : SeapoPym DAG v1.0
-   **Physique** : Biologie seule (0D distribué)
-   **Sortie** : `seapopym_pacific_no_transport.zarr`

This permet d'isoler l'impact du transport et de vérifier que les divergences avec Seapodym viennent bien du transport (ou non).

### Étape 3 : Analyse et Comparaison (À faire ultérieurement)

**Notebook** : `data/notebook/article_04_analysis_pacific.ipynb`

-   Comparaison DAG (Transport) vs Seapodym (Ref) -> Validation principale.
-   Comparaison DAG (Transport) vs DAG (NoTransport) -> Impact du transport.

---

## 2. Détails Techniques de la Simulation (`article_03`)

### Calcul du Pas de Temps (CFL)

Le notebook doit calculer dynamiquement le $\Delta t$ :
$$ \Delta t*{cfl} = \frac{0.5 \times \Delta x*{min}}{V*{max}} $$
Si $\Delta t*{cfl} < 1 \text{ jour}$, le contrôleur utilisera ce pas de temps pour la boucle de simulation, tout en interpolant les forçages journaliers.

### Configuration du Blueprint (Avec Transport)

Le groupe fonctionnel `Zooplankton` doit inclure les unités de transport :

```python
# Unités supplémentaires par rapport au 0D
{
    "func": compute_advection_tendency, # ou version numba
    "input_mapping": {"biomass": "biomass", "u": "current_u", "v": "current_v", ...},
    ...
},
{
    "func": compute_diffusion_tendency,
    "input_mapping": {"biomass": "biomass", "D": "diffusivity", ...},
    ...
}
```

**Attention** : Le transport s'applique à la fois à la **biomasse** (state) et à la **production** (si modélisée comme state transporté). Dans la version actuelle LMTL simplifiée, seule la biomasse totale ou par cohorte est transportée. Il faudra vérifier si `production` doit aussi être transportée ou si elle est consommée localement. Dans Seapodym-LMTL standard, les cohortes de fourrage sont transportées.

### Paramétrisation

-   **Diffusivité** : Constante $500 m^2/s$.
-   **Paramètres Bio** : Identiques à l'expérience 0D.

---

## 3. Données de Sortie

Les résultats seront sauvegardés dans `data/article/data/` sous format Zarr, avec les dimensions : `(time, latitude, longitude)`.

---

## 4. Checklist Simulation (`article_03`)

-   [ ] Charger `seapodym_lmtl_forcings_pacific.zarr`.
-   [ ] Calculer CFL et $\Delta t$.
-   [ ] Configurer Blueprint **AVEC Transport**.
-   [ ] Exécuter Run A (Transport).
-   [ ] Sauvegarder Run A.
-   [ ] Configurer Blueprint **SANS Transport**.
-   [ ] Exécuter Run B (No Transport).
-   [ ] Sauvegarder Run B.
