# Analyse de l'Architecture Multi-Groupes et Multi-Forçages

Ce document analyse la faisabilité et la pertinence de la restructuration du projet `seapopym-message` pour supporter des groupes fonctionnels dynamiques, des forçages N-dimensionnels et des unités de calcul contextuelles.

## 1. Analyse des Propositions

### A. Groupes Fonctionnels (State, Params, Units)
**Proposition :** Définir des entités "Groupes" encapsulant leur propre état, paramètres et logique (liste d'unités).
**Analyse :**
*   **Pertinence :** Crucial pour l'évolution du modèle. Cela permet de passer d'un code "codé en dur" pour une espèce à un moteur générique capable de simuler un écosystème.
*   **Faisabilité :** Élevée.
    *   Le `State` global devient une composition de sous-états (Namespacing). Ex: `biomass` devient `group_A/biomass`, `group_B/biomass`.
    *   Les `Params` doivent être structurés de la même manière ou passés comme arguments spécifiques à chaque groupe.
    *   **Challenge :** Il faut une abstraction (classe `FunctionalGroup`) qui configure ces éléments avant la construction du Kernel.

### B. Forçages Globaux N-Dimensions (Xarray/Dask)
**Proposition :** Utiliser Xarray/Dask pour gérer des forçages complexes (ex: 3D, 4D) et conserver les métadonnées.
**Analyse :**
*   **Pertinence :** Indispensable pour gérer la profondeur (Depth) et le temps (Time) de manière flexible sans tout charger en mémoire RAM.
*   **Faisabilité :** Moyenne (Attention à la barrière JAX).
    *   **Le Problème :** JAX (utilisé dans les Kernels) ne sait pas manipuler des objets Xarray/Dask directement à l'intérieur d'une fonction compilée (`jit`).
    *   **La Solution :** Une séparation stricte des responsabilités.
        1.  **Host (CPU) :** Le `ForcingManager` utilise Xarray/Dask pour "préparer" les données (slicing, interpolation, réduction de dimension, calcul de moyenne jour/nuit).
        2.  **Device (GPU/TPU/JIT) :** Le Kernel reçoit des `jax.numpy.array` purs.
    *   Cela implique que la logique de "quelle couche de profondeur utiliser" doit être résolue *avant* l'appel au Kernel JAX, ou que le Kernel reçoive le champ 3D complet et un index de profondeur.

### C. Unités Contextuelles et "Linkage" Dynamique
**Proposition :** Des fonctions unitaires génériques (ex: `grow()`) qui peuvent être appliquées à n'importe quel groupe via un mapping dynamique des entrées/sorties.
**Analyse :**
*   **Pertinence :** C'est le cœur de la modularité. Cela évite de dupliquer le code (`grow_tuna`, `grow_sardine`). Une fonction `predation(predator, prey)` peut être configurée pour `(Tuna, Sardine)` ou `(Shark, Tuna)`.
*   **Faisabilité :** Élevée via un système de "Binding" ou "Mapping".
    *   La signature de la fonction est statique : `def grow(biomass, temperature, rate): ...`
    *   La configuration du Groupe définit le mapping :
        *   `biomass` -> `state['group_A/biomass']`
        *   `temperature` -> `forcings['temp_100m']`
        *   `rate` -> `params['group_A/growth_rate']`

## 2. Architecture Proposée

Voici comment structurer le code pour répondre à ces besoins.

### 2.1. Le Concept de `FunctionalGroup`

C'est une classe de configuration (Python pur, pas JAX) qui définit l'identité du groupe.

```python
@dataclass
class FunctionalGroup:
    name: str
    units: list[Unit]  # Les comportements de ce groupe

    # Mapping des noms internes de la Unit vers les noms globaux du State/Forcing
    # Ex: La unit attend "temperature", ce groupe lui fournit "temperature_layer_1"
    variable_map: dict[str, str]

    # Paramètres spécifiques à ce groupe
    params: dict[str, float]
```

### 2.2. Le "Kernel Compiler" (L'assembleur)

Au lieu d'exécuter les unités directement, le Kernel doit maintenant "compiler" la logique pour chaque groupe.

1.  **Aplatissement du State :** Le state global reste un dictionnaire plat pour la performance JAX, mais les clés sont préfixées : `{"tuna_biomass": ..., "sardine_biomass": ...}`.
2.  **Résolution des Dépendances :**
    *   Quand le Groupe A exécute `update_biomass(biomass, recruitment)`, le système regarde son `variable_map`.
    *   Il injecte `state["tuna_biomass"]` et `state["tuna_recruitment"]`.
3.  **Interactions Inter-Groupes :**
    *   Si le Groupe A a une unité `predation(prey_biomass)`, on peut mapper `prey_biomass` vers `state["sardine_biomass"]`.

### 2.3. Gestion des Forçages (Le "Smart" ForcingManager)

Le `ForcingManager` doit devenir plus intelligent pour gérer les dimensions N.

*   **Configuration :** On définit des "Stratégies de Forçage".
    *   *Direct :* "Prends la variable `temp` telle quelle."
    *   *Slicing :* "Prends `temp` à `depth=100`."
    *   *Aggregation :* "Prends `temp`, pondère par `day_length` à `depth=0` et `(1-day_length)` à `depth=200`."
*   **Exécution :** Cette préparation se fait en Python/Xarray *avant* de passer les tableaux à JAX.

## 3. Exemple de Flux de Données

Imaginez une fonction unitaire simple :
```python
# Signature statique
def compute_growth(biomass, temperature, r):
    return biomass * r * temperature
```

**Configuration :**
*   **Groupe 1 (Surface)** :
    *   Map: `biomass` -> `g1_bio`, `temperature` -> `sst`, `r` -> `g1_r`
*   **Groupe 2 (Deep)** :
    *   Map: `biomass` -> `g2_bio`, `temperature` -> `temp_200m`, `r` -> `g2_r`

**Au moment de l'exécution (JAX Kernel) :**
Le système appelle automatiquement :
1.  `compute_growth(state['g1_bio'], forcings['sst'], params['g1_r'])` -> écrit dans `state['g1_bio']`
2.  `compute_growth(state['g2_bio'], forcings['temp_200m'], params['g2_r'])` -> écrit dans `state['g2_bio']`

## 4. Conclusion et Recommandations

Cette approche est **très pertinente** et alignée avec les standards des modèles complexes (comme ceux basés sur des agents ou des systèmes multi-physiques).

**Étapes recommandées :**

1.  **Refactoriser `Unit` :** Séparer la définition de la fonction (logique pure) de la définition des entrées/sorties (noms de variables). Ajouter un concept de "Variable interne" vs "Variable liée".
2.  **Créer `FunctionalGroup` :** Une classe pour tenir la configuration.
3.  **Mettre à jour `ForcingManager` :** Intégrer Xarray pour le pré-traitement (Slicing/Interpolation) avant l'envoi aux workers.
4.  **Adapter le `Kernel` :** Il doit itérer sur les groupes et appliquer les mappings de variables.

C'est un chantier conséquent mais qui rendra le framework extrêmement puissant et générique.
