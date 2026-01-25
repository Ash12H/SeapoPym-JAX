# Axe 1 : Blueprint & Data

## Objectifs

Définir le format déclaratif du modèle, le système de registre de fonctions et la validation des données.

## État des Discussions

### 1. Format de Configuration (YAML/Dict)

_Objectif : Définir la structure de données qui remplace l'API impérative `register_unit`._

**Décision Validée (2026-01-24) : Architecture "Split"**
Nous séparons la définition de la physique de la définition de l'expérience.

1.  **Le Blueprint (Fichier Modèle `*.yaml`)** :
    - Définit la topologie du graphe et les contrats d'interface.
    - Ne contient AUCUNE valeur de donnée massive.
    - Déclare les dimensions requises ("Semantic Dimensions"), ex: `spatial`, `cohort`.
    - Déclare les unités attendues (Pint).
    - Structure :
        ```yaml
        id: "tuna-model-v1"
        declarations:
            state: { biomass: { units: "g", dims: ["spatial", "cohort"] } }
            parameters: { growth_rate: { units: "1/d" } } # Pas de valeur ici
        process:
            - func: "biol:growth"
              inputs: { ... }
        ```

2.  **La Configuration (Fichier Expérience `run.yaml`)** :
    - Injecte les valeurs et les fichiers de données.
    - Mappe les contrats abstraits vers du concret.
    - Structure :
        ```yaml
        model: "./models/tuna-model-v1.yaml"
        inputs:
            parameters: { growth_rate: 0.1 } # Valeur scalaire
            forcings: { temp: "/data/sst.nc" } # Fichier avec dims explicites
        ```

_(Discussion close sur le format global)_

### 2. Registre de Fonctions

_Objectif : Mapper les noms de fonctions (Strings) vers le code exécutable (Python/JAX)._

**Approche suggérée :**
Utilisation d'un décorateur `@seapopym.functional` qui enregistre la fonction dans un catalogue global lors de l'import.
Le Blueprint ne stocke que la clé : `func: "seapopym.library.growth.arrhenius"`.

_(Espace pour nos échanges à venir)_

### 3. Data & Validation

_Objectif : Garantir la cohérence physique avant compilation._

### 3. Data, Dimensions & Broadcasting

_Objectif : Gestion des dimensions (Broadcasting) et validation._

**Discussion actuelle :**

- Les fichiers de forçage (NetCDF/Zarr) portent leurs propres dimensions explicites.
- **Le Modèle doit être "Broadcastable"** :
    - Idéalement, les noyaux biologiques sont écrits en 0D (point-wise).
    - Le Compilateur utilise `jax.vmap` pour les appliquer aux dimensions des données d'entrée (Forçages).
    - Exception : Les noyaux "Grid" (Transport) qui requièrent explicitement les dimensions spatiales (`lat`, `lon`) pour les calculs de voisinage.
- **Validation** : Chargement "Lazy" des métadonnées des forçages pour vérifier la compatibilité avec le Blueprint avant exécution.

**Points clés :**

- Utilisation de `pint` pour parser et valider les unités.
- Vérification des signatures de fonctions (Inputs requis vs Inputs disponibles).

**Décision Validée (2026-01-24) : Règle de Cohérence Dimensionnelle**

1.  **Input Utilisateur** : L'ordre des dimensions dans les fichiers de forçage n'importe pas (Xarray gère par nom).
2.  **Alignement Préalable** : Le système doit garantir que tous les forçages partagent les mêmes coordonnées (tailles, valeurs) pour les dimensions communes. Pas de broadcast magique entre grilles différentes.
3.  **Layout Canonique (Lien Axe 2)** : Le Compilateur transposera toutes les données dans un ordre figé (ex: `T, C, Z, Y, X`) avant d'appeler JAX.

## Décisions Validées

1.  **Format Split** : `model.yaml` (Physique) vs `run.yaml` (Config/Data).
2.  **Décorateur** : Utilisation de `@functional` pour le registre.
3.  **Binding Sémantique** : Le modèle déclare des besoins dimensionnels ("cohort"), le config fournit des données alignées nominalement.
4.  **Pré-Validation** : Xarray/Pint valident bien avant JAX.
