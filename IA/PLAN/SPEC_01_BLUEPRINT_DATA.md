# SPEC 01 : Blueprint & Data

**Version** : 1.1
**Date** : 2026-01-25
**Statut** : Validé

---

## 1. Vue d'ensemble

Ce module définit le format déclaratif des modèles, le système de registre de fonctions et la validation des données. C'est le point d'entrée de l'architecture SeapoPym-JAX.

### 1.1 Objectifs

- Permettre la définition déclarative d'un modèle via fichiers (YAML/JSON) ou directement en Python
- Séparer la physique (modèle) de l'expérience (configuration)
- Enregistrer et résoudre les fonctions de calcul
- Valider la cohérence avant exécution

### 1.2 Dépendances

- **Amont** : Aucune (point d'entrée)
- **Aval** : Axe 2 (Compilateur) consomme le graphe validé

---

## 2. Formats d'Entrée

Le système accepte trois formats équivalents pour le Blueprint et la Config :

| Format        | Extension       | Cas d'usage                           |
| ------------- | --------------- | ------------------------------------- |
| YAML          | `.yaml`, `.yml` | Fichiers de configuration lisibles    |
| JSON          | `.json`         | Interopérabilité avec d'autres outils |
| Python `dict` | -               | Prototypage en notebook, tests        |

### 2.1 Chargement Unifié

```python
from seapopym import Blueprint, Config

# Depuis YAML
blueprint = Blueprint.load("model.yaml")

# Depuis JSON
blueprint = Blueprint.load("model.json")

# Depuis dict Python
blueprint = Blueprint.from_dict({
    "id": "test-model",
    "declarations": {...},
    "process": [...]
})

# Détection automatique par extension
def load_blueprint(source: str | dict) -> Blueprint:
    if isinstance(source, dict):
        return Blueprint.from_dict(source)
    if source.endswith('.json'):
        return Blueprint.from_json(source)
    return Blueprint.from_yaml(source)
```

---

## 3. Architecture "Split"

L'architecture sépare deux structures distinctes :

### 3.1 Le Blueprint (Définition du Modèle)

**Fichier** : `*.model.yaml` ou `*.model.json`

**Rôle** : Définir la topologie du graphe et les contrats d'interface.

**Contraintes** :

- Ne contient AUCUNE valeur de donnée massive
- Déclare les dimensions sémantiques requises
- Déclare les unités attendues (Pint)

**Structure avec groupes fonctionnels** :

```yaml
id: "ecosystem-model-v1"
version: "1.0.0"

# Déclarations des variables (hiérarchie par groupe fonctionnel)
declarations:
  state:
    tuna:
      biomass:
        units: "g"
        dims: ["Y", "X", "C"]
        description: "Biomasse de thon par cohorte"
    zooplankton:
      biomass:
        units: "g"
        dims: ["Y", "X", "C"]
        description: "Biomasse de zooplancton par cohorte"

  parameters:
    tuna:
      growth_rate:
        units: "1/d"
      predation_rate:
        units: "1/d"
    zooplankton:
      growth_rate:
        units: "1/d"

  forcings:
    temperature:
      units: "degC"
      dims: ["T", "Y", "X"]

  # Variables dérivées (intermédiaires calculés)
  derived:
    temperature_gillooly:
      units: "dimensionless"
      dims: ["T", "Y", "X"]
      description: "Température transformée pour processus biologiques"

# Chaîne de traitement
process:
  # Transformation de forçage
  - func: "biol:gillooly_transform"
    inputs:
      temp: "forcings.temperature"
    outputs:
      temp_bio:
        target: "derived.temperature_gillooly"
        type: "derived"

  # Croissance zooplancton
  - func: "biol:growth"
    inputs:
      biomass: "state.zooplankton.biomass"
      rate: "parameters.zooplankton.growth_rate"
      temp: "derived.temperature_gillooly"
    outputs:
      tendency:
        target: "tendencies.zooplankton.growth"
        type: "tendency"

  # Prédation (outputs multiples)
  - func: "biol:predation"
    inputs:
      prey_biomass: "state.zooplankton.biomass"
      predator_biomass: "state.tuna.biomass"
      rate: "parameters.tuna.predation_rate"
    outputs:
      prey_loss:
        target: "tendencies.zooplankton.predation"
        type: "tendency"
      predator_gain:
        target: "tendencies.tuna.predation"
        type: "tendency"

  # Vieillissement (modification directe du state)
  - func: "biol:aging"
    inputs:
      biomass: "state.tuna.biomass"
    outputs:
      biomass:
        target: "state.tuna.biomass"
        type: "state"
```

### 3.2 Types d'Output

Les fonctions peuvent produire différents types d'output :

| Type         | Comportement                                                     | Exemple                 |
| ------------ | ---------------------------------------------------------------- | ----------------------- |
| `tendency`   | Intégré via `S += tendency * dt`                                 | Flux de croissance      |
| `derived`    | Stocké temporairement, utilisable par les processus suivants     | Température transformée |
| `diagnostic` | Sauvegardé en sortie, pas utilisé dans le calcul                 | Flux pour analyse       |
| `state`      | Écrit directement dans le state (opérations non-différentielles) | Aging, migration        |

### 3.3 La Configuration (Expérience)

**Fichier** : `*.run.yaml` ou `*.run.json`

**Rôle** : Injecter les valeurs concrètes et mapper les contrats abstraits.

**Structure** :

```yaml
model: "./models/ecosystem-model-v1.model.yaml"

# Valeurs des paramètres (hiérarchie correspondante)
parameters:
  tuna:
    growth_rate:
      value: 0.08
      trainable: true
      bounds: [0.01, 0.2]
    predation_rate:
      value: 0.05
      trainable: false
  zooplankton:
    growth_rate:
      value: 0.15
      trainable: false

# Chemins des données (ou arrays en mémoire)
forcings:
  temperature: "/data/sst.nc"

# État initial
initial_state:
  tuna:
    biomass: "/data/tuna_biomass_init.nc"
  zooplankton:
    biomass: "/data/zoo_biomass_init.nc"

# Métadonnées d'exécution
execution:
  time_range: ["2020-01-01", "2020-12-31"]
  dt: "1d"
  output_path: "/results/run_001/"
```

### 3.4 Création Directe en Python (Mode Notebook)

Pour le prototypage rapide sans fichiers :

```python
import numpy as np
from seapopym import Blueprint, Config, compile_model

# Blueprint en Python pur
blueprint = Blueprint.from_dict({
    "id": "test-model",
    "version": "0.1.0",
    "declarations": {
        "state": {
            "biomass": {"units": "g", "dims": ["Y", "X", "C"]}
        },
        "parameters": {
            "growth_rate": {"units": "1/d"}
        },
        "forcings": {
            "temperature": {"units": "degC", "dims": ["T", "Y", "X"]}
        }
    },
    "process": [
        {
            "func": "biol:growth",
            "inputs": {
                "biomass": "state.biomass",
                "rate": "parameters.growth_rate",
                "temp": "forcings.temperature"
            },
            "outputs": {
                "tendency": {"target": "tendencies.growth", "type": "tendency"}
            }
        }
    ]
})

# Config avec données en mémoire (pas de fichiers)
config = Config.from_dict({
    "parameters": {
        "growth_rate": {"value": 0.1}
    },
    "forcings": {
        # Array NumPy directement
        "temperature": np.random.uniform(15, 25, size=(365, 20, 30))
    },
    "initial_state": {
        "biomass": np.ones((20, 30, 5)) * 100.0
    },
    "execution": {
        "dt": "1d"
    }
})

# Compilation et exécution
model = compile_model(blueprint, config, backend="jax")
result = model.run()
```

---

## 4. Registre de Fonctions

### 4.1 Décorateur `@functional`

Les fonctions de calcul sont enregistrées via un décorateur qui les inscrit dans un catalogue global.

**Syntaxe pour output simple** :

```python
from seapopym import functional
import jax.numpy as jnp

@functional(
    name="biol:growth",
    backend="jax",
    core_dims={"biomass": ["C"]},
    out_dims=["C"],
    units={
        "biomass": "g",
        "rate": "1/d",
        "temp": "degC",
        "return": "g/d"
    }
)
def growth(biomass, rate, temp):
    """Calcule la tendance de croissance."""
    return biomass * rate * jnp.exp(temp / 10)
```

**Syntaxe pour outputs multiples** :

```python
@functional(
    name="biol:predation",
    backend="jax",
    core_dims={"prey_biomass": ["C"], "predator_biomass": ["C"]},
    outputs=["prey_loss", "predator_gain"],  # Noms des outputs (ordre du tuple)
    units={
        "prey_biomass": "g",
        "predator_biomass": "g",
        "rate": "1/d",
        "prey_loss": "g/d",
        "predator_gain": "g/d"
    }
)
def predation(prey_biomass, predator_biomass, rate):
    """
    Calcule les flux de prédation.

    Returns:
        tuple: (prey_loss, predator_gain)
    """
    flux = rate * prey_biomass * predator_biomass
    return -flux, +flux  # Tuple ordonné selon `outputs`
```

**Syntaxe pour forçage dérivé** :

```python
@functional(
    name="biol:gillooly_transform",
    backend="jax",
    units={
        "temp": "degC",
        "return": "dimensionless"
    }
)
def gillooly_transform(temp):
    """Transforme la température selon Gillooly et al."""
    T_ref = 15.0
    E_a = 0.63  # eV
    k_B = 8.617e-5  # eV/K
    return jnp.exp(-E_a / k_B * (1 / (temp + 273.15) - 1 / (T_ref + 273.15)))
```

### 4.2 Paramètres du Décorateur

| Paramètre   | Type                   | Obligatoire | Description                                      |
| ----------- | ---------------------- | ----------- | ------------------------------------------------ |
| `name`      | `str`                  | Oui         | Identifiant unique (format `namespace:function`) |
| `backend`   | `str`                  | Oui         | `"jax"` ou `"numpy"`                             |
| `core_dims` | `dict[str, list[str]]` | Non         | Dimensions non-broadcastables par input          |
| `out_dims`  | `list[str]`            | Non         | Dimensions de sortie (output simple)             |
| `outputs`   | `list[str]`            | Non         | Noms des outputs (outputs multiples)             |
| `units`     | `dict[str, str]`       | Non         | Unités attendues par argument                    |

### 4.3 Résolution Backend

Le registre maintient deux catalogues séparés :

```python
REGISTRY = {
    "jax": {
        "biol:growth": growth_jax,
        "biol:predation": predation_jax,
        "biol:gillooly_transform": gillooly_jax,
    },
    "numpy": {
        "biol:growth": growth_numpy,
    }
}
```

**Règle de résolution** :

1. Le Runner spécifie le backend actif (`"jax"` ou `"numpy"`)
2. Le registre retourne la fonction correspondante
3. Si la fonction n'existe pas pour ce backend → `FunctionNotFoundError`

### 4.4 Fonctions Vectorisées

Les fonctions doivent être écrites en **vectorisé natif** (broadcasting NumPy/JAX), pas en 0D scalaire.

**Exemple correct** :

```python
def growth(biomass, rate, temp):
    # biomass: (Y, X, C), rate: scalar, temp: (Y, X)
    # Broadcasting automatique sur (Y, X)
    return biomass * rate * jnp.exp(temp[..., None] / 10)
```

**Core Dims** : Les dimensions déclarées dans `core_dims` indiquent que la fonction opère explicitement sur ces axes (ex: `roll` sur l'axe cohorte).

---

## 5. Groupes Fonctionnels

### 5.1 Convention de Nommage Hiérarchique

Les groupes fonctionnels (espèces, compartiments) sont organisés hiérarchiquement :

```
state.{groupe}.{variable}
parameters.{groupe}.{paramètre}
tendencies.{groupe}.{processus}
```

**Exemples** :

- `state.tuna.biomass`
- `state.zooplankton.biomass`
- `parameters.tuna.growth_rate`
- `tendencies.zooplankton.predation`

### 5.2 Avantages

- Chaque groupe peut avoir une structure différente (nombre de cohortes, dimensions)
- Les processus peuvent référencer explicitement les interactions inter-groupes
- Le blueprint reste lisible même avec de nombreuses espèces

### 5.3 Alternative : Dimension Espèce

Pour des modèles où toutes les espèces ont la même structure :

```yaml
declarations:
  state:
    biomass:
      dims: ["Y", "X", "C", "F"] # F = functional_group
```

Choisir selon le cas d'usage : hiérarchie si structures hétérogènes, dimension si homogènes.

---

## 6. Dimensions Canoniques

### 6.1 Standard de Nommage

| Abréviation | Nom Complet | Description                                  |
| ----------- | ----------- | -------------------------------------------- |
| `T`         | `time`      | Dimension temporelle                         |
| `E`         | `ensemble`  | Membres d'ensemble (batch)                   |
| `C`         | `cohort`    | Classes d'âge/taille                         |
| `Z`         | `depth`     | Profondeur verticale                         |
| `Y`         | `latitude`  | Axe latitudinal                              |
| `X`         | `longitude` | Axe longitudinal                             |
| `F`         | `functional_group` | Groupes fonctionnels (si dimension plutôt que hiérarchie) |

### 6.2 Ordre Canonique

L'ordre interne utilisé par le Compilateur (Axe 2) est :

```
(E, T, F, C, Z, Y, X)
```

Les données utilisateur peuvent avoir n'importe quel ordre ; le Compilateur transpose automatiquement.

### 6.3 Mapping Utilisateur

L'utilisateur peut fournir un mapping de renommage dans la configuration :

```yaml
dimension_mapping:
  lat: "Y"
  lon: "X"
  time: "T"
  age_class: "C"
  species: "F"
```

---

## 7. Validation

### 7.1 Pipeline en 6 Étapes

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. RÉCEPTION BLUEPRINT                                          │
│    - Parse YAML/JSON/dict                                       │
│    - Vérifie la syntaxe                                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. CONSTRUCTION DU GRAPHE                                       │
│    - Résout les fonctions dans le registre                      │
│    - Vérifie les signatures (arguments requis vs disponibles)   │
│    - Vérifie les dimensions IN/OUT                              │
│    - Vérifie les unités (Pint)                                  │
│    - Vérifie la cohérence des outputs multiples                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. VALIDATION DU GRAPHE                                         │
│    - Graphe valide → retourne l'objet Graph                     │
│    - Graphe invalide → lève ValidationError avec détails        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. RÉCEPTION DES DONNÉES                                        │
│    - Charge les métadonnées (lazy) des fichiers NetCDF/Zarr     │
│    - Accepte les arrays NumPy en mémoire                        │
│    - Parse les valeurs scalaires                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. VALIDATION DES DONNÉES                                       │
│    - Vérifie les unités vs déclarations                         │
│    - Vérifie les dimensions vs déclarations                     │
│    - Vérifie l'alignement des coordonnées (grilles compatibles) │
│    - Vérifie l'absence de NaN non masqués                       │
│    - Vérifie la présence de tous les objets requis              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. RETOUR OBJET EXÉCUTABLE                                      │
│    - CompiledModel prêt pour le Runner (Axe 3)                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Erreurs de Validation

| Code   | Type                       | Description                           |
| ------ | -------------------------- | ------------------------------------- |
| `E101` | `FunctionNotFoundError`    | Fonction absente du registre          |
| `E102` | `SignatureMismatchError`   | Arguments manquants ou superflus      |
| `E103` | `DimensionMismatchError`   | Dimensions incompatibles              |
| `E104` | `UnitMismatchError`        | Unités incompatibles (Pint)           |
| `E105` | `GridAlignmentError`       | Coordonnées non alignées              |
| `E106` | `MissingDataError`         | Donnée requise absente                |
| `E107` | `OutputCountMismatchError` | Nombre d'outputs déclarés ≠ retournés |

---

## 8. Gestion des Masques

### 8.1 Stratégie

- Les NaN sont remplacés par `0.0` en prétraitement (Axe 2)
- L'utilisateur fournit un masque booléen comme input standard
- Le masque peut être statique `(Y, X)` ou dynamique `(T, Y, X)`

### 8.2 Convention

**Responsabilité du développeur** : Chaque fonction susceptible d'être polluée par les valeurs `0.0` (ex: moyennes, réductions) doit utiliser explicitement le masque.

```python
@functional(name="stats:mean", backend="jax")
def masked_mean(data, mask):
    """Moyenne masquée."""
    return jnp.sum(data * mask) / jnp.sum(mask)
```

---

## 9. Tests Unitaires

Les fonctions `@functional` sont testées de manière isolée avec des données mock.

```python
import jax.numpy as jnp

def test_growth():
    biomass = jnp.ones((10, 10, 5))  # Y, X, C
    rate = 0.1
    temp = jnp.full((10, 10), 20.0)  # Y, X

    result = growth(biomass, rate, temp)

    assert result.shape == (10, 10, 5)
    assert jnp.all(result > 0)

def test_predation_outputs():
    prey = jnp.ones((10, 10, 5)) * 100
    predator = jnp.ones((10, 10, 3)) * 50
    rate = 0.01

    prey_loss, predator_gain = predation(prey, predator, rate)

    # Conservation de masse
    assert jnp.allclose(jnp.sum(prey_loss), -jnp.sum(predator_gain))
```

---

## 10. Interfaces

### 10.1 API Python

```python
from seapopym import Blueprint, Config, compile_model

# Depuis fichiers
blueprint = Blueprint.load("model.yaml")
config = Config.load("run.yaml")

# Ou depuis dict (notebook)
blueprint = Blueprint.from_dict({...})
config = Config.from_dict({...})

# Compilation (étapes 1-6)
model = compile_model(blueprint, config, backend="jax")

# Prêt pour exécution (Axe 3)
model.run()
```

### 10.2 Objets Retournés

| Objet           | Description                                |
| --------------- | ------------------------------------------ |
| `Blueprint`     | Graphe abstrait (topologie + contrats)     |
| `Config`        | Valeurs concrètes + chemins/arrays données |
| `CompiledModel` | Graphe résolu + données prêtes             |

---

## 11. Liens avec les Autres Axes

| Axe                 | Interaction                                                     |
| ------------------- | --------------------------------------------------------------- |
| Axe 2 (Compilateur) | Reçoit le `Blueprint` + `Config`, produit les arrays transposés |
| Axe 3 (Engine)      | Reçoit le `CompiledModel`, exécute via Runner                   |
| Axe 5 (Auto-Diff)   | Lit le flag `trainable: true` dans Config                       |

---

## 12. Questions Ouvertes (V2+)

- Support de blueprints composites (héritage/imports entre modèles)
- Versioning sémantique des blueprints
- Validation de reproductibilité (seed, ordre des opérations)
- Génération automatique de documentation depuis le blueprint
