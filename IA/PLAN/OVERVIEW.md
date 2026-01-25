# OVERVIEW - SeapoPym JAX

**Document de référence pour l'implémentation**

---

## 1. Vision du Projet

SeapoPym-JAX est un framework de modélisation déclaratif permettant de définir des modèles à base d'équations aux dérivées partielles (EDP) et de les exécuter efficacement via JAX. L'objectif est de séparer la déclaration scientifique (quoi calculer) de l'exécution technique (comment calculer).

---

## 2. Conventions Transversales

### 2.1 Dimensions Canoniques

| Abréviation | Nom Complet | Description |
|-------------|-------------|-------------|
| `E` | ensemble | Membres d'ensemble (batch) |
| `T` | time | Dimension temporelle |
| `F` | functional_group | Groupes fonctionnels |
| `C` | cohort | Classes d'âge/taille |
| `Z` | depth | Profondeur verticale |
| `Y` | latitude | Axe latitudinal |
| `X` | longitude | Axe longitudinal |

### 2.2 Ordre Canonique

```
(E, T, F, C, Z, Y, X)
```

Toutes les données sont transposées dans cet ordre par le Compilateur.

### 2.3 Mask

Le mask est stocké dans `forcings["mask"]` et accessible uniformément dans :
- Le forward pass (`step_fn`)
- La loss function (`model.forcings["mask"]`)

### 2.4 Backends

| Backend | Usage | Boucle temporelle |
|---------|-------|-------------------|
| `jax` | Production, optimisation | `jax.lax.scan` |
| `numpy` | Tests, prototypage | Boucle `for` Python |

---

## 3. Structure des Packages

```
seapopym/
├── __init__.py
├── blueprint/              # Axe 1 : Blueprint & Data
│   ├── __init__.py
│   ├── schema.py           # Classes Blueprint, Config
│   ├── registry.py         # Registre @functional
│   ├── validation.py       # Pipeline de validation
│   └── loaders.py          # Chargement YAML/JSON/dict
│
├── compiler/               # Axe 2 : Compilateur
│   ├── __init__.py
│   ├── inference.py        # Inférence de shapes
│   ├── transpose.py        # Transposition canonique
│   ├── preprocessing.py    # Stripping, NaN, masques
│   └── model.py            # CompiledModel
│
├── engine/                 # Axe 3 : Moteur d'exécution
│   ├── __init__.py
│   ├── step.py             # Step kernel
│   ├── runners.py          # StreamingRunner, GradientRunner
│   ├── backends.py         # JAXBackend, NumpyBackend
│   └── io.py               # I/O asynchrone
│
├── parallel/               # Axe 4 : Parallélisme
│   ├── __init__.py
│   ├── batch.py            # BatchRunner, vmap
│   └── sharding.py         # Configuration mesh GPU
│
├── optim/                  # Axe 5 : Auto-Diff
│   ├── __init__.py
│   ├── loss.py             # Fonctions de loss
│   ├── checkpointing.py    # jax.remat wrapper
│   └── training.py         # Boucle d'optimisation
│
├── functions/              # Bibliothèque de fonctions @functional
│   ├── __init__.py
│   ├── biology.py          # growth, aging, predation, ...
│   ├── physics.py          # transport, diffusion, ...
│   └── utils.py            # Fonctions utilitaires
│
└── tests/
    ├── unit/               # Tests unitaires par module
    ├── integration/        # Tests end-to-end
    └── fixtures/           # Données de test, modèle toy
```

---

## 4. Phases d'Implémentation

### 4.1 Diagramme de Dépendances

```
Phase 1 (Blueprint)
      │
      ↓
Phase 2 (Compiler)
      │
      ↓
Phase 3 (Engine)
      │
      ├──────────┬──────────┐
      ↓          ↓          ↓
Phase 4      Phase 5    [Tests E2E]
(Parallel)   (AutoDiff)
```

### 4.2 Détail des Phases

| Phase | Axe | SPEC | Livrables |
|-------|-----|------|-----------|
| 1 | Blueprint & Data | `SPEC_01_BLUEPRINT_DATA.md` | `Blueprint`, `Config`, `@functional`, validation |
| 2 | Compilateur | `SPEC_02_COMPILER.md` | `CompiledModel`, transposition, preprocessing |
| 3 | Engine | `SPEC_03_ENGINE.md` | `step_fn`, `StreamingRunner`, backends |
| 4 | Parallélisme | `SPEC_04_PARALLELISM.md` | `BatchRunner`, sharding config |
| 5 | Auto-Diff | `SPEC_05_AUTODIFF.md` | `GradientRunner`, loss, checkpointing |

### 4.3 Critères de Complétion par Phase

**Phase 1** :
- [ ] `Blueprint.load()` depuis YAML/JSON/dict
- [ ] `Config.load()` depuis YAML/JSON/dict
- [ ] Décorateur `@functional` avec registre
- [ ] Validation du graphe (fonctions, dims, unités)
- [ ] Tests unitaires du registre et de la validation

**Phase 2** :
- [ ] Inférence de shapes depuis métadonnées
- [ ] Transposition vers ordre canonique
- [ ] Preprocessing NaN → 0.0 + mask
- [ ] `CompiledModel` avec pytrees
- [ ] Tests unitaires de chaque étape

**Phase 3** :
- [ ] `step_fn` générique
- [ ] `JAXBackend` avec `lax.scan`
- [ ] `NumpyBackend` avec boucle for
- [ ] `StreamingRunner` avec chunking
- [ ] I/O asynchrone
- [ ] **Test d'intégration E2E** : Blueprint → Compile → Run

**Phase 4** :
- [ ] `BatchRunner` avec `vmap`
- [ ] Configuration sharding déclarative
- [ ] Tests sur grille de paramètres

**Phase 5** :
- [ ] `GradientRunner`
- [ ] Checkpointing (`jax.remat`)
- [ ] Fonctions de loss (MSE, MAE, composite)
- [ ] Boucle d'optimisation avec Optax
- [ ] **Test d'intégration** : Calibration sur modèle toy

---

## 5. Modèle Toy de Validation

Un modèle minimal pour tester chaque phase sans données réelles.

### 5.1 Définition

```yaml
# toy_model.yaml
id: "toy-growth"
version: "0.1.0"

declarations:
  state:
    biomass:
      units: "g"
      dims: ["Y", "X"]
  parameters:
    growth_rate:
      units: "1/d"
  forcings:
    temperature:
      units: "degC"
      dims: ["T", "Y", "X"]
    mask:
      dims: ["Y", "X"]

process:
  - func: "biol:simple_growth"
    inputs:
      biomass: "state.biomass"
      rate: "parameters.growth_rate"
      temp: "forcings.temperature"
    outputs:
      tendency:
        target: "tendencies.growth"
        type: "tendency"
```

### 5.2 Fonction Associée

```python
@functional(name="biol:simple_growth", backend="jax")
def simple_growth(biomass, rate, temp):
    """Croissance exponentielle simple."""
    return biomass * rate * (temp / 20.0)
```

### 5.3 Données de Test

```python
# Grille 10x10, 30 jours
config = Config.from_dict({
    "parameters": {"growth_rate": {"value": 0.1}},
    "forcings": {
        "temperature": np.random.uniform(15, 25, (30, 10, 10)),
        "mask": np.ones((10, 10))
    },
    "initial_state": {
        "biomass": np.ones((10, 10)) * 100.0
    },
    "execution": {"dt": "1d"}
})
```

### 5.4 Utilisation par Phase

| Phase | Test avec modèle toy |
|-------|---------------------|
| 1 | Charger blueprint, valider graphe |
| 2 | Compiler, vérifier shapes et transposition |
| 3 | Exécuter 30 pas, vérifier croissance biomasse |
| 4 | Exécuter batch de 10 simulations |
| 5 | Calibrer `growth_rate` sur données synthétiques |

---

## 6. Tests d'Intégration

### 6.1 Test E2E Phase 3

```python
def test_e2e_basic_simulation():
    """Test complet : Blueprint → Compile → Run."""
    # 1. Chargement
    blueprint = Blueprint.load("tests/fixtures/toy_model.yaml")
    config = Config.load("tests/fixtures/toy_config.yaml")

    # 2. Compilation
    model = compile_model(blueprint, config, backend="jax")

    # 3. Exécution
    result = model.run()

    # 4. Vérifications
    assert result.state["biomass"].shape == (10, 10)
    assert jnp.all(result.state["biomass"] > 100.0)  # Croissance
    assert not jnp.any(jnp.isnan(result.state["biomass"]))
```

### 6.2 Test E2E Phase 5

```python
def test_e2e_calibration():
    """Test complet : Calibration sur données synthétiques."""
    # 1. Générer observations synthétiques avec growth_rate=0.15
    true_rate = 0.15
    observations = generate_synthetic(growth_rate=true_rate)

    # 2. Partir d'une valeur initiale différente
    config = Config.from_dict({
        "parameters": {"growth_rate": {"value": 0.05, "trainable": True}},
        ...
    })

    # 3. Calibration
    model = compile_model(blueprint, config)
    result = model.optimize(
        observations=observations,
        optimizer=optax.adam(0.01),
        n_epochs=100
    )

    # 4. Vérifier convergence
    assert abs(result.optimal_params["growth_rate"] - true_rate) < 0.02
```

---

## 7. Checklist de Démarrage de Phase

Avant de commencer une nouvelle phase :

1. [ ] Relire le SPEC correspondant
2. [ ] Relire ce document OVERVIEW (conventions, structure)
3. [ ] Vérifier que la phase précédente est complète (critères ci-dessus)
4. [ ] Créer la structure de packages si inexistante
5. [ ] Implémenter le test du modèle toy en premier
6. [ ] Implémenter les fonctionnalités
7. [ ] Exécuter les tests unitaires
8. [ ] (Phase 3+) Exécuter le test d'intégration E2E

---

## 8. Fichiers de Référence

| Document | Contenu |
|----------|---------|
| `OVERVIEW.md` | Ce document (référence transversale) |
| `00_COHERENCE_ANALYSIS.md` | Historique des décisions d'architecture |
| `SPEC_01_BLUEPRINT_DATA.md` | Spécification Axe 1 |
| `SPEC_02_COMPILER.md` | Spécification Axe 2 |
| `SPEC_03_ENGINE.md` | Spécification Axe 3 |
| `SPEC_04_PARALLELISM.md` | Spécification Axe 4 |
| `SPEC_05_AUTODIFF.md` | Spécification Axe 5 |
