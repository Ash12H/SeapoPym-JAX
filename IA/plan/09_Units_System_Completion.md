# Plan: Complétion du Système d'Unités

## Problèmes Identifiés

### 1. Bug Critique
- Le notebook produit des valeurs astronomiques (1.076e+152)
- Cause: Confusion entre unités temporelles (jour vs seconde)
- `compute_production_initialization` attend NPP en unités/s mais reçoit unités/jour
- Le TimeIntegrator multiplie par dt en secondes → explosion exponentielle

### 2. Système d'Unités Incomplet
- ✅ `DataNode` supporte les unités
- ✅ `register_forcing()` et `register_parameter()` acceptent les unités
- ✅ Le Controller standardise automatiquement (pint-xarray)
- ❌ `register_unit()` ne permet PAS de définir les unités des outputs
- ❌ Le notebook n'utilise PAS les attributs `units` dans les DataArrays
- ❌ Pas de documentation des conventions d'unités

## Objectifs

1. **Documenter les conventions d'unités**
   - Toutes les variables internes en unités SI (secondes, mètres, etc.)
   - Les tendances doivent être en [unité_variable/seconde]
   - Les forçages peuvent être en unités conventionnelles (convertis automatiquement)

2. **Compléter l'API Blueprint**
   - Ajouter `output_units` à `register_unit()`
   - Permettre la spécification d'unités pour chaque output

3. **Mettre à jour LMTL**
   - Documenter les unités attendues/produites dans chaque fonction
   - Corriger `compute_production_initialization` pour gérer les unités correctement
   - Ajouter validation/conversion si nécessaire

4. **Mettre à jour le notebook**
   - Ajouter attributs `units` aux forçages
   - Spécifier les unités dans le Blueprint
   - Vérifier que la simulation converge vers des valeurs réalistes

5. **Tests**
   - Ajouter tests unitaires pour la validation des unités
   - Tester les conversions automatiques
   - Tester que les tendances ont les bonnes unités

## Architecture des Changements

### Phase 1: Documentation et Conventions
```
IA/plan/01_units_and_parameters.md  ← Conventions d'unités
seapopym/lmtl/core.py                ← Documenter unités dans docstrings
```

### Phase 2: Extension Blueprint
```
seapopym/blueprint/core.py           ← Ajouter output_units parameter
seapopym/blueprint/nodes.py          ← Déjà OK (DataNode.units existe)
```

### Phase 3: Correction LMTL
```
seapopym/lmtl/core.py                ← Fix compute_production_initialization
                                       Documenter toutes les fonctions
```

### Phase 4: Mise à Jour Notebook
```
notebook/05_LMTL_Model_Demo.ipynb    ← Ajouter units aux DataArrays
                                       Spécifier units dans Blueprint
```

### Phase 5: Tests
```
tests/test_units_parameters.py       ← Tests validation unités
tests/test_lmtl.py                   ← Ajouter tests avec unités
```

## Détails Techniques

### Convention d'Unités (SI)
```
Temps:        seconde (s)
Taux:         s^-1
Température:  degC (convertible)
Longueur:     mètre (m)
Masse:        g (gram) - Note: gC not used as it's not recognized by pint
Production:   g/m²/s
Biomasse:     g/m²
Tendances:    [unité_variable]/s
```

**Note**: We use "g" (gram) instead of "gC" (gram Carbon) because pint doesn't recognize "gC" as a standard unit. This is a common convention in marine ecology but requires custom unit definition in pint, which we avoid for simplicity.

### Modification Blueprint.register_unit()

```python
def register_unit(
    self,
    func: Callable[..., Any],
    output_mapping: dict[str, str],
    input_mapping: dict[str, str] | None = None,
    output_tendencies: dict[str, str] | None = None,
    output_units: dict[str, str] | None = None,  # NOUVEAU
    scope: str = "local",
    name: str | None = None,
) -> None:
    ...
    # Pour chaque output
    for key, var_name in final_output_mapping.items():
        is_tendency_of = output_tendencies.get(key)
        units = output_units.get(key) if output_units else None  # NOUVEAU

        output_node = DataNode(
            name=var_name,
            is_tendency_of=is_tendency_of,
            units=units  # NOUVEAU
        )
    ...
```

### Correction compute_production_initialization

**Option 1: Documenter que NPP doit être en /s**
```python
def compute_production_initialization(
    primary_production: xr.DataArray,  # [gC/m²/s]
    cohorts: xr.DataArray,
    E: float,  # dimensionless
) -> dict[str, xr.DataArray]:
    """
    Returns:
        Tendency in [gC/m²/s] for cohort 0
    """
    ...
```

**Option 2: Détecter et convertir automatiquement**
```python
def compute_production_initialization(...):
    # Vérifier unités de primary_production
    npp_units = primary_production.attrs.get('units', None)

    if npp_units and 'day' in npp_units:
        # Convertir jour → seconde
        tendency_rate = E * primary_production / 86400.0
    else:
        tendency_rate = E * primary_production
    ...
```

**Choix recommandé: Option 1** (plus explicite, moins de magie)
- Le Blueprint doit spécifier units='gC/m²/s' pour primary_production
- Le Controller convertira automatiquement si nécessaire

### Mise à Jour Notebook

```python
# Avant
temperature = xr.DataArray(temp_data, coords=..., dims=..., name="temperature")

# Après
temperature = xr.DataArray(
    temp_data,
    coords=...,
    dims=...,
    name="temperature",
    attrs={'units': 'degC'}  # AJOUTER
)

# Et dans configure_model
bp.register_forcing(
    "temperature",
    dims=(...),
    units="degC"  # AJOUTER
)
bp.register_forcing(
    "primary_production",
    dims=(...),
    units="gC/m²/day"  # AJOUTER (sera converti auto en /s)
)
```

## Plan d'Exécution

1. ✅ Créer documentation conventions (ce fichier)
2. Modifier `Blueprint.register_unit()` pour accepter `output_units`
3. Modifier `Blueprint.register_group()` pour passer `output_units` aux unités
4. Documenter les unités dans toutes les fonctions LMTL (docstrings)
5. Mettre à jour le notebook pour utiliser les unités
6. Ajouter tests de validation d'unités
7. Exécuter le notebook et vérifier les résultats
8. Corriger si nécessaire

## Résultat Attendu

- Les unités sont clairement documentées partout
- Le Blueprint peut spécifier les unités attendues pour tous les inputs/outputs
- Le Controller convertit automatiquement vers les unités SI
- Le notebook converge vers des valeurs réalistes (biomasse ~O(1-100), pas 1e152)
- Les tests valident la cohérence dimensionnelle
