# Xarray Support & Unit Aliasing

## Amélioration 1 : Support Xarray dans les Sensing Units

### Problème
Les Sensing Units (`extract_layer`, `diel_migration`) utilisaient la sélection par index (`forcing_3d[layer_index]`), ce qui est fragile car cela suppose un ordre de dimensions précis.

### Solution
Utilisation de `xarray.DataArray` pour :
- **Sélection robuste** par nom de dimension (pas d'index)
```python
forcing_nd.isel(depth=layer_index)  # Au lieu de forcing_nd[layer_index]
```

- **Auto-détection** des noms de dimension (depth, Z, z, lev, level)
- **Préservation des métadonnées** (coords, dims, attrs)

### Architecture
1. **ForcingManager** : Nouvelle méthode `prepare_timestep_xarray()` qui retourne `dict[str, xr.DataArray]`
2. **Sensing Units** : Déclarées avec `compiled=False` pour accepter xarray
3. **Conversion** : Units biologiques reçoivent toujours `jnp.ndarray` (via `prepare_timestep()`)

### Exemple
```python
# Forçage 3D avec métadonnées
temp_3d = xr.DataArray(
    data,
    dims=("depth", "lat", "lon"),
    coords={"depth": [0, 50, 100], ...}
)

# Extraction robuste (auto-détecte 'depth')
result = extract_layer.execute(
    state={},
    forcings={"forcing_nd": temp_3d},
    params={"layer_index": 0}
)
# result.dims == ("lat", "lon")
```

---

## Amélioration 2 : Système d'Alias (UnitInstance)

### Problème
Impossible de réutiliser la même `Unit` plusieurs fois dans un `FunctionalGroup` (conflit de noms).

### Solution
Classe `UnitInstance` qui encapsule une `Unit` avec :
- **Alias unique** : Nom spécifique à cette instance
- **Paramètres locaux** : Paramètres spécifiques à l'instance

### Utilisation
```python
from seapopym_message.core.group import UnitInstance

FunctionalGroup(
    name="tuna",
    units=[
        UnitInstance(extract_layer, alias="extract_temp"),
        UnitInstance(extract_layer, alias="extract_salinity"),
    ],
    variable_map={
        # Temperature
        "extract_temp.forcing_nd": "forcing/temperature_3d",
        "extract_temp.forcing_2d": "tuna/temperature",

        # Salinity
        "extract_salinity.forcing_nd": "forcing/salinity_3d",
        "extract_salinity.forcing_2d": "tuna/salinity",
    }
)
```

### Kernel Compilation
Le `Kernel` :
1. Détecte `UnitInstance` vs `Unit`
2. Utilise l'alias pour le mapping : `"alias.var_name"` → global name
3. Renomme l'unité liée : `"{group}/{alias}"`

### Résultat
Deux Units distinctes dans le Kernel :
- `tuna/extract_temp` : lit `forcing/temperature_3d` → écrit `tuna/temperature`
- `tuna/extract_salinity` : lit `forcing/salinity_3d` → écrit `tuna/salinity`

---

## Impact sur l'Architecture

### Rétrocompatibilité
- ✅ `prepare_timestep()` existe toujours (retourne `jnp.ndarray`)
- ✅ `FunctionalGroup` accepte `Unit` ou `UnitInstance`
- ✅ Units existantes non affectées

### Performances
- Sensing Units : `compiled=False` (pas de JIT), mais pas critique (calculs simples)
- Units biologiques : toujours JIT'd avec `jnp.ndarray`

### Tests
- `tests/unit/core/test_xarray_and_aliasing.py` : Validation complète

---

## Prochaines Étapes

Cette architecture permet maintenant de créer des groupes très flexibles :

```python
# Groupe avec perception multi-forcing
epipelagic = FunctionalGroup(
    name="epi",
    units=[
        UnitInstance(extract_layer, alias="get_temp"),
        UnitInstance(extract_layer, alias="get_sal"),
        compute_physiology_unit,  # Utilise temp ET sal
        compute_growth,
    ],
    variable_map={
        "get_temp.forcing_nd": "env/temp_3d",
        "get_temp.forcing_2d": "epi/temp",
        "get_sal.forcing_nd": "env/sal_3d",
        "get_sal.forcing_2d": "epi/sal",
        "temperature": "epi/temp",  # physiology unit
        "salinity": "epi/sal",      # physiology unit
    }
)
```
