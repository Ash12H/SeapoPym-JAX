# Memory Writer Coordinates Fix

## Problème

Le `MemoryWriter` reconstruit un `xarray.Dataset` à partir de données accumulées, mais :
- Les dimensions blueprint n'incluent pas `T` (state instantané)
- Les données accumulées ont shape `(n_steps, ...)` avec dimension temporelle
- Les coordonnées `T` de `model.coords` représentent TOUTE la simulation, pas les timesteps réellement exportés
- Incohérence lors de la création des DataArrays

## Solution

**Passer les coordonnées réelles au Writer** lors de l'initialisation :
- Le `StreamingRunner` accumule les timestamps pendant l'exécution
- Il passe les coords réelles (spatiales + temporelles) au Writer
- Le `MemoryWriter` utilise ces coords pour construire le Dataset
- Application de l'ordre canonique des dimensions (`E, T, F, C, Z, Y, X`)

## Plan d'implémentation

### 1. Modifier le Protocol `OutputWriter`
- Ajouter paramètre optionnel `coords: dict[str, Array] | None` à `initialize()`

### 2. Modifier `StreamingRunner.run()`
- Accumuler les timestamps réels pendant la boucle de chunks
- Préparer un dict de coords (spatiales depuis `model.coords` + temporelles accumulées)
- Passer ce dict au `writer.initialize()`

### 3. Modifier `MemoryWriter`
- Stocker les coords passés dans `initialize()`
- Dans `finalize()` :
  - Résoudre les dimensions depuis le graph
  - Ajouter `T` si nécessaire
  - Appliquer `get_canonical_order()` pour respecter l'ordre canonique
  - Utiliser les coords stockés pour créer le Dataset

### 4. Modifier `DiskWriter`
- Accepter le paramètre `coords` dans `initialize()` (pour compatibilité Protocol)
- Non utilisé pour l'instant (peut servir pour métadonnées futures)

### 5. Test avec `full_model_0d.py`
- Exécuter l'exemple avec export en mémoire
- Vérifier les dimensions et coordonnées du Dataset retourné

## Notes techniques

- **Ordre canonique** : `("E", "T", "F", "C", "Z", "Y", "X")` défini dans `model.py`
- **Utilitaire** : `get_canonical_order(dims)` dans `transpose.py`
- **time_grid** : Toujours présent, contient `.coords` avec vraies dates (datetime64)
- **Coords spatiales** : Déjà extraites des forcings lors de la compilation
