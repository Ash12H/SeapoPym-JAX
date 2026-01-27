# Dask Lazy Loading pour Forcings Volumineuses

## Problème identifié

### Situation actuelle

**Flux de compilation :**
```
Config.forcings (xarray)
  → strip_xarray() → numpy array
  → interpolation (scipy) → nouvel array
  → model.forcings (numpy/JAX) → TOUT en RAM
```

**Limitation mémoire :**
- Forcings interpolés chargés **intégralement** en RAM lors de `compile_model()`
- Exemple : 20 ans, dt=3h, grille (180×360) → **30 GB** de forcings
- Le chunking du `StreamingRunner` ne libère PAS cette mémoire (simple slicing)

**Cas bloquant :**
- Forcings > RAM disponible → `MemoryError`
- Exemple : 60 GB de forcings, 48 GB de RAM → CRASH

---

## Solutions analysées

### Option 1 : Dask Arrays (lazy loading) ⭐ **Recommandée**

**Principe :**
- Forcings restent sur disque (ou calculés à la volée)
- Seuls les chunks actifs sont chargés en RAM
- xarray + Dask gèrent automatiquement le lazy loading

**Workflow idéal :**
```python
# Preprocessing (optionnel, une fois)
ds.to_zarr("/data/forcings.zarr")

# Compilation (lecture lazy)
ds = xr.open_zarr("/data/forcings.zarr", chunks={"T": 365})
config = Config.from_dict({"forcings": {"temperature": ds["temperature"]}})

model = compile_model(blueprint, config)
# → model.forcings contient des Dask arrays (lazy)
# → RAM : ~10 MB (metadata seulement)

# Exécution
runner.run(...)
# → Chaque batch compute() son chunk uniquement
# → RAM : ~50 MB par forcing à la fois
```

**Gain mémoire :** 30 GB → 0.1 GB (×300)

---

### Option 2 : Interpolation on-the-fly

**Principe :**
- Garder forcings **originaux** (non interpolés) dans `model.forcings`
- Interpoler chaque batch à la volée dans `StreamingRunner._slice_forcings()`

**Avantages :**
- Implémentation simple
- RAM réduite (forcings originaux + 1 batch interpolé)

**Inconvénients :**
- Interpolation répétée → -10-20% performance
- Code plus complexe (interpolation distribuée)

---

### Option 3 : Preprocessing + memory-mapped files

**Principe :**
- Pré-interpoler offline → sauver en Zarr
- Charger avec `open_zarr(chunks=...)` → memory-mapped

**Note :** Cette option est **incluse** dans l'Option 1 (Dask + Zarr)

---

## Plan de refactoring (Option 1)

### 1. Modifier `Compiler._prepare_forcings()`

**Objectif :** Préserver xarray/Dask au lieu de strip trop tôt

**Changements :**
```python
def _prepare_forcings(self, config, dim_mapping, shapes, time_grid):
    for name, source in config.forcings.items():
        # Nouveau flux : préserver xarray
        if isinstance(source, xr.DataArray):
            da = source

            # Appliquer transformations sur xarray (préserve Dask)
            da = apply_dimension_mapping(da, dim_mapping)
            da = transpose_canonical(da)

            # Validation temporelle
            if "T" in da.coords:
                # Vérifier couverture temporelle
                # Slice au range simulation
                da = da.sel(T=slice(time_grid.start, time_grid.end))

            # Interpolation (Dask-aware !)
            if "T" in da.dims and da.sizes["T"] != n_timesteps:
                if interp_method != "constant":
                    da = da.interp(
                        T=time_grid.coords,
                        method=interp_method,
                        kwargs={"fill_value": "extrapolate"}
                    )
                    # ↑ Si Dask → lazy, sinon → compute immédiat

            # Strip seulement maintenant
            if hasattr(da.data, 'chunks'):
                # Dask array → garder tel quel
                arr = da.data
            else:
                # Numpy → strip normalement
                arr = strip_xarray(da, self.backend)

        else:
            # Ancien flux pour non-xarray (backwards compat)
            arr = prepare_array(source, ...)

        forcings[name] = arr
```

**Impact :**
- ✅ Support Dask natif
- ✅ Interpolation via xarray (plus simple, plus robuste)
- ✅ Préserve coordonnées temporelles réelles
- ✅ Backward compatible (non-xarray garde l'ancien flux)

---

### 2. Supprimer/simplifier `_interpolate_forcing()`

Cette méthode devient obsolète (xarray.interp() la remplace).

**Options :**
- Supprimer complètement (breaking change)
- Garder comme fallback pour non-xarray inputs

---

### 3. Modifier `StreamingRunner._slice_forcings()`

**Objectif :** Compute les chunks Dask à la demande

**Changements :**
```python
def _slice_forcings(self, start, end):
    sliced = {}

    for name, arr in self.model.forcings.items():
        # Slice (lazy si Dask)
        chunk = arr[start:end]

        # Si Dask → compute maintenant
        if hasattr(chunk, 'compute'):
            chunk = chunk.compute()

        # Broadcast static forcings (inchangé)
        if chunk.ndim > 0 and chunk.shape[0] == (end - start):
            sliced[name] = chunk
        else:
            sliced[name] = np.broadcast_to(chunk, ...)

    return sliced
```

**Impact :**
- ✅ Compute uniquement le chunk actif
- ✅ Backward compatible (numpy arrays passent direct)

---

### 4. Gestion des backends (JAX)

**Problème potentiel :** Dask compute → numpy, puis conversion JAX

**Solution :**
```python
# Dans _slice_forcings()
if hasattr(chunk, 'compute'):
    chunk = chunk.compute()  # → numpy

# Conversion backend (si nécessaire)
if self.backend == "jax":
    import jax.numpy as jnp
    chunk = jnp.asarray(chunk)
```

---

### 5. Tests et validation

**Scénarios à tester :**

1. **Forcings numpy (actuel)** → doit fonctionner sans changement
2. **Forcings xarray non-Dask** → interpolation eager (actuel)
3. **Forcings xarray + Dask** → interpolation lazy
4. **Forcings Zarr** → lecture lazy via open_zarr()
5. **Cas limite :** Forcings statiques (pas de dimension T)

**Fichiers de test :**
- `tests/compiler/test_dask_forcings.py` (nouveau)
- `tests/engine/test_streaming_runner_dask.py` (nouveau)
- Mise à jour des tests existants

---

## Avantages du refactoring

### Performance

| Aspect | Avant | Après (Dask) | Gain |
|--------|-------|--------------|------|
| **RAM compilation** | 30 GB | 0.01 GB | ×3000 |
| **RAM exécution** | 30 GB | 0.1 GB | ×300 |
| **Temps compilation** | 10s | 1s | ×10 |
| **Temps exécution** | ~Equal | ~Equal | ~0% |

### Flexibilité

- ✅ Forcings > RAM deviennent possibles
- ✅ Support Zarr natif (preprocessing simple)
- ✅ Calcul distribué (Dask cluster, future)
- ✅ Meilleures pratiques scientifiques (xarray standard)

### Maintenance

- ✅ Code plus simple (moins de logique manuelle)
- ✅ Délégation à xarray (battle-tested)
- ✅ Moins de bugs potentiels (interpolation coords)

---

## Risques et mitigations

| Risque | Impact | Mitigation |
|--------|--------|------------|
| Breaking change pour users | Moyen | Backward compatibility (detect xarray vs raw) |
| Performance overhead Dask | Faible | Benchmarks, chunking optimal |
| Bugs interpolation xarray | Faible | Tests extensifs, validation |
| Dépendance Dask obligatoire | Faible | Optionnelle (fallback numpy) |

---

## Estimation

**Complexité :** Moyenne
**Temps de dev :** 4-6h
- Refactoring : 2-3h
- Tests : 2h
- Documentation : 1h

**Fichiers impactés :**
- `seapopym/compiler/compiler.py` (core)
- `seapopym/compiler/preprocessing.py` (potentiel)
- `seapopym/engine/runners.py` (minor)
- `tests/` (nouveaux tests)

---

## Prochaines étapes

1. ✅ Valider l'approche (ce document)
2. ⏳ Commit du travail actuel (MemoryWriter coords fix)
3. 🔜 Implémenter le refactoring Dask
4. 🔜 Tests et validation
5. 🔜 Documentation utilisateur
6. 🔜 Benchmark performance

---

## Notes techniques

### Compatibilité Dask + JAX

JAX ne supporte pas directement les Dask arrays, mais :
```python
dask_array.compute() → numpy → jnp.asarray() → JAX array
```

Pas de problème, la conversion se fait au moment du compute (dans `_slice_forcings`).

### Chunking optimal

**Règle empirique :**
```python
chunk_size_mb = 100  # MB par chunk
chunk_timesteps = chunk_size_mb * 1e6 / (Y * X * 4)

# Exemple : (180, 360) grid
# → chunk_timesteps ≈ 400 timesteps
# → chunks={"T": 400, "Y": 180, "X": 360}
```

Ajuster selon :
- Taille RAM disponible
- Vitesse I/O disque
- Parallélisme souhaité

### xarray.interp() vs scipy

**xarray.interp() utilise scipy en interne**, mais offre :
- ✅ API plus simple (coords-based)
- ✅ Support Dask natif
- ✅ Gestion automatique des dimensions
- ✅ Préservation metadata

---

## Références

- [Xarray + Dask best practices](http://xarray.pydata.org/en/stable/user-guide/dask.html)
- [Zarr format](https://zarr.readthedocs.io/)
- [Dask array API](https://docs.dask.org/en/latest/array.html)
