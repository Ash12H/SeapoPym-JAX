# SPEC 02 : Compilateur

**Version** : 1.0
**Date** : 2026-01-25
**Statut** : Validé

---

## 1. Vue d'ensemble

Le Compilateur (Linker) transforme le graphe déclaratif (Blueprint + Config) en structures de données prêtes pour l'exécution JAX. Il gère l'alignement mémoire, dimensionnel et le prétraitement des données.

### 1.1 Objectifs

- Résoudre les dimensions sémantiques en tailles concrètes
- Transposer les données dans l'ordre canonique
- Préparer les arrays NumPy/JAX (stripping Xarray)
- Gérer les NaN et produire les masques

### 1.2 Dépendances

- **Amont** : Axe 1 (Blueprint) fournit le graphe validé + Config
- **Aval** : Axe 3 (Engine) consomme les arrays préparés

---

## 2. Pipeline de Compilation

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT : Blueprint validé + Config                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ ÉTAPE 1 : INFÉRENCE DE SHAPE                                    │
│ - Charge les métadonnées (lazy) des fichiers                    │
│ - Résout les dimensions sémantiques → tailles entières          │
│ - Fige les static_argnums pour jax.jit                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ ÉTAPE 2 : RENOMMAGE DES DIMENSIONS                              │
│ - Applique le mapping utilisateur (si fourni)                   │
│ - Convertit vers les noms canoniques (T, E, C, Z, Y, X)         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ ÉTAPE 3 : TRANSPOSITION CANONIQUE                               │
│ - Transpose tous les DataArrays vers l'ordre (E, T, C, Z, Y, X) │
│ - Garantit la contiguïté mémoire (C-order)                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ ÉTAPE 4 : STRIPPING & PRÉTRAITEMENT                             │
│ - xarray.DataArray → numpy.ndarray / jax.Array                  │
│ - Remplacement des NaN par 0.0                                  │
│ - Génération du masque binaire                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ ÉTAPE 5 : EMPAQUETAGE                                           │
│ - Construction des pytrees JAX                                  │
│ - Séparation State / Forcings / Parameters                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT : CompiledModel (pytrees prêts pour le Runner)           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Inférence de Shape

### 3.1 Mécanisme

L'inférence n'est pas "devinée" mais **lue** depuis les métadonnées.

```python
import xarray as xr

def infer_shapes(config: Config) -> dict[str, int]:
    """Lit les tailles des dimensions depuis les fichiers."""
    shapes = {}

    # Lecture lazy des métadonnées
    for name, path in config.forcings.items():
        ds = xr.open_dataset(path, chunks={})  # Lazy
        for dim, size in ds.sizes.items():
            if dim in shapes and shapes[dim] != size:
                raise GridAlignmentError(f"Dimension {dim} incohérente")
            shapes[dim] = size

    return shapes
```

### 3.2 Dimensions Statiques vs Temps

| Type | Comportement | Exemple |
|------|--------------|---------|
| **Statique** | Figée à la compilation | `Y=180`, `X=360`, `C=10` |
| **Temps** | Chunké par le Runner | `T=365` par chunk |

Le temps n'est pas "infini" en V1 (offline uniquement). Le Chunked Runner (Axe 3) découpe la dimension temporelle en chunks de taille fixe.

---

## 4. Layout Canonique

### 4.1 Ordre des Dimensions

L'ordre canonique interne est :

```
(E, T, C, Z, Y, X)
```

| Position | Dimension | Rationale |
|----------|-----------|-----------|
| 0 | `E` (ensemble) | Axe de batch, parallélisable via vmap |
| 1 | `T` (time) | Axe de scan, itéré séquentiellement |
| 2 | `C` (cohort) | Axe biologique, core_dim fréquent |
| 3 | `Z` (depth) | Axe vertical |
| 4 | `Y` (latitude) | Axe spatial |
| 5 | `X` (longitude) | Axe spatial (dernier = stride 1) |

### 4.2 Transposition Automatique

```python
def transpose_canonical(da: xr.DataArray, target_dims: list[str]) -> xr.DataArray:
    """Transpose un DataArray vers l'ordre canonique."""
    # Filtre les dimensions présentes
    present_dims = [d for d in target_dims if d in da.dims]
    return da.transpose(*present_dims)
```

### 4.3 Contiguïté Mémoire

La transposition garantit que les arrays sont **C-contiguous** (row-major), optimisant :
- L'accès mémoire séquentiel sur GPU
- La coalescence des lectures VRAM
- La compatibilité avec `jax.jit`

---

## 5. Stripping & Prétraitement

### 5.1 Conversion Xarray → Arrays

```python
def strip_xarray(ds: xr.Dataset, backend: str) -> dict[str, Array]:
    """Convertit un Dataset en dictionnaire d'arrays."""
    arrays = {}
    for name, da in ds.data_vars.items():
        # Transpose
        da = transpose_canonical(da, CANONICAL_DIMS)
        # Extrait les valeurs
        values = da.values  # numpy.ndarray
        if backend == "jax":
            values = jnp.asarray(values)
        arrays[name] = values
    return arrays
```

### 5.2 Gestion des NaN

JAX ne tolère pas les NaN de manière fiable (comportement indéfini dans certaines opérations).

**Stratégie** :

```python
def preprocess_nan(data: Array, fill_value: float = 0.0) -> tuple[Array, Array]:
    """Remplace les NaN et génère un masque."""
    mask = ~jnp.isnan(data)
    data_clean = jnp.where(mask, data, fill_value)
    return data_clean, mask
```

### 5.3 Masque Explicite

Le masque est traité comme un input standard du modèle :

```yaml
# Dans la Config
forcings:
  temperature: "/data/sst.nc"
  mask: "/data/land_mask.nc"  # Fourni par l'utilisateur
```

**Formes supportées** :
- Statique : `(Y, X)` — masque terre/mer fixe
- Dynamique : `(T, Y, X)` — masque variable (ex: glace de mer)

Le broadcasting gère automatiquement l'extension aux autres dimensions.

---

## 6. Empaquetage en Pytrees

### 6.1 Structure de Sortie

Le Compilateur produit des **pytrees** JAX organisés par catégorie :

```python
@dataclass
class CompiledModel:
    # Graphe de fonctions résolu
    graph: ProcessGraph

    # Données empaquetées
    state: dict[str, Array]       # Variables évolutives (biomass, ...)
    forcings: dict[str, Array]    # Données d'entrée (temperature, ...)
    parameters: dict[str, Array]  # Constantes (growth_rate, ...)
    mask: Array                   # Masque binaire

    # Métadonnées
    shapes: dict[str, int]        # Tailles des dimensions
    coords: dict[str, Array]      # Coordonnées (lat, lon, time, ...)
    dt: float                     # Pas de temps en secondes

    # Configuration
    backend: str                  # "jax" ou "numpy"
    trainable_params: list[str]   # Paramètres optimisables (Axe 5)
```

### 6.2 Séparation des Rôles

| Catégorie | Mutabilité | Exemple |
|-----------|------------|---------|
| `state` | Évolue à chaque pas | `biomass`, `concentration` |
| `forcings` | Lu depuis les données | `temperature`, `current_u` |
| `parameters` | Constant pendant le run | `growth_rate`, `mortality` |
| `mask` | Constant | `land_mask` |

---

## 7. Renommage des Dimensions

### 7.1 Mapping Utilisateur

L'utilisateur peut fournir un dictionnaire de correspondance :

```yaml
# Dans la Config
dimension_mapping:
  lat: "Y"
  lon: "X"
  time: "T"
  age_class: "C"
  member: "E"
```

### 7.2 Application

```python
def apply_dimension_mapping(ds: xr.Dataset, mapping: dict[str, str]) -> xr.Dataset:
    """Renomme les dimensions selon le mapping utilisateur."""
    rename_dict = {old: new for old, new in mapping.items() if old in ds.dims}
    return ds.rename(rename_dict)
```

---

## 8. Interfaces

### 8.1 API Python

```python
from seapopym.compiler import Compiler

# Configuration du compilateur
compiler = Compiler(
    backend="jax",
    canonical_order=["E", "T", "C", "Z", "Y", "X"],
    fill_nan=0.0
)

# Compilation
compiled_model = compiler.compile(blueprint, config)

# Accès aux données préparées
state = compiled_model.state
forcings = compiled_model.forcings
```

### 8.2 Méthodes Principales

| Méthode | Description |
|---------|-------------|
| `compile(blueprint, config)` | Pipeline complet → `CompiledModel` |
| `infer_shapes(config)` | Lecture des métadonnées → `dict[str, int]` |
| `prepare_arrays(dataset)` | Transpose + strip + preprocess |

---

## 9. Gestion des Erreurs

| Code | Type | Description |
|------|------|-------------|
| `E201` | `ShapeInferenceError` | Impossible de lire les métadonnées |
| `E202` | `GridAlignmentError` | Dimensions incohérentes entre fichiers |
| `E203` | `MissingDimensionError` | Dimension requise absente des données |
| `E204` | `TransposeError` | Échec de transposition |

---

## 10. Performance

### 10.1 Chargement Lazy

Le Compilateur utilise Xarray avec `chunks={}` pour ne charger que les métadonnées :

```python
ds = xr.open_dataset(path, chunks={})  # Lazy, pas de lecture mémoire
shapes = ds.sizes  # Accès aux tailles sans charger les données
```

Les données ne sont effectivement chargées que lors de l'appel `.values` ou `.compute()`.

### 10.2 Formats Supportés

| Format | Librairie | Lazy Load |
|--------|-----------|-----------|
| NetCDF4 | `netcdf4` | Oui |
| Zarr | `zarr` | Oui (recommandé pour gros volumes) |
| GRIB | `cfgrib` | Partiel |

---

## 11. Liens avec les Autres Axes

| Axe | Interaction |
|-----|-------------|
| Axe 1 (Blueprint) | Reçoit `Blueprint` + `Config` validés |
| Axe 3 (Engine) | Fournit `CompiledModel` avec pytrees prêts |
| Axe 4 (Parallelism) | L'ordre canonique facilite le sharding batch (E) |
| Axe 5 (Auto-Diff) | Identifie les `trainable_params` dans le pytree |

---

## 12. Questions Ouvertes (V2+)

- Support du chargement incrémental (streaming Zarr chunk par chunk)
- Compression/décompression à la volée
- Cache de compilation pour éviter re-transposition
