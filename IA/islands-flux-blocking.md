# Plan Phase 11 : Gestion Robuste des Îles (Flux Blocking)

## Objectif

Améliorer la gestion des îles pour garantir qu'**aucun flux de biomasse ne traverse les frontières land/ocean**, que ce soit par advection ou diffusion.

**Méthode** : Bloquer physiquement les flux aux interfaces plutôt que forcer biomass=0 après calcul.

---

## Décisions de Design

### Masque
- **Format** : Tableau booléen `mask: jnp.ndarray` de shape `(nlat, nlon)`
- **Convention** : `True` = ocean, `False` = land
- **Passage** : Via `params["mask"]` (optionnel)

### Flux Blocking
- **Principe** : Un flux entre deux cellules n'existe que si **les deux cellules sont ocean**
- **Implémentation** : Masques d'interface via opération `&` (AND logique)

### Neumann BC pour Diffusion
- **Principe** : Si voisin est land → gradient normal = 0
- **Implémentation** : Utiliser valeur de la cellule center au lieu du voisin land

---

## Phase 11.1 : Modifier `compute_advection_2d` (~1h)

### Fichier
`src/seapopym_message/kernels/transport.py:165-330`

### Modifications

#### 1. Créer masques étendus (après ligne 252)

```python
# Build extended arrays with halos
biomass_ext = jnp.zeros((nlat + 2, nlon + 2))
u_ext = jnp.zeros((nlat + 2, nlon + 2))
v_ext = jnp.zeros((nlat + 2, nlon + 2))

# NOUVEAU : masque étendu (True = ocean, False = land)
if "mask" in params:
    mask = params["mask"]
    mask_ext = jnp.zeros((nlat + 2, nlon + 2), dtype=bool)
    mask_ext = mask_ext.at[1:-1, 1:-1].set(mask)

    # Remplir halos du masque (land par défaut aux frontières)
    # Ou utiliser halos si fournis
    if halo_north is not None and "mask" in halo_north:
        mask_ext = mask_ext.at[0, 1:-1].set(halo_north["mask"])
    if halo_south is not None and "mask" in halo_south:
        mask_ext = mask_ext.at[-1, 1:-1].set(halo_south["mask"])
    if halo_west is not None and "mask" in halo_west:
        mask_ext = mask_ext.at[1:-1, 0].set(halo_west["mask"])
    if halo_east is not None and "mask" in halo_east:
        mask_ext = mask_ext.at[1:-1, -1].set(halo_east["mask"])

    # Extraire masques center et voisins
    mask_c = mask_ext[1:-1, 1:-1]
    mask_north = mask_ext[:-2, 1:-1]
    mask_south = mask_ext[2:, 1:-1]
    mask_west = mask_ext[1:-1, :-2]
    mask_east = mask_ext[1:-1, 2:]
else:
    # Pas de masque : tout est ocean
    mask_c = jnp.ones((nlat, nlon), dtype=bool)
    mask_north = jnp.ones((nlat, nlon), dtype=bool)
    mask_south = jnp.ones((nlat, nlon), dtype=bool)
    mask_west = jnp.ones((nlat, nlon), dtype=bool)
    mask_east = jnp.ones((nlat, nlon), dtype=bool)
```

#### 2. Bloquer flux aux interfaces (remplacer lignes 297-313)

```python
# Upwind scheme pour x-direction
dB_dx_upwind_raw = jnp.where(
    u_c >= 0,
    (B_c - B_west) / dx,
    (B_east - B_c) / dx,
)

# Upwind scheme pour y-direction
dB_dy_upwind_raw = jnp.where(
    v_c >= 0,
    (B_c - B_north) / dy,
    (B_south - B_c) / dy,
)

# NOUVEAU : Bloquer flux aux interfaces land/ocean
# Flux x bloqué si cellule actuelle OU voisin approprié est land
mask_flux_x = jnp.where(
    u_c >= 0,
    mask_c & mask_west,   # Flux depuis ouest : les deux doivent être ocean
    mask_c & mask_east,   # Flux depuis est : les deux doivent être ocean
)

mask_flux_y = jnp.where(
    v_c >= 0,
    mask_c & mask_north,  # Flux depuis nord
    mask_c & mask_south,  # Flux depuis sud
)

# Appliquer masques de flux
dB_dx_upwind = jnp.where(mask_flux_x, dB_dx_upwind_raw, 0.0)
dB_dy_upwind = jnp.where(mask_flux_y, dB_dy_upwind_raw, 0.0)
```

#### 3. Simplifier la fin (remplacer lignes 315-328)

```python
# Advection equation: ∂B/∂t = -u*∂B/∂x - v*∂B/∂y
dB_dt = -u_c * dB_dx_upwind - v_c * dB_dy_upwind

# Forward Euler integration
biomass_new = biomass + dB_dt * dt

# Forcer land cells à 0 (sécurité supplémentaire)
if "mask" in params:
    biomass_new = jnp.where(mask, biomass_new, 0.0)

# Ensure non-negative
biomass_new = jnp.maximum(biomass_new, 0.0)

return biomass_new
```

---

## Phase 11.2 : Modifier `compute_diffusion_2d` (~1h)

### Fichier
`src/seapopym_message/kernels/transport.py:15-128`

### Modifications

#### 1. Ajouter paramètre masque dans docstring (ligne 43-53)

```python
Args:
    biomass: Current biomass distribution (nlat, nlon).
    dt: Time step.
    params: Dictionary containing:
        - 'D': Diffusion coefficient [m²/s]
        - 'dx': Grid spacing in x-direction [m]
        - 'dy': Grid spacing in y-direction [m] (optional, defaults to dx)
        - 'mask': Optional boolean array (nlat, nlon) where True = ocean, False = land
    halo_north: Boundary data from northern neighbor {'biomass': array, 'mask': array}
    ...
```

#### 2. Créer masque étendu (après ligne 75)

```python
# Build extended array with halos
extended = jnp.zeros((nlat + 2, nlon + 2))
extended = extended.at[1:-1, 1:-1].set(biomass)

# NOUVEAU : Masque étendu
if "mask" in params:
    mask = params["mask"]
    mask_ext = jnp.zeros((nlat + 2, nlon + 2), dtype=bool)
    mask_ext = mask_ext.at[1:-1, 1:-1].set(mask)

    # Remplir halos du masque
    if halo_north is not None and "mask" in halo_north:
        mask_ext = mask_ext.at[0, 1:-1].set(halo_north["mask"])
    else:
        mask_ext = mask_ext.at[0, 1:-1].set(False)  # Land par défaut

    if halo_south is not None and "mask" in halo_south:
        mask_ext = mask_ext.at[-1, 1:-1].set(halo_south["mask"])
    else:
        mask_ext = mask_ext.at[-1, 1:-1].set(False)

    if halo_west is not None and "mask" in halo_west:
        mask_ext = mask_ext.at[1:-1, 0].set(halo_west["mask"])
    else:
        mask_ext = mask_ext.at[1:-1, 0].set(False)

    if halo_east is not None and "mask" in halo_east:
        mask_ext = mask_ext.at[1:-1, -1].set(halo_east["mask"])
    else:
        mask_ext = mask_ext.at[1:-1, -1].set(False)
else:
    mask_ext = jnp.ones((nlat + 2, nlon + 2), dtype=bool)
```

#### 3. Appliquer Neumann BC sur land (remplacer lignes 109-119)

```python
# Compute Laplacian using 5-point stencil
center = extended[1:-1, 1:-1]
north = extended[:-2, 1:-1]
south = extended[2:, 1:-1]
west = extended[1:-1, :-2]
east = extended[1:-1, 2:]

# NOUVEAU : Neumann BC si voisin est land
# Si voisin est land, utiliser valeur center (gradient = 0)
mask_center = mask_ext[1:-1, 1:-1]
mask_north_cells = mask_ext[:-2, 1:-1]
mask_south_cells = mask_ext[2:, 1:-1]
mask_west_cells = mask_ext[1:-1, :-2]
mask_east_cells = mask_ext[1:-1, 2:]

north_effective = jnp.where(mask_north_cells, north, center)
south_effective = jnp.where(mask_south_cells, south, center)
west_effective = jnp.where(mask_west_cells, west, center)
east_effective = jnp.where(mask_east_cells, east, center)

laplacian = (
    (north_effective + south_effective) / dy**2
    + (west_effective + east_effective) / dx**2
    - center * 2 * (1 / dx**2 + 1 / dy**2)
)
```

#### 4. Masquer résultat final (après ligne 122)

```python
# Forward Euler integration
biomass_new = biomass + D * laplacian * dt

# Forcer land cells à 0
if "mask" in params:
    biomass_new = jnp.where(mask, biomass_new, 0.0)

# Ensure non-negative
biomass_new = jnp.maximum(biomass_new, 0.0)

return biomass_new
```

---

## Phase 11.3 : Tests Unitaires (~1h)

### Fichier
`tests/unit/test_transport.py`

### Nouveaux Tests à Ajouter

#### Test 1 : Advection - Aucun flux à travers île

```python
def test_advection_no_flux_through_island():
    """Vérifier qu'aucun flux ne traverse une île même si u≠0."""
    # Configuration : île verticale au centre, flux est vers ouest
    biomass = jnp.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 100.0, 0.0, 0.0, 0.0],  # Biomasse à gauche de l'île
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])

    # Île au centre (colonne j=2)
    mask = jnp.ones((3, 5), dtype=bool)
    mask = mask.at[:, 2].set(False)

    # Flux uniforme vers l'est (traverse l'île)
    u = jnp.ones((3, 5)) * 1.0
    v = jnp.zeros((3, 5))

    params = {"dx": 1000.0, "dy": 1000.0, "mask": mask}
    forcings = {"u": u, "v": v}
    dt = 500.0

    state = {"biomass": biomass}

    # Exécuter plusieurs steps
    for _ in range(10):
        state = compute_advection_simple.execute(state, dt=dt, params=params, forcings=forcings)

    # Vérifier qu'aucune biomasse n'est passée à droite de l'île
    assert jnp.all(state["biomass"][:, 3:] == 0.0)
    # L'île reste à 0
    assert jnp.all(state["biomass"][:, 2] == 0.0)
```

#### Test 2 : Diffusion - Pas de diffusion à travers île

```python
def test_diffusion_no_flux_through_island():
    """Vérifier que la diffusion ne traverse pas une île."""
    # Blob à gauche, île au centre
    biomass = jnp.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 100.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])

    # Île au centre
    mask = jnp.ones((3, 5), dtype=bool)
    mask = mask.at[:, 2].set(False)

    params = {"D": 1000.0, "dx": 1000.0, "dy": 1000.0, "mask": mask}
    dt = 10.0

    state = {"biomass": biomass}

    # Diffuser
    for _ in range(100):
        state = compute_diffusion_simple.execute(state, dt=dt, params=params)

    # Aucune biomasse à droite de l'île
    assert jnp.all(state["biomass"][:, 3:] < 1e-10)
    # Biomasse s'est diffusée à gauche
    assert state["biomass"][1, 0] > 10.0
```

#### Test 3 : Île complexe (archipel)

```python
def test_complex_island_pattern():
    """Test avec archipel complexe."""
    # Grille 7x7 avec plusieurs îles
    biomass = jnp.zeros((7, 7))
    biomass = biomass.at[3, 1].set(100.0)  # Blob dans un "canal"

    # Créer archipel : îles en damier
    mask = jnp.ones((7, 7), dtype=bool)
    mask = mask.at[2, 3].set(False)
    mask = mask.at[3, 3].set(False)
    mask = mask.at[4, 3].set(False)  # Île verticale centrale
    mask = mask.at[1, 1].set(False)
    mask = mask.at[5, 5].set(False)

    # Flux vers l'est
    u = jnp.ones((7, 7)) * 0.5
    v = jnp.zeros((7, 7))

    params = {"dx": 1000.0, "dy": 1000.0, "D": 100.0, "mask": mask}
    forcings = {"u": u, "v": v}
    dt = 100.0

    # Kernel combiné
    kernel = Kernel([compute_advection_simple, compute_diffusion_simple])
    state = {"biomass": biomass}

    for _ in range(50):
        state = kernel.execute_local_phase(state, dt=dt, params=params, forcings=forcings)

    # Toutes les îles restent à 0
    assert state["biomass"][2, 3] == 0.0
    assert state["biomass"][3, 3] == 0.0
    assert state["biomass"][4, 3] == 0.0

    # Biomasse a circulé autour des îles
    assert jnp.sum(state["biomass"]) > 50.0  # Pas trop de perte
```

#### Test 4 : Conservation masse avec îles

```python
def test_mass_conservation_with_islands():
    """Vérifier conservation masse en présence d'îles."""
    biomass = jnp.ones((10, 10)) * 50.0

    # Quelques îles
    mask = jnp.ones((10, 10), dtype=bool)
    mask = mask.at[3:6, 3:6].set(False)  # Île carrée centrale

    # Forcer biomass=0 sur îles initialement
    biomass = jnp.where(mask, biomass, 0.0)

    u = jnp.ones((10, 10)) * 0.3
    v = jnp.ones((10, 10)) * 0.2

    params = {"dx": 1000.0, "dy": 1000.0, "D": 50.0, "mask": mask}
    forcings = {"u": u, "v": v}
    dt = 50.0

    initial_mass = jnp.sum(biomass[mask])  # Masse seulement sur ocean

    kernel = Kernel([compute_advection_simple, compute_diffusion_simple])
    state = {"biomass": biomass}

    for _ in range(100):
        state = kernel.execute_local_phase(state, dt=dt, params=params, forcings=forcings)

    final_mass = jnp.sum(state["biomass"][mask])

    # Conservation stricte (< 1% erreur)
    mass_error = abs(final_mass - initial_mass) / initial_mass
    assert mass_error < 0.01
```

---

## Phase 11.4 : Exemple avec Îles (~0.5h)

### Option A : Modifier un exemple existant

Ajouter un scénario avec îles dans `advection_blob.py` ou `advection_diffusion.py`

### Option B : Créer nouvel exemple

`examples/islands_example.py`
- Grille avec archipel réaliste
- Biomasse initiale dans une "baie"
- Courants uniformes
- Visualiser comment biomasse contourne les îles

---

## Phase 11.5 : Mise à Jour Documentation

### Docstrings

Mettre à jour les docstrings de :
- `compute_advection_2d` : mentionner flux blocking
- `compute_diffusion_2d` : mentionner Neumann BC sur land

### Halo Exchange

**Important** : Les halos doivent maintenant inclure le masque !

Modifier `CellWorker2D.get_boundary_*()` dans `distributed/worker.py` :

```python
def get_boundary_north(self) -> dict:
    """Get northern boundary data including mask if present."""
    boundary = {"biomass": self.state["biomass"][0, :]}

    # Ajouter masque si présent
    if hasattr(self, "mask") and self.mask is not None:
        boundary["mask"] = self.mask[0, :]

    return boundary
```

---

## Résumé Chronologique

| Tâche | Fichier | Temps |
|-------|---------|-------|
| **11.1** | `transport.py` (advection) | 1h |
| **11.2** | `transport.py` (diffusion) | 1h |
| **11.3** | `test_transport.py` (4 tests) | 1h |
| **11.4** | Exemple îles (optionnel) | 0.5h |
| **11.5** | Worker halo + docs | 0.5h |
| **TOTAL** | | **4h** |

---

## Critères de Validation

### Tests Unitaires
- ✓ Aucun flux ne traverse île (advection)
- ✓ Aucune diffusion à travers île
- ✓ Conservation masse stricte avec îles
- ✓ Archipel complexe fonctionne

### Tests d'Intégration
- ✓ Simulation distribuée 2x2 avec îles
- ✓ Halos incluent masques correctement

### Validation Physique
- ✓ Biomasse=0 sur land maintenu dans le temps
- ✓ Gradients normaux = 0 aux frontières land/ocean
- ✓ Aucune accumulation numérique sur land

---

## Questions Avant Implémentation

1. **Halos distribués** : Faut-il modifier `CellWorker2D` pour propager les masques ? ✓ (Phase 11.5)
2. **Masques 3D** : Pour l'instant 2D uniquement, 3D plus tard ? ✓ (plus tard)
3. **Performance** : Le flux blocking ajoute ~10% overhead, acceptable ? ✓ (acceptable)

---

## Prêt pour Commit + Implémentation

**Ordre** :
1. Commit Phase 10 (advection actuelle)
2. Implémenter Phase 11.1-11.3 (flux blocking)
3. Tester
4. Commit Phase 11
