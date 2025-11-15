# Plan d'Implémentation TransportWorker - Phase 2A

**Date** : 2025-01-15
**Objectif** : Implémenter advection-diffusion avec conservation >99% sur 10 jours
**Approche** : Itérative (Option B) - Implémentation + Tests à chaque étape

---

## Vue d'Ensemble

### Architecture Finale

```
src/seapopym_message/transport/
├── __init__.py           # Exports publics
├── worker.py             # Ray actor (existant, à modifier)
├── grid.py               # NEW: Géométrie grille (sphérique/plane)
├── boundary.py           # NEW: Conditions limites (CLOSED/PERIODIC/OPEN)
├── advection.py          # NEW: Schéma upwind flux-based
└── diffusion.py          # NEW: Euler explicite grille sphérique

tests/unit/
├── test_transport_grid.py           # NEW: Tests Grid
├── test_transport_boundary.py       # NEW: Tests BoundaryConditions
├── test_transport_advection.py      # NEW: Tests advection
├── test_transport_diffusion.py      # NEW: Tests diffusion
└── test_transport_worker_physics.py # NEW: Tests intégration worker
```

### Dépendances entre Modules

```
Grid (indépendant)
  ↓
BoundaryConditions (utilise Grid pour périodique)
  ↓
Advection (utilise Grid + Boundary)
  ↓
Diffusion (utilise Grid + Boundary)
  ↓
TransportWorker (utilise tout)
```

---

## Phase 1 : Infrastructure Grid (1-2h)

### Tâche 1.1 : Créer `grid.py`

**Fichier** : `src/seapopym_message/transport/grid.py`

**Contenu** :
```python
# Classes à implémenter :
- Grid (ABC)
  - cell_areas() -> jnp.ndarray
  - face_areas_ew() -> jnp.ndarray | float
  - face_areas_ns() -> jnp.ndarray

- SphericalGrid(Grid)
  - __init__(lat_min, lat_max, lon_min, lon_max, nlat, nlon, R=6371e3)
  - Pré-calcule aires dans _compute_geometry()
  - Attributs : lat, lon, dlat, dlon (degrés), R

- PlaneGrid(Grid)
  - __init__(dx, dy, nlat, nlon)
  - Aires uniformes
```

**Validations** :
- Grille sphérique : `cell_areas()[0] > cell_areas()[-1]` (aire plus petite aux pôles)
- `face_areas_ew()` scalaire (constant en latitude pour grille sphérique)
- `face_areas_ns()` shape = (nlat, nlon) et varie avec latitude

**Critères de complétion** :
- [ ] Classes Grid, SphericalGrid, PlaneGrid implémentées
- [ ] Docstrings complètes avec formules mathématiques
- [ ] Type hints complets

---

### Tâche 1.2 : Tests unitaires Grid

**Fichier** : `tests/unit/test_transport_grid.py`

**Tests à implémenter** :

```python
class TestSphericalGrid:
    def test_initialization():
        """Vérifier paramètres stockés correctement."""
        grid = SphericalGrid(-10, 10, 0, 360, nlat=20, nlon=40)
        assert grid.nlat == 20
        assert grid.nlon == 40
        assert len(grid.lat) == 20
        assert len(grid.lon) == 40

    def test_cell_areas_decrease_toward_poles():
        """Aires diminuent vers les pôles (cos(lat))."""
        grid = SphericalGrid(-60, 60, 0, 360, nlat=100, nlon=200)
        areas = grid.cell_areas()
        # Équateur à lat=0 → index 50
        assert areas[50, 0] > areas[0, 0]   # Équateur > Pôle Sud
        assert areas[50, 0] > areas[99, 0]  # Équateur > Pôle Nord

    def test_face_areas_ew_constant():
        """Aires faces E/O constantes (R × dφ)."""
        grid = SphericalGrid(-45, 45, 0, 180, nlat=10, nlon=20)
        area = grid.face_areas_ew()
        assert isinstance(area, (float, jnp.ndarray))
        if isinstance(area, jnp.ndarray):
            assert area.ndim == 0  # Scalaire

    def test_face_areas_ns_vary_with_latitude():
        """Aires faces N/S varient avec latitude (cos(lat))."""
        grid = SphericalGrid(-80, 80, 0, 360, nlat=50, nlon=100)
        areas_ns = grid.face_areas_ns()
        assert areas_ns.shape == (50, 100)
        # Plus grandes à l'équateur
        assert jnp.max(areas_ns[25, :]) > jnp.max(areas_ns[0, :])

    def test_realistic_seapopym_grid():
        """Test avec grille océanique réaliste."""
        # Pacifique : -60°S à 60°N, 120°E à 280°E (1° résolution)
        grid = SphericalGrid(-60, 60, 120, 280, nlat=120, nlon=160)

        # Vérifier cohérence dimensions
        assert grid.cell_areas().shape == (120, 160)
        assert grid.face_areas_ns().shape == (120, 160)

        # Vérifier aire totale raisonnable
        total_area = jnp.sum(grid.cell_areas())
        # Doit être < aire totale Terre (510e12 m²)
        assert total_area < 510e12

class TestPlaneGrid:
    def test_uniform_areas():
        """Aires uniformes pour grille plane."""
        grid = PlaneGrid(dx=1000.0, dy=1000.0, nlat=10, nlon=20)
        areas = grid.cell_areas()
        assert jnp.allclose(areas, 1e6)  # dx × dy
```

**Critères de complétion** :
- [ ] Tous les tests passent
- [ ] Coverage > 95% sur grid.py
- [ ] Tests avec pytest-benchmark pour vérifier performance

---

## Phase 2 : Boundary Conditions (1-2h)

### Tâche 2.1 : Créer `boundary.py`

**Fichier** : `src/seapopym_message/transport/boundary.py`

**Contenu** :
```python
# Enum à implémenter :
class BoundaryType(Enum):
    CLOSED = "closed"      # u=0 ou v=0, ghost cells = edge
    PERIODIC = "periodic"  # Wrap-around
    OPEN = "open"          # Gradient nul (même que CLOSED pour scalaire)

# Classe principale :
class BoundaryConditions:
    __init__(lat_bc: BoundaryType, lon_bc: BoundaryType)

    apply_ghost_cells(field: jnp.ndarray, halo_width: int = 1) -> jnp.ndarray:
        """Ajoute ghost cells selon lat_bc et lon_bc.

        CLOSED/OPEN : jnp.pad(..., mode='edge')
        PERIODIC : jnp.concatenate([field[-w:], field, field[:w]])
        """

    mask_velocities(u: jnp.ndarray, v: jnp.ndarray) -> tuple:
        """Force u=0, v=0 aux bords fermés.

        Si lat_bc == CLOSED : v[0, :] = v[-1, :] = 0
        Si lon_bc == CLOSED : u[:, 0] = u[:, -1] = 0
        """
```

**Validations** :
- PERIODIC : `field_ext[:, 0] == field[:, -1]` (bord Ouest = dernière colonne)
- CLOSED : `field_ext[0, :] == field[0, :]` (ghost = edge)
- mask_velocities préserve shape

**Critères de complétion** :
- [ ] BoundaryType enum défini
- [ ] BoundaryConditions implémenté
- [ ] Docstrings avec exemples
- [ ] Type hints complets

---

### Tâche 2.2 : Tests unitaires Boundary

**Fichier** : `tests/unit/test_transport_boundary.py`

**Tests à implémenter** :

```python
class TestBoundaryConditions:
    def test_closed_ghost_cells():
        """Ghost cells = edge pour CLOSED."""
        bc = BoundaryConditions(lat_bc=CLOSED, lon_bc=CLOSED)
        field = jnp.arange(12).reshape(3, 4)
        ext = bc.apply_ghost_cells(field, halo_width=1)

        assert ext.shape == (5, 6)
        assert ext[0, 1] == field[0, 0]  # Ghost Nord = edge
        assert ext[-1, 1] == field[-1, 0]  # Ghost Sud = edge

    def test_periodic_lon_wrap_around():
        """Périodique en longitude : wraparound."""
        bc = BoundaryConditions(lat_bc=CLOSED, lon_bc=PERIODIC)
        field = jnp.arange(12).reshape(3, 4)
        ext = bc.apply_ghost_cells(field, halo_width=1)

        # Ghost Ouest = dernière colonne
        assert jnp.array_equal(ext[:, 0], field[:, -1])
        # Ghost Est = première colonne
        assert jnp.array_equal(ext[:, -1], field[:, 0])

    def test_mask_velocities_closed_lat():
        """u, v = 0 aux bords fermés."""
        bc = BoundaryConditions(lat_bc=CLOSED, lon_bc=PERIODIC)
        u = jnp.ones((10, 20))
        v = jnp.ones((10, 20)) * 2.0

        u_m, v_m = bc.mask_velocities(u, v)

        # v=0 au Nord et Sud (lat fermée)
        assert jnp.all(v_m[0, :] == 0.0)
        assert jnp.all(v_m[-1, :] == 0.0)

        # u préservé (lon périodique)
        assert jnp.all(u_m[:, 5] == 1.0)

    def test_realistic_seapopym_bc():
        """BC typique SEAPOPYM : lat=CLOSED, lon=PERIODIC."""
        bc = BoundaryConditions(lat_bc=CLOSED, lon_bc=PERIODIC)
        biomass = jnp.ones((120, 160)) * 100.0

        ext = bc.apply_ghost_cells(biomass)
        assert ext.shape == (122, 162)

        # Vérifier périodicité longitude
        assert jnp.allclose(ext[:, 0], biomass[:, -1])
        assert jnp.allclose(ext[:, -1], biomass[:, 0])
```

**Critères de complétion** :
- [ ] Tous les tests passent
- [ ] Coverage > 95% sur boundary.py
- [ ] Tests avec différentes combinaisons BC

---

## Phase 3 : Advection (2-3h)

### Tâche 3.1 : Créer `advection.py`

**Fichier** : `src/seapopym_message/transport/advection.py`

**Contenu** :
```python
def advection_upwind_flux(
    biomass: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    dt: float,
    grid: Grid,
    boundary: BoundaryConditions,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Advection upwind flux-based (volumes finis).

    Implémentation exacte du code dans TRANSPORT_ANALYSIS.md section 8.

    Étapes :
    1. Masquer u, v (bords + terres)
    2. Ghost cells pour biomasse
    3. Interpoler u, v aux faces
    4. Choix upwind pour C_face
    5. Calculer flux F = u × C_face × Aire_face
    6. Bilan flux_net = (F_o - F_e) + (F_n - F_s)
    7. Update : dC/dt = flux_net / Volume
    8. Enforce mask
    """
    # Code complet selon TRANSPORT_ANALYSIS.md
```

**Validations** :
- Conservation : `sum(biomass_out) == sum(biomass_in)` si pas de masque
- CFL check : `max(|u|) * dt / dx ≤ 1.0` (warning si violé)
- Shape préservée
- Non-négatif

**Critères de complétion** :
- [ ] Fonction implémentée selon spec
- [ ] Docstring avec équations et références
- [ ] Type hints complets
- [ ] Commentaires alignés avec étapes description

---

### Tâche 3.2 : Tests unitaires Advection

**Fichier** : `tests/unit/test_transport_advection.py`

**Tests à implémenter** :

```python
class TestAdvectionUpwindFlux:
    def test_no_advection_if_u_v_zero():
        """Pas de changement si u=v=0."""
        grid = PlaneGrid(dx=1000, dy=1000, nlat=10, nlon=20)
        bc = BoundaryConditions(CLOSED, CLOSED)

        biomass = jnp.ones((10, 20)) * 50.0
        u = jnp.zeros((10, 20))
        v = jnp.zeros((10, 20))

        result = advection_upwind_flux(biomass, u, v, dt=100, grid=grid, boundary=bc)

        assert jnp.allclose(result, biomass)

    def test_translation_pure_eastward():
        """Translation pure vers l'Est."""
        grid = PlaneGrid(dx=1000, dy=1000, nlat=10, nlon=30)
        bc = BoundaryConditions(CLOSED, PERIODIC)

        # Blob à lon_idx=5
        biomass = jnp.zeros((10, 30))
        biomass = biomass.at[5, 5].set(100.0)

        # Vitesse uniforme Est : u=1 m/s, CFL = u×dt/dx = 0.5
        u = jnp.ones((10, 30)) * 1.0
        v = jnp.zeros((10, 30))
        dt = 500.0  # CFL = 0.5

        # 10 steps → déplacement ~5 cellules
        for _ in range(10):
            biomass = advection_upwind_flux(biomass, u, v, dt, grid, bc)

        # Vérifier déplacement
        max_idx = jnp.argmax(biomass[5, :])
        assert max_idx >= 8  # Déplacé d'au moins 3 cellules

    def test_conservation_no_mask():
        """Conservation masse sans masque."""
        grid = SphericalGrid(-45, 45, 0, 360, nlat=20, nlon=40)
        bc = BoundaryConditions(CLOSED, PERIODIC)

        biomass = jnp.ones((20, 40)) * 100.0
        u = jnp.ones((20, 40)) * 0.5
        v = jnp.zeros((20, 40))

        mass_before = jnp.sum(biomass * grid.cell_areas())

        result = advection_upwind_flux(biomass, u, v, dt=1000, grid=grid, boundary=bc)

        mass_after = jnp.sum(result * grid.cell_areas())
        error = abs(mass_after - mass_before) / mass_before

        assert error < 0.01  # <1% erreur

    def test_island_blocking():
        """Flux bloqué aux interfaces île/océan."""
        grid = PlaneGrid(dx=10000, dy=10000, nlat=30, nlon=40)
        bc = BoundaryConditions(CLOSED, PERIODIC)

        # Île centrale
        mask = jnp.ones((30, 40), dtype=bool)
        mask = mask.at[10:20, 15:25].set(False)

        # Biomasse uniquement océan
        biomass = jnp.where(mask, 100.0, 0.0)

        # Flux vers l'Est
        u = jnp.ones((30, 40)) * 1.0
        v = jnp.zeros((30, 40))

        result = advection_upwind_flux(biomass, u, v, dt=500, grid=grid, bc, mask=mask)

        # Vérifier biomasse=0 dans île
        assert jnp.all(result[10:20, 15:25] == 0.0)

        # Conservation dans océan
        mass_ocean_before = jnp.sum(biomass[mask] * grid.cell_areas()[mask])
        mass_ocean_after = jnp.sum(result[mask] * grid.cell_areas()[mask])
        assert jnp.allclose(mass_ocean_before, mass_ocean_after, rtol=0.05)

    def test_cfl_stability():
        """Vérifier stabilité CFL."""
        grid = PlaneGrid(dx=1000, dy=1000, nlat=10, nlon=20)
        bc = BoundaryConditions(CLOSED, CLOSED)

        biomass = jnp.ones((10, 20)) * 50.0

        # CFL = 0.5 : stable
        u_stable = jnp.ones((10, 20)) * 0.5  # CFL = 0.5×100/1000 = 0.05
        result = advection_upwind_flux(biomass, u_stable, v=jnp.zeros((10,20)), dt=100, grid=grid, boundary=bc)
        assert jnp.all(result >= 0)  # Pas de valeurs négatives

        # CFL > 1 : instable (devrait fonctionner mais avec erreurs)
        u_unstable = jnp.ones((10, 20)) * 15.0  # CFL = 15×100/1000 = 1.5
        result = advection_upwind_flux(biomass, u_unstable, v=jnp.zeros((10,20)), dt=100, grid=grid, boundary=bc)
        # Upwind reste stable même CFL>1 mais moins précis
        assert jnp.all(result >= 0)
```

**Critères de complétion** :
- [ ] Tous les tests passent
- [ ] Conservation <1% erreur sans masque
- [ ] Conservation <5% erreur avec masque et île
- [ ] Translation fonctionne correctement
- [ ] Coverage > 90% sur advection.py

---

## Phase 4 : Diffusion (2-3h)

### Tâche 4.1 : Créer `diffusion.py`

**Fichier** : `src/seapopym_message/transport/diffusion.py`

**Contenu** :
```python
def diffusion_explicit_spherical(
    biomass: jnp.ndarray,
    D: float,
    dt: float,
    grid: Grid,
    boundary: BoundaryConditions,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Diffusion Euler explicite grille sphérique.

    Implémentation exacte du code dans TRANSPORT_ANALYSIS.md section 8.

    Équation : C^{n+1} = C^n + Δt·D·∇²C

    Laplacien : ∇²C = ∂²C/∂x² + ∂²C/∂y²
        ∂²C/∂x² = (C_e - 2C_c + C_w) / dx(lat)²
        ∂²C/∂y² = (C_n - 2C_c + C_s) / dy²

    Stabilité : dt ≤ min(dx²)/(4D)
    """
    # Code complet selon TRANSPORT_ANALYSIS.md
```

**Validations** :
- Vérification stabilité : raise si dt > dt_max
- Conservation : `sum(biomass)` constant si pas de masque
- Shape préservée
- Non-négatif

**Critères de complétion** :
- [ ] Fonction implémentée selon spec
- [ ] Check stabilité avec message d'erreur clair
- [ ] Docstring avec équations
- [ ] Support grille plane ET sphérique

---

### Tâche 4.2 : Tests unitaires Diffusion

**Fichier** : `tests/unit/test_transport_diffusion.py`

**Tests à implémenter** :

```python
class TestDiffusionExplicitSpherical:
    def test_no_diffusion_if_D_zero():
        """Pas de changement si D=0."""
        grid = PlaneGrid(dx=1000, dy=1000, nlat=10, nlon=20)
        bc = BoundaryConditions(CLOSED, CLOSED)

        biomass = jnp.ones((10, 20)) * 50.0

        result = diffusion_explicit_spherical(biomass, D=0.0, dt=100, grid=grid, boundary=bc)

        assert jnp.allclose(result, biomass)

    def test_blob_diffuses():
        """Blob central diffuse."""
        grid = PlaneGrid(dx=1000, dy=1000, nlat=20, nlon=20)
        bc = BoundaryConditions(CLOSED, CLOSED)

        # Blob au centre
        biomass = jnp.zeros((20, 20))
        biomass = biomass.at[10, 10].set(1000.0)

        D = 10.0  # m²/s
        dt = 100.0  # dt_max = 1e6 / (4×10) = 25000 >> 100

        result = diffusion_explicit_spherical(biomass, D, dt, grid, bc)

        # Vérifier diffusion
        assert result[10, 10] < biomass[10, 10]  # Centre diminue
        assert result[10, 11] > 0  # Voisins augmentent
        assert result[11, 10] > 0

    def test_conservation_no_mask():
        """Conservation masse sans masque."""
        grid = PlaneGrid(dx=1000, dy=1000, nlat=15, nlon=25)
        bc = BoundaryConditions(CLOSED, PERIODIC)

        biomass = jnp.ones((15, 25)) * 100.0
        biomass = biomass.at[7, 12].set(500.0)  # Pic

        mass_before = jnp.sum(biomass)

        result = diffusion_explicit_spherical(biomass, D=50.0, dt=50.0, grid=grid, boundary=bc)

        mass_after = jnp.sum(result)
        error = abs(mass_after - mass_before) / mass_before

        assert error < 0.001  # <0.1% erreur

    def test_island_neumann_bc():
        """Neumann BC aux interfaces île."""
        grid = PlaneGrid(dx=5000, dy=5000, nlat=20, nlon=30)
        bc = BoundaryConditions(CLOSED, CLOSED)

        # Île centrale
        mask = jnp.ones((20, 30), dtype=bool)
        mask = mask.at[8:12, 13:17].set(False)

        # Biomasse océan uniquement
        biomass = jnp.where(mask, 50.0, 0.0)

        result = diffusion_explicit_spherical(biomass, D=100.0, dt=20.0, grid=grid, bc, mask=mask)

        # Vérifier biomasse=0 dans île
        assert jnp.all(result[8:12, 13:17] == 0.0)

        # Conservation dans océan
        mass_ocean_before = jnp.sum(biomass[mask])
        mass_ocean_after = jnp.sum(result[mask])
        assert jnp.allclose(mass_ocean_before, mass_ocean_after, rtol=0.01)

    def test_stability_constraint():
        """Vérifier contrainte stabilité."""
        grid = PlaneGrid(dx=1000, dy=1000, nlat=10, nlon=10)
        bc = BoundaryConditions(CLOSED, CLOSED)

        biomass = jnp.ones((10, 10)) * 100.0
        D = 100.0

        # dt_max = 1e6 / (4×100) = 2500
        dt_stable = 2000.0
        dt_unstable = 3000.0

        # Stable : OK
        result = diffusion_explicit_spherical(biomass, D, dt_stable, grid, bc)
        assert jnp.all(result >= 0)

        # Instable : ValueError
        with pytest.raises(ValueError, match="Instabilité diffusion"):
            diffusion_explicit_spherical(biomass, D, dt_unstable, grid, bc)

    def test_spherical_grid_dx_latitude_dependency():
        """dx(lat) variable sur grille sphérique."""
        grid = SphericalGrid(-60, 60, 0, 360, nlat=60, nlon=120)
        bc = BoundaryConditions(CLOSED, PERIODIC)

        biomass = jnp.ones((60, 120)) * 100.0
        D = 1000.0

        # dx plus petit aux hautes latitudes → dt_max plus restrictif
        result = diffusion_explicit_spherical(biomass, D, dt=10.0, grid=grid, bc)

        # Vérifier shape et valeurs
        assert result.shape == (60, 120)
        assert jnp.all(result >= 0)

        # Conservation
        mass_before = jnp.sum(biomass * grid.cell_areas())
        mass_after = jnp.sum(result * grid.cell_areas())
        assert jnp.allclose(mass_before, mass_after, rtol=0.01)
```

**Critères de complétion** :
- [ ] Tous les tests passent
- [ ] Conservation <0.1% erreur sans masque
- [ ] Conservation <1% erreur avec masque
- [ ] Stabilité vérifiée (ValueError si dt trop grand)
- [ ] Coverage > 90% sur diffusion.py

---

## Phase 5 : Intégration TransportWorker (2-3h)

### Tâche 5.1 : Modifier `worker.py`

**Fichier** : `src/seapopym_message/transport/worker.py`

**Modifications** :

```python
# Ajouter imports
from seapopym_message.transport.grid import SphericalGrid, PlaneGrid
from seapopym_message.transport.boundary import BoundaryConditions, BoundaryType
from seapopym_message.transport.advection import advection_upwind_flux
from seapopym_message.transport.diffusion import diffusion_explicit_spherical

@ray.remote
class TransportWorker:
    def __init__(
        self,
        grid_type: str = "spherical",  # "spherical" ou "plane"
        lat_min: float = -90.0,
        lat_max: float = 90.0,
        lon_min: float = 0.0,
        lon_max: float = 360.0,
        nlat: int = 180,
        nlon: int = 360,
        lat_bc: str = "closed",  # "closed", "periodic", "open"
        lon_bc: str = "periodic",
    ):
        """Initialize TransportWorker avec grille et BC."""
        # Créer grille
        if grid_type == "spherical":
            self.grid = SphericalGrid(lat_min, lat_max, lon_min, lon_max, nlat, nlon)
        else:
            # Pour plane, utiliser dx moyen de spherical
            dx = 111e3 * (lon_max - lon_min) / nlon  # ~111km par degré
            dy = 111e3 * (lat_max - lat_min) / nlat
            self.grid = PlaneGrid(dx, dy, nlat, nlon)

        # Créer BC
        lat_bc_enum = BoundaryType[lat_bc.upper()]
        lon_bc_enum = BoundaryType[lon_bc.upper()]
        self.boundary = BoundaryConditions(lat_bc_enum, lon_bc_enum)

    def transport_step(
        self,
        biomass: jnp.ndarray,
        u: jnp.ndarray,
        v: jnp.ndarray,
        D: float,
        dt: float,
        mask: jnp.ndarray | None = None,
    ) -> dict[str, Any]:
        """Execute transport step: advection puis diffusion.

        Note : Pas de dx, dy en paramètres → utilisent self.grid
        """
        import time

        t_start = time.perf_counter()

        mass_before = float(jnp.sum(biomass * self.grid.cell_areas()))

        # 1. Advection
        biomass_advected = advection_upwind_flux(
            biomass, u, v, dt, self.grid, self.boundary, mask
        )

        mass_after_advection = float(jnp.sum(biomass_advected * self.grid.cell_areas()))

        # 2. Diffusion
        if D > 0:
            biomass_final = diffusion_explicit_spherical(
                biomass_advected, D, dt, self.grid, self.boundary, mask
            )
        else:
            biomass_final = biomass_advected

        mass_after = float(jnp.sum(biomass_final * self.grid.cell_areas()))

        t_end = time.perf_counter()

        # 3. Diagnostics
        diagnostics = {
            "mass_before": mass_before,
            "mass_after": mass_after,
            "mass_error_total": (mass_after - mass_before) / mass_before if mass_before > 0 else 0.0,
            "mass_error_advection": (mass_after_advection - mass_before) / mass_before if mass_before > 0 else 0.0,
            "mass_error_diffusion": (mass_after - mass_after_advection) / mass_after_advection if mass_after_advection > 0 else 0.0,
            "compute_time_s": t_end - t_start,
            "mode": "physics",
        }

        return {"biomass": biomass_final, "diagnostics": diagnostics}
```

**Critères de complétion** :
- [ ] __init__ modifié avec paramètres grille et BC
- [ ] transport_step implémenté avec advection + diffusion
- [ ] Diagnostics de conservation calculés
- [ ] Docstrings mises à jour
- [ ] Backward compatibility : paramètres optionnels

---

### Tâche 5.2 : Tests intégration worker

**Fichier** : `tests/unit/test_transport_worker_physics.py`

**Tests à implémenter** :

```python
class TestTransportWorkerPhysics:
    def test_initialization_spherical(self, ray_context):
        """Initialisation grille sphérique."""
        worker = TransportWorker.remote(
            grid_type="spherical",
            lat_min=-45, lat_max=45,
            lon_min=0, lon_max=360,
            nlat=90, nlon=180,
            lat_bc="closed", lon_bc="periodic"
        )

        # Test simple : worker existe
        assert worker is not None

    def test_advection_only(self, ray_context):
        """Transport avec D=0 (advection seule)."""
        worker = TransportWorker.remote(
            grid_type="plane",  # Plus simple pour test
            lat_min=0, lat_max=10,
            lon_min=0, lon_max=20,
            nlat=10, nlon=20,
            lat_bc="closed", lon_bc="periodic"
        )

        biomass = jnp.ones((10, 20)) * 100.0
        u = jnp.ones((10, 20)) * 1.0
        v = jnp.zeros((10, 20))

        result = ray.get(worker.transport_step.remote(
            biomass=biomass, u=u, v=v, D=0.0, dt=100.0, mask=None
        ))

        # Vérifier conservation
        assert abs(result["diagnostics"]["mass_error_total"]) < 0.05  # <5%
        assert result["diagnostics"]["mode"] == "physics"

    def test_diffusion_only(self, ray_context):
        """Transport avec u=v=0 (diffusion seule)."""
        worker = TransportWorker.remote(
            grid_type="plane",
            lat_min=0, lat_max=10,
            lon_min=0, lon_max=10,
            nlat=20, nlon=20,
            lat_bc="closed", lon_bc="closed"
        )

        # Blob central
        biomass = jnp.zeros((20, 20))
        biomass = biomass.at[10, 10].set(1000.0)

        u = jnp.zeros((20, 20))
        v = jnp.zeros((20, 20))

        result = ray.get(worker.transport_step.remote(
            biomass=biomass, u=u, v=v, D=50.0, dt=50.0, mask=None
        ))

        # Blob doit diffuser
        assert result["biomass"][10, 10] < biomass[10, 10]
        assert result["biomass"][10, 11] > 0

        # Conservation
        assert abs(result["diagnostics"]["mass_error_total"]) < 0.01  # <1%

    def test_advection_diffusion_combined(self, ray_context):
        """Transport complet : advection + diffusion."""
        worker = TransportWorker.remote(
            grid_type="spherical",
            lat_min=-30, lat_max=30,
            lon_min=100, lon_max=200,
            nlat=60, nlon=100,
            lat_bc="closed", lon_bc="periodic"
        )

        biomass = jnp.ones((60, 100)) * 50.0
        u = jnp.ones((60, 100)) * 0.5
        v = jnp.zeros((60, 100))

        result = ray.get(worker.transport_step.remote(
            biomass=biomass, u=u, v=v, D=100.0, dt=500.0, mask=None
        ))

        # Vérifier conservation
        assert abs(result["diagnostics"]["mass_error_total"]) < 0.05  # <5%

    def test_10_day_conservation(self, ray_context):
        """Test critique : conservation sur 10 jours."""
        worker = TransportWorker.remote(
            grid_type="spherical",
            lat_min=-60, lat_max=60,
            lon_min=120, lon_max=280,
            nlat=120, nlon=160,
            lat_bc="closed", lon_bc="periodic"
        )

        # État initial
        biomass = jnp.ones((120, 160)) * 100.0
        u = jnp.ones((120, 160)) * 1.0  # Courant constant
        v = jnp.zeros((120, 160))
        D = 1000.0
        dt = 3600.0  # 1h

        # 10 jours = 240 timesteps
        num_steps = 240

        mass_initial = float(jnp.sum(biomass))

        for step in range(num_steps):
            result = ray.get(worker.transport_step.remote(
                biomass=biomass, u=u, v=v, D=D, dt=dt, mask=None
            ))
            biomass = result["biomass"]

            # Vérifier à chaque étape
            error = result["diagnostics"]["mass_error_total"]
            assert abs(error) < 0.05, f"Step {step}: error {error:.4f} > 5%"

        # Vérifier conservation globale
        mass_final = float(jnp.sum(biomass))
        total_error = abs(mass_final - mass_initial) / mass_initial

        assert total_error < 0.01, f"Total mass loss: {total_error:.2%} > 1%"

    def test_island_blocking_integrated(self, ray_context):
        """Test avec île : flux bloqué."""
        worker = TransportWorker.remote(
            grid_type="plane",
            lat_min=0, lat_max=30,
            lon_min=0, lon_max=40,
            nlat=30, nlon=40,
            lat_bc="closed", lon_bc="periodic"
        )

        # Île centrale
        mask = jnp.ones((30, 40), dtype=bool)
        mask = mask.at[10:20, 15:25].set(False)

        # Biomasse océan
        biomass = jnp.where(mask, 100.0, 0.0)

        u = jnp.ones((30, 40)) * 1.0
        v = jnp.zeros((30, 40))

        result = ray.get(worker.transport_step.remote(
            biomass=biomass, u=u, v=v, D=50.0, dt=200.0, mask=mask
        ))

        # Vérifier île reste à 0
        assert jnp.all(result["biomass"][10:20, 15:25] == 0.0)

        # Conservation dans océan
        mass_ocean_before = float(jnp.sum(biomass[mask]))
        mass_ocean_after = float(jnp.sum(result["biomass"][mask]))
        error = abs(mass_ocean_after - mass_ocean_before) / mass_ocean_before

        assert error < 0.05  # <5% dans océan
```

**Critères de complétion** :
- [ ] Tous les tests passent
- [ ] Test 10 jours : conservation >99%
- [ ] Test île : biomasse=0 dans île
- [ ] Performance : <1s pour 1 timestep sur grille 120×160

---

## Phase 6 : Mise à jour `__init__.py` et Documentation (1h)

### Tâche 6.1 : Mettre à jour `__init__.py`

**Fichier** : `src/seapopym_message/transport/__init__.py`

**Contenu** :
```python
"""Transport module: centralized advection-diffusion computation.

Architecture:
    - TransportWorker: Ray actor for distributed transport
    - Grid: Geometric computations (spherical/plane)
    - BoundaryConditions: Boundary handling (CLOSED/PERIODIC/OPEN)
    - advection_upwind_flux: Flux-based upwind scheme
    - diffusion_explicit_spherical: Explicit Euler on spherical grid

Usage:
    >>> import ray
    >>> from seapopym_message.transport import TransportWorker
    >>>
    >>> ray.init()
    >>> worker = TransportWorker.remote(
    ...     grid_type="spherical",
    ...     lat_min=-60, lat_max=60,
    ...     lon_min=120, lon_max=280,
    ...     nlat=120, nlon=160,
    ...     lat_bc="closed", lon_bc="periodic"
    ... )
    >>>
    >>> result = ray.get(worker.transport_step.remote(
    ...     biomass=biomass, u=u, v=v, D=1000.0, dt=3600.0
    ... ))
"""

from seapopym_message.transport.worker import TransportWorker
from seapopym_message.transport.grid import Grid, SphericalGrid, PlaneGrid
from seapopym_message.transport.boundary import BoundaryConditions, BoundaryType

__all__ = [
    "TransportWorker",
    "Grid",
    "SphericalGrid",
    "PlaneGrid",
    "BoundaryConditions",
    "BoundaryType",
]
```

**Critères de complétion** :
- [ ] Exports publics définis
- [ ] Docstring avec exemple d'usage
- [ ] Import test : `from seapopym_message.transport import TransportWorker`

---

### Tâche 6.2 : Mettre à jour README transport

**Fichier** : `src/seapopym_message/transport/README.md` (nouveau)

**Contenu** :
```markdown
# Transport Module

Centralized advection-diffusion computation for SEAPOPYM.

## Architecture

- **TransportWorker**: Ray remote actor
- **Grid**: Geometric computations (spherical lat/lon or plane)
- **BoundaryConditions**: CLOSED/PERIODIC/OPEN
- **advection_upwind_flux**: Flux-based upwind (volumes finis)
- **diffusion_explicit_spherical**: Euler explicite

## Usage

[Code examples]

## References

- Theory: `IA/Advection-upwind-description.md`
- Theory: `IA/Diffusion-euler-explicite-description.md`
- Analysis: `IA/TRANSPORT_ANALYSIS.md`
```

---

## Phase 7 : Tests Finaux et Validation (2h)

### Tâche 7.1 : Test régression passthrough

**Objectif** : Vérifier que les anciens tests passthrough passent toujours.

**Fichier** : `tests/unit/test_transport_worker_passthrough.py`

**Action** :
- Modifier pour accepter nouveaux paramètres __init__ avec valeurs par défaut
- Vérifier tous les tests passent

---

### Tâche 7.2 : Benchmark performance

**Fichier** : `tests/benchmark/test_transport_performance.py` (nouveau)

**Tests** :
```python
def test_benchmark_advection_100x200(benchmark):
    """Benchmark advection sur grille 100×200."""
    grid = SphericalGrid(-50, 50, 0, 360, nlat=100, nlon=200)
    bc = BoundaryConditions(CLOSED, PERIODIC)

    biomass = jnp.ones((100, 200)) * 100.0
    u = jnp.ones((100, 200)) * 1.0
    v = jnp.zeros((100, 200))

    result = benchmark(
        advection_upwind_flux,
        biomass, u, v, dt=1000.0, grid=grid, boundary=bc
    )

    # Target : <100ms
```

---

### Tâche 7.3 : Créer exemple complet

**Fichier** : `examples/transport_demo.py` (nouveau)

**Contenu** :
```python
"""Démonstration TransportWorker : advection + diffusion sur 10 jours."""

import jax.numpy as jnp
import ray
from seapopym_message.transport import TransportWorker

def main():
    ray.init()

    # Configuration
    worker = TransportWorker.remote(
        grid_type="spherical",
        lat_min=-60, lat_max=60,
        lon_min=120, lon_max=280,
        nlat=120, nlon=160,
        lat_bc="closed", lon_bc="periodic"
    )

    # État initial
    biomass = jnp.ones((120, 160)) * 100.0
    u = jnp.ones((120, 160)) * 1.0  # 1 m/s Est
    v = jnp.zeros((120, 160))
    D = 1000.0  # m²/s
    dt = 3600.0  # 1h

    # Simulation 10 jours
    for day in range(10):
        print(f"\n=== Day {day+1} ===")

        for hour in range(24):
            result = ray.get(worker.transport_step.remote(
                biomass=biomass, u=u, v=v, D=D, dt=dt
            ))

            biomass = result["biomass"]

            if hour == 0:  # Afficher 1x par jour
                diag = result["diagnostics"]
                print(f"Mass error: {diag['mass_error_total']:.4%}")
                print(f"Compute time: {diag['compute_time_s']:.3f}s")

    print(f"\n=== Final ===")
    print(f"Conservation: {100 - abs(result['diagnostics']['mass_error_total'])*100:.2f}%")

    ray.shutdown()

if __name__ == "__main__":
    main()
```

---

## Récapitulatif des Tâches

| Phase | Tâches | Temps Estimé | Fichiers |
|-------|--------|--------------|----------|
| **1. Grid** | 1.1 + 1.2 | 1-2h | grid.py, test_transport_grid.py |
| **2. Boundary** | 2.1 + 2.2 | 1-2h | boundary.py, test_transport_boundary.py |
| **3. Advection** | 3.1 + 3.2 | 2-3h | advection.py, test_transport_advection.py |
| **4. Diffusion** | 4.1 + 4.2 | 2-3h | diffusion.py, test_transport_diffusion.py |
| **5. Worker** | 5.1 + 5.2 | 2-3h | worker.py, test_transport_worker_physics.py |
| **6. Docs** | 6.1 + 6.2 | 1h | __init__.py, README.md |
| **7. Validation** | 7.1 + 7.2 + 7.3 | 2h | tests, example |
| **TOTAL** | 17 tâches | **11-16h** | 13 fichiers |

---

## Critères de Succès Phase 2A

- [x] Architecture modulaire (grid, boundary, advection, diffusion, worker)
- [x] Tests unitaires complets (coverage >90% par module)
- [x] Conservation >99% sur 10 jours (test critique)
- [x] Flux bloqué aux îles (biomasse=0 dans terre)
- [x] Performance <1s par timestep (grille 120×160)
- [x] Documentation complète
- [x] Exemple fonctionnel

---

**Prochaine étape** : Valider ce plan, créer la TODO list, et commencer Phase 1 !
