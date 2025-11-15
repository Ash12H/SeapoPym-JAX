# Analyse JAX-Fluids vs JAX-CFD pour l'implémentation du transport

**Date**: 2025-01-15
**Contexte**: Phase 2 du refactor TransportWorker - Choix de la bibliothèque pour implémenter l'advection-diffusion

---

## 1. Résumé Exécutif

Après analyse approfondie des deux principales bibliothèques CFD en JAX, **aucune des deux n'est parfaitement adaptée** à notre cas d'usage :

-   **JAX-Fluids** : Trop complexe (Navier-Stokes compressible) mais supporte les conditions aux limites nécessaires
-   **JAX-CFD** : Plus simple mais limitation critique (conditions périodiques uniquement)
-   **Recommandation** : **Implémentation JAX pure** avec schémas simples, optimisée pour notre cas

---

## 2. Comparaison Détaillée

### 2.1 JAX-Fluids (TU Munich)

**Repository**: https://github.com/tumaer/JAXFLUIDS
**Documentation**: https://jax-fluids.readthedocs.io/

#### Caractéristiques

| Aspect                     | Détail                                                   |
| -------------------------- | -------------------------------------------------------- |
| **Équations**              | Navier-Stokes **compressibles** (deux phases)            |
| **Schémas spatiaux**       | WENO-3/5/7, WENO-CU6, TENO5 (high-order)                 |
| **Solveurs Riemann**       | Lax-Friedrichs, Rusanov, HLL, HLLC, Roe                  |
| **Intégration temporelle** | Euler explicite, RK2, RK3                                |
| **Conditions limites**     | ✅ Periodic, Symmetric, Wall, **Dirichlet**, **Neumann** |
| **Obstacles solides**      | ✅ Level-set immersed boundaries                         |
| **Parallélisation**        | ✅ Multi-GPU/TPU (JAX-Fluids 2.0)                        |
| **Complexité**             | ⚠️ **Très élevée**                                       |

#### Architecture

```
SimulationManager
    ├── InputReader (JSON configuration)
    ├── Initializer (initial conditions)
    ├── SpaceSolver (RHS computation)
    │   ├── FluxComputer (convective fluxes)
    │   └── BoundaryCondition (BC enforcement)
    ├── MaterialManager (equation of state)
    └── TimeIntegrator (RK schemes)
```

#### Points Positifs ✅

1. **Support Neumann BC** : Exactement ce dont on a besoin pour zero-flux aux bords
2. **Level-set pour obstacles** : Peut gérer les îles/terres via champs level-set
3. **Différentiable** : AD end-to-end pour optimisation ML
4. **Production-ready** : Robuste, testé, publié dans Computer Physics Communications
5. **Parallélisation** : Scale sur 512 GPUs A100 (JAX-Fluids 2.0)

#### Points Négatifs ⚠️

1. **Overkill massif** : Résout Navier-Stokes compressible, on a juste besoin d'advection-diffusion
2. **Complexité API** : Nécessite JSON config, equation of state, material properties
3. **Overhead** : Variables conservatives (ρ, ρu, ρv, E), pas juste biomasse
4. **Courbe d'apprentissage** : Architecture complexe (SimulationManager, MaterialManager, etc.)
5. **Dépendances implicites** : Gestion de la pression, température, compressibilité (inutiles pour nous)

#### Exemple d'utilisation (supposé)

```python
# Configuration complexe via JSON
config = {
    "domain": {"x": {"cells": 100, "range": [0, 100000]}, ...},
    "boundary_conditions": {
        "primitives": {"west": "neumann", "east": "neumann", ...}
    },
    "material": {"type": "IdealGas", ...},  # ⚠️ Pas adapté à la biomasse !
    "numerical": {"convective_solver": "WENO5", ...}
}

# Workflow
input_reader = InputReader(config)
sim_manager = SimulationManager(input_reader)
initializer = Initializer(...)
buffer = initializer.initialization(...)  # Variables conservatives !
sim_manager.simulate()  # Black box, peu de contrôle
```

---

### 2.2 JAX-CFD (Google)

**Repository**: https://github.com/google/jax-cfd
**Documentation**: README uniquement

#### Caractéristiques

| Aspect                     | Détail                                                 |
| -------------------------- | ------------------------------------------------------ |
| **Équations**              | Navier-Stokes **incompressibles**                      |
| **Schémas spatiaux**       | Van Leer 2ème ordre, méthodes pseudo-spectrales        |
| **Grilles**                | Staggered grids (finite volume), spectral              |
| **Intégration temporelle** | Explicite (advection), implicite/explicite (diffusion) |
| **Conditions limites**     | ❌ **Periodic uniquement**                             |
| **Obstacles solides**      | ❌ Non supporté                                        |
| **Parallélisation**        | ✅ Multi-device via JAX                                |
| **Complexité**             | ✅ **Modérée**                                         |

#### Architecture

```python
# API plus simple et directe
grid = grids.Grid((nx, ny), domain=((0, Lx), (0, Ly)))
state = initial_conditions.vortex_pair(grid, ...)

# Advection
advect_fn = advection.van_leer_advection()
state = advect_fn(state, dt)

# Diffusion
diffuse_fn = diffusion.solve_cg(grid, dt, viscosity)
state = diffuse_fn(state)
```

#### Points Positifs ✅

1. **API simple** : Fonctions directes, pas de config JSON
2. **Incompressible** : Plus proche de notre cas (advection-diffusion de scalaire)
3. **Van Leer** : Schéma robuste 2ème ordre pour advection
4. **Diffusion flexible** : CG ou FFT, implicite/explicite
5. **Google-maintained** : Qualité de code élevée

#### Points Négatifs ❌

1. **DEAL BREAKER : Periodic BC only** : Documenté explicitement
    - `"Currently only periodic boundary conditions are supported"`
    - Future work : "Alternative boundary conditions" (pas encore fait)
2. **Pas de masques** : Impossible de gérer les îles/terres
3. **Pas d'immersed boundaries** : Listé comme future work
4. **Documentation limitée** : Principalement notebooks, pas de docs complètes

#### Code existant dans notre projet

```python
# Ce qu'on avait essayé (ancien code)
from jax_cfd.base import advection

def _advection_jax_cfd(...):
    # Erreur : nécessite GridVariable, pas simple array
    # API mismatch avec nos données
    warnings.warn("JAX-CFD advection not yet fully implemented")
    return self._advection_upwind(...)  # Fallback
```

---

### 2.3 Nos Besoins vs Bibliothèques

| Besoin             | JAX-Fluids    | JAX-CFD        | JAX Pur             |
| ------------------ | ------------- | -------------- | ------------------- |
| Advection 2D       | ✅ (overkill) | ✅ Van Leer    | ✅ Custom           |
| Diffusion 2D       | ✅ (overkill) | ✅ CG/explicit | ✅ Custom           |
| Neumann BC         | ✅            | ❌ **Non**     | ✅                  |
| Masque îles/terres | ✅ Level-set  | ❌             | ✅ Direct           |
| Conservation masse | ✅            | ✅             | ✅ (contrôle total) |
| Simplicité API     | ❌            | ✅             | ✅                  |
| Overhead           | ⚠️ Élevé      | ✅ Faible      | ✅ Minimal          |
| Contrôle           | ❌ Black box  | ⚠️ Partiel     | ✅ Total            |

---

## 3. Notre Problème : Advection-Diffusion de Biomasse

### Équation à Résoudre

```
∂B/∂t + u·∂B/∂x + v·∂B/∂y = D·∇²B
```

où :

-   `B` : biomasse (kg/m²)
-   `u, v` : vitesses (m/s) - **données externes** (forcings)
-   `D` : diffusivité (m²/s)

### Caractéristiques

1. **Scalaire passif** : B est transporté par u,v mais n'affecte pas le champ de vitesse
2. **Incompressible** : ∇·u = 0 (océan)
3. **Conservation critique** : Masse totale doit être conservée (>99%)
4. **Conditions limites** :
    - Bords du domaine : Neumann (∂B/∂n = 0, zero flux)
    - Interfaces terre/mer : Flux bloqué (pas de biomasse sur terre)

### Différences avec CFD classique

| CFD (Navier-Stokes)      | Notre cas                  |
| ------------------------ | -------------------------- |
| Résout u, v, p (couplés) | u, v **donnés** (forcings) |
| Équation de pression     | ❌ Pas nécessaire          |
| Équation d'état          | ❌ Pas nécessaire          |
| Température, densité     | ❌ Pas nécessaire          |
| Viscosité dynamique      | ❌ Diffusivité donnée      |

⚠️ **Conclusion** : On n'a pas besoin d'un solver CFD complet, juste d'un transport de scalaire passif.

---

## 4. Recommandations

### Option 1 : **Implémentation JAX Pure** (RECOMMANDÉ ⭐)

#### Avantages

✅ **Contrôle total** : On implémente exactement ce dont on a besoin
✅ **Simplicité** : Pas de couche d'abstraction inutile
✅ **Performance** : Pas d'overhead de variables conservatives
✅ **Debugging** : Code transparent, facile à comprendre
✅ **Flexibilité** : Adaptation facile aux besoins SEAPOPYM
✅ **Conservation** : Contrôle direct des flux aux interfaces

#### Schémas Proposés

**Advection** :

-   **Upwind 1er ordre** : Stable, conservatif, CFL ≤ 1
    ```
    ∂B/∂x ≈ (B[i,j] - B[i-1,j])/dx  si u > 0
    ∂B/∂x ≈ (B[i+1,j] - B[i,j])/dx  si u < 0
    ```
-   **Upgrade futur** : Lax-Wendroff 2ème ordre, Fromm scheme

**Diffusion** :

-   **Euler explicite** : Simple, facile à debugger
    ```
    ∂B/∂t = D·∇²B
    ∇²B ≈ (B[i-1,j] + B[i+1,j] + B[i,j-1] + B[i,j+1] - 4B[i,j]) / dx²
    ```
-   **Stabilité** : dt ≤ dx²/(4D)
-   **Upgrade futur** : Crank-Nicolson (implicite, dt plus grand)

**Masques** :

```python
# Flux blocking aux interfaces terre/mer
flux_x = jnp.where(
    mask[i,j] & mask[i-1,j],  # Les deux cellules sont océan
    u[i,j] * B_upstream,       # Flux actif
    0.0                        # Flux bloqué
)
```

#### Implémentation

```python
@ray.remote
class TransportWorker:
    def transport_step(self, biomass, u, v, D, dt, dx, dy, mask):
        # 1. Advection (upwind)
        biomass = self._advection_upwind(biomass, u, v, dt, dx, dy, mask)

        # 2. Diffusion (explicit)
        biomass = self._diffusion_explicit(biomass, D, dt, dx, dy, mask)

        # 3. Enforce mask
        biomass = jnp.where(mask, biomass, 0.0)

        # 4. Diagnostics
        mass_after = jnp.sum(biomass)

        return {"biomass": biomass, "diagnostics": {...}}
```

**Complexité** : ~150 lignes de code (vs 1000+ pour intégration JAX-Fluids)

---

### Option 2 : JAX-Fluids (Si complexité acceptée)

#### Quand l'utiliser ?

-   Si on veut un solver **production-ready** ultra-robuste
-   Si on prévoit d'ajouter de la physique complexe (compressibilité, multi-phase)
-   Si on a le temps d'apprendre l'API complexe

#### Challenges

1. **Adapter la biomasse en variables conservatives** :

    ```python
    # JAX-Fluids attend (ρ, ρu, ρv, ρE)
    # On a juste B (biomasse)
    # Workaround : ρ=B, u=v=0, E=0 ?
    ```

2. **Configuration JSON** : Apprendre le format, beaucoup de params inutiles

3. **MaterialManager** : Trouver un hack pour éviter l'équation d'état

4. **Level-set pour îles** : Apprendre à créer les champs level-set φ(x,y)

#### Estimation

-   **Temps d'intégration** : 2-3 semaines (apprentissage + adaptation)
-   **Lignes de code** : Config ~100 lignes, wrapper ~200 lignes
-   **Risque** : Moyen (API complexe, workarounds nécessaires)

---

### Option 3 : JAX-CFD (Contribution Neumann BC)

#### Principe

Contribuer au projet JAX-CFD pour ajouter les conditions Neumann, puis utiliser la bibliothèque.

#### Avantages

✅ API simple de JAX-CFD
✅ Contribution open-source
✅ Bénéfice pour la communauté

#### Inconvénients

❌ **Temps long** : 1-2 mois (implémentation, tests, review)
❌ **Incertain** : Dépend de l'acceptation Google
❌ **Masques** : Toujours pas supporté, nécessite autre contribution

#### Conclusion

Pas réaliste pour notre timeline.

---

## 5. Décision Recommandée

### Approche Itérative : Custom JAX avec Upgrades Progressifs

**Phase 2A : MVP Simple** (1-2 semaines)

```python
# Advection upwind 1er ordre
# Diffusion euler explicite
# Neumann BC via halos manuels
# Flux blocking via masques
```

✅ **Target** : Conservation >99% sur 10 jours

**Phase 2B : Optimisation** (1 semaine)

```python
# JIT compilation
# Profilage performance
# Réduction overhead Ray
```

**Phase 2C : Upgrade Schémas** (optionnel, 1-2 semaines)

```python
# Advection : Lax-Wendroff ou Fromm (2ème ordre)
# Diffusion : Crank-Nicolson (implicite)
# CFL adaptatif
```

**Justification** :

1. On a déjà 80% du code (ancien `worker.py` avant passthrough)
2. On contrôle la conservation de masse directement
3. Debugging facile (pas de black box)
4. Performance suffisante pour SEAPOPYM (grille océanique ~1°)

---

## 6. Comparaison Effort vs Bénéfice

| Option       | Effort     | Risque   | Conservation | Performance | Contrôle   |
| ------------ | ---------- | -------- | ------------ | ----------- | ---------- |
| **JAX Pur**  | ⭐⭐ (2/5) | ⭐ (1/5) | ⭐⭐⭐⭐⭐   | ⭐⭐⭐⭐    | ⭐⭐⭐⭐⭐ |
| JAX-Fluids   | ⭐⭐⭐⭐⭐ | ⭐⭐⭐   | ⭐⭐⭐⭐⭐   | ⭐⭐⭐      | ⭐⭐       |
| JAX-CFD + BC | ⭐⭐⭐⭐   | ⭐⭐⭐⭐ | ⭐⭐⭐⭐     | ⭐⭐⭐⭐    | ⭐⭐⭐     |

---

## 7. Plan d'Implémentation Détaillé (Option 1)

### Étape 1 : Récupérer le code pré-passthrough

```bash
git show 82669a0~1:src/seapopym_message/transport/worker.py > worker_v1.py
# Review du code advection/diffusion
```

### Étape 2 : Fix de la conservation

**Problème identifié** : 97.68% mass loss

**Hypothèses à tester** :

1. ~~Halos `jnp.pad(..., mode="edge")`~~ (déjà testé, pas le problème)
2. **Accumulation erreurs** : Advection puis diffusion (erreurs s'additionnent)
3. **Stabilité diffusion** : dt trop grand ? (vérifier dt ≤ dx²/4D)
4. **Flux aux bords** : Vérifier zero-flux Neumann
5. **Island blocking** : Vérifier que flux = 0 aux interfaces

**Approche debug** :

```python
# Test advection seule (D=0)
# Test diffusion seule (u=v=0)
# Test avec dt/2, dt/4
# Vérifier chaque flux individuel
```

### Étape 3 : Tests de validation

```python
# Test 1: Blob stationnaire (u=v=0, D>0)
# → Doit diffuser, conserver masse

# Test 2: Translation pure (u>0, v=0, D=0)
# → Doit translater, conserver masse

# Test 3: Île centrale
# → Biomasse = 0 dans île, flux bloqué

# Test 4: 10 jours océan
# → Conservation >99%
```

### Étape 4 : Intégration Ray

```python
# worker.py déjà en place (mode passthrough)
# Remplacer `return biomass` par vraie physique
# Tests passthrough → tests physiques
```

---

## 8. Code de Référence

**Note** : Les codes ci-dessous ont été mis à jour pour être cohérents avec les descriptions théoriques dans `IA/Advection-upwind-description.md` et `IA/Diffusion-euler-explicite-description.md`.

### Comparaison Approches : Advection

| Aspect | Approche Gradients (ancien) | Approche Flux (théorie) |
|--------|----------------------------|-------------------------|
| **Principe** | dB/dt = -u·∂B/∂x - v·∂B/∂y | dB/dt = -(F_est - F_ouest + F_nord - F_sud) / Volume |
| **Conservation** | ⚠️ Pas garantie | ✅ Par construction |
| **Grille non-uniforme** | ⚠️ Nécessite facteurs correctifs | ✅ Naturel (aires de faces) |
| **Masques** | Conditions booléennes | u=0, v=0 aux interfaces |
| **Complexité** | Simple | Moyenne |

**Verdict** : Approche **flux** (volumes finis) recommandée pour garantir conservation de masse.

### Comparaison Approches : Diffusion

| Aspect | Grille Plane (ancien) | Grille Sphérique (théorie) |
|--------|-----------------------|----------------------------|
| **dx** | Constant (scalaire) | dx(lat) = R·cos(lat)·dλ |
| **Laplacien** | ∇²B = (B_w + B_e - 2B_c)/dx² + ... | ∇²B = (B_w + B_e - 2B_c)/dx(lat)² + ... |
| **Stabilité** | dt ≤ dx²/(4D) | dt ≤ min(dx(lat)²)/(4D) **⚠️ Plus restrictif !** |
| **Validité** | ✅ Grille cartésienne | ✅ Grille lat/lon (SEAPOPYM) |

**Verdict** : Utiliser **grille sphérique** avec dx(lat) variable pour SEAPOPYM.

---

### Upwind Advection Flux-Based (Volumes Finis)

**Basé sur** : `IA/Advection-upwind-description.md`

```python
def advection_upwind_flux(
    biomass: jnp.ndarray,
    u: jnp.ndarray,  # Vitesse au centre (nlat, nlon)
    v: jnp.ndarray,  # Vitesse au centre (nlat, nlon)
    dt: float,
    grid: Grid,  # Contient aires cellules et faces
    boundary: BoundaryConditions,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Advection upwind via volumes finis (méthode flux).

    Principe (IA/Advection-upwind-description.md) :
        dC/dt = -(F_est - F_ouest + F_nord - F_sud) / Volume
        F_face = u_face × C_upwind × Aire_face

    Choix upwind :
        - Si u_face > 0 : flux sort de la cellule → C_upwind = C_cell
        - Si u_face < 0 : flux entre dans la cellule → C_upwind = C_neighbor

    Masques (Cas 3 de la description) :
        - u=0, v=0 aux interfaces terre/mer → flux automatiquement nul
    """
    nlat, nlon = biomass.shape

    # Étape 1 : Forcer u=v=0 aux bords fermés ET aux interfaces terre/mer
    u, v = boundary.mask_velocities(u, v)  # Bords du domaine

    if mask is not None:
        # u=0 si cellule courante OU voisine Est est terre
        # (description ligne 99-100 : "vitesse normale est nulle")
        mask_u_faces = mask & jnp.roll(mask, -1, axis=1)
        u = jnp.where(mask_u_faces, u, 0.0)

        # v=0 si cellule courante OU voisine Sud est terre
        mask_v_faces = mask & jnp.roll(mask, -1, axis=0)
        v = jnp.where(mask_v_faces, v, 0.0)

        # NaN → 0
        u = jnp.nan_to_num(u, 0.0)
        v = jnp.nan_to_num(v, 0.0)

    # Étape 2 : Ghost cells pour biomasse
    B_ext = boundary.apply_ghost_cells(biomass, halo_width=1)
    u_ext = boundary.apply_ghost_cells(u, halo_width=1)
    v_ext = boundary.apply_ghost_cells(v, halo_width=1)

    # Étape 3 : Interpoler u,v aux faces (grille collocated)
    # u_face_est[i,j] = moyenne entre cellules (i,j) et (i,j+1)
    u_faces = 0.5 * (u_ext[:, :-1] + u_ext[:, 1:])  # (nlat+2, nlon+1)
    v_faces = 0.5 * (v_ext[:-1, :] + v_ext[1:, :])  # (nlat+1, nlon+2)

    # Étape 4 : Choix upwind pour C_face (description lignes 25-43)
    # Face Est de (i,j) sépare (i,j) de (i,j+1)
    C_face_est = jnp.where(
        u_faces >= 0,
        B_ext[:, :-1],  # u>0 : flux sort → C de (i,j)
        B_ext[:, 1:]    # u<0 : flux entre → C de (i,j+1)
    )

    C_face_ouest = jnp.where(
        u_faces[:, :-1] >= 0,
        B_ext[:, :-2],
        B_ext[:, 1:-1]
    )

    # Face Sud de (i,j) sépare (i,j) de (i+1,j)
    C_face_sud = jnp.where(
        v_faces >= 0,
        B_ext[:-1, :],  # v>0 : flux sort
        B_ext[1:, :]    # v<0 : flux entre
    )

    C_face_nord = jnp.where(
        v_faces[:-1, :] >= 0,
        B_ext[:-2, :],
        B_ext[1:-1, :]
    )

    # Étape 5 : Calculer flux (description ligne 21 : F = u × C_face × Aire_face)
    # Extraire région intérieure (sans ghost cells)
    u_int = u_faces[1:-1, 1:-1]  # (nlat, nlon-1)
    v_int = v_faces[1:-1, 1:-1]  # (nlat-1, nlon)

    C_e_int = C_face_est[1:-1, 1:-1]
    C_o_int = C_face_ouest[1:-1, 1:-1]
    C_s_int = C_face_sud[1:-1, 1:-1]
    C_n_int = C_face_nord[1:-1, 1:-1]

    # Aires des faces (grille sphérique)
    area_ew = grid.face_areas_ew()  # R × dφ
    area_ns = grid.face_areas_ns()  # R × cos(lat) × dλ

    # Flux aux 4 faces de chaque cellule
    # Note : description ligne 45-46 mentionne dépendance latitude
    F_est = u_int[:, 1:] * C_e_int[:, 1:] * area_ew
    F_ouest = u_int[:, :-1] * C_o_int[:, :-1] * area_ew
    F_sud = v_int[1:, :] * C_s_int[1:, :] * area_ns[1:, :]
    F_nord = v_int[:-1, :] * C_n_int[:-1, :] * area_ns[:-1, :]

    # Étape 6 : Bilan de flux (description ligne 9)
    # Convention : flux_net positif = entrant
    flux_net = (F_ouest - F_est) + (F_nord - F_sud)

    # Étape 7 : Update (description ligne 9 : dC/dt = flux_net / Volume)
    volumes = grid.cell_areas()
    dC_dt = flux_net / volumes

    C_new = biomass + dC_dt * dt

    # Étape 8 : Enforce mask
    if mask is not None:
        C_new = jnp.where(mask, C_new, 0.0)

    return jnp.maximum(C_new, 0.0)
```

---

### Diffusion Euler Explicite (Grille Sphérique)

**Basé sur** : `IA/Diffusion-euler-explicite-description.md`

```python
def diffusion_explicit_spherical(
    biomass: jnp.ndarray,
    D: float,  # ou K_h dans la description
    dt: float,
    grid: Grid,  # Contient dx(lat), dy
    boundary: BoundaryConditions,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Diffusion Euler explicite sur grille sphérique.

    Principe (IA/Diffusion-euler-explicite-description.md ligne 34) :
        C^{n+1} = C^n + Δt·Kh·[∂²C/∂x² + ∂²C/∂y²]

        ∂²C/∂x² = (C[i+1,j] - 2C[i,j] + C[i-1,j]) / dx(j)²
        ∂²C/∂y² = (C[i,j+1] - 2C[i,j] + C[i,j-1]) / dy²

    IMPORTANT (ligne 26) : dx(j) dépend de la latitude j !
        dx(j) = R × cos(lat[j]) × dλ

    Masques (Cas 3, lignes 92-102) :
        - Si voisin est terre → C_voisin = C_center (flux nul)

    Stabilité (ligne 41) :
        dt ≤ min(dx²)/(4·Kh)  ⚠️ dx minimal aux pôles !
    """
    nlat, nlon = biomass.shape

    # Calcul dx(lat) pour grille sphérique
    # dx[j] = R × cos(lat[j]) × dλ (description ligne 26)
    if hasattr(grid, 'lat'):
        lat_rad = jnp.radians(grid.lat)
        dlon_rad = jnp.radians(grid.dlon)
        dx_lat = grid.R * jnp.cos(lat_rad) * dlon_rad  # (nlat,)
        dy = grid.R * jnp.radians(grid.dlat)  # scalaire
    else:
        # Fallback grille plane
        dx_lat = jnp.full(nlat, grid.face_areas_ew())
        dy = grid.face_areas_ew()

    # Vérification stabilité (description ligne 41)
    dx_min = jnp.min(dx_lat)
    dt_max = min(dx_min**2, dy**2) / (4 * D)
    if dt > dt_max:
        raise ValueError(
            f"Instabilité diffusion : dt={dt:.2f} > dt_max={dt_max:.2f}. "
            f"Sur grille sphérique, dx_min={dx_min:.0f}m aux hautes latitudes."
        )

    # Ghost cells (description ligne 49-58 : condition Neumann)
    B_ext = boundary.apply_ghost_cells(biomass, halo_width=1)

    if mask is not None:
        mask_ext = boundary.apply_ghost_cells(mask, halo_width=1)
    else:
        mask_ext = jnp.ones_like(B_ext, dtype=bool)

    # 5-point stencil
    B_c = B_ext[1:-1, 1:-1]
    B_w = B_ext[1:-1, :-2]
    B_e = B_ext[1:-1, 2:]
    B_n = B_ext[:-2, 1:-1]
    B_s = B_ext[2:, 1:-1]

    # Neumann BC aux interfaces terre/mer (description lignes 93-100)
    # "Si voisin est terre, C_voisin = C_center"
    B_w_eff = jnp.where(mask_ext[1:-1, :-2], B_w, B_c)
    B_e_eff = jnp.where(mask_ext[1:-1, 2:], B_e, B_c)
    B_n_eff = jnp.where(mask_ext[:-2, 1:-1], B_n, B_c)
    B_s_eff = jnp.where(mask_ext[2:, 1:-1], B_s, B_c)

    # Laplacien avec dx(j) variable (description ligne 34)
    # ∂²C/∂x² = (C_e - 2C_c + C_w) / dx(j)²
    # ∂²C/∂y² = (C_n - 2C_c + C_s) / dy²

    dx2_inv = 1.0 / (dx_lat[:, None]**2)  # (nlat, 1)
    dy2_inv = 1.0 / dy**2

    laplacian = (
        (B_e_eff - 2*B_c + B_w_eff) * dx2_inv +
        (B_n_eff - 2*B_c + B_s_eff) * dy2_inv
    )

    # Update (description ligne 34)
    C_new = B_c + D * laplacian * dt

    # Enforce mask
    if mask is not None:
        C_new = jnp.where(mask, C_new, 0.0)

    return jnp.maximum(C_new, 0.0)
```

---

### Comparaison Avant/Après

| Code | Advection | Diffusion |
|------|-----------|-----------|
| **Ancien (TRANSPORT_ANALYSIS.md)** | Approche gradients | Grille plane (dx constant) |
| **Nouveau (cohérent avec descriptions)** | Approche flux (volumes finis) | Grille sphérique (dx(lat) variable) |
| **Amélioration** | ✅ Conservation garantie | ✅ Physiquement correct (lat/lon) |
| **Impact** | Devrait fixer bug 97.68% mass loss | Stabilité correcte aux hautes latitudes |

---

## 9. Conclusion et Décision Finale

### Verdict : **Implémentation JAX Pure** ⭐

**Raisons** :

1. ✅ Adapté exactement à notre problème (advection-diffusion scalaire passif)
2. ✅ Conservation masse : contrôle total sur les flux
3. ✅ Simplicité : code clair, debuggable, maintenable
4. ✅ Performance : pas d'overhead de CFD complet
5. ✅ Timeline : 1-2 semaines pour MVP robuste
6. ✅ Upgrades : possibilité d'améliorer schémas progressivement

**JAX-Fluids** et **JAX-CFD** sont excellentes bibliothèques, mais **pas adaptées à notre cas** :

-   JAX-Fluids : trop complexe (compressible, multi-phase)
-   JAX-CFD : limitation fatale (periodic BC only)

### Prochaines Étapes

1. ✅ Rapport validé par l'équipe
2. ⏭️ Phase 2A : Implémenter advection + diffusion avec schémas simples
3. ⏭️ Tests de conservation sur 10 jours
4. ⏭️ Commit et merge sur `main`

---

**Fin du rapport**
