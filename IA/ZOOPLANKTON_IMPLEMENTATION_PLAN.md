# Plan d'Implémentation : Modèle de Zooplancton SEAPODYM-LMTL

**Date** : 2025-01-17
**Objectif** : Intégrer le modèle biologique de zooplancton à 2 compartiments dans SEAPOPYM-MESSAGE
**Référence** : `Annexe A – Formulation du modèle sans transport.md`

---

## 📋 Vue d'ensemble

### Modèle biologique

**2 compartiments** :

1. **Biomasse adulte B** (sans âge) : `∂B/∂t = R - λB`
2. **Production juvénile p(τ)** (avec âge) : `∂p/∂t + ∂p/∂τ = -μp`

**Caractéristiques** :

-   Mortalité thermosensible : `λ(T) = λ₀ exp(γ_λ (T - T_ref))`
-   Source depuis NPP : `p(τ=0) = E × NPP`
-   Recrutement par absorption totale (α→∞) dans fenêtre `[τ_r(T), τ_r0]`
-   Transport des 2 compartiments (advection + diffusion)

### Paramètres

```python
{
    "lambda_0": 1/150,        # Mortalité de base [jour⁻¹]
    "gamma_lambda": 0.15,     # Sensibilité thermique mortalité [°C⁻¹]
    "T_ref": 0.0,            # Température de référence [°C]
    "E": 0.1668,             # Efficacité transfert PP→production
    "tau_r0": 10.38,         # Âge maximal recrutement [jours]
    "gamma_tau_r": 0.11,     # Sensibilité thermique recrutement [°C⁻¹]
    "dt": 1.0,               # Pas de temps = pas d'âge [jour]
    "n_ages": 11,            # Classes d'âge [0..10]
}
```

### État du système

```python
state = {
    "biomass": jnp.ndarray,      # (nlat, nlon) - B [kg/m²]
    "production": jnp.ndarray,   # (n_ages, nlat, nlon) - p_a [kg/m²]
}
```

**Champs à transporter** : 12 (1 biomasse + 11 production)

---

## 🏗️ Architecture technique

### Workflow 6 phases (extension du workflow actuel)

```
EventScheduler.step():

  1. PHASE BIOLOGIE (parallel sur CellWorkers)
     ├─ Unit: age_production        # Vieillissement + source NPP
     ├─ Unit: compute_recruitment   # R = Σ production recrutée
     └─ Unit: update_biomass        # Euler implicite B^{n+1}

  2. COLLECTE (parallel)
     ├─ _collect_global_biomass()       # Existant
     └─ _collect_global_production()    # NOUVEAU

  3. PHASE TRANSPORT (centralisé TransportWorker)
     ├─ Transporter biomass             # Existant
     └─ Transporter production[0..10]   # NOUVEAU (boucle sur âges)

  4. REDISTRIBUTION (parallel)
     ├─ _redistribute_biomass()         # Existant
     └─ _redistribute_production()      # NOUVEAU

  5. AGRÉGATION DIAGNOSTICS
     └─ Ajouter diagnostics zooplancton
```

### Modules à créer/modifier

```
src/seapopym_message/
├── kernels/
│   └── zooplankton.py              # NOUVEAU
│       ├── compute_tau_r()
│       ├── compute_mortality()
│       ├── age_production()
│       ├── compute_recruitment()
│       └── update_biomass()
│
├── distributed/
│   ├── scheduler.py                # MODIFIER
│   │   ├── _collect_global_production()
│   │   ├── _redistribute_production()
│   │   └── step() - étendre phase transport
│   │
│   └── worker.py                   # MODIFIER
│       ├── get_production()
│       └── set_production()
│
tests/
├── unit/
│   └── test_zooplankton.py         # NOUVEAU
│
└── integration/
    └── test_bio_transport_zoo.py   # NOUVEAU

examples/
└── zooplankton_demo.ipynb          # NOUVEAU
```

---

## 📅 Phases d'implémentation

### **Phase 1** : Fonctions auxiliaires de base

**Fichier** : `src/seapopym_message/kernels/zooplankton.py`

#### 1.1 Calcul âge minimal de recrutement

```python
def compute_tau_r(temperature: jnp.ndarray, params: dict) -> jnp.ndarray:
    """Calcule τ_r(T) = τ_r0 × exp(-γ_τr × (T - T_ref)).

    Args:
        temperature: Champ de température [°C], shape (nlat, nlon)
        params: Dictionnaire avec tau_r0, gamma_tau_r, T_ref

    Returns:
        Âge minimal de recrutement [jours], shape (nlat, nlon)

    Note:
        τ_r diminue avec T (recrutement plus précoce en eaux chaudes)
    """
    tau_r0 = params["tau_r0"]
    gamma_tau_r = params["gamma_tau_r"]
    T_ref = params["T_ref"]

    tau_r = tau_r0 * jnp.exp(-gamma_tau_r * (temperature - T_ref))

    return tau_r
```

#### 1.2 Calcul mortalité thermosensible

```python
def compute_mortality(temperature: jnp.ndarray, params: dict) -> jnp.ndarray:
    """Calcule λ(T) = λ₀ × exp(γ_λ × (T - T_ref)).

    Args:
        temperature: Champ de température [°C], shape (nlat, nlon)
        params: Dictionnaire avec lambda_0, gamma_lambda, T_ref

    Returns:
        Mortalité [jour⁻¹], shape (nlat, nlon)

    Note:
        λ augmente avec T (mortalité plus élevée en eaux chaudes)
    """
    lambda_0 = params["lambda_0"]
    gamma_lambda = params["gamma_lambda"]
    T_ref = params["T_ref"]

    # Température limitée à T_ref minimum (comme dans SEAPODYM-LMTL)
    T_effective = jnp.maximum(temperature, T_ref)

    mortality = lambda_0 * jnp.exp(gamma_lambda * (T_effective - T_ref))

    return mortality
```

#### 1.3 Tests unitaires Phase 1

**Fichier** : `tests/unit/test_zooplankton.py`

```python
def test_tau_r_decreases_with_temperature():
    """τ_r doit diminuer quand T augmente."""

def test_tau_r_value_at_tref():
    """τ_r(T_ref) = τ_r0."""

def test_mortality_increases_with_temperature():
    """λ doit augmenter quand T augmente."""

def test_mortality_value_at_tref():
    """λ(T_ref) = λ₀."""
```

**Critères de validation Phase 1** :

-   ✅ τ_r(T=0°C) = 10.38 jours
-   ✅ τ_r(T=20°C) < 10.38 jours
-   ✅ λ(T=0°C) = 1/150 jour⁻¹
-   ✅ λ(T=20°C) > 1/150 jour⁻¹
-   ✅ Tests passent avec `uv run pytest tests/unit/test_zooplankton.py`

---

### **Phase 2** : Units biologiques (vieillissement + recrutement)

#### 2.1 Unit : AgeProduction

```python
@unit(
    name="age_production",
    inputs=["production"],
    outputs=["production"],
    scope="local",
    forcings=["npp", "temperature"],
)
def age_production(production, dt, params, forcings):
    """Vieillissement de la production avec source NPP et absorption.

    Algorithme :
    1. production[0] ← E × NPP (nouvelle génération)
    2. Pour age=1..n_ages-1 :
       - Si age < τ_r(T) : production[age] ← production[age-1] (vieillissement)
       - Si age ≥ τ_r(T) : production[age] ← 0 (absorbé → recruté)

    Args:
        production: Production par âge, shape (n_ages, nlat, nlon)
        dt: Pas de temps [jour]
        params: Paramètres du modèle
        forcings: {"npp": NPP [kg/m²/jour], "temperature": T [°C]}

    Returns:
        Production mise à jour, shape (n_ages, nlat, nlon)
    """
    npp = forcings["npp"]
    temperature = forcings["temperature"]

    n_ages = params["n_ages"]
    E = params["E"]

    # Calculer âge minimal de recrutement (en jours)
    tau_r = compute_tau_r(temperature, params)

    # Initialiser nouvelle production
    production_new = jnp.zeros_like(production)

    # Classe d'âge 0 : source depuis NPP
    production_new = production_new.at[0].set(E * npp)

    # Classes d'âge 1 à n_ages-1 : vieillissement avec absorption
    for age in range(1, n_ages):
        # Masque : 1 si âge < τ_r (survit), 0 si âge ≥ τ_r (recruté)
        survives = jnp.where(age < tau_r, 1.0, 0.0)

        # Vieillissement : passe de age-1 à age
        production_new = production_new.at[age].set(
            production[age - 1] * survives
        )

    return production_new
```

#### 2.2 Unit : ComputeRecruitment

```python
@unit(
    name="compute_recruitment",
    inputs=["production"],
    outputs=["recruitment"],
    scope="local",
    forcings=["temperature"],
)
def compute_recruitment(production, dt, params, forcings):
    """Calcule le recrutement R = Σ production absorbée.

    Avec α→∞, toute production atteignant τ_r(T) est recrutée.

    Args:
        production: Production par âge, shape (n_ages, nlat, nlon)
        dt: Pas de temps [jour]
        params: Paramètres du modèle
        forcings: {"temperature": T [°C]}

    Returns:
        Recrutement R [kg/m²/jour], shape (nlat, nlon)
    """
    temperature = forcings["temperature"]
    n_ages = params["n_ages"]

    # Calculer âge minimal de recrutement
    tau_r = compute_tau_r(temperature, params)

    # Sommer la production des classes d'âge ≥ τ_r
    recruitment = jnp.zeros_like(production[0])

    for age in range(n_ages):
        # Masque : 1 si âge ≥ τ_r (recruté), 0 sinon
        is_recruited = jnp.where(age >= tau_r, 1.0, 0.0)
        recruitment += production[age] * is_recruited

    return recruitment
```

#### 2.3 Unit : UpdateBiomass

```python
@unit(
    name="update_biomass",
    inputs=["biomass", "recruitment"],
    outputs=["biomass"],
    scope="local",
    forcings=["temperature"],
)
def update_biomass(biomass, recruitment, dt, params, forcings):
    """Mise à jour biomasse avec schéma Euler implicite.

    Équation : B^{n+1} = (B^n + Δt R) / (1 + Δt λ(T))

    Args:
        biomass: Biomasse adulte [kg/m²], shape (nlat, nlon)
        recruitment: Recrutement [kg/m²/jour], shape (nlat, nlon)
        dt: Pas de temps [jour]
        params: Paramètres du modèle
        forcings: {"temperature": T [°C]}

    Returns:
        Biomasse mise à jour [kg/m²], shape (nlat, nlon)
    """
    temperature = forcings["temperature"]

    # Calculer mortalité thermosensible
    mortality = compute_mortality(temperature, params)

    # Euler implicite (stable inconditionnellement)
    biomass_new = (biomass + dt * recruitment) / (1.0 + dt * mortality)

    return biomass_new
```

#### 2.4 Tests unitaires Phase 2

```python
def test_age_production_source_from_npp():
    """production[0] doit être E × NPP."""

def test_age_production_aging():
    """Production doit vieillir correctement avant τ_r."""

def test_age_production_absorption():
    """Production ≥ τ_r doit être absorbée (mise à 0)."""

def test_recruitment_sum():
    """R doit être la somme des classes recrutées."""

def test_biomass_update_no_recruitment():
    """Avec R=0, B doit décroître exponentiellement."""

def test_biomass_update_steady_state():
    """À l'équilibre : R = λB."""

def test_conservation_mass():
    """Masse totale (B + Σp_a) sans transport."""
```

**Critères de validation Phase 2** :

-   ✅ `production[0]` correctement alimenté par NPP
-   ✅ Vieillissement avant τ_r fonctionne
-   ✅ Absorption après τ_r fonctionne
-   ✅ Recrutement R calculé correctement
-   ✅ Biomasse évolue selon Euler implicite
-   ✅ Conservation masse (variation = NPP entrant - mortalité sortante)

---

### **Phase 3** : Extension CellWorker2D

**Fichier** : `src/seapopym_message/distributed/worker.py`

#### 3.1 Ajout méthodes get/set production

```python
class CellWorker2D:

    def get_production(self) -> jnp.ndarray:
        """Get current production field for this patch.

        Returns:
            Production array with shape (n_ages, nlat, nlon).
            Returns zeros if 'production' not in state.
        """
        n_ages = self.params.get("n_ages", 11)
        return self.state.get(
            "production",
            jnp.zeros((n_ages, self.nlat, self.nlon))
        )

    def set_production(self, production: jnp.ndarray) -> None:
        """Set production field for this patch (after transport).

        Args:
            production: Production array with shape (n_ages, nlat, nlon).
        """
        self.state["production"] = jnp.array(production)
```

#### 3.2 Tests unitaires Phase 3

```python
def test_worker_get_production_initial():
    """get_production doit retourner zeros si non initialisé."""

def test_worker_set_get_production():
    """set_production puis get_production doit retourner la même chose."""
```

**Critères de validation Phase 3** :

-   ✅ `get_production()` retourne (n_ages, nlat, nlon)
-   ✅ `set_production()` met à jour l'état correctement
-   ✅ Tests passent

---

### **Phase 4** : Extension EventScheduler (transport multi-champ)

**Fichier** : `src/seapopym_message/distributed/scheduler.py`

#### 4.1 Méthode collecte production

```python
def _collect_global_production(self) -> jnp.ndarray:
    """Assemble global production field from all workers.

    Collects production patches from each worker and assembles them
    into a global grid according to worker topology.

    Returns:
        Global production array with shape (n_ages, global_nlat, global_nlon).

    Raises:
        ValueError: If transport not enabled or topology not available.
    """
    if not self.transport_enabled:
        raise ValueError("Cannot collect production: transport not enabled")

    # Get n_ages from forcing_params or use default
    n_ages = self.forcing_params.get("n_ages", 11)

    # Collect production from all workers in parallel
    futures = [worker.get_production.remote() for worker in self.workers]
    patches = ray.get(futures)  # List of (n_ages, nlat_local, nlon_local)

    # Initialize global production grid
    production_global = jnp.zeros(
        (n_ages, self.global_nlat, self.global_nlon),
        dtype=jnp.float32
    )

    # Assemble patches into global grid
    for patch, topology in zip(patches, self.worker_topology, strict=True):
        lat_start = topology["lat_start"]
        lat_end = topology["lat_end"]
        lon_start = topology["lon_start"]
        lon_end = topology["lon_end"]

        # Insert patch into global grid (all ages at once)
        production_global = production_global.at[
            :, lat_start:lat_end, lon_start:lon_end
        ].set(patch)

    return production_global
```

#### 4.2 Méthode redistribution production

```python
def _redistribute_production(self, production_global: jnp.ndarray) -> None:
    """Redistribute global production field to all workers.

    Extracts patches from the global production grid and distributes
    them to the corresponding workers according to topology.

    Args:
        production_global: Global production array, shape (n_ages, global_nlat, global_nlon).

    Raises:
        ValueError: If transport not enabled or topology not available.
    """
    if not self.transport_enabled:
        raise ValueError("Cannot redistribute production: transport not enabled")

    # Extract and send patches to workers
    futures = []
    for worker, topology in zip(self.workers, self.worker_topology, strict=True):
        lat_start = topology["lat_start"]
        lat_end = topology["lat_end"]
        lon_start = topology["lon_start"]
        lon_end = topology["lon_end"]

        # Extract patch from global grid (all ages)
        patch = production_global[:, lat_start:lat_end, lon_start:lon_end]

        # Send to worker (non-blocking)
        futures.append(worker.set_production.remote(patch))

    # Wait for all workers to receive their production
    ray.get(futures)
```

#### 4.3 Modification step() - phase transport

```python
def step(self) -> dict[str, Any]:
    """Execute one synchronized timestep across all workers.

    [...docstring existant...]
    """
    # [...code existant jusqu'à PHASE 2-4 Transport...]

    # PHASE 2-4: Transport (if enabled)
    transport_diag = None
    if self.transport_enabled:
        assert self.transport_worker is not None

        # PHASE 2: Collect global biomass + production
        biomass_global = self._collect_global_biomass()
        production_global = self._collect_global_production()  # NOUVEAU

        # PHASE 3: Transport step
        transport_forcings = self._get_transport_forcings(forcings_ref)

        # Transport biomasse
        biomass_result = ray.get(
            self.transport_worker.transport_step.remote(
                biomass=biomass_global,
                u=transport_forcings["u"],
                v=transport_forcings["v"],
                D=transport_forcings["D"],
                dt=self.dt,
                mask=transport_forcings.get("mask"),
            )
        )
        biomass_global = biomass_result["biomass"]
        transport_diag = biomass_result["diagnostics"]

        # Transport production (boucle sur classes d'âge)
        n_ages = self.forcing_params.get("n_ages", 11)
        production_transported = jnp.zeros_like(production_global)

        for age in range(n_ages):
            prod_result = ray.get(
                self.transport_worker.transport_step.remote(
                    biomass=production_global[age],  # (nlat, nlon)
                    u=transport_forcings["u"],
                    v=transport_forcings["v"],
                    D=transport_forcings["D"],
                    dt=self.dt,
                    mask=transport_forcings.get("mask"),
                )
            )
            production_transported = production_transported.at[age].set(
                prod_result["biomass"]
            )

        # PHASE 4: Redistribute biomass + production
        self._redistribute_biomass(biomass_global)
        self._redistribute_production(production_transported)  # NOUVEAU

    # [...reste du code existant...]
```

#### 4.4 Tests unitaires Phase 4

```python
def test_collect_redistribute_production_conservation():
    """Collecte puis redistribution doit conserver production totale."""

def test_transport_all_ages():
    """Transport de toutes les classes d'âge."""

def test_integrated_bio_transport_zoo():
    """Test intégration complète bio + transport zooplancton."""
```

**Critères de validation Phase 4** :

-   ✅ `_collect_global_production()` assemble correctement
-   ✅ `_redistribute_production()` redistribue correctement
-   ✅ Conservation masse production lors collect/redistribute
-   ✅ Transport de toutes les classes d'âge fonctionne
-   ✅ Workflow complet 6 phases opérationnel

---

### **Phase 5** : Tests d'intégration

**Fichier** : `tests/integration/test_bio_transport_zoo.py`

#### 5.1 Test conservation masse totale

```python
def test_conservation_total_mass_bio_transport():
    """Test conservation de B + Σp_a avec biologie + transport.

    Configuration :
    - 2×2 workers
    - Grille 20×20
    - NPP constant
    - Température constante
    - Transport avec courant + diffusion
    - 100 timesteps

    Vérification :
    - Masse initiale = Masse finale + Masse perdue par mortalité
    - Masse entrante (NPP) = Masse sortante (mortalité) à l'équilibre
    """
```

#### 5.2 Test état d'équilibre

```python
def test_steady_state_without_transport():
    """Test atteinte de l'équilibre sans transport.

    À l'équilibre sans transport :
    - R ≈ λB (recrutement = mortalité)
    - Production stable dans chaque classe d'âge
    """
```

#### 5.3 Test sensibilité thermique

```python
def test_temperature_effect_on_recruitment():
    """Test effet température sur recrutement.

    Vérifier :
    - τ_r diminue avec T (recrutement plus précoce)
    - λ augmente avec T (mortalité plus élevée)
    - Biomasse finale différente entre eau froide et eau chaude
    """
```

**Critères de validation Phase 5** :

-   ✅ Conservation masse totale > 99.9%
-   ✅ Équilibre R ≈ λB atteint (écart < 1%)
-   ✅ Effet température cohérent (τ_r↓ et λ↑ quand T↑)

---

### **Phase 6** : Notebook de démonstration

**Fichier** : `examples/zooplankton_demo.ipynb`

#### Sections du notebook

1. **Introduction et équations**

    - Présentation du modèle
    - Paramètres
    - Diagramme conceptuel

2. **Configuration du système**

    - Grille 50×50
    - 2×2 workers
    - Forcings synthétiques (NPP et T)

3. **Cas 1 : Biologie seule (sans transport)**

    - Initialisation à 0
    - Spin-up 100 jours
    - Évolution vers équilibre
    - Graphiques : B(t), R(t), Σp_a(t)

4. **Cas 2 : Biologie + Transport**

    - Blob de NPP localisé
    - Gradient de température
    - Courant vers l'est
    - Animation dispersion biomasse
    - Conservation masse

5. **Cas 3 : Sensibilité thermique**

    - Comparaison eau froide vs eau chaude
    - Impact sur τ_r et λ
    - Impact sur biomasse finale

6. **Diagnostics**
    - Production par classe d'âge
    - Spectre d'âge spatial
    - Comparaison avec/sans transport

**Critères de validation Phase 6** :

-   ✅ Notebook exécute sans erreur
-   ✅ Visualisations claires et informatives
-   ✅ Conservation masse vérifiée graphiquement
-   ✅ Effets température visibles

---

## 📊 Tests et validation

### Tests unitaires

```bash
# Phase 1 : Fonctions auxiliaires
uv run pytest tests/unit/test_zooplankton.py::test_tau_r -v
uv run pytest tests/unit/test_zooplankton.py::test_mortality -v

# Phase 2 : Units biologiques
uv run pytest tests/unit/test_zooplankton.py::test_age_production -v
uv run pytest tests/unit/test_zooplankton.py::test_recruitment -v
uv run pytest tests/unit/test_zooplankton.py::test_biomass_update -v

# Phase 3 : CellWorker
uv run pytest tests/unit/test_worker.py::test_production_methods -v

# Phase 4 : EventScheduler
uv run pytest tests/unit/test_scheduler.py::test_collect_production -v
uv run pytest tests/unit/test_scheduler.py::test_transport_production -v
```

### Tests d'intégration

```bash
uv run pytest tests/integration/test_bio_transport_zoo.py -v
```

### Validation scientifique

**Critères** :

1. ✅ Conservation masse totale (B + Σp_a) > 99.9%
2. ✅ Équilibre sans transport : R ≈ λB (écart < 1%)
3. ✅ τ_r(T=0°C) = 10.38 jours
4. ✅ τ_r(T=20°C) ≈ 10.38 × exp(-0.11 × 20) ≈ 1.16 jours
5. ✅ λ(T=0°C) = 1/150 jour⁻¹
6. ✅ λ(T=20°C) ≈ (1/150) × exp(0.15 × 20) ≈ 0.134 jour⁻¹
7. ✅ Biomasse finale cohérente avec littérature

---

## 📝 Documentation

### Docstrings

Toutes les fonctions doivent avoir :

-   Description claire
-   Args avec types et unités
-   Returns avec types et unités
-   Notes sur équations/références
-   Exemples si pertinent

### Fichiers de documentation

**À créer** :

-   `IA/ZOOPLANKTON_EQUATIONS.md` : Dérivation mathématique détaillée
-   `IA/ZOOPLANKTON_PARAMETERS.md` : Guide des paramètres et calibration
-   `IA/ZOOPLANKTON_VALIDATION.md` : Résultats de validation

**À mettre à jour** :

-   `README.md` : Ajouter section zooplancton
-   `IA/INTEGRATION_PLAN.md` : Référencer intégration zooplancton

---

## ⏱️ Timeline estimée

| Phase     | Tâche                            | Durée estimée | Dépendances |
| --------- | -------------------------------- | ------------- | ----------- |
| 1         | Fonctions auxiliaires + tests    | 1h            | -           |
| 2         | Units biologiques + tests        | 2h            | Phase 1     |
| 3         | Extension CellWorker + tests     | 1h            | Phase 2     |
| 4         | Extension EventScheduler + tests | 2h            | Phase 3     |
| 5         | Tests d'intégration              | 1h            | Phase 4     |
| 6         | Notebook démonstration           | 2h            | Phase 5     |
| **Total** |                                  | **9h**        |             |

**Note** : Timeline pour développement initial. Validation scientifique et calibration nécessiteront du temps additionnel.

---

## 🚀 Ordre d'exécution

1. **Créer le module** : `src/seapopym_message/kernels/zooplankton.py` (Phase 1)
2. **Tester fonctions de base** : `tests/unit/test_zooplankton.py` (Phase 1)
3. **Implémenter Units** dans `zooplankton.py` (Phase 2)
4. **Tester Units** (Phase 2)
5. **Modifier CellWorker2D** (Phase 3)
6. **Tester CellWorker2D** (Phase 3)
7. **Modifier EventScheduler** (Phase 4)
8. **Tester EventScheduler** (Phase 4)
9. **Tests d'intégration** (Phase 5)
10. **Créer notebook** (Phase 6)
11. **Commit et documentation** finale

---

## ✅ Checklist avant commit

Avant chaque commit majeur :

-   [ ] Tous les tests unitaires passent
-   [ ] Tous les tests d'intégration passent
-   [ ] `uv run ruff format` appliqué
-   [ ] `uv run ruff check --fix` appliqué
-   [ ] `uv run mypy` sans erreur
-   [ ] Docstrings complètes (pydocstyle)
-   [ ] Conservation masse vérifiée
-   [ ] Commit message descriptif

---

## 📚 Références

1. **Annexe A – Formulation du modèle sans transport.md** : Document source
2. **IA/INTEGRATION_PLAN.md** : Architecture transport actuelle
3. **IA/TRANSPORT_IMPLEMENTATION_PLAN.md** : Implémentation transport physique

---

## 🎯 Résultat attendu

À la fin de l'implémentation :

✅ Modèle de zooplancton SEAPODYM-LMTL fonctionnel
✅ Intégration complète biologie + transport
✅ Conservation masse > 99.9%
✅ Tests couvrant tous les cas
✅ Notebook de démonstration opérationnel
✅ Documentation complète
✅ Prêt pour calibration et validation scientifique
