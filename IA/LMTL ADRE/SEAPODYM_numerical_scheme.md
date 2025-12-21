# SEAPODYM - Schéma Numérique et Résolution du Système

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Équations résolues](#équations-résolues)
3. [Schéma numérique](#schéma-numérique)
4. [Résolution du système linéaire](#résolution-du-système-linéaire)
5. [Implémentation dans le code](#implémentation-dans-le-code)
6. [Flux de calcul complet](#flux-de-calcul-complet)

---

## Vue d'ensemble

SEAPODYM résout des **équations d'advection-diffusion-réaction (ADRE)** pour modéliser le transport et la dynamique des populations de proies dans l'océan. Le modèle suit deux types de variables d'état :

- **Production** : distribution spatiale des cohortes d'âge (larvaire à juvénile)
- **Biomasse** : biomasse totale adulte (avec mortalité)

### Architecture numérique

```
┌─────────────────────────────────────────────────────────┐
│ ÉQUATIONS ADRE (continues)                              │
│ ∂ρ/∂t + ∇·(u⃗ρ) = ∇·(D∇ρ) + R(ρ)                       │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ DISCRÉTISATION (différences finies implicites)         │
│ Schéma ADI (Alternating Direction Implicit)            │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ SYSTÈMES LINÉAIRES TRIDIAGONAUX                        │
│ Direction longitude puis latitude                       │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ ALGORITHME DE THOMAS                                    │
│ Résolution O(n) - linearSolver.cpp                     │
└─────────────────────────────────────────────────────────┘
```

---

## Équations résolues

### 1. Équation ADRE générale

Pour une densité de population ρ(x, y, t), SEAPODYM résout :

```
∂ρ/∂t + ∂(uρ)/∂x + ∂(vρ)/∂y = ∂/∂x(D ∂ρ/∂x) + ∂/∂y(D ∂ρ/∂y) + R(ρ)
```

Où :
- **ρ(x, y, t)** : densité de population (g/m²)
- **u, v** : composantes du courant océanique (°/pas de temps)
- **D** : coefficient de diffusion (°²/pas de temps)
- **R(ρ)** : terme de réaction (mortalité, recrutement)

### 2. Terme d'advection

```
Advection = ∂(uρ)/∂x + ∂(vρ)/∂y
```

**Source dans le code** : `modelPrey.cpp:1435-1448`

Conversion des courants de m/s vers °/pas de temps :

```cpp
double mtodeg = 1 / (EARTH_RADIUS * M_PI / 180.0);  // m → degrés
double daytosec = 3600.0 * 24;                       // jour → secondes
double c = mtodeg * daytosec * simStep_;             // facteur de conversion

// U en degrés/pas de temps
forcingList_.at(varKeys::U).maskAndScale(activeLayers, mask_, fillval, c);
// V en degrés/pas de temps (opposé car latitudes décroissantes)
forcingList_.at(varKeys::V).maskAndScale(activeLayers, mask_, fillval, -c);
```

### 3. Terme de diffusion

```
Diffusion = D(∂²ρ/∂x² + ∂²ρ/∂y²)
```

**Source dans le code** : `modelPrey.cpp:1458`

```cpp
double m2todeg2 = mtodeg * mtodeg;
double diffCoeff = param_.tlDiffCoeff_ * m2todeg2 * daytosec * simStep_;
```

### 4. Terme de réaction

**Pour la production** : R = 0 (transport passif)

**Pour la biomasse** : R = -μ(T)ρ (mortalité dépendante de la température)

```cpp
// functionalGroup.cpp:461
biomass_.mortalityComp(lim_, invLambdaMax_, invLambdaRate_, dT, temperatureAvg);
```

**Loi de mortalité** :
```
μ(T) = μ₀ · exp(-λ⁻¹ · T)
```

Où T a été transformé selon Gillooly et al. (2002) :
```cpp
// modelPrey.cpp:991
T_transformed = T / (1.0 + T / 273.0)
```

---

## Schéma numérique

### 1. Discrétisation temporelle : Euler implicite

Pour passer du temps t à t+Δt :

```
(ρⁿ⁺¹ - ρⁿ)/Δt = Advection(ρⁿ⁺¹) + Diffusion(ρⁿ⁺¹) + Reaction(ρⁿ⁺¹)
```

Le schéma **implicite** garantit la stabilité inconditionnelle (pas de restriction CFL stricte).

### 2. Splitting directionnel : ADI (Alternating Direction Implicit)

Le problème 2D est décomposé en deux étapes 1D :

#### Étape 1 : Direction X (longitude)

```
(ρ* - ρⁿ)/(Δt/2) = Lₓ(ρ*) + Lᵧ(ρⁿ)
```

où Lₓ contient advection et diffusion en x.

#### Étape 2 : Direction Y (latitude)

```
(ρⁿ⁺¹ - ρ*)/(Δt/2) = Lₓ(ρ*) + Lᵧ(ρⁿ⁺¹)
```

**Avantage** : chaque étape donne un système **tridiagonal** 1D au lieu d'un système 2D complexe.

### 3. Discrétisation spatiale : Différences finies centrées

Pour une cellule (i, j) :

**Advection en x** :
```
∂(uρ)/∂x ≈ (u_{i+1/2}ρ_{i+1} - u_{i-1/2}ρ_{i-1}) / (2Δx)
```

**Diffusion en x** :
```
∂²ρ/∂x² ≈ (ρ_{i+1} - 2ρᵢ + ρ_{i-1}) / (Δx)²
```

### 4. Système tridiagonal résultant

Après discrétisation, pour chaque ligne (latitude fixe) ou colonne (longitude fixe), on obtient :

```
aⱼ·ρⱼ₋₁ + bⱼ·ρⱼ + cⱼ·ρⱼ₊₁ = rhsⱼ
```

Sous forme matricielle :

```
┌                                    ┐ ┌   ┐   ┌     ┐
│ b₀  c₀   0   0  ...  0             │ │ρ₀ │   │rhs₀ │
│ a₁  b₁  c₁   0  ...  0             │ │ρ₁ │   │rhs₁ │
│  0  a₂  b₂  c₂  ...  0             │ │ρ₂ │ = │rhs₂ │
│ ...                                │ │...│   │ ... │
│  0  ...  0  aₙ  bₙ                 │ │ρₙ │   │rhsₙ │
└                                    ┘ └   ┘   └     ┘
```

#### Coefficients pour l'advection-diffusion

```
aⱼ = -D/(Δx)² + u/(2Δx)          (flux depuis j-1)
bⱼ = 1/Δt + 2D/(Δx)² + μ         (terme diagonal)
cⱼ = -D/(Δx)² - u/(2Δx)          (flux vers j+1)
rhsⱼ = ρⁿⱼ/Δt                     (valeur au temps précédent)
```

---

## Résolution du système linéaire

### Algorithme de Thomas (tridiagonal)

**Fichier** : `src/linearSolver.cpp`

L'algorithme de Thomas résout le système tridiagonal en **O(n)** opérations (vs O(n³) pour une résolution générale).

#### Phase 1 : Décomposition LU et substitution forward

**Version classique** (`tridiag()`, lignes 74-97) :

```cpp
bet = b(jmin);
u(jmin) = rhs(jmin) / bet;

for (int jl = jmin + 1; jl <= jmax; ++jl) {
    gam_(jl) = c(jl - 1) / bet;              // Facteur de décomposition
    bet = b(jl) - a(jl) * gam_(jl);          // Nouveau pivot

    if (toolkit::isSmall(bet, tol_)) {
        throw SeapodymException(..., "Algorithm fails [zero pivot]");
    }

    u(jl) = (rhs(jl) - a(jl) * u(jl - 1)) / bet;
}
```

**Interprétation** :
- `gam_` : coefficients de la décomposition LU
- `bet` : pivots diagonaux après élimination
- `u` : solution temporaire (sera modifiée en phase 2)

#### Phase 2 : Substitution backward

```cpp
for (int jl = jmax - 1; jl >= jmin; --jl) {
    u(jl) = u(jl) - gam_(jl + 1) * u(jl + 1);
}
```

Cette phase remonte de la dernière cellule vers la première pour obtenir la solution finale.

### Optimisation : Pré-calcul de l'inverse des pivots

**Version optimisée** (`tridiagOpt()`, lignes 111-149) :

Les coefficients `a`, `b`, `c` dépendent uniquement de U, V, D qui ne changent pas pendant les sous-itérations temporelles. On peut donc **pré-calculer** les inverses des pivots.

**Pré-calcul** (`betInvComp()`, lignes 152-172) :

```cpp
dvector bet(jmin, jmax);
bet(jmin) = 1.0 / b(jmin);

for (int jl = jmin + 1; jl <= jmax; ++jl) {
    bet(jl) = b(jl) - c(jl - 1) * a(jl) * bet(jl - 1);
    bet(jl) = 1.0 / bet(jl);  // ← Stocke directement l'inverse
}
return bet;
```

**Résolution avec pivots pré-calculés** :

```cpp
// Forward
for (int jl = jmin + 1; jl <= jmax; ++jl) {
    gam_(jl) = rhs(jl) - gam_(jl - 1) * a(jl) * bet(jl - 1);
}

// Backward
u(jmax) = gam_(jmax) * bet(jmax);
for (int jl = jmax - 1; jl >= jmin; --jl) {
    u(jl) = (gam_(jl) - c(jl) * u(jl + 1)) * bet(jl);
}
```

**Gain** : Évite les divisions répétées (opération coûteuse) lors des multiples résolutions avec le même système.

### Conditions limites

Les conditions aux limites sont gérées par le type de domaine :

**Source** : `modelPrey.cpp:330-348`

```cpp
switch (bcs_) {
    case bcsType::OPEN:      // Flux libre aux bords
        map_ = new openDomain(xlon, ylat, mask);
        break;
    case bcsType::CLOSED:    // Flux nul aux bords (terre)
        map_ = new closedDomain(xlon, ylat, mask);
        break;
    case bcsType::CEW:       // Périodique en longitude (Cyclic East-West)
        map_ = new cewDomain(xlon, ylat, mask);
        break;
}
```

Les conditions limites modifient les coefficients `a`, `b`, `c` aux bords du domaine.

---

## Implémentation dans le code

### 1. Calcul des coefficients ADRE

**Fichier** : `src/population.cpp` (non lu encore, mais appelé dans `functionalGroup.cpp`)

```cpp
// functionalGroup.cpp:429
production_.advectionDiffusionComp(lim_, zonCurrentAvg, merCurrentAvg, diffCoeff);

// functionalGroup.cpp:459
biomass_.advectionDiffusionComp(lim_, zonCurrentAvg, merCurrentAvg, diffCoeff);
biomass_.mortalityComp(lim_, invLambdaMax_, invLambdaRate_, dT, temperatureAvg);
```

Ces fonctions remplissent `adreCoeffs_` avec les matrices `a`, `b`, `c`.

### 2. Construction du système linéaire

```cpp
// functionalGroup.cpp:431, 463
production_.solver_->computeSystem(production_.adreCoeffs_);
biomass_.solver_->computeSystem(biomass_.adreCoeffs_);
```

Cette étape :
1. Assemble les coefficients tridiagonaux
2. Pré-calcule les inverses des pivots (optimisation)

### 3. Résolution pour chaque cohorte

**Production** (multiples cohortes en parallèle MPI) :

```cpp
// functionalGroup.cpp:434-441
for (auto cl = production_.begin(); cl != production_.end(); ++cl) {
    (*cl).transport(startDate);  // Résout ADRE pour cette cohorte
}
```

**Biomasse** (une seule cohorte sur rang 0) :

```cpp
// functionalGroup.cpp:465
biomass_.pop_[0].transport(startDate);
```

### 4. Appel au solveur linéaire

À l'intérieur de `cohort::transport()` :

```cpp
// Pseudo-code basé sur la structure observée
for (int iter = 0; iter < nbIterations_; ++iter) {
    // Résolution en X (longitude)
    for (int j = 0; j < nlat; ++j) {
        solver_->tridiagOpt(a_x, bet_x, c_x, rhs_x, u_x);
    }

    // Résolution en Y (latitude)
    for (int i = 0; i < nlon; ++i) {
        solver_->tridiagOpt(a_y, bet_y, c_y, rhs_y, u_y);
    }
}
```

**Itérations multiples** : `modelPrey.cpp:383`

```cpp
nbIterations_ = (simStep_ * 24) / param_.integrationStep_;
```

Exemple : simStep = 7 jours, integrationStep = 1 heure → 168 itérations par pas de temps.

### 5. Optimisations numériques

**Early exit si densité nulle** (`linearSolver.cpp:48-53`) :

```cpp
if (max(rhs) < tol_) {
    return 0;  // Pas de résolution inutile
}
```

**Détection de pivot nul** (`linearSolver.cpp:82-94`) :

```cpp
if (toolkit::isSmall(bet, tol_)) {
    std::ostringstream msg;
    msg << "bet is closed to zero at index " << jl << ": " << bet << endl;
    console::warning(msg.str());
    throw SeapodymException(..., "Algorithm fails [zero pivot]");
}
```

Protection contre :
- CFL trop élevé
- Conditions limites mal définies
- Coefficients de diffusion négatifs

---

## Flux de calcul complet

### Boucle temporelle principale

**Fichier** : `modelPrey.cpp:1411-1760`

```
┌────────────────────────────────────────────────────────────┐
│ POUR CHAQUE PAS DE TEMPS t                                 │
└────────────────────────────────────────────────────────────┘
    │
    ├─→ [1] Lecture des forçages physiques
    │   ├─ U, V : courants (m/s)
    │   ├─ T : température (°C)
    │   ├─ NPP : production primaire (mg C/m²/jour)
    │   └─ Conversion d'unités vers °/pas de temps
    │
    ├─→ [2] Moyennes pondérées par longueur du jour
    │   └─ Couche jour vs nuit (migration verticale)
    │
    ├─→ [3] SI PRODUCTION ACTIVÉE :
    │   ├─ Distribuer NPP → cohorte larvaire (âge 0)
    │   │   └─ functionalGroup.cpp:248
    │   │
    │   ├─ Calculer temps de recrutement Tr(x,y,T)
    │   │   └─ functionalGroup.cpp:340
    │   │
    │   ├─ Transférer cohortes matures → recruitment_
    │   │   └─ functionalGroup.cpp:367
    │   │
    │   ├─ Transport de TOUTES les cohortes
    │   │   └─ functionalGroup.cpp:423
    │   │       ├─ Calcul coefficients ADRE
    │   │       ├─ Construction système linéaire
    │   │       └─ Résolution tridiagonale
    │   │
    │   └─ Vieillissement des cohortes
    │       └─ population.ageing()
    │
    ├─→ [4] SI BIOMASSE ACTIVÉE :
    │   ├─ Ajouter recrutement → biomasse
    │   │   └─ functionalGroup.cpp:290
    │   │
    │   ├─ Transport avec mortalité
    │   │   └─ functionalGroup.cpp:446
    │   │       ├─ Calcul advection-diffusion
    │   │       ├─ Calcul mortalité μ(T)
    │   │       ├─ Construction système linéaire
    │   │       └─ Résolution tridiagonale
    │   │
    │   └─ Écriture outputs (NetCDF, CSV)
    │
    └─→ [5] Sauvegarde restart si demandé
```

### Détail de la résolution ADRE

```
functionalGroup::transportProd() / transportBiom()
    │
    ├─→ population::advectionDiffusionComp()
    │   └─ Remplit adreCoeffs_ avec a, b, c
    │
    ├─→ population::mortalityComp() [biomasse uniquement]
    │   └─ Ajoute μ(T) au terme diagonal b
    │
    ├─→ solver_->computeSystem(adreCoeffs_)
    │   └─ linearSolver::betInvComp()
    │       └─ Pré-calcule bet⁻¹ pour optimisation
    │
    └─→ BOUCLE sur nbIterations_
        └─ cohort::transport()
            │
            ├─→ SPLITTING X (longitude)
            │   └─ POUR chaque ligne (latitude fixe)
            │       └─ solver_->tridiagOpt(a_x, bet_x, c_x, rhs, u)
            │           ├─ Forward substitution
            │           └─ Backward substitution
            │
            └─→ SPLITTING Y (latitude)
                └─ POUR chaque colonne (longitude fixe)
                    └─ solver_->tridiagOpt(a_y, bet_y, c_y, rhs, u)
                        ├─ Forward substitution
                        └─ Backward substitution
```

---

## Résumé : Schéma numérique complet

### Équation continue

```
∂ρ/∂t + u·∂ρ/∂x + v·∂ρ/∂y = D(∂²ρ/∂x² + ∂²ρ/∂y²) - μ(T)ρ
```

### Discrétisation

1. **Temporelle** : Euler implicite inconditionnellement stable
2. **Spatiale** : Différences finies centrées d'ordre 2
3. **Splitting** : ADI (Alternating Direction Implicit)

### Algorithme

1. **Assemblage** : Système tridiagonal par direction
2. **Résolution** : Algorithme de Thomas O(n)
3. **Optimisation** : Pré-calcul des pivots inverses

### Complexité

- **Temps** : O(n·m·k) où n×m = grille spatiale, k = itérations temporelles
- **Mémoire** : O(n·m) - stockage de la densité

### Stabilité

- **Schéma implicite** → stable quel que soit Δt
- **Condition CFL relâchée** → pas de temps adaptatifs possibles
- **Contrôle des pivots** → détection d'instabilités

---

## Fichiers sources concernés

| Fichier | Rôle |
|---------|------|
| `modelPrey.cpp:1411-1760` | Boucle temporelle principale |
| `modelPrey.cpp:1435-1458` | Conversion unités forcings |
| `functionalGroup.cpp:322-365` | Temps de recrutement Tr(T) |
| `functionalGroup.cpp:367-415` | Nouveau recrutement |
| `functionalGroup.cpp:423-476` | Appel transport ADRE |
| `linearSolver.cpp:40-107` | Algorithme Thomas classique |
| `linearSolver.cpp:111-149` | Algorithme Thomas optimisé |
| `linearSolver.cpp:152-172` | Pré-calcul pivots inverses |

---

## Références scientifiques implicites

1. **Gillooly et al. (2002)** - Loi température-développement
   - Nature 417, http://dx.doi.org/10.1038/417070a
   - `modelPrey.cpp:983-992`

2. **Algorithme de Thomas (1949)** - Résolution tridiagonale
   - Aussi appelé TDMA (TriDiagonal Matrix Algorithm)

3. **Schéma ADI** - Peaceman & Rachford (1955), Douglas & Gunn (1964)
   - Splitting directionnel pour problèmes paraboliques 2D

4. **Différences finies implicites** - Stabilité inconditionnelle
   - Richtmyer & Morton (1967), Numerical Solution of PDEs

---

## Notes pour investigation future

### Questions à clarifier

1. **Implémentation exacte du splitting ADI** :
   - Lire `src/population.cpp` et `src/cohort.cpp`
   - Vérifier l'ordre des directions (X puis Y ou Y puis X)
   - Confirmer le traitement des termes croisés

2. **Conditions limites détaillées** :
   - Lire `src/openDomain.cpp`, `src/closedDomain.cpp`, `src/cewDomain.cpp`
   - Comprendre comment elles modifient les coefficients aux bords

3. **Gestion de la migration verticale** :
   - Comment les couches jour/nuit sont-elles couplées ?
   - Moyenne pondérée par longueur du jour : `modelPrey.cpp:1453-1455`

4. **Nombre d'itérations** :
   - Relation entre `simStep_`, `integrationStep_` et précision numérique
   - Y a-t-il un critère de convergence ou nombre fixe ?

### Fichiers à analyser

- `src/population.cpp` : calcul détaillé des coefficients ADRE
- `src/cohort.cpp` : implémentation de `transport()`
- `src/domain.cpp` et dérivés : conditions limites
- `src/adreCoeffs.hpp` : structure des coefficients

---

**Document créé le** : 2025-11-13
**Basé sur l'analyse de** : `modelPrey.cpp`, `functionalGroup.cpp`, `linearSolver.cpp`
**Objectif** : Comprendre le schéma numérique et les équations résolues par SEAPODYM
