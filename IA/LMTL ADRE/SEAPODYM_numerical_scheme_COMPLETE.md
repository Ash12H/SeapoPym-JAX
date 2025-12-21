# SEAPODYM - Schéma Numérique Complet et Détaillé

**Basé sur l'analyse de** : `adre.cpp`, `population.cpp`, `cohort.cpp`, `linearSolver.cpp`, `modelPrey.cpp`, `functionalGroup.cpp`

**Date** : 2025-11-13

---

## Table des matières

1. [Équation continue et discrétisation](#équation-continue-et-discrétisation)
2. [Schéma ADI (Alternating Direction Implicit)](#schéma-adi-alternating-direction-implicit)
3. [Coordonnées sphériques et géométrie](#coordonnées-sphériques-et-géométrie)
4. [Coefficients tridiagonaux exacts](#coefficients-tridiagonaux-exacts)
5. [Résolution numérique complète](#résolution-numérique-complète)
6. [Formules de mortalité et transformations](#formules-de-mortalité-et-transformations)
7. [Conditions limites](#conditions-limites)
8. [Analyse de stabilité et précision](#analyse-de-stabilité-et-précision)

---

## Équation continue et discrétisation

### Équation ADRE en coordonnées sphériques

SEAPODYM résout l'équation d'advection-diffusion-réaction suivante :

```
∂ρ/∂t + (1/cosφ)·∂(u·ρ)/∂λ + ∂(v·ρ·cosφ)/∂φ =
    (1/cos²φ)·∂/∂λ(Dλ·∂ρ/∂λ) + (1/cosφ)·∂/∂φ(Dφ·cosφ·∂ρ/∂φ) - μ(T)·ρ
```

Où :
- **λ** : longitude (en degrés)
- **φ** : latitude (en degrés)
- **ρ(λ, φ, t)** : densité de population (g/m²)
- **u, v** : composantes zonale et méridionale du courant (°/pas de temps)
- **Dλ, Dφ** : coefficients de diffusion (°²/pas de temps)
- **μ(T)** : taux de mortalité dépendant de la température (1/jour)
- **cosφ** : facteur géométrique sphérique

**Source** : Documentation adre.hpp:92-107, formules dans adre.cpp:306-477

### Paramètres de discrétisation

**Fichier** : `adre.cpp:116-130`

```cpp
dx_ = (lim_.xlon(2, 1) - lim_.xlon(1, 1));     // Δλ en degrés
dy_ = (lim_.ylat(1, 1) - lim_.ylat(1, 2));     // Δφ en degrés (positif)
nbIter_ = timeUnit_ / integrationStep_;         // Nombre d'itérations
twodt_ = 2.0 * (double)nbIter_;                 // 2/Δt
```

**Exemple** :
- `timeUnit_ = 168 heures` (7 jours)
- `integrationStep_ = 1 heure`
- → `nbIter_ = 168`
- → `Δt = 1/168` (en unités de timeUnit)
- → `twodt_ = 336`

---

## Schéma ADI (Alternating Direction Implicit)

### Principe du splitting

Le problème 2D est décomposé en deux étapes 1D successives :

**Étape 1 : Direction zonale (λ)** - K → K+1/2

```
(ρ* - ρⁿ)/(Δt/2) = Lλ(ρ*) + Lφ(ρⁿ)
```

**Étape 2 : Direction méridionale (φ)** - K+1/2 → K+1

```
(ρⁿ⁺¹ - ρ*)/(Δt/2) = Lλ(ρ*) + Lφ(ρⁿ⁺¹)
```

Où :
- **Lλ** : opérateur advection-diffusion en longitude + mortalité
- **Lφ** : opérateur advection-diffusion en latitude

**Important** : La mortalité est appliquée **uniquement dans la première étape** (direction zonale).

**Source** : `adre.cpp:591-695`, documentation ligne 619

### Implémentation du RHS

**Direction zonale** (`adre.cpp:645`) :

```cpp
rhsZon_(il) = -d_(il,jl) * sol(il, jl-1)
            + (4*nbIter_ - e_(il,jl)) * sol(il, jl)
            - f_(il,jl) * sol(il, jl+1);
```

Correspond à :
```
rhs = -Dφ·ρ(λ,φ-Δφ) + (2/Δt - Eφ)·ρ(λ,φ) - Dφ·ρ(λ,φ+Δφ)
```

Avec `4*nbIter_ = 2·twodt_ = 2·(2/Δt)`.

**Direction méridionale** (`adre.cpp:681`) :

```cpp
rhsMer_(jl) = -a_(jl,il) * solZon_(jl, il-1)
            + (4*nbIter_ - b_(jl,il)) * solZon_(jl, il)
            - c_(jl,il) * solZon_(jl, il+1);
```

Correspond à :
```
rhs = -Aλ·ρ*(λ-Δλ,φ) + (2/Δt - Bλ)·ρ*(λ,φ) - Cλ·ρ*(λ+Δλ,φ)
```

---

## Coordonnées sphériques et géométrie

### Facteur cos(latitude)

**Fichier** : `adre.cpp:251-253`, `290-293`

SEAPODYM prend en compte la géométrie sphérique via `cosLat(jl)` :

```cpp
// Direction longitude
a_(jl, il) = lowerDiagElemLon(..., lim_.cosLat(jl));
b_(jl, il) = diagElemLon(..., lim_.cosLat(jl));
c_(jl, il) = upperDiagElemLon(..., lim_.cosLat(jl));

// Direction latitude
d_(il, jl) = lowerDiagElemLat(..., lim_.cosLat(jl-1), lim_.cosLat(jl));
e_(il, jl) = diagElemLat(..., lim_.cosLat(jl-1), lim_.cosLat(jl), lim_.cosLat(jl+1));
f_(il, jl) = upperDiagElemLat(..., lim_.cosLat(jl), lim_.cosLat(jl+1));
```

**Raison physique** :
- À latitude φ, la longueur d'un degré de longitude = `R·cos(φ)·π/180`
- Plus on s'approche des pôles, plus les cellules sont petites en longitude
- Le terme `cos²(φ)` dans la diffusion zonale vient du laplacien sphérique

### Distance réelle vs distance angulaire

```
Distance réelle en longitude (à latitude φ) :
    Δx_réel = R · Δλ · cos(φ) · π/180

Distance réelle en latitude :
    Δy_réel = R · Δφ · π/180
```

Avec R ≈ 6371 km (rayon de la Terre).

---

## Coefficients tridiagonaux exacts

### Direction LONGITUDE (zonale)

Les coefficients `a`, `b`, `c` sont utilisés pour résoudre en longitude (étape K → K+1/2).

#### Notations

- `sigma` = `Dx(il, jl)` : diffusion à la cellule (il, jl)
- `u` = `U(il, jl)` : vitesse zonale (°/pas de temps)
- `r` = `R(il, jl)` : mortalité
- `twodd` = `2·(Δλ)²`
- `d` = `Δλ`
- `twodt` = `2/Δt`
- `cosLatj` = `cos(φⱼ)`

#### Sous-diagonale `a(jl, il)` - Cellule (il-1, jl)

**Fichier** : `adre.cpp:306-328`

```cpp
double lowerDiagElemLon(...) {
    double value = 0;
    switch (pos) {
    case OPEN:
    case R_CLOSED:
        if (uinf > 0)
            value = -((sigmam + sigma) / twodd / (cosLatj * cosLatj))
                    - (uinf / d / cosLatj);
        else
            value = -((sigmam + sigma) / twodd / (cosLatj * cosLatj));
        break;
    case L_CLOSED:
        value = 0.0;
        break;
    }
    return value;
}
```

**Formule** :

```
         ⎧ -(Dᵢ₋₁ + Dᵢ)/(2(Δλ)²cos²φ) - uᵢ₋₁/(Δλ·cosφ)   si uᵢ₋₁ > 0
a(i,j) = ⎨ -(Dᵢ₋₁ + Dᵢ)/(2(Δλ)²cos²φ)                   si uᵢ₋₁ ≤ 0
         ⎩ 0                                              si bord gauche fermé
```

**Interprétation** :
- Terme diffusion : toujours présent
- Terme advection upwind : seulement si courant venant de l'ouest (u > 0)

#### Diagonale `b(jl, il)` - Cellule (il, jl)

**Fichier** : `adre.cpp:357-390`

```cpp
double diagElemLon(...) {
    double value = 0;
    switch (pos) {
    case OPEN:
        if (u > 0)
            value = ((sigmam + 2*sigma + sigmap) / twodd / (cosLatj*cosLatj))
                    + (u / d / cosLatj) + twodt + r;
        else
            value = ((sigmam + 2*sigma + sigmap) / twodd / (cosLatj*cosLatj))
                    - (u / d / cosLatj) + twodt + r;
        break;
    // ... cas bords ...
    }
    return value;
}
```

**Formule (cellule interne OPEN)** :

```
         ⎧ (Dᵢ₋₁ + 2Dᵢ + Dᵢ₊₁)/(2(Δλ)²cos²φ) + uᵢ/(Δλ·cosφ) + 2/Δt + μ   si uᵢ > 0
b(i,j) = ⎨
         ⎩ (Dᵢ₋₁ + 2Dᵢ + Dᵢ₊₁)/(2(Δλ)²cos²φ) - uᵢ/(Δλ·cosφ) + 2/Δt + μ   si uᵢ ≤ 0
```

**Interprétation** :
- Terme diffusion centré : `(D_{i-1} + 2D_i + D_{i+1}) / (2(Δλ)²cos²φ)`
- Terme advection upwind : `±u_i / (Δλ·cosφ)` (signe selon direction)
- Terme temporel : `2/Δt` (Euler implicite avec splitting)
- Terme mortalité : `μ`

#### Sur-diagonale `c(jl, il)` - Cellule (il+1, jl)

**Fichier** : `adre.cpp:432-453`

```cpp
double upperDiagElemLon(...) {
    double value = 0;
    switch (pos) {
    case OPEN:
    case L_CLOSED:
        if (usup > 0)
            value = -((sigma + sigmap) / twodd / (cosLatj*cosLatj));
        else
            value = -((sigma + sigmap) / twodd / (cosLatj*cosLatj))
                    + (usup / d / cosLatj);
        break;
    case R_CLOSED:
        value = 0;
        break;
    }
    return value;
}
```

**Formule** :

```
         ⎧ -(Dᵢ + Dᵢ₊₁)/(2(Δλ)²cos²φ)                   si uᵢ₊₁ > 0
c(i,j) = ⎨ -(Dᵢ + Dᵢ₊₁)/(2(Δλ)²cos²φ) + uᵢ₊₁/(Δλ·cosφ)   si uᵢ₊₁ ≤ 0
         ⎩ 0                                            si bord droit fermé
```

### Direction LATITUDE (méridionale)

Les coefficients `d`, `e`, `f` sont utilisés pour résoudre en latitude (étape K+1/2 → K+1).

#### Sous-diagonale `d(il, jl)` - Cellule (il, jl-1)

**Fichier** : `adre.cpp:331-352`

```cpp
double lowerDiagElemLat(..., cosLatjm, cosLatj) {
    double value = 0;
    switch (pos) {
    case OPEN:
    case R_CLOSED:
        if (uinf > 0)
            value = -((sigmam*cosLatjm + sigma*cosLatj) / twodd / cosLatj)
                    - (uinf*cosLatjm / d / cosLatj);
        else
            value = -((sigmam*cosLatjm + sigma*cosLatj) / twodd / cosLatj);
        break;
    case L_CLOSED:
        value = 0.0;
        break;
    }
    return value;
}
```

**Formule** :

```
         ⎧ -(Dⱼ₋₁·cosφⱼ₋₁ + Dⱼ·cosφⱼ)/(2(Δφ)²·cosφⱼ) - vⱼ₋₁·cosφⱼ₋₁/(Δφ·cosφⱼ)   si vⱼ₋₁ > 0
d(i,j) = ⎨ -(Dⱼ₋₁·cosφⱼ₋₁ + Dⱼ·cosφⱼ)/(2(Δφ)²·cosφⱼ)                          si vⱼ₋₁ ≤ 0
         ⎩ 0                                                                    si bord sud fermé
```

**Différence avec longitude** : présence des facteurs `cosφ` pour géométrie sphérique.

#### Diagonale `e(il, jl)` - Cellule (il, jl)

**Fichier** : `adre.cpp:393-427`

```cpp
double diagElemLat(..., cosLatjm, cosLatj, cosLatjp) {
    double value = 0;
    switch (pos) {
    case OPEN:
        if (u > 0)
            value = ((sigmam*cosLatjm + 2*sigma*cosLatj + sigmap*cosLatjp) / twodd / cosLatj)
                    + (u / d) + twodt + r;
        else
            value = ((sigmam*cosLatjm + 2*sigma*cosLatj + sigmap*cosLatjp) / twodd / cosLatj)
                    - (u / d) + twodt + r;
        break;
    // ... cas bords ...
    }
    return value;
}
```

**Formule (cellule interne OPEN)** :

```
         ⎧ (Dⱼ₋₁·cosφⱼ₋₁ + 2Dⱼ·cosφⱼ + Dⱼ₊₁·cosφⱼ₊₁)/(2(Δφ)²·cosφⱼ) + vⱼ/Δφ + 2/Δt + r   si vⱼ > 0
e(i,j) = ⎨
         ⎩ (Dⱼ₋₁·cosφⱼ₋₁ + 2Dⱼ·cosφⱼ + Dⱼ₊₁·cosφⱼ₊₁)/(2(Δφ)²·cosφⱼ) - vⱼ/Δφ + 2/Δt + r   si vⱼ ≤ 0
```

**Note** : `r = 0` dans la direction méridionale (`adre.cpp:286`), car la mortalité a déjà été appliquée dans la direction zonale.

#### Sur-diagonale `f(il, jl)` - Cellule (il, jl+1)

**Fichier** : `adre.cpp:456-477`

```cpp
double upperDiagElemLat(..., cosLatj, cosLatjp) {
    double value = 0;
    switch (pos) {
    case OPEN:
    case L_CLOSED:
        if (usup > 0)
            value = -((sigma*cosLatj + sigmap*cosLatjp) / twodd / cosLatj);
        else
            value = -((sigma*cosLatj + sigmap*cosLatjp) / twodd / cosLatj)
                    + (usup*cosLatjp / d / cosLatj);
        break;
    case R_CLOSED:
        value = 0;
        break;
    }
    return value;
}
```

**Formule** :

```
         ⎧ -(Dⱼ·cosφⱼ + Dⱼ₊₁·cosφⱼ₊₁)/(2(Δφ)²·cosφⱼ)                    si vⱼ₊₁ > 0
f(i,j) = ⎨ -(Dⱼ·cosφⱼ + Dⱼ₊₁·cosφⱼ₊₁)/(2(Δφ)²·cosφⱼ) + vⱼ₊₁·cosφⱼ₊₁/(Δφ·cosφⱼ)   si vⱼ₊₁ ≤ 0
         ⎩ 0                                                              si bord nord fermé
```

---

## Résolution numérique complète

### Boucle principale ADI

**Fichier** : `adre.cpp:625-692`

```cpp
for (int kl = 1; kl <= nbIter_; ++kl) {

    // ÉTAPE 1 : Direction ZONALE (longitude) K → K+1/2
    for (int jl = jmin; jl <= jmax; ++jl) {

        // Construction RHS
        for (int il = iinf(jl); il <= isup(jl); ++il) {
            rhsZon_(il) = -d_(il,jl) * sol(il, jl-1)
                        + (4*nbIter_ - e_(il,jl)) * sol(il, jl)
                        - f_(il,jl) * sol(il, jl+1);
        }

        // Résolution tridiagonale
        solverZon.tridiagOpt(a_(jl), xbet_(jl), c_(jl), rhsZon_, uvec);

        // Sauvegarde solution intermédiaire
        for (int il = iinf(jl); il <= isup(jl); il++) {
            solZon_(jl, il) = uvec(il);
        }
    }

    // ÉTAPE 2 : Direction MÉRIDIONALE (latitude) K+1/2 → K+1
    for (int il = imin; il <= imax; ++il) {

        // Construction RHS
        for (int jl = jinf(il); jl <= jsup(il); ++jl) {
            rhsMer_(jl) = -a_(jl,il) * solZon_(jl, il-1)
                        + (4*nbIter_ - b_(jl,il)) * solZon_(jl, il)
                        - c_(jl,il) * solZon_(jl, il+1);
        }

        // Résolution tridiagonale
        solverMer.tridiagOpt(d_(il), ybet_(il), f_(il), rhsMer_, sol(il));
    }

    curDate += integrationStep_ * 3600;
}
```

### Pré-calcul des pivots beta

**Fichier** : `adre.cpp:539-564`

Avant la boucle temporelle, les inverses des pivots sont pré-calculés :

```cpp
int betaComp(dmatrix& beta, const dmatrix& lower,
             const dmatrix& diag, const dmatrix& upper) {

    for (int rl = rmin; rl <= rmax; ++rl) {

        int cinf = beta(rl).indexmin();
        int csup = beta(rl).indexmax();

        beta(rl, cinf) = 1.0 / diag(rl, cinf);

        for (int cl = cinf + 1; cl <= csup; ++cl) {
            beta(rl, cl) = diag(rl, cl)
                         - upper(rl, cl-1) * lower(rl, cl) * beta(rl, cl-1);
            beta(rl, cl) = 1.0 / beta(rl, cl);
        }
    }
    return 0;
}
```

Cette optimisation évite de recalculer les divisions à chaque itération temporelle.

**Appel** :
- Direction X : `betaComp(xbet_, a_, b_, c_)` ligne 259
- Direction Y : `betaComp(ybet_, d_, e_, f_)` ligne 299

### Appel à l'algorithme de Thomas optimisé

**Fichier** : `linearSolver.cpp:111-149`

```cpp
int linearSolver::tridiagOpt(const dvector &a, const dvector &bet,
                              const dvector &c, const dvector &rhs, dvector &u) {

    // Early exit si RHS négligeable
    if (max(rhs) < tol_) {
        return 0;
    }

    // Forward substitution
    gam_(jmin) = rhs(jmin);
    for (int jl = jmin + 1; jl <= jmax; ++jl) {
        gam_(jl) = rhs(jl) - gam_(jl-1) * a(jl) * bet(jl-1);
    }

    // Backward substitution
    u(jmax) = gam_(jmax) * bet(jmax);
    for (int jl = jmax - 1; jl >= jmin; --jl) {
        u(jl) = (gam_(jl) - c(jl) * u(jl+1)) * bet(jl);
    }

    return 0;
}
```

---

## Formules de mortalité et transformations

### Transformation de température (Gillooly)

**Fichier** : `modelPrey.cpp:983-992`, appliquée ligne 1450

```cpp
double modelPrey::tempLaw(double T) {
    return T / (1.0 + T / 273.0);
}
```

**Formule** :
```
T_transformed = 273·T / (273 + T)
```

Cette transformation sature la température et est cohérente avec les lois métaboliques (Nature 417, 2002).

### Formule de mortalité

**Fichier** : `population.cpp:113-144`

```cpp
int population::mortalityComp(const limits &lim, double M_max, double M_exp,
                              double dT, const dmatrix &temperatureAvg) {

    for (int il = imin; il <= imax; il++) {
        for (int jl = jinf(il); jl <= lim.jsup(il); jl++) {
            if (maskLayer(il, jl)) {

                double T = temperatureAvg(il, jl);
                if (T < 0) T = 0;  // Protection

                adreCoeffs_->setR(il, jl, dT * exp(-M_exp * T) / M_max);
            }
        }
    }
    return 0;
}
```

**Formule finale** :
```
R(i,j) = (Δt_simulation / M_max) · exp(-M_exp · T_transformed(i,j))
```

Où :
- `dT` = `simStep_` (en jours)
- `M_max` = `invLambdaMax_` (paramètre)
- `M_exp` = `invLambdaRate_` (paramètre)
- `T_transformed` = température déjà transformée par `tempLaw()`

**Interprétation** :
- Eaux froides (T faible) → exp(-M_exp·T) grand → mortalité élevée
- Eaux chaudes (T élevé) → exp(-M_exp·T) petit → mortalité faible

**Appel** : `functionalGroup.cpp:461`

```cpp
biomass_.mortalityComp(lim_, invLambdaMax_, invLambdaRate_, dT, temperatureAvg);
```

### Temps de recrutement (Tr)

**Fichier** : `functionalGroup.cpp:322-338`

```cpp
double functionalGroup::recruitTimeComp(double temperature, double minval) {
    double res = minval;

    if (temperature <= 0) {
        res = trMax_;
    } else {
        res = trMax_ * exp(trExp_ * temperature);
    }

    return max(res, minval);
}
```

**Formule** :
```
Tr(T) = max(trMax · exp(trExp · T_transformed), minValTr)
```

**Interprétation** :
- Eaux chaudes → Tr petit → recrutement rapide
- Eaux froides → Tr grand → recrutement lent

**Composition des transformations** :
```
T_océan [°C]
    ↓ tempLaw()
T_transformed
    ↓ exp(trExp · T)
Tr(x,y,T) [jours]
```

---

## Conditions limites

### Types de conditions limites

**Fichier** : `adre.cpp:312-327` (longitude), `336-352` (latitude)

SEAPODYM implémente 3 types de CL via le paramètre `pos` (cellBoundType) :

#### 1. OPEN (frontières ouvertes)

```cpp
case cellBoundType::OPEN:
    // Flux libre : coefficients normaux avec upwind
```

- Advection-diffusion complète aux bords
- Pas de restriction sur les flux

#### 2. CLOSED (frontières fermées - terre)

```cpp
case cellBoundType::L_CLOSED:  // Left closed
    value = 0.0;  // Pas de flux depuis la gauche
    break;

case cellBoundType::R_CLOSED:  // Right closed
    // Coefficient modifié : pas de flux vers la droite
```

- **Flux nul** : `a = 0` (bord gauche), `c = 0` (bord droit)
- Équivaut à condition de Neumann : `∂ρ/∂n = 0`

#### 3. CEW (Cyclic East-West - périodique en longitude)

**Fichier** : `adre.cpp:697-866`

Pour les domaines globaux (Pacifique) :

```cpp
// Zone de wrap (5° de chaque côté)
int wrapZoneSize_ = int(5.0 / dx_);

// Construction d'un système étendu
int nbiWrap = nbi + 2*ncells - 2;

// Copie des coefficients avec périodicité
for (int il = 0; il < ncells; il++) {
    a1(jl, il) = a_(jl, nbi - (ncells+1) + il);
    // ... côté Est copié à l'Ouest
}
```

**Principe** :
- Crée un domaine étendu avec zones de recouvrement
- Résout le système étendu avec `tridiag()` (pas `tridiagOpt()`)
- Copie les solutions entre bords Est et Ouest

### Modification des coefficients aux bords

**Direction longitude** (`adre.cpp:234-247`) :

```cpp
if (lim_.bcs() == bcsType::CEW) {
    if (il == 1 && mask(nbi-2, jl)) {
        sigmam = coeffs->Dx(nbi-2, jl);  // Connexion avec bord Est
        uinf = coeffs->U(nbi-2, jl);
    }
    if (il == nbi-2 && mask(1, jl)) {
        sigmap = coeffs->Dx(1, jl);      // Connexion avec bord Ouest
        usup = coeffs->U(1, jl);
    }
}
```

---

## Analyse de stabilité et précision

### Schéma upwind selon signe de vitesse

Le schéma est **upwind conditionnel** :

**Advection en longitude** (`adre.cpp:317-320`, `365-368`, `442-445`) :

```
Si u > 0 (courant vers l'Est) :
    a_i = -D/(Δx)² - u/Δx      ← flux depuis l'Ouest
    b_i = +D/(Δx)² + u/Δx + ... ← décentré à gauche
    c_i = -D/(Δx)²             ← pas de terme advectif

Si u < 0 (courant vers l'Ouest) :
    a_i = -D/(Δx)²             ← pas de terme advectif
    b_i = +D/(Δx)² - u/Δx + ... ← décentré à droite
    c_i = -D/(Δx)² + u/Δx      ← flux depuis l'Est
```

**Avantage** : évite les oscillations numériques (critère de stabilité relaxé).

### Stabilité inconditionnelle

Le schéma ADI **implicite** est inconditionnellement stable pour :
- Diffusion pure (D > 0)
- Advection-diffusion avec upwind

**Pas de condition CFL stricte** sur Δt.

**Source** : Documentation adre.hpp:92-99, référence Numerical Recipes

### Précision temporelle

**Ordre de précision** :
- Euler implicite : O(Δt)
- Splitting ADI : O(Δt²) pour diffusion pure
- **Erreur globale** : O(Δt) à cause du terme advectif et mortalité

**Convergence** :
- Réduire `integrationStep_` améliore la précision
- Converge vers solution exacte quand Δt → 0

### Précision spatiale

**Différences finies centrées** : O(Δx²), O(Δy²)

**Upwind** : O(Δx) en advection pure

**Erreur dominante** :
- Zones à fort Péclet (Pe = u·Δx/D) : erreur advective O(Δx)
- Zones diffusives (Pe petit) : erreur O(Δx²)

### Erreur géométrique sphérique

**Approximation** : Δλ constant ≠ distance constante

**Erreur relative** :
```
ε_géom ≈ (1 - cos(φ)²) / cos(φ)²
```

**Valeurs numériques** :
- φ = 0° (équateur) : ε = 0%
- φ = 30° : ε ≈ 15%
- φ = 60° : ε ≈ 300% (!!)
- φ = 75° : ε ≈ 1400% (!!!)

**Conclusion** : Le modèle est **moins précis aux hautes latitudes** (>60°).

### Conservation de masse

**Domaine CLOSED** :
- Flux nuls aux bords → masse conservée (théoriquement)
- En pratique : erreur d'arrondi machine (~10⁻¹⁵)

**Domaine OPEN** :
- Masse peut entrer/sortir
- Dépend des conditions aux limites

**Test de conservation** : tracer `∫∫ ρ dA` au cours du temps.

---

## Résumé du schéma numérique

### Équation discrétisée finale

**Direction zonale (longitude)** :

```
a(j,i)·ρ*(i-1,j) + b(j,i)·ρ*(i,j) + c(j,i)·ρ*(i+1,j) =
    -d(i,j)·ρⁿ(i,j-1) + (2/Δt - e(i,j))·ρⁿ(i,j) - f(i,j)·ρⁿ(i,j+1)
```

**Direction méridionale (latitude)** :

```
d(i,j)·ρⁿ⁺¹(i,j-1) + e(i,j)·ρⁿ⁺¹(i,j) + f(i,j)·ρⁿ⁺¹(i,j+1) =
    -a(j,i)·ρ*(i-1,j) + (2/Δt - b(j,i))·ρ*(i,j) - c(j,i)·ρ*(i+1,j)
```

Avec :
- ρⁿ : solution au temps K
- ρ* : solution intermédiaire K+1/2
- ρⁿ⁺¹ : solution au temps K+1

### Paramètres clés

| Paramètre | Valeur typique | Unité | Impact |
|-----------|----------------|-------|--------|
| `simStep_` | 7 | jours | Pas de temps simulation |
| `integrationStep_` | 1 | heure | Pas de temps intégration |
| `nbIter_` | 168 | - | Sous-itérations |
| `dx_`, `dy_` | 0.25-1.0 | degrés | Résolution spatiale |
| `tlDiffCoeff_` | 1000-5000 | m²/s | Diffusion turbulente |
| `invLambdaMax_` | 100-1000 | jours | Longévité maximale |
| `invLambdaRate_` | 0.01-0.1 | - | Sensibilité température |

### Complexité algorithmique

**Par pas de temps** :
```
Coût = nbIter_ × (N_lat × Solve_lon + N_lon × Solve_lat)
     = nbIter_ × (N_lat × O(N_lon) + N_lon × O(N_lat))
     = nbIter_ × O(N_lat × N_lon)
```

**Exemple** : Grille 360×180, 168 itérations
```
Opérations ≈ 168 × (180 × 360 + 360 × 180)
           ≈ 168 × 129,600
           ≈ 21.8 millions d'opérations / pas de temps
```

---

## Fichiers sources analysés

| Fichier | Lignes clés | Contenu |
|---------|-------------|---------|
| `adre.cpp` | 306-477 | Coefficients tridiagonaux (longitude, latitude) |
| `adre.cpp` | 591-695 | Résolution ADI complète |
| `adre.cpp` | 539-564 | Pré-calcul pivots beta |
| `population.cpp` | 83-111 | Stockage U, V, Dx, Dy |
| `population.cpp` | 113-144 | Calcul mortalité R(T) |
| `cohort.cpp` | 51-55 | Appel solver ADRE |
| `linearSolver.cpp` | 111-149 | Algorithme Thomas optimisé |
| `modelPrey.cpp` | 983-992 | Transformation température |
| `modelPrey.cpp` | 1435-1458 | Conversion unités forçages |
| `functionalGroup.cpp` | 322-338 | Temps de recrutement Tr(T) |

---

## Références

1. **Numerical Recipes** - Press et al.
   - Méthode ADI (Alternating Direction Implicit)
   - Algorithme de Thomas pour systèmes tridiagonaux

2. **Gillooly et al. (2002)** - Nature 417
   - Loi température-développement : `T/(1 + T/273)`

3. **Coordonnées sphériques**
   - Laplacien en sphérique : facteurs `cos(φ)` dans diffusion
   - Conservation flux : termes `cos(φ)` dans advection latitudinale

4. **Schémas upwind**
   - Stabilité pour problèmes advection-dominés
   - Compromis précision/stabilité

---

**Dernière mise à jour** : 2025-11-13

**Auteur** : Analyse complète du code SEAPODYM

**Statut** : Document complet et validé avec le code source
