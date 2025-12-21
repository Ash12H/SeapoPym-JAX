# SEAPODYM - Transformations critiques et altérations de la solution

## Vue d'ensemble

Ce document répertorie **tous les éléments** identifiés dans le code qui transforment, modifient ou altèrent les données avant/pendant la résolution numérique. Ces transformations peuvent introduire des **non-linéarités**, **biais**, ou **erreurs** qui affectent la solution finale.

---

## 1. Transformations de température

### 1.1 Loi de température de Gillooly

**Fichier** : `modelPrey.cpp:983-992`

```cpp
double modelPrey::tempLaw(double T) {
    // Development is related to temperature by a model given by
    // Gillooly et. al., Effects of size and temperature on developmental time,
    // Nature 417, 2002, http://dx.doi.org/10.1038/417070a

    return T / (1.0 + T / 273.0);
}
```

**Application** : `modelPrey.cpp:1450`

```cpp
// Temperature is rescaled following Gillooly et. al., 2002
forcingList_.at(varKeys::T).maskAndScale(activeLayers, mask_, fillval, &tempLaw);
```

**Impact** :
- ✅ **Non-linéarité** : transformation hyperbolique
- ✅ **Saturation** : T → ∞ ⇒ T_transformed → 273
- ✅ **Sensibilité** : forte aux basses températures, faible aux hautes
- ⚠️ **Unités** : suppose T en Kelvin ou Celsius ?

**Forme mathématique** :
```
T_transformed = T / (1 + T/273) = 273T / (273 + T)
```

Dérivée :
```
dT_transformed/dT = 273² / (273 + T)²
```

### 1.2 Temps de recrutement dépendant de T

**Fichier** : `functionalGroup.cpp:322-338`

```cpp
double functionalGroup::recruitTimeComp(double temperature, double minval) {
    double res = minval;

    if (temperature <= 0) {
        res = trMax_;
    } else {
        res = trMax_ * exp(trExp_ * temperature);  // ← Exponentielle !
    }

    return max(res, minval);
}
```

**Application** : `functionalGroup.cpp:340-365`

```cpp
int functionalGroup::updateRecruitTime(const dmatrix &temperatureAvg) {
    for (int il = imin; il <= imax; ++il) {
        for (int jl = jinf(il); jl <= jsup(il); ++jl) {
            if (maskLayer(il, jl)) {
                (*Tr_)(il, jl) = recruitTimeComp(temperatureAvg(il, jl), minValTr_);
            }
        }
    }
    return 0;
}
```

**Impact** :
- ✅ **Non-linéarité forte** : exponentielle
- ✅ **Sensibilité extrême** : petite variation de T → grande variation de Tr
- ✅ **Effet spatial** : recrutement plus rapide en eaux chaudes
- ⚠️ **Plafonnement** : `max(res, minval)` évite Tr trop petits
- ⚠️ **Test température** : si T ≤ 0, Tr = trMax (protection)

**Forme mathématique** :
```
Tr(T) = max(trMax · exp(trExp · T_transformed), minValTr)
```

Avec T_transformed déjà modifié par `tempLaw()` !

---

## 2. Conversions d'unités critiques

### 2.1 Courants océaniques : m/s → °/pas de temps

**Fichier** : `modelPrey.cpp:1432-1448`

```cpp
// Lenght of 1 arc degree on the sphere with earth radius
// = EARTH_RADIUS * pi / 180.0 meter arc length

// meter to "degree" i.e. meter to length of one arc degree
double mtodeg = 1 / (EARTH_RADIUS * M_PI / 180.0);
// meter square to "degree" square
double m2todeg2 = mtodeg * mtodeg;
// simStep_ units is days = 3600.0 * 24 seconds
double daytosec = 3600.0 * 24;
// convertion from m/s to "angle" / simulation step
double c = mtodeg * daytosec * simStep_;

double fillval = -99.0;
// Convertion from m/s to degree/time step
forcingList_.at(varKeys::U).maskAndScale(activeLayers, mask_, fillval, c);
// Take oposite because latitudes are in decreasing order
forcingList_.at(varKeys::V).maskAndScale(activeLayers, mask_, fillval, -c);
```

**Impact** :
- ⚠️ **Approximation sphérique** : 1° ≠ distance constante (dépend de la latitude)
- ⚠️ **Signe V inversé** : `-c` car latitudes décroissantes
- ⚠️ **Valeur de remplissage** : `fillval = -99.0` pour cellules terre
- ⚠️ **Dépendance à simStep_** : conversion change selon le pas de temps

**Calcul détaillé** :
```
EARTH_RADIUS = 6371000 m (approximation)
1° = EARTH_RADIUS * π/180 ≈ 111195 m

mtodeg = 1/111195 ≈ 8.99e-6 °/m
c = 8.99e-6 * 86400 * simStep_ [°/pas de temps]
```

Pour `simStep_ = 7 jours` :
```
c ≈ 8.99e-6 * 86400 * 7 ≈ 5.43 °/(m/s)/semaine
```

Un courant de **1 m/s** déplace de **5.43°** en une semaine !

### 2.2 Diffusion : m²/s → °²/pas de temps

**Fichier** : `modelPrey.cpp:1458`

```cpp
// Change unit of diffusion coefficient from m^2/s to "degree^2" / time step
double diffCoeff = param_.tlDiffCoeff_ * m2todeg2 * daytosec * simStep_;
```

**Impact** :
- ⚠️ **Double approximation** : (mtodeg)² amplifie l'erreur sphérique
- ⚠️ **Dépendance quadratique** : erreur en O(lat²) si approximation plane

### 2.3 NPP : conversion vers g/m²

**Fichier** : `modelPrey.cpp:1475`

```cpp
// Integrate production (/day) over time step
// Convert NPP to g/m2
forcingList_.at(varKeys::NPP).maskAndScale(1, mask_, -99.0,
                                           param_.tlNppUnitConvert_ * simStep_);
```

**Impact** :
- ⚠️ **Unités initiales inconnues** : dépend de `tlNppUnitConvert_`
- ⚠️ **Intégration temporelle** : multiplie par `simStep_` (suppose taux constant)

### 2.4 Carbone → poids humide

**Fichier** : `functionalGroup.cpp:280`

```cpp
production_[larvaePos].density_(il, jl) = pp(il, jl) * c2ww;
```

Où `c2ww = param_.tlCarbonToWetWeight_`

**Impact** :
- ⚠️ **Coefficient empirique** : varie selon les espèces
- ⚠️ **Hypothèse constante** : rapport C/poids humide fixe spatialement

---

## 3. Moyennes pondérées par longueur du jour

### 3.1 Calcul de la longueur du jour

**Fichier** : `modelPrey.cpp:640-663`

```cpp
int modelPrey::updateDayLength(int doy) {
    matDayLength_.initialize();

    for (int il = map_->imin()(1); il <= map_->imax()(1); il++) {
        for (int jl = map_->jinf()(1, il); jl <= map_->jsup()(1, il); jl++) {
            if (mask_(il, jl)) {
                matDayLength_(il, jl) = date::dayLength(ylat_(il, jl), doy, calendar_);
            }
        }
    }
    return 0;
}
```

**Impact** :
- ⚠️ **Variation saisonnière** : change chaque jour de l'année
- ⚠️ **Gradient latitudinal** : jour polaire vs équateur
- ⚠️ **Fonction non-linéaire** : dépend de sin(déclinaison solaire)

### 3.2 Moyenne pondérée U, V, T

**Fichier** : `modelPrey.cpp:1453-1455`

```cpp
forcingList_.at(varKeys::U).averageLayer(dayLayer, nightLayer, matDayLength_, mask_, zonCurrentAvg);
forcingList_.at(varKeys::V).averageLayer(dayLayer, nightLayer, matDayLength_, mask_, merCurrentAvg);
forcingList_.at(varKeys::T).averageLayer(dayLayer, nightLayer, matDayLength_, mask_, temperatureAvg);
```

**Forme supposée** :
```
U_avg = (dayLength/24) * U_day + (1 - dayLength/24) * U_night
```

**Impact** :
- ⚠️ **Hypothèse migration verticale** : organismes changent de couche jour/nuit
- ⚠️ **Moyenne linéaire** : ne capture pas les non-linéarités du transport
- ⚠️ **Couplage spatial-temporel** : dayLength varie spatialement

---

## 4. Transfert d'énergie entre groupes et couches

### 4.1 Calcul des coefficients de transfert

**Fichier** : `modelPrey.cpp:868-981`

```cpp
std::map<std::string, dvector> modelPrey::energyTransferMap() {
    int nbTotalGroup = param_.tlNbTotalGroup_;
    int nbLayer = map_->nlay();

    std::map<std::string, dvector> energyMap;
    // ... initialisation ...

    for (int ll = 1; ll <= nbLayer; ll++) {
        int nF_count = 0;
        double sum_E_remaining = 0;

        for (/* each group */) {
            if (dayLayer > ll || nightLayer > ll) {
                F_missing[groupName] = 1;
                nF_count++;
                sum_E_remaining += energy;
            }
        }

        for (/* each group */) {
            if (F_missing[groupName]) {
                energyMap[groupName](ll) = 0.0;
            } else {
                // ← Redistribution de l'énergie !
                energyMap[groupName](ll) = energy + sum_E_remaining / (nbTotalGroup - nF_count);
            }
        }
    }
    return energyMap;
}
```

**Impact** :
- ⚠️ **Redistribution énergétique** : énergie des groupes absents redistribuée
- ⚠️ **Dépendance aux couches** : change selon profondeur
- ⚠️ **Couplage entre groupes** : un groupe affecte les autres
- ⚠️ **Non-linéarité** : division par `(nbTotalGroup - nF_count)`

**Exemple** :
```
Couche 1 : 3 groupes présents, energie = [0.3, 0.3, 0.4]
Couche 3 : 1 groupe absent (0.3)
→ Les 2 restants reçoivent : [0.3 + 0.15, 0.4 + 0.15] = [0.45, 0.55]
```

### 4.2 Application à la biomasse

**Fichier** : `functionalGroup.cpp:290-320`

```cpp
int functionalGroup::distributeEnergyBiom(double E) {
    for (int il = imin; il <= imax; ++il) {
        for (int jl = jinf(il); jl <= jsup(il); ++jl) {
            if (maskLayer(il, jl)) {
                int nlayer = mask_(il, jl);
                if (nlayer > nbLayer_) nlayer = nbLayer_;

                if (nlayer >= dayLayer_) {
                    // ← Double coefficient énergétique !
                    biomass_.pop_[0].density_(il, jl) +=
                        recruitment_(il, jl) * E * energyTransfer_(nlayer);
                }
            }
        }
    }
    return 0;
}
```

**Impact** :
- ⚠️ **Produit E × E'(nlayer)** : deux coefficients multiplicatifs
- ⚠️ **Dépendance bathymétrique** : `nlayer` = nombre de couches océaniques
- ⚠️ **Troncation** : `if (nlayer > nbLayer_) nlayer = nbLayer_`

---

## 5. Masquage et gestion des valeurs manquantes

### 5.1 Lecture et transposition des grilles NetCDF

**Fichier** : `modelPrey.cpp:158-248`

```cpp
bool transpose;
if (row_stdname == "latitude") {
    transpose = true;
    console::debug("transpose data");
} else {
    transpose = false;
}

// ...

bool flipud = false;
// SEAPODYM handle latitudes in decreasing order
if (transpose) {
    if (rows[nrow - 1] - rows[0] > 0) {
        flipud = true;
        console::warning("reverse order of rows");
        // ... inversion du vecteur rows ...
    }
}
```

**Impact** :
- ⚠️ **Ordre des données** : transpose + flipud peuvent altérer l'interprétation
- ⚠️ **Convention latitudes** : SEAPODYM assume latitudes décroissantes
- ⚠️ **Erreur silencieuse** : si convention non respectée

### 5.2 Application du masque et remplissage

**Fichier** : `modelPrey.cpp:1445-1448` (exemple)

```cpp
double fillval = -99.0;
forcingList_.at(varKeys::U).maskAndScale(activeLayers, mask_, fillval, c);
```

**Impact** :
- ⚠️ **Valeurs sur terre** : remplacées par `fillval = -99.0`
- ⚠️ **Propagation potentielle** : si masque incorrect, -99 peut entrer dans les calculs
- ⚠️ **Détection** : dépend de la rigueur de `maskAndScale()`

### 5.3 Gestion des NaN et valeurs extrêmes

**Fichier** : `linearSolver.cpp:48-53`

```cpp
// Skip resolution if rhs is very small
if (max(rhs) < tol_) {
    return 0;
}
```

**Impact** :
- ✅ **Protection** : évite résolution avec densités nulles
- ⚠️ **Seuil** : `tol_ = std::numeric_limits<double>::epsilon()` très petit
- ⚠️ **Early exit** : solution = 0 si max(rhs) < epsilon

---

## 6. Conditions limites et domaines

### 6.1 Choix du type de domaine

**Fichier** : `modelPrey.cpp:330-348`

```cpp
switch (bcs_) {
    case bcsType::OPEN:
        map_ = new openDomain(xlon, ylat, mask);
        break;
    case bcsType::CLOSED:
        map_ = new closedDomain(xlon, ylat, mask);
        break;
    case bcsType::CEW:
        map_ = new cewDomain(xlon, ylat, mask);
        break;
    default:
        throw SeapodymException(...);
}
```

**Impact** :
- ⚠️ **Flux aux bords** : OPEN vs CLOSED change radicalement la solution
- ⚠️ **Périodicité** : CEW (Cyclic East-West) crée continuité Pacifique
- ⚠️ **Conservation masse** : CLOSED conserve, OPEN peut perdre/gagner de la biomasse

---

## 7. Mortalité dépendante de la température

**Fichier** : `functionalGroup.cpp:461`

```cpp
biomass_.mortalityComp(lim_, invLambdaMax_, invLambdaRate_, dT, temperatureAvg);
```

**Forme supposée** (à confirmer dans `population.cpp`) :
```
μ(T) = invLambdaMax · exp(-invLambdaRate · T_transformed)
```

**Impact** :
- ✅ **Non-linéarité** : exponentielle de la température
- ✅ **Terme puits** : R = -μ(T)ρ dans l'équation ADRE
- ⚠️ **Rétroaction** : zones froides → mortalité élevée → accumulation réduite

---

## 8. Intégration temporelle avec sous-itérations

**Fichier** : `modelPrey.cpp:383`

```cpp
nbIterations_ = (simStep_ * 24) / param_.integrationStep_;
```

**Exemple** :
- `simStep_ = 7 jours`
- `integrationStep_ = 1 heure`
- → `nbIterations_ = 168`

**Impact** :
- ⚠️ **Accumulation erreurs** : 168 résolutions successives
- ⚠️ **Stabilité** : schéma implicite stable mais erreur de troncature
- ⚠️ **Coût calcul** : 168× résolutions tridiagonales par pas de temps

---

## 9. Lecture de forçages : interpolation temporelle

**Fichier** : `modelPrey.cpp:1425-1427`

```cpp
forcingList_.at(varKeys::U).read(activeLayers, curDate);
forcingList_.at(varKeys::V).read(activeLayers, curDate);
forcingList_.at(varKeys::T).read(activeLayers, curDate);
```

**Impact** (à confirmer dans `forcing.cpp`) :
- ⚠️ **Interpolation** : si `curDate` entre deux pas de temps de forçage
- ⚠️ **Nearest neighbor** : ou interpolation linéaire ?
- ⚠️ **Cohérence temporelle** : U, V, T doivent être synchrones

---

## 10. Vieillissement et décalage des cohortes

**Fichier** : `modelPrey.cpp:1490`

```cpp
fGroup.production_.ageing();
```

**Impact** (à confirmer dans `population.cpp`) :
- ⚠️ **Shift des indices** : cohorte i → cohorte i+1
- ⚠️ **Perte de masse** : si dernière cohorte "sort" du système
- ⚠️ **Conservation** : dépend de l'implémentation

---

## Résumé : Hiérarchie des altérations

### Niveau 1 : Transformations physiques (justifiées scientifiquement)

| Transformation | Fichier:ligne | Justification | Risque |
|----------------|---------------|---------------|--------|
| `tempLaw()` | `modelPrey.cpp:983` | Gillooly et al. 2002 | Faible si T > 0 |
| Tr(T) exponentielle | `functionalGroup.cpp:334` | Développement biologique | Moyen (sensible) |
| Mortalité μ(T) | `functionalGroup.cpp:461` | Métabolisme | Moyen |
| Transfert énergie | `modelPrey.cpp:868` | Réseau trophique | Moyen |

### Niveau 2 : Conversions d'unités (approximations géométriques)

| Conversion | Fichier:ligne | Approximation | Erreur |
|------------|---------------|---------------|--------|
| m/s → °/step | `modelPrey.cpp:1435` | Sphère → plan | O(lat²) |
| m²/s → °²/step | `modelPrey.cpp:1458` | Sphère → plan | O(lat⁴) |
| Carbone → poids | `functionalGroup.cpp:280` | Ratio constant | Variable |

### Niveau 3 : Moyennes et couplages (hypothèses écologiques)

| Opération | Fichier:ligne | Hypothèse | Impact |
|-----------|---------------|-----------|--------|
| Moyenne jour/nuit | `modelPrey.cpp:1453` | Migration verticale | Fort |
| Pondération dayLength | `modelPrey.cpp:656` | Sine(déclinaison) | Moyen |

### Niveau 4 : Numérique (erreurs de troncature)

| Source | Fichier:ligne | Type | Ordre |
|--------|---------------|------|-------|
| Euler implicite | Schéma ADI | Troncature temporelle | O(Δt) |
| Différences finies | Schéma spatial | Troncature spatiale | O(Δx²) |
| Sous-itérations | `modelPrey.cpp:383` | Accumulation | O(n·Δt) |

### Niveau 5 : Données et I/O (erreurs d'interprétation)

| Opération | Fichier:ligne | Risque | Détection |
|-----------|---------------|--------|-----------|
| Transpose/flipud | `modelPrey.cpp:158-248` | Grille inversée | Visuel |
| Masque terre/mer | `modelPrey.cpp:1445` | Valeurs -99 | Logs |
| Interpolation forçages | `modelPrey.cpp:1425` | Temporelle | ? |

---

## Recommandations pour investigation

### Tests de sensibilité prioritaires

1. **`tempLaw()`** :
   - Comparer solution avec/sans transformation
   - Vérifier unités de T en entrée (K ou °C ?)

2. **Tr(T)** :
   - Cartographier Tr spatial
   - Identifier zones de recrutement extrême (trop rapide/lent)

3. **Conversions sphériques** :
   - Tester sur grille haute latitude (>60°)
   - Comparer avec conversion exacte (distance géodésique)

4. **Moyenne jour/nuit** :
   - Comparer avec résolution 3D (sans moyenne)
   - Vérifier aux équinoxes vs solstices

5. **Transfert d'énergie** :
   - Tracer coefficients par couche
   - Vérifier conservation énergétique totale

### Fichiers à analyser en priorité

1. **`src/forcing.cpp`** : interpolation temporelle
2. **`src/population.cpp`** : calcul mortalité, ageing()
3. **`src/date.cpp`** : calcul dayLength()
4. **`src/toolkit.cpp`** : isSmall(), autres utilitaires

### Validation numérique

1. **Test de convergence** :
   - Réduire `integrationStep_` : solution doit converger
   - Réduire Δx, Δy : convergence O(Δx²) ?

2. **Conservation de masse** :
   - Domaine CLOSED : masse totale constante ?
   - Tracer bilan : recrutement - mortalité - flux bords

3. **Comparaison analytique** :
   - Cas simple : diffusion pure, solution analytique connue
   - Cas simple : advection pure, déplacement gaussienne

---

**Document créé le** : 2025-11-13
**Objectif** : Identifier toutes les transformations qui altèrent la solution SEAPODYM
**Statut** : À compléter avec `population.cpp`, `forcing.cpp`, `date.cpp`
