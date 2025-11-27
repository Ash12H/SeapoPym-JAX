# Analyse de compute_recruitment_tendency

## Code actuel

```python
def compute_recruitment_tendency(
    production: xr.DataArray,
    recruitment_age: xr.DataArray,
    cohort_ages: xr.DataArray,
    dt: float,
) -> dict[str, xr.DataArray]:
    will_be_recruited = (cohort_ages + dt) >= recruitment_age
    recruited_production = production.where(will_be_recruited, 0.0)
    production_sink = -recruited_production / dt
    biomass_source = recruited_production.sum(dim="cohort") / dt
    return {"recruitment_sink": production_sink, "recruitment_source": biomass_source}
```

## Comportement détaillé

### Exemple 1: τ_r = 10.38 jours, max_cohort = 10 jours

cohort_ages = [0, 1, 2, ..., 10] jours = [0, 86400, ..., 864000] s
τ_r = 896832 s (10.38 jours)
dt = 86400 s (1 jour)

Test de recrutement:
```
will_be_recruited = (cohort_ages + 86400) >= 896832

cohort 9:  (777600 + 86400) = 864000 < 896832 → False
cohort 10: (864000 + 86400) = 950400 >= 896832 → True
```

✅ **Correct** : Seule la cohorte 10 est recrutée

Production_sink:
```
production_sink[10] = -P[10] / dt
```

Biomass_source:
```
biomass_source = P[10] / dt
```

### Validation avec TimeIntegrator et Aging

État: P = [0.1, 0.1, ..., 0.1, 0.1]

**Aging alone:**
- aging_flux[10] = (P[9] - P[10]) / dt = 0

**Recruitment:**
- recruitment_sink[10] = -P[10] / dt

**Total tendency:**
```
total[10] = aging_flux[10] + recruitment_sink[10]
          = 0 - P[10]/dt
          = -P[10]/dt
```

**After TimeIntegrator:**
```
P[10](t+dt) = P[10] + dt × (-P[10]/dt)
            = P[10] - P[10]
            = 0
```

✅ **Correct** : La cohorte 10 se vide complètement

### Exemple 2: τ_r = 5 jours, max_cohort = 10 jours

cohort_ages = [0, 1, ..., 10] jours
τ_r = 432000 s (5 jours)
dt = 86400 s (1 jour)

Test de recrutement:
```
cohort 0: (0 + 86400) = 86400 < 432000 → False
cohort 4: (345600 + 86400) = 432000 >= 432000 → True
cohort 5: (432000 + 86400) = 518400 >= 432000 → True
...
cohort 10: (864000 + 86400) = 950400 >= 432000 → True
```

❌ **PROBLÈME** : Cohortes 4-10 sont TOUTES recrutées !

État stable supposé: P = [0.1, 0.1, 0.1, 0.1, 0, 0, ..., 0]

**Cohorte 4:**
- aging_flux[4] = (P[3] - P[4]) / dt = (0.1 - 0.1) / dt = 0
- recruitment_sink[4] = -P[4] / dt = -0.1/dt

Total: -0.1/dt → P[4](t+dt) = 0.1 - 0.1 = 0 ✅

**Cohorte 5:**
- aging_flux[5] = (P[4] - P[5]) / dt = (0.1 - 0) / dt = 0.1/dt
- recruitment_sink[5] = -P[5] / dt = 0

Total: 0.1/dt → P[5](t+dt) = 0 + 0.1 = 0.1 ❌

**PROBLÈME IDENTIFIÉ** :
- La cohorte 5 reçoit un influx depuis P[4] (aging)
- Mais P[5] est déjà à 0, donc recruitment_sink = 0
- P[5] se remplit alors qu'elle devrait rester vide !

## Root Cause

Le recrutement draine `P[i] / dt` SEULEMENT si P[i] > 0.

Mais l'aging continue d'injecter `P[i-1] / dt` même si i est au-delà de τ_r.

**Conséquence** : Les cohortes au-delà de τ_r se remplissent progressivement via l'aging.

## Solution Correcte

### Option A: Modifier l'aging pour qu'il s'arrête au recrutement

```python
def compute_aging_tendency(...):
    # Identifier la dernière cohorte non-recrutée
    is_recruited = (cohort_ages + dt) >= recruitment_age
    last_non_recruited = (~is_recruited).sum() - 1

    # L'aging normal jusqu'à cette cohorte
    # Pour la dernière cohorte non-recrutée, l'outflux va vers le recrutement
    ...
```

Problème : L'aging a besoin de connaître recruitment_age, ce qui couple les fonctions.

### Option B: Modifier le recrutement pour drainer aussi l'influx

```python
def compute_recruitment_tendency(...):
    will_be_recruited = (cohort_ages + dt) >= recruitment_age

    # Drainer toute production existante
    recruited_production_stock = production.where(will_be_recruited, 0.0)

    # PLUS drainer l'influx qui arrive via aging
    # (C'est compliqué car on ne connait pas encore l'aging ici)
    ...
```

Problème : Le recrutement a besoin de connaître l'aging, ce qui couple aussi.

### Option C: Utiliser une contrainte de positivité + masque

```python
# Dans TimeIntegrator
new_production = old_production + dt * (aging + source + recruitment)

# Après intégration, forcer à 0 les cohortes recrutées
is_recruited = (cohort_ages >= recruitment_age)
new_production = new_production.where(~is_recruited, 0.0)
```

C'est une solution "ad-hoc" mais simple.

### Option D (RECOMMANDÉE): Le recrutement bloque aussi l'aging

Le recrutement ne draine pas seulement P[i], mais AUSSI bloque l'influx depuis P[i-1]:

```python
def compute_recruitment_tendency(...):
    will_be_recruited = (cohort_ages + dt) >= recruitment_age

    # Shift pour savoir quelle cohorte ALIMENTE une cohorte recrutée
    will_feed_recruited = will_be_recruited.shift(cohort=-1, fill_value=False)

    # Le recrutement draine:
    # 1. La production existante des cohortes recrutées
    recruited_stock = production.where(will_be_recruited, 0.0)

    # 2. L'influx qui IRAIT vers les cohortes recrutées
    recruited_influx = production.where(will_feed_recruited, 0.0)

    production_sink = -(recruited_stock + recruited_influx) / dt
    biomass_source = (recruited_stock + recruited_influx).sum(dim="cohort") / dt

    return {...}
```

Non, c'est trop compliqué et crée un double comptage.

## Solution SIMPLE et CORRECTE

**PRINCIPE** : Les cohortes ne devraient PAS s'étendre au-delà de τ_r.

Si τ_r = 5 jours, les cohortes doivent être [0, 1, 2, 3, 4] jours SEULEMENT.

Ensuite, l'aging fait naturellement sortir P[4] vers "hors grille" et le recrutement capture ce flux.

**Implémentation actuelle est correcte TANT QUE** max(cohort_ages) ≈ τ_r.

Le problème vient du SETUP dans le notebook, pas du code!
