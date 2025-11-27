# Analyse de compute_production_initialization

## Code actuel

```python
def compute_production_initialization(
    primary_production: xr.DataArray,  # [g/m²/s]
    cohorts: xr.DataArray,
    E: float,
) -> dict[str, xr.DataArray]:
    tendency_rate = E * primary_production
    tendency = tendency_rate.expand_dims(cohort=[0])
    tendency = tendency.reindex(cohort=cohorts, fill_value=0.0)
    return {"output": tendency}
```

## Comportement détaillé

### Exemple

Entrées:
- primary_production = 1.157e-5 g/m²/s (converti de 1 g/m²/day)
- E = 0.1
- cohorts = [0, 86400, 172800, ...] seconds

Étape 1: Calcul du taux
```
tendency_rate = 0.1 × 1.157e-5 = 1.157e-6 g/m²/s
```

Étape 2: Expand à cohort=0
```
tendency = [1.157e-6]  (1 élément avec cohort=0)
```

Étape 3: Reindex sur toutes les cohortes
```
tendency = [1.157e-6, 0, 0, 0, ...]  (N éléments)
```

### Validation

Après TimeIntegrator:
```
P[0](t+dt) = P[0](t) + dt × tendency[0]
           = P[0](t) + 86400 × 1.157e-6
           = P[0](t) + 0.1
```

✅ **Correct** : Ajout de 0.1 g/m² par jour à l'âge 0

### Unités

- Input NPP: [g/m²/s]
- Output tendency: [g/m²/s]
- TimeIntegrator multiplie par dt [s]
- Résultat final: [g/m²]

✅ **Cohérent dimensionnellement**

## Test numérique

État initial: P = [0, 0, 0, ...]
NPP = 1.157e-6 g/m²/s
E = 0.1
dt = 86400 s

**Pas 1:**
- tendency[0] = 1.157e-6 g/m²/s
- P[0](t+dt) = 0 + 86400 × 1.157e-6 = 0.1 g/m²

**Pas 2:**
- Sans aging ni recruitment, P[0] continue d'accumuler
- P[0](t+2dt) = 0.1 + 0.1 = 0.2 g/m²

✅ Production s'accumule linéairement à l'âge 0

## Conclusion

`compute_production_initialization` est **correct**.

Il ajoute exactement E × NPP à l'âge 0, et rien aux autres âges.
