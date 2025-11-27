# Analyse de compute_aging_tendency

## Code actuel

```python
def compute_aging_tendency(
    production: xr.DataArray,
    dt: float,
) -> dict[str, xr.DataArray]:
    shifted = production.shift(cohort=1, fill_value=0.0)
    tendency = (shifted - production) / dt
    return {"aging_flux": tendency}
```

## Comportement détaillé

### Exemple avec 5 cohortes

État: P = [1, 2, 3, 4, 5]

Après `shift(cohort=1, fill_value=0)`:
```
shifted = [0, 1, 2, 3, 4]
```

Tendency:
```
tendency = (shifted - production) / dt
         = ([0, 1, 2, 3, 4] - [1, 2, 3, 4, 5]) / dt
         = [-1, -1, -1, -1, -1] / dt
```

### Interprétation physique

Pour la cohorte i:
```
aging_flux[i] = (P[i-1] - P[i]) / dt
```

- **Influx** : P[i-1] / dt (production venant de la cohorte précédente)
- **Outflux** : -P[i] / dt (production qui vieillit vers la cohorte suivante)
- **Net** : (P[i-1] - P[i]) / dt

### Cas spécial: première cohorte (i=0)

```
shifted[0] = 0  (fill_value)
aging_flux[0] = (0 - P[0]) / dt = -P[0] / dt
```

✅ Correct : Pas d'influx, seulement outflux vers cohorte 1

### Cas spécial: dernière cohorte (i=N)

```
aging_flux[N] = (P[N-1] - P[N]) / dt
```

Le flux sortant de P[N] "disparaît" de la grille.
Question : **Où va-t-il ?**

**Réponse attendue** : Il devrait aller vers le recrutement (biomasse).

## Validation avec TimeIntegrator

Après intégration:
```
P[i](t+dt) = P[i](t) + dt × aging_flux[i]
           = P[i](t) + dt × (P[i-1] - P[i]) / dt
           = P[i](t) + P[i-1] - P[i]
           = P[i-1]
```

✅ **Parfait** : La production de chaque cohorte devient égale à celle de la cohorte précédente.
C'est exactement l'advection en âge !

## Test numérique

État initial: P = [0, 0, 0, 1, 0] (pulse à cohorte 3)

**Pas 1:**
- aging_flux = ([--, 0, 0, 0, 1] - [0, 0, 0, 1, 0]) / dt
             = [0, 0, 0, -1, 1] / dt
- P(t+dt) = [0, 0, 0, 1, 0] + dt × [0, 0, 0, -1, 1] / dt
          = [0, 0, 0, 0, 1]

✅ Le pulse s'est déplacé de cohorte 3 → cohorte 4

**Pas 2:**
- P(t+2dt) = [0, 0, 0, 0, 0]

❌ **PROBLÈME** : Le pulse a disparu ! Il est "sorti" de la grille.

## Conclusion

`compute_aging_tendency` est **mathématiquement correct** pour l'advection.

**Mais** : Il ne gère pas la condition aux limites (boundary condition) à la dernière cohorte.

Le flux sortant `P[N]` devrait être capturé par le recrutement, pas perdu.

## Solution proposée

Le recrutement doit capturer **exactement** le flux sortant de l'aging à la dernière cohorte:

```
recruitment_flux = P[last_cohort] / dt  (si cette cohorte est au-delà de τ_r)
```

Pas plus, pas moins. Juste le flux qui sort naturellement par aging.
