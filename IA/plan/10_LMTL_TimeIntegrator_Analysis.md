# Analyse du TimeIntegrator

## Comportement actuel

Le TimeIntegrator (`euler_forward`) applique:

```python
for target_var, tendency_names in tendency_map.items():
    tendencies = [all_results[t] for t in tendency_names]
    total_tendency = sum(tendencies)
    new_state[target_var] = state[target_var] + dt * total_tendency
```

## Pour la variable "production"

Le `tendency_map` contient:
```python
{
    "production": [
        "Micronekton/production_source_npp",    # Source NPP
        "Micronekton/production_aging",          # Aging flux
        "Micronekton/production_sink"            # Recruitment sink
    ]
}
```

Donc:
```
P(t+dt) = P(t) + dt × (source_npp + aging_flux + recruitment_sink)
```

## Application par cohorte

Pour la cohorte i:
```
P_i(t+dt) = P_i(t) + dt × [
    source_npp[i]          # E × NPP si i=0, sinon 0
    + aging_flux[i]        # (P[i-1] - P[i]) / dt
    + recruitment_sink[i]  # -P[i]/dt si recruited, sinon 0
]
```

## Exemple: cohorte 5 avec τ_r = 5j

État à t:
- P_4 = 0.1 g/m²
- P_5 = 0.1 g/m²
- dt = 86400 s

Tendances calculées:
1. **source_npp[5]** = 0 (pas à l'âge 0)
2. **aging_flux[5]** = (P_4 - P_5) / dt = (0.1 - 0.1) / 86400 = 0
3. **recruitment_sink[5]** = -P_5 / dt = -0.1 / 86400

Total:
```
P_5(t+dt) = 0.1 + 86400 × (0 + 0 - 0.1/86400)
          = 0.1 + 86400 × (-1.157e-6)
          = 0.1 - 0.1
          = 0
```

✅ **Correct si P_4 = P_5** (état stable)

## Exemple problématique: cohorte 5 en transition

État à t:
- P_4 = 0.1 g/m²
- P_5 = 0.05 g/m² (en train de se vider)

Tendances:
1. **source_npp[5]** = 0
2. **aging_flux[5]** = (0.1 - 0.05) / 86400 = +5.8e-7 (influx depuis P_4)
3. **recruitment_sink[5]** = -0.05 / 86400 = -5.8e-7 (drainage par recrutement)

Total:
```
P_5(t+dt) = 0.05 + 86400 × (5.8e-7 - 5.8e-7)
          = 0.05
```

❌ **PROBLÈME**: P_5 ne change pas car influx = outflux !

Mais P_4 subit aussi le recrutement:
- aging_flux[4] = (P_3 - P_4) / 86400
- recruitment_sink[4] = -P_4 / 86400

Si P_3 = P_4 = 0.1:
```
P_4(t+dt) = 0.1 + 86400 × (0 - 0.1/86400) = 0
```

✅ P_4 se vide correctement

## Conclusion sur le TimeIntegrator

Le TimeIntegrator fonctionne **correctement**. Il somme simplement les tendances.

Le problème est dans la **logique des tendances**:
- L'aging crée un flux P_i → P_{i+1}
- Le recruitment draine P_i si recruited
- Ces deux processus s'additionnent, ce qui est **correct mathématiquement**

L'équation correcte pour une cohorte recrutée est:
```
dP_i/dt = flux_in - flux_out - recruitment
        = P_{i-1}/dt - P_i/dt - P_i/dt
        = (P_{i-1} - 2×P_i) / dt
```

Non ! C'est faux. Revoyons la logique.

## Correction de la logique

### Option A: Aging sans boundary

```
dP_i/dt |_aging = (P_{i-1} - P_i) / dt
```

Cela crée un flux continu. Pour la dernière cohorte:
```
dP_N/dt |_aging = (P_{N-1} - P_N) / dt
```

P_N se remplit depuis P_{N-1}. Ensuite, le recrutement draine P_N.

### Option B: Aging avec boundary au recrutement

```
dP_i/dt |_aging = (P_{i-1} - P_i) / dt  si i < i_recruit
                = P_{i-1} / dt          si i = i_recruit (pas d'outflux, il va vers B)
                = 0                     si i > i_recruit
```

Dans cette version, l'aging s'arrête à la frontière de recrutement.

## Choix de design

**Option A** (actuelle) est plus simple:
- Aging traite toutes les cohortes uniformément
- Recruitment agit comme un sink additionnel

**Option B** est plus physique:
- L'aging s'arrête au recrutement
- Pas de double comptage

**Recommandation**: Garder Option A mais corriger le recrutement pour qu'il ne draine que l'EXCÈS de production qui franchit la frontière.
