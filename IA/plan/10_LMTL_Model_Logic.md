# Logique Mathématique du Modèle LMTL

## Vue d'ensemble

Le modèle LMTL simule la dynamique de population structurée par âge:
- **Production** P(a, t) : biomasse à l'âge a au temps t [g/m²]
- **Biomasse** B(t) : biomasse adulte (recrutée) [g/m²]

## Équations de continuité

### 1. Production structurée par âge

```
∂P/∂t + ∂P/∂a = Source(a) - Sink(a)
```

Avec:
- **Advection temporelle**: ∂P/∂t (changement au cours du temps)
- **Advection en âge**: ∂P/∂a (vieillissement)
- **Source à l'âge 0**: S(0) = E × NPP
- **Sink par recrutement**: R(a) = P(a) si a ≥ τ_r, 0 sinon

### 2. Biomasse adulte

```
dB/dt = Recruitment - Mortality
      = ∫[a≥τ_r] P(a) da / τ_r - λ × B
```

## Discrétisation

### Grille de cohortes

- Cohortes: a_i = i × dt, i = 0, 1, ..., N
- Spacing: Δa = dt (le pas de temps)
- Maximum: a_max ≈ τ_r (ne pas dépasser l'âge de recrutement)

### Équations discrètes

#### Aging tendency (advection en âge)

```
dP_i/dt |_aging = (P_{i-1} - P_i) / dt
```

- Flux entrant: P_{i-1} (production de la cohorte précédente)
- Flux sortant: P_i (production qui vieillit vers la cohorte suivante)
- Pour i=0: P_{-1} = 0 (pas d'influx, seulement NPP)
- Pour i=N: P_N vieillit "hors de la grille" → recrutement

#### Production initialization (source NPP)

```
dP_0/dt |_source = E × NPP
```

- Ajout à l'âge 0 seulement
- E: efficacité de transfert [sans dimension]
- NPP: production primaire [g/m²/s]

#### Recruitment tendency (sink)

```
dP_i/dt |_recruit = -P_i / dt  si a_i + dt ≥ τ_r
                  = 0          sinon

dB/dt |_recruit = Σ[a_i + dt ≥ τ_r] P_i / dt
```

Logique:
- Identifier les cohortes qui **franchiront** τ_r au pas de temps suivant
- Drainer complètement ces cohortes (sink = -P_i/dt)
- Transférer à la biomasse (source = somme des drains)

#### Mortality tendency

```
dB/dt |_mortality = -λ(T) × B

où λ(T) = λ_0 × exp(γ_λ × (T - T_ref))
```

## Intégration temporelle (Euler explicite)

Le TimeIntegrator applique:

```
P_i(t+dt) = P_i(t) + dt × [dP_i/dt |_aging + dP_i/dt |_source + dP_i/dt |_recruit]
B(t+dt) = B(t) + dt × [dB/dt |_recruit + dB/dt |_mortality]
```

## Comportement attendu

### Cas 1: τ_r = 10.38 jours, cohortes 0-10 jours

- La dernière cohorte (i=10, a=10j) vieillira vers a=11j > τ_r
- Elle sera recrutée: P_10 → 0, contribution à B
- Les cohortes 0-9 restent avec production non-nulle

### Cas 2: τ_r = 5 jours, cohortes 0-10 jours

- Les cohortes i ≥ 5 ont a_i ≥ 5j
- Test: a_i + dt ≥ τ_r
  - i=4: 4j + 1j = 5j ≥ 5j → **RECRUTÉ**
  - i=5: 5j + 1j = 6j ≥ 5j → **RECRUTÉ**
  - etc.
- **Résultat attendu**: P_i = 0 pour i ≥ 4
- **État stable**: P_0 = P_1 = P_2 = P_3 = E × NPP, P_i≥4 = 0

## Problèmes potentiels identifiés

### 1. Double drainage (aging + recruitment)

Si une cohorte est recrutée ET vieillit, elle subit deux tendances négatives:
- Aging: -P_i/dt (flux sortant)
- Recruitment: -P_i/dt (drain complet)
- **Total: -2 × P_i/dt** → valeurs négatives!

**Solution**: Le recrutement doit remplacer l'aging pour les cohortes recrutées, pas s'y ajouter.

### 2. Aging hors de la grille

L'aging de la dernière cohorte (i=N) crée un flux P_N qui "sort" de la grille.
Où va-t-il?
- Option A: Perdu (boundary condition)
- Option B: Capturé par le recrutement

**Design actuel**: Le recrutement capture ce flux.

### 3. Cohort spacing = dt

Avec Δa = dt = 1 jour, l'advection en âge est exacte:
- En 1 jour, P_i vieillit complètement vers P_{i+1}
- Pas de résidu dans P_i après aging

Mais si dt ≠ Δa, il faudrait interpolation!

## Tests à effectuer

1. **Aging seul** (sans recruitment, sans source):
   - État initial: P_5 = 1, autres = 0
   - Après 1 pas: P_6 = 1, P_5 = 0
   - Après 5 pas: P_10 = 1, autres = 0

2. **Source seule** (sans aging, sans recruitment):
   - État initial: P_i = 0
   - NPP = 0.1 g/m²/day, E = 0.1
   - Après 1 pas: P_0 = E × NPP × dt = 0.1 × 0.1 × 1 = 0.01 g/m²

3. **Recruitment seul** (sans aging, sans source):
   - État initial: P_5 = 1 g/m², τ_r = 5 jours
   - Après 1 pas: P_5 = 0, B = 1 g/m²

4. **Modèle complet avec τ_r = 5j**:
   - Équilibre: flux entrant (NPP) = flux sortant (recruitment)
   - E × NPP = somme des recrutements
   - Production accumulée dans cohortes 0-4, nulle pour 5+
