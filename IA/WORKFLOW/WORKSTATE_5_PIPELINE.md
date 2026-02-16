# Workflow State — Brique 5 : Pipeline CMA-ES → NUTS

## Informations generales

- **Projet** : SeapoPym v0.2 — Chaîne CMA-ES → NUTS
- **Brique** : 5/5 — Pipeline CMA-ES → NUTS
- **Etape courante** : 5. Exécution (en cours)
- **Role actif** : Facilitateur
- **Derniere mise a jour** : 2026-02-16
- **Branche** : `jax`

## Objectif global

Assembler IPOP-CMA-ES et NUTS en un pipeline complet pour estimer les paramètres du modèle LMTL 0D (twin experiment). CMA-ES identifie les modes (y compris les paramètres à faible gradient), NUTS affine les paramètres à gradient exploitable.

---

## Contexte : problèmes d'identifiabilité découverts

### 1. Gradient quasi-nul pour τ_r₀ / γ_τ_r

Le recrutement utilise une sigmoïde très serrée (transition ±1 jour, k = ln(99)/86400 ≈ 5.32e-5 s⁻¹). Avec des cohortes journalières, seules ~2 cohortes sont dans la zone de transition. Résultat :

| Paramètre     | Sensibilité normalisée |grad × val| | Facteur vs λ₀ |
|---------------|----------------------------------------|----------------|
| lambda_0      | ~7                                     | référence      |
| gamma_lambda  | ~17                                    | ~2.4×           |
| efficiency    | ~7                                     | ~1×            |
| **tau_r_0**   | **~0.008**                             | **~1000× plus faible** |
| **gamma_tau_r** | **~0.020**                           | **~350× plus faible**  |

**Conséquences :**
- NUTS (gradient-based) ne peut pas explorer efficacement τ_r₀ / γ_τ_r
- Avec 2000 warmup + acceptance_rate=0.95 : ~160/200 divergences, ces paramètres font du random walk
- CMA-ES (gradient-free) n'a pas ce problème : il évalue la loss par population (finite differences implicites)

**Solutions envisagées et rejetées :**
- Augmenter le nombre de cohortes (résolution sub-journalière) → trop coûteux pour le transport 2D
- Élargir la pente de la sigmoïde → changerait la physique du modèle
- Meilleur warmup → amélioration marginale (162 → 160 divergences)

### 2. Compensation entre paramètres (équifinalité)

Les paramètres λ₀ et efficiency se compensent : le flux entrant (E × NPP) et sortant (λ₀ × biomasse) s'ajustent mutuellement pour produire la même biomasse à l'équilibre.

**Indices :**
- Analyse Sobol : ordre total >> ordre 1 pour ces paramètres → interactions fortes
- CMA-ES trouve des modes avec λ₀ et E aux bornes mais une loss quasi-identique
- Il existe une "vallée" (ou "banane") dans l'espace (E, λ₀) plutôt qu'un minimum ponctuel

**Solutions possibles :**
- Reparamétriser (ratio E/λ₀, ou biomasse d'équilibre) — change la sémantique biologique
- Observations plus riches (production par cohorte, pas seulement biomasse totale)
- Priors informatifs sur E ou λ₀ issus de la littérature
- Accepter l'équifinalité et la quantifier via NUTS (la corrélation postérieure EST un résultat)

---

## Travail réalisé (commits existants)

| Commit    | Description |
|-----------|-------------|
| `3e92b6a` | Exemple IPOP-CMA-ES twin experiment sur LMTL 0D |
| `538aef9` | Exemple NUTS twin experiment + reparamétrisation unit-space |
| `4c2d377` | Script comparaison sigmoïde vs seuil dur |
| `a61de08` | Remplacement du seuil de recrutement `>=` par sigmoïde dans `lmtl.py` |
| `f9295a2` | Tuning NUTS warmup=2000, acceptance_rate=0.95 |

## Modifications non commitées

### 1. `seapopym/optimization/prior.py`
- Ajout `_bounds_arrays()` : évalue les bornes eagerly (hors JIT) pour éviter `ConcretizationTypeError` avec `HalfNormal`

### 2. `seapopym/optimization/likelihood.py`
- `reparameterize_log_posterior()` : pré-calcule les bornes avant la closure JIT (inline `from_unit` avec bornes pre-computed)

### 3. `examples/ipop_cmaes_lmtl_0d.py`
- Paramètres CMA-ES mis aux défauts Hansen (pycma, Auger & Hansen 2005) :
  - `INITIAL_POPSIZE = 4 + floor(3 * ln(n))` → 8 pour n=5
  - `N_GENERATIONS = 100 + 150 * (n+3)² / √popsize` → ~3020 pour n=5
  - `N_RESTARTS = 5`
- Note : n'a pas encore été exécuté avec ces réglages

### 4. `examples/pipeline_cmaes_nuts_lmtl_0d.py` (nouveau, non tracké)
- Pipeline complet CMA-ES → NUTS :
  - Stage 1 : IPOP-CMA-ES sur tous les paramètres
  - Stage 2 : NUTS uniquement sur (λ₀, γ_λ, efficiency), τ_r₀/γ_τ_r fixés depuis CMA-ES
- Priors NUTS : HalfNormal pour λ₀ et γ_λ, Uniform pour efficiency
- Visualisation : biomass comparison, parameter recovery bars, NUTS traces, corner plot loss landscape

---

## Implémentation technique clé

### Sigmoïde de recrutement (`seapopym/functions/lmtl.py`)

```python
_RECRUITMENT_TRANSITION_DAYS = 1.0
_K_SIGMOID = float(jnp.log(99.0)) / (_RECRUITMENT_TRANSITION_DAYS * 86400.0)

# Dans aging_flow et recruitment_flow :
recruit_fraction = jax.nn.sigmoid(_K_SIGMOID * (cohort_ages - rec_age))
aging_outflow = (1.0 - recruit_fraction) * base_outflow    # non recruté
flux_to_biomass = recruit_fraction * base_outflow            # recruté
# Conservation : aging + recruitment = base_outflow
```

### Reparamétrisation unit-space (`prior.py` + `likelihood.py`)

```python
# prior.py — PriorSet._bounds_arrays() : évalue hors JIT
# prior.py — PriorSet.to_unit(params), from_unit(params_unit), log_det_jacobian()

# likelihood.py — reparameterize_log_posterior(log_posterior_fn, prior_set)
# Pré-calcule bounds eagerly, puis closure JIT-safe :
#   params_phys[name] = params_unit[name] * (high - low) + low
#   return log_posterior_fn(params_phys) + log_det_jac
```

### Réglages Hansen pour CMA-ES

```python
N_PARAMS = 5
INITIAL_POPSIZE = 4 + int(3 * math.log(N_PARAMS))          # 8
N_GENERATIONS = int(100 + 150 * (N_PARAMS + 3)**2 / math.sqrt(INITIAL_POPSIZE))  # ~3020
N_RESTARTS = 5  # (Auger & Hansen 2005 utilisent 9 dans leurs benchmarks)
# IPOP : popsize double à chaque restart → 8, 16, 32, 64, 128
```

---

## Prochaines étapes

1. **Exécuter IPOP-CMA-ES** avec les réglages Hansen sur GPU (le budget est ~3020 générations × 5 restarts avec pop doublée, soit ~750k évaluations) pour vérifier l'identifiabilité des paramètres
2. **Analyser les résultats** : est-ce que CMA-ES retrouve les vrais paramètres ? Combien de modes distincts ? Quelle compensation E/λ₀ ?
3. **Profil de vraisemblance** (optionnel) : fixer un paramètre, optimiser les autres → quantifie l'équifinalité
4. **Pipeline complet** : si CMA-ES identifie correctement les modes, exécuter NUTS en stage 2 pour affiner les paramètres à gradient exploitable
5. **Corner plot amélioré** : utiliser échelle log pour la loss et zoomer autour du minimum (le corner plot actuel est tout jaune car les bornes sont trop larges)

## Références

- Hansen, N. (2016). *The CMA Evolution Strategy: A Tutorial*. arXiv:1604.00772
- Auger, A. & Hansen, N. (2005). *A Restart CMA Evolution Strategy With Increasing Population Size*. CEC 2005.
- pycma defaults : `popsize = 4 + floor(3*ln(n))`, `maxiter = 100 + 150*(n+3)²/√popsize`
