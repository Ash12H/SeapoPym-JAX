Le modèle décrit un système zooplancton à 2 compartiments inspiré de SEAPODYM-LMTL :

1. Biomasse adulte B (sans âge)

-   Dynamique : ∂B/∂t = R - λB
-   Mortalité thermosensible : λ(T) = λ₀ exp(γ_λ (T - T_ref))
-   Reçoit le recrutement R depuis la production

2. Production juvénile p(t,τ) (avec âge)

-   Équation de McKendrick-Von Foerster : ∂p/∂t + ∂p/∂τ = -μ(t,τ)p
-   Source : production primaire nette p(t,0) = E × NPP(t)
-   Fenêtre de recrutement : A(t) = [τ_r(T), τ_{r0}]
-   Transfert vers B via taux μ = α Π(τ ∈ A)

3. Schéma numérique

-   Euler implicite pour B : B^{n+1} = (B^n + Δ R) / (1 + Δ λ)
-   Lagrangien pour p : p*a^{n+1} = p*{a-1}^n exp(-μ_a Δ)
-   Recrutement : R = Σ p_a (1 - S_a)

---

❓ Questions pour clarifier l'intégration

A. Forcings et données

1. Avez-vous des champs NPP(x,y,t) et T(x,y,t) disponibles ?

> Oui j'ai des champs NPP et T disponiblent.

2. Format des données (NetCDF, résolution spatiale/temporelle) ?

> Netcdf et/ou zarr mais je passe par xarray quoi qu'il arrive.

B. Paramètres du modèle

Pouvez-vous me donner les valeurs typiques :

-   Mortalité : λ₀, γ_λ, T_ref
-   Production : E (efficacité de transfert)
-   Recrutement : α, τ*{r0}, γ*{τr}
-   Pas de temps/âge : Δt, Δτ (le doc suggère Δt = Δτ)

> lambda_0 = 1/150 ; gamma_lambda = 0.15 ; E = 0.1668 ; tau_r0 = 10.38 ; gamma_tau_r = 0.11 ; l'âge est une dimension au même titre que le temps donc on a une matrice allant de l'âge 1 à l'âge tau_r_0 car on recrute quoi qu'il arrive à cet age max ; Δt = Δτ

C. Structure d'âge

3. Combien de classes d'âge pour la production ? (τ\_{r0} typique ?)

> J'ai répondu plus tot : tau_r_0 / delta_tau

4. L'âge maximal correspond-il à un nombre fixe de jours ?

> Oui

D. Couplage avec le transport

5. Question cruciale : La production p(t,τ) doit-elle être transportée (advection/diffusion) ?

    - Si OUI : chaque classe d'âge est transportée → complexité++
    - Si NON : seule la biomasse B est transportée → plus simple

> Oui on transporte la production ET la biomasse. Comment gérer de manière simple le fait qu'on transporte n'importe quelle masse ? Je souhaite appliquer l'advection à la production puis à la biomasse et je ne souhaite pas réécrire la fonction d'advection.

6. Ou bien : production locale uniquement (p ne se déplace pas, seul B est mobile) ?

> Les deux se déplacent

E. Initialisation

7. Comment initialiser B et p au t=0 ?

    - B₀ uniforme ? Depuis données ?
    - p₀ = 0 partout (spin-up) ?

> Pour le moment à 0 mais plus tard on peut fournir un état initial.

F. Diagnostics

8. Quelles variables voulez-vous en sortie ?

    - B (biomasse adulte)
    - R (recrutement)
    - p par classe d'âge ?
    - NPP absorbée ?

> Je souhaite obtenir la biomasse.

---

🏗️ Proposition d'intégration (préliminaire)

En attendant vos réponses, voici 2 architectures possibles :

Option A : Production locale (p non transportée)

CellWorker
├─ State: [biomass, production_age_0, ..., production_age_N]
├─ Kernel:
│ ├─ Unit 1: compute_production_aging (local)
│ ├─ Unit 2: compute_recruitment (local)
│ └─ Unit 3: update_biomass (local)
└─ Forcings: [NPP, Temperature]

TransportWorker
└─ Transporte uniquement "biomass"

Avantages : Simple, production suit NPP locale
Inconvénients : Pas de dispersion larvaire

Option B : Production transportée (complexe)

CellWorker
├─ State: [biomass, prod_age_0, ..., prod_age_N]
├─ Kernel: biologie + vieillissement

TransportWorker
└─ Transporte [biomass, prod_age_0, ..., prod_age_N]
→ N+1 champs à transporter

Avantages : Réaliste (dispersion larvaire)
Inconvénients : Coûteux (N champs), complexité algorithmique

---

🎯 Ma recommandation (après vos réponses)

Une fois que j'aurai vos réponses, je pourrai :

1. Créer les Units biologique :

    - AgeProduction : vieillissement de p
    - ComputeRecruitment : calcul de R depuis p
    - UpdateBiomass : intégration Euler implicite de B

2. Adapter le ForcingManager :

    - Charger NPP et Température
    - Interpolation temporelle si nécessaire

3. Créer un notebook de démonstration :

    - Cas test avec forçages synthétiques
    - Vérification conservation de masse
    - Sensibilité à la température

4. Tests unitaires :

    - Validation des équations
    - Conservation de masse (B + Σp_a constant sans mortalité)

---

Pouvez-vous me répondre aux questions A-F pour que je vous propose une implémentation détaillée ? 🚀
