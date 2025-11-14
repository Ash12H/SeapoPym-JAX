Bien ! Voyons ce qu'il faut pour implémenter l'advection. Voici mes propositions :

Analyse du problème

Équation d'advection : ∂B/∂t = -∇·(u\*B) où u = (u, v) est le champ de vitesse (courants)

Défis techniques :

1. Schéma numérique : **Upwind** vs centré (stabilité/précision)
2. Condition CFL pour l'advection : |u|\*dt/dx <= 1
3. Données de courants : Comment les fournir ? (statiques, dynamiques, interpolées)
4. Halo exchange : Comme diffusion, scope='global'
5. Combinaison advection-diffusion : Ordre des opérations (splitting)

Propositions pour le plan

Option A : Advection simple (upwind, courants constants)

-   Schéma upwind 1er ordre (stable mais diffusif)
-   Courants fournis dans params (u, v constants ou grilles 2D)
-   Tests unitaires avec translation d'un blob
-   ~4h

Option B : Advection précise (semi-Lagrangien)

-   Schéma semi-Lagrangien (conservatif, peu diffusif)
-   Interpolation bilinéaire pour backtracking
-   Plus complexe mais meilleur pour océan
-   ~6-8h

> **NOTE** : Trop complexe pour le moment, on reste sur un schéma simple et on utilisera plus tard un worker pour faire spécifiquement le transport.

Option C : Advection + diffusion couplée

-   Operator splitting (advection puis diffusion)
-   Ou advection-diffusion combinée
-   Nécessite attention à la stabilité
-   ~6h

Ma recommandation

Je propose Option A + tests, puis itérer vers B si nécessaire :

Phase "Advection 2D" (~5h)

1. Implémenter compute_advection_2d (~2h)

    - Schéma upwind 1er ordre
    - Halo exchange pour u, v, biomass
    - Vérification CFL
    - Dans transport.py

2. Tests unitaires (~1.5h)

    - Test translation uniforme (u constant)
    - Test rotation (champ de vitesse circulaire)
    - Test conservation de masse
    - Test avec/sans halos

3. Exemple "advection blob" (~1h)

    - Blob advecté par courant uniforme
    - Visualisation trajectoire
    - Comparer avec solution analytique

4. Optionnel : combiné advection+diffusion (~0.5h)

    - Kernel avec les deux processus
    - Exemple "courant + diffusion"

Questions pour vous

1. Schéma numérique : Upwind 1er ordre suffit pour commencer ? Ou vous voulez directement semi-Lagrangien ?

-   On commence avec upwind pour le moment. On fait quelque chose de simple.

2. Format des courants :

    - Courants constants (scalaires u, v) ?
    - Grilles 2D statiques (u[i,j], v[i,j]) ?
    - Dépendants du temps (u[i,j,t]) ?

-   Le courant est un grille dont la dimension est égale à celle de l'espace + du temps. Donc si on modélise uniquement de la 2D elle est en time+lat+lon. Est-ce qu'on peut faire quelque chose comme ça ? Qu'est-ce que ça implique de devoir généraliser à N dimensions (exemple: je souhaite intégrer un courant vertical) ?

3. Priorité : Advection seule d'abord, ou directement advection-diffusion couplée ?

-   Qu'est-ce que ça implique ? Peut on appliquer l'un puis l'autre dans l'ordre de notre choix ? Ou est-ce plus intéressant/recommandé de faire les deux ensemble ou en splitting ?

4. Tests : Quels cas de test vous semblent essentiels ?

-   Je souhaite vérfieri : que la masse est conservée / que le comportement est bon lorsqu'il y a une "ile" représentée par un bloc de valeurs nulles

---

# Suite

Merci pour vos réponses détaillées ! Je vais analyser vos besoins et proposer un plan adapté.

Analyse de vos réponses

1. Schéma upwind 1er ordre ✓ - Simple et stable pour commencer

2. Courants time-dépendants (time, lat, lon)

-   Implique d'interpoler les courants à chaque timestep
-   Généralisation N dimensions (+ vertical) : ajouter dimension depth → grille 4D (time, depth, lat, lon)
-   Impact : Plus de données en mémoire, interpolation temporelle nécessaire

3. Advection vs Advection-Diffusion
   Excellente question ! Voici les implications :

Option A : Séparé (operator splitting)

# Dans le kernel

biomass_adv = compute_advection_2d(biomass, dt, u, v, ...)
biomass_final = compute_diffusion_2d(biomass_adv, dt, D, ...)

Avantages :

-   Modularité (réutiliser les Units existantes)
-   Flexibilité (activer/désactiver chaque processus)
-   Debug facile (tester indépendamment)

Inconvénients :

-   Erreur de splitting O(dt) (mais acceptable si dt petit)
-   Double halo exchange (coût communication)

Option B : Couplé (advection-diffusion ensemble)

# Équation : ∂B/∂t = -∇·(u*B) + D*∇²B

biomass_final = compute_advection_diffusion_2d(biomass, dt, u, v, D, ...)

Avantages :

-   Plus précis numériquement
-   Un seul halo exchange

Inconvénients :

-   Moins flexible
-   Code plus complexe
-   Couplage stabilité CFL (advection ET diffusion)

Ma recommandation : Splitting (Option A)

-   Votre architecture Unit/Kernel est faite pour ça !
-   Plus maintenable et extensible
-   Ordre conseillé : Advection → Diffusion (Strang splitting classique)

4. Tests avec "îles" - Excellent cas test ! Conditions aux limites avec u·n = 0 (pas de flux à travers îles)

Plan proposé : Phase "Advection 2D"

⏺ Phase 8 : Advection 2D (~6h)

8.1. Structure des données courants (~0.5h)

-   Créer classe/dataclass CurrentField pour gérer interpolation temporelle
-   Stockage : dict avec u[time, lat, lon], v[time, lat, lon]
-   Méthode interpolate_at_time(t) → retourne u[lat,lon], v[lat,lon]

    8.2. Implémenter compute_advection_2d (~2.5h)

-   Schéma upwind 1er ordre : ∂B/∂t = -u*∂B/∂x - v*∂B/∂y
-   Upwind directionnel (gauche si u>0, droite si u<0)
-   Halo exchange pour biomass (u,v peuvent être locaux si grille complète)
-   Vérification CFL : max(|u|, |v|) \* dt / min(dx, dy) <= 1
-   Gestion "îles" : masque avec biomass=0 → u·n=0

    8.3. Tests unitaires (~2h)

-   Test 1 : Translation uniforme (u constant, v=0) → blob se déplace
-   Test 2 : Conservation masse (∫B dA constant)
-   Test 3 : Rotation (u = -y, v = x) → blob tourne
-   Test 4 : Avec "île" (zone u=v=0) → vérifier contournement
-   Test 5 : Halo exchange (domaine décomposé)
-   Test 6 : CFL violation → erreur ou warning

    8.4. Exemple "advection blob" (~1h)

-   Blob advecté par courant uniforme ou rotationnel
-   Visualisation multi-temps (comme diffusion_blob.py)
-   Comparer position analytique vs numérique

Questions de design

Q1 : Stockage des courants

Comment fournir les courants au système ? Deux approches :

A. Dans params (comme D pour diffusion)
params = {
"u": u_field, # jnp.array shape (ntime, nlat, nlon)
"v": v_field,
"time_coords": jnp.array([0, 3600, 7200, ...]), # timestamps
}

-   Simple, mais réplication sur chaque worker
-   OK si grille petite

B. Object Ray partagé (pour grandes grilles)
current_field_ref = ray.put(CurrentField(u, v, time_coords))
params = {"current_field_ref": current_field_ref}

-   Mémoire partagée, un seul exemplaire
-   Plus complexe

Ma recommandation : Commencer avec A (dans params), optimiser avec B si nécessaire.

Q2 : Interpolation temporelle

Si simulation à t=5400s et courants disponibles à t=3600s et t=7200s :

-   Interpolation linéaire : u(t) = u1 + (u2-u1) \* (t-t1)/(t2-t1)
-   Ou "nearest neighbor" (plus simple, moins précis)

Préférence ?

Q3 : Ordre advection-diffusion

Dans le Kernel, quelle séquence ?

-   Strang splitting : Advection(dt/2) → Diffusion(dt) → Advection(dt/2) (2ème ordre)
-   Simple splitting : Advection(dt) → Diffusion(dt) (1er ordre, plus simple)

Pour commencer, simple splitting suffit ?

Q4 : Généralisation 3D (vertical)

Pas pour cette phase, mais pour anticiper :

-   Ajouter dimension depth → grilles 4D
-   Advection 3D : ∂B/∂t = -u*∂B/∂x - v*∂B/∂y - w\*∂B/∂z
-   Décomposition domaine 3D (workers 3D)

On peut structurer le code pour faciliter extension future ?

Proposition de démarrage

Voulez-vous que je commence par :

1. Créer la structure CurrentField pour gérer l'interpolation temporelle ?
2. Implémenter directement compute_advection_2d avec courants statiques d'abord (time-independent), puis ajouter
   interpolation ?
3. Autre approche ?
