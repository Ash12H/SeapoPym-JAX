Voici une explication de la réalisation d'une advection _upwind_ (ou "amont") sur une grille 2D.

L'advection est le processus de transport d'une propriété (comme la température, la salinité, ou un traceur, que nous appellerons $C$) par un courant $(u, v)$. Le schéma _upwind_ est une méthode numérique pour calculer ce transport.

Son principe fondamental est simple : **la valeur de $C$ qui traverse la face d'une cellule est la valeur de $C$ de la cellule située _en amont_ du flux.**

Pour ce faire, on calcule les _flux_ de $C$ qui entrent et sortent de chaque cellule de la grille $(i, j)$ par ses quatre faces (Est, Ouest, Nord, Sud). La variation de $C$ dans la cellule $(i, j)$ est la somme de ces flux (flux entrants - flux sortants).

$\frac{dC(i, j)}{dt} = \frac{1}{\text{Volume}(i,j)} \cdot (F_{\text{ouest}} + F_{\text{sud}} - F_{\text{est}} - F_{\text{nord}})$

Le défi est de déterminer la valeur de $C$ à utiliser pour calculer chaque flux.

---

### Le Principe : Le Choix Upwind

Concentrons-nous sur le calcul du flux à travers la **face Est** de la cellule $(i, j)$.

-   Cette face la sépare de la cellule $(i+1, j)$.
-   La vitesse sur cette face est $u_e$.
-   Le flux $F_{\text{est}}$ sera $u_e \cdot C_{\text{face}} \cdot \text{Aire}_{\text{face}}$.

Quelle valeur utiliser pour $C_{\text{face}}$ ?

-   **Cas 1 : $u_e > 0$**

    -   Le courant va de l'Ouest vers l'Est (de $i$ vers $i+1$).
    -   Le flux _sort_ de la cellule $(i, j)$.
    -   La propriété $C$ transportée est celle _en amont_ du flux, c'est-à-dire celle de la cellule $(i, j)$.
    -   Donc, $C_{\text{face}} = C(i, j)$.

-   **Cas 2 : $u_e < 0$**
    -   Le courant va de l'Est vers l'Ouest (de $i+1$ vers $i$).
    -   Le flux _entre_ dans la cellule $(i, j)$.
    -   La propriété $C$ transportée est celle _en amont_ du flux, c'est-à-dire celle de la cellule $(i+1, j)$.
    -   Donc, $C_{\text{face}} = C(i+1, j)$.

On applique cette même logique pour les 4 faces :

-   **Flux Est ($F_e$) :** $C_{\text{face}} = C(i, j)$ si $u_e > 0$, sinon $C(i+1, j)$.
-   **Flux Ouest ($F_w$) :** $C_{\text{face}} = C(i-1, j)$ si $u_w > 0$, sinon $C(i, j)$.
-   **Flux Nord ($F_n$) :** $C_{\text{face}} = C(i, j)$ si $v_n > 0$, sinon $C(i, j+1)$.
-   **Flux Sud ($F_s$) :** $C_{\text{face}} = C(i, j-1)$ si $v_s > 0$, sinon $C(i, j)$.

**Note sur la grille Lat/Lon :**
Sur une grille lat/lon, les cellules n'ont pas la même taille. L'aire de la cellule $(i, j)$ et les aires de ses faces (Nord/Sud vs Est/Ouest) changent avec la latitude. Le calcul de flux doit en tenir compte : $F = u \cdot C_{\text{face}} \cdot \text{Aire}_{\text{face}}$. Par exemple, l'aire d'une face Nord/Sud (traversée par $v$) dépend du $R \cdot \cos(\text{latitude}) \cdot \Delta\lambda$. Le principe _upwind_ pour choisir $C_{\text{face}}$ reste identique.

---

### 1. Cas Simple : Domaine "Normal" (Limites Fermées)

Dans ce cas, le domaine a des murs sur les quatre côtés (par exemple, un lac).

-   **Comment modéliser un "mur" ?** On impose une vitesse _nulle_ à travers ce mur.
-   **Limite Ouest (colonne $i=1$) :** On force la vitesse sur la face ouest de toutes les cellules de cette colonne à être nulle : $u_w(1, j) = 0$ pour tout $j$.
-   **Limite Est (colonne $i=N$) :** $u_e(N, j) = 0$ pour tout $j$.
-   **Limite Sud (ligne $j=1$) :** $v_s(i, 1) = 0$ pour tout $i$.
-   **Limite Nord (ligne $j=M$) :** $v_n(i, M) = 0$ pour tout $i$.

**Conséquence :**
Lorsque l'algorithme calcule les flux pour une cellule de bord (par exemple, la cellule $(1, j)$ à l'Ouest), le flux $F_{\text{ouest}}$ sera :
$F_{\text{ouest}} = u_w \cdot C_{\text{face}} \cdot \text{Aire}_{\text{face}} = 0 \cdot C_{\text{face}} \cdot \text{Aire}_{\text{face}} = 0$.
Il n'y a donc **aucun flux** qui traverse les frontières du domaine. Les calculs des flux internes (Est, Nord, Sud pour cette cellule) se font normalement.

---

### 2. Cas 2 : Limites Périodiques (Gauche/Droite) et Fermées (Haut/Bas)

C'est le cas typique d'un modèle global (la Terre "boucle" sur elle-même) ou d'un canal.

-   **Limites Haut/Bas (Nord/Sud) Fermées :**

    -   Identique au Cas 1. On impose $v_s(i, 1) = 0$ et $v_n(i, M) = 0$.
    -   Les flux $F_{\text{sud}}$ pour la ligne $j=1$ et $F_{\text{nord}}$ pour la ligne $j=M$ sont nuls.

-   **Limites Gauche/Droite (Est/Ouest) Périodiques :**

    -   La grille "se referme" sur elle-même. La cellule à l'Est de la dernière colonne ($i=N$) est la première colonne ($i=1$). La cellule à l'Ouest de la première colonne ($i=1$) est la dernière colonne ($i=N$).
    -   **Pour la cellule $(1, j)$ (bord Ouest) :**
        -   $F_e, F_n, F_s$ sont calculés normalement.
        -   Pour $F_w$, la cellule "en amont" (si $u_w > 0$) est la cellule $(N, j)$.
        -   Le choix upwind pour $C_{\text{face,ouest}}$ devient : $C_{\text{face}} = C(N, j)$ si $u_w > 0$, sinon $C(1, j)$.
    -   **Pour la cellule $(N, j)$ (bord Est) :**
        -   $F_w, F_n, F_s$ sont calculés normally.
        -   Pour $F_e$, la cellule "en amont" (si $u_e < 0$) est la cellule $(1, j)$.
        -   Le choix upwind pour $C_{\text{face,est}}$ devient : $C_{\text{face}} = C(N, j)$ si $u_e > 0$, sinon $C(1, j)$.

    > En pratique, on gère cela avec un "remplissage fantôme" (ghost cells) où la colonne "0" est une copie de la colonne $N$, et la colonne "N+1" est une copie de la colonne 1.

---

### 3. Cas 3 : Cas 2 + Terres (Murs Internes)

L'ajout de "terres" (continents, îles) revient à ajouter des **murs internes** dans le domaine.

-   **Comment ça marche ?** C'est une généralisation du Cas 1.
-   On utilise un **masque terre-mer** (un tableau 2D qui vaut 1 pour l'eau et 0 pour la terre).
-   Avant de calculer l'advection, on **modifie les champs de vitesse $u$ et $v$** :
    -   Partout où une face de cellule se trouve entre une cellule "mer" et une cellule "terre", la vitesse _normale_ (perpendiculaire) à cette face est **mise à zéro**.
    -   Par exemple, si la cellule $(i, j)$ est de l'eau mais la cellule $(i+1, j)$ est de la terre : la vitesse $u_e$ sur la face Est de $(i, j)$ est $u_e = 0$.
    -   De même, si $(i, j)$ est de l'eau et $(i, j-1)$ est de la terre : la vitesse $v_s$ sur la face Sud de $(i, j)$ est $v_s = 0$.

**Conséquence :**
Le code d'advection _upwind_ (décrit au début) peut alors être exécuté **sans aucune modification**.

Quand il calculera le flux pour la cellule d'eau $(i, j)$ à côté de la terre $(i+1, j)$, il trouvera :
$F_{\text{est}} = u_e \cdot C_{\text{face}} \cdot \text{Aire}_{\text{face}} = 0 \cdot C_{\text{face}} \cdot \text{Aire}_{\text{face}} = 0$.

Les "terres" sont donc simplement traitées comme des frontières fermées internes au domaine, où le flux est automatiquement nul parce que la vitesse normale est nulle. Le calcul de $C$ n'est effectué que pour les cellules "mer".

---

---

---

---

---

---

---

---

# Questions

1. Convention de Grille : Staggered vs Collocated

Question fondamentale : Où sont localisées les vitesses u,v par rapport à la biomasse ?

Option A : Grille Collocated (Plus Simple)

Biomasse B, vitesses u,v au CENTRE des cellules (i,j)
┌─────┬─────┬─────┐
│ B,u,v│ B,u,v│ B,u,v│
├─────┼─────┼─────┤
│ B,u,v│ B,u,v│ B,u,v│
└─────┴─────┴─────┘

-   Avantage : Simple, tout au même endroit
-   Inconvénient : Besoin d'interpoler u,v aux faces pour calculer flux
-   Exemple : u_face_est[i,j] = 0.5 \* (u[i,j] + u[i,j+1])

Option B : Grille Staggered (Arakawa C, Plus Physique)

Biomasse B au centre, u sur faces E/O, v sur faces N/S
v[i,j]
↓
┌───┬───┬───┐
│ B → u B → u B →
├───┼───┼───┤
│ B → u B → u B →
↓ ↓

-   Avantage : Vitesses déjà aux faces → pas d'interpolation → meilleure conservation
-   Inconvénient : Gestion indices plus complexe

Vos données SEAPOPYM : Les forcings u,v océaniques viennent comment ? Centres de cellules ou faces ?

> Réponse : Je pense que la vitesse est au centre de la cellule. Est-ce qu'il faut proposer une autre méthode si la vitesse n'est pas au centre ? Est-ce que c'est bloquant pour JAX ? Il me semble que la bibliothèque XGCM permet de nativement prendre en compte ces grilles : https://xgcm.readthedocs.io/en/latest/ .

---

2. Aires des Faces sur Grille Lat/Lon

Sur une grille lat/lon, les cellules n'ont pas toutes la même taille. Voici le calcul :

Géométrie Sphérique

# Paramètres

R = 6371e3 # Rayon Terre (m)
lat = jnp.linspace(lat_min, lat_max, nlat) # Latitudes centres cellules (degrés)
lon = jnp.linspace(lon_min, lon_max, nlon) # Longitudes centres cellules (degrés)

dlat = (lat_max - lat_min) / nlat # Espacement latitude (degrés)
dlon = (lon_max - lon_min) / nlon # Espacement longitude (degrés)

# Conversion degrés → radians

dlat_rad = jnp.radians(dlat)
dlon_rad = jnp.radians(dlon)
lat_rad = jnp.radians(lat)

# Aires des cellules (nlat, nlon)

# Aire ≈ R² × cos(lat) × dλ × dφ

cell*areas = R\*\_2 * jnp.cos(lat*rad)[:, None] * dlon*rad * dlat_rad

# Aires des faces Nord/Sud (perpendiculaires à v)

# Face N/S à latitude fixe : longueur = R × cos(lat) × dλ, hauteur = 1

face*areas_ns = R * jnp.cos(lat*rad)[:, None] * dlon_rad # (nlat, nlon)

# Aires des faces Est/Ouest (perpendiculaires à u)

# Face E/O : longueur = R × dφ, hauteur = 1

face_areas_ew = R \* dlat_rad # Scalaire (identique partout)

Question : Est-ce que cette formulation sphérique est OK pour SEAPOPYM, ou utilisez-vous une approximation plane ?

> Réponse : Je préfère utiliser les coordonnées sphériques en effet. Mais c'est une simple propriété de notre grille n'est-ce pas ? Si jamais je souhaite modifier le comportement il suffira de définir un autre type de grille et donc une nouvelle fonction pour calculer la surface ?

---

3. Ghost Cells vs Boundary Conditions

Dans votre Advection-upwind-description.md, vous mentionnez :

Cas 1 : Limites Fermées

u_face_ouest[i=0] = 0 # Pas de flux au bord Ouest

Cas 2 : Limites Périodiques E/O + Fermées N/S

# Cellule (i=0, j) : son voisin Ouest est (i=nlon-1, j)

B_ghost_ouest[j] = B[nlon-1, j]

Votre cas SEAPOPYM :

-   Longitude : Périodique (Terre boucle) ou fermé ?
-   Latitude : Fermé (Nord/Sud = murs) ?

Si périodique E/O, le code change :

# Périodique en longitude

biomass_periodic = jnp.concatenate([biomass[:, -1:], biomass, biomass[:, :1]], axis=1)

# Maintenant biomass_periodic[:, 0] est la cellule Ouest de biomass[:, 0]

# Et biomass_periodic[:, -1] est la cellule Est de biomass[:, -1]

> Réponse : J'aimerai pouvoir choisir un comportement en fonction de la dimension parmis CLOSED / OPEN / PERIODIC . Ainsi je peux dire lat=CLOSED et lon=PERIODIC. Mais je pourrai aussi dans le futur choisir depth=CLOSED

---

4. Masques : Zeroing Velocity vs Flux Blocking

Votre approche (doc) : Modifier u,v directement

# Avant calcul flux

u_masked = jnp.where(mask[:, :-1] & mask[:, 1:], u, 0.0)
v_masked = jnp.where(mask[:-1, :] & mask[1:, :], v, 0.0)

Question : Les forcings u,v océaniques que vous recevez ont-ils déjà u=v=0 sur terre, ou faut-il les forcer ?

Si u,v sont déjà nuls sur terre, alors le masking est juste pour garantir B=0 sur terre après update.

> Réponse : En général c'est effectivement soit des 0 soit des Nans. Mais c'est une sécurité intéressante de forcer la mise à 0.

---

5. Stabilité CFL

Pour l'advection upwind :
CFL = max(|u|) × dt / dx ≤ 1

Sur grille lat/lon, dx varie avec latitude :
dx[lat] = R × cos(lat) × dlon_rad # Plus petit aux pôles !
dy = R × dlat_rad # Constant

# CFL par cellule

CFL*x = jnp.abs(u) * dt / dx[:, None]
CFL*y = jnp.abs(v) * dt / dy

CFL_max = jnp.max(jnp.maximum(CFL_x, CFL_y))

Question : Voulez-vous un dt adaptatif (calculé par le worker) ou dt fixe imposé par l'utilisateur avec vérification ?

> Réponse : La gestion du dt et du calcul CFL sera faite en amont de la simulation. Pas besoin de l'inclure dans le développement ici.

---

6. Ordre d'Application : Advection puis Diffusion

Actuellement le plan est :
B1 = advection(B0, u, v, dt)
B2 = diffusion(B1, D, dt)

Alternative : Operator Splitting de Strang (2ème ordre en temps)
B1 = diffusion(B0, D, dt/2)
B2 = advection(B1, u, v, dt)
B3 = diffusion(B2, D, dt/2)

Plus précis mais 3 étapes au lieu de 2. Pour MVP, OK de rester simple ?

> Réponse : Oui on reste sur quelque chose de simple. On pourra rafiner plus tard.

---

7. Stockage en Mémoire : Shape des Arrays

Proposition actuelle :
biomass.shape = (nlat, nlon) # B au centre
u.shape = (nlat, nlon) # u au centre (si collocated)
v.shape = (nlat, nlon) # v au centre (si collocated)
mask.shape = (nlat, nlon) # mask au centre

Si staggered :
biomass.shape = (nlat, nlon)
u.shape = (nlat, nlon+1) # u sur faces E/O
v.shape = (nlat+1, nlon) # v sur faces N/S
mask.shape = (nlat, nlon)

Quel format correspondent à vos données CellWorker2D actuelles ?

> Réponse : si je comprend bien ta question elle fait référence à ce dont on a parlé dans la question 1.

---

Mes Questions Récapitulatives

Pour que je puisse écrire le code exact adapté à SEAPOPYM :

1. Grille : Collocated (simple) ou Staggered (physique) ?
    > Réponse : Je pense que la vitesse est au centre de la cellule. Est-ce qu'il faut proposer une autre méthode si la vitesse n'est pas au centre ? Est-ce que c'est bloquant pour JAX ? Il me semble que la bibliothèque XGCM permet de nativement prendre en compte ces grilles : https://xgcm.readthedocs.io/en/latest/ .
2. Géométrie : Sphérique (avec cos(lat)) ou plane (dx=dy=constant) ?
    > Réponse : Je préfère utiliser les coordonnées sphériques en effet. Mais c'est une simple propriété de notre grille n'est-ce pas ? Si jamais je souhaite modifier le comportement il suffira de définir un autre type de grille et donc une nouvelle fonction pour calculer la surface ?
3. Limites E/O : Périodiques ou fermées ?
    > Réponse : J'aimerai pouvoir choisir un comportement en fonction de la dimension parmis CLOSED / OPEN / PERIODIC . Ainsi je peux dire lat=CLOSED et lon=PERIODIC. Mais je pourrai aussi dans le futur choisir depth=CLOSED
4. Limites N/S : Fermées (u=v=0) ?
    > Réponse : En général c'est effectivement soit des 0 soit des Nans. Mais c'est une sécurité intéressante de forcer la mise à 0.
5. Forcings u,v : Déjà zéro sur terre ou faut-il masquer ?
    > Réponse : La gestion du dt et du calcul CFL sera faite en amont de la simulation. Pas besoin de l'inclure dans le développement ici.
6. dt : Adaptatif (worker calcule) ou fixe (vérification CFL) ?
    > Réponse : Oui on reste sur quelque chose de simple. On pourra rafiner plus tard.
7. Shape des arrays : Format actuel des données ?
    > Réponse : si je comprend bien ta question elle fait référence à ce dont on a parlé dans la question 1.
