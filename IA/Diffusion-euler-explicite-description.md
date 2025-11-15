Voici l'explication pour une diffusion de type **Euler explicite**.

La diffusion modélise le "mélange" d'une propriété, tendant à lisser les gradients. L'équation physique est (en 2D, avec un coefficient de diffusion $K_h$) :

$\frac{\partial C}{\partial t} = K_h \cdot \left( \frac{\partial^2 C}{\partial x^2} + \frac{\partial^2 C}{\partial y^2} \right)$

Le terme $\left( \frac{\partial^2 C}{\partial x^2} + \frac{\partial^2 C}{\partial y^2} \right)$ est le **Laplacien** de $C$, $\nabla^2 C$.

---

### Le Principe : Euler Explicite et Différences Centrales

Pour résoudre cela numériquement :

1.  **Dans le temps (Euler Explicite) :** On utilise la valeur au temps $n$ pour calculer la valeur au temps $n+1$.
    $\frac{C^{n+1}(i, j) - C^n(i, j)}{\Delta t} = (\text{Terme de diffusion au temps } n)$

2.  **Dans l'espace (Laplacien) :** On discrétise le Laplacien en utilisant des **différences finies centrales**. Cela est différent de l'advection _upwind_ ; la diffusion n'a pas de direction privilégiée, elle mélange dans tous les sens.

L'approximation standard pour la dérivée seconde en $x$ est :
$\frac{\partial^2 C}{\partial x^2} \approx \frac{C(i+1, j) - 2C(i, j) + C(i-1, j)}{dx(j)^2}$

Et pour la dérivée seconde en $y$ :
$\frac{\partial^2 C}{\partial y^2} \approx \frac{C(i, j+1) - 2C(i, j) + C(i, j-1)}{dy^2}$

-   $dx(j)$ est la distance physique entre les centres des cellules en longitude (elle dépend de la latitude $j$).
-   $dy$ est la distance physique entre les centres des cellules en latitude (généralement constante).

### La Règle de Mise à Jour (Le Cœur du Calcul)

En combinant les deux, la règle de mise à jour **pour une cellule interne** $(i, j)$ est :

$$
C^{n+1}(i, j) = C^n(i, j) + \Delta t \cdot K_h \cdot \left[ \frac{C^n(i+1, j) - 2C^n(i, j) + C^n(i-1, j)}{dx(j)^2} + \frac{C^n(i, j+1) - 2C^n(i, j) + C^n(i, j-1)}{dy^2} \right]
$$

**Points clés :**

-   **Explicite :** La nouvelle valeur $C^{n+1}$ ne dépend que des valeurs à l'ancien pas de temps $C^n$.
-   **Centré :** Pour calculer le changement en $(i, j)$, on regarde les voisins directs de tous les côtés : $(i+1, j), (i-1, j), (i, j+1), (i, j-1)$.
-   **Stabilité :** Ce schéma est soumis à une contrainte de stabilité (type CFL) très stricte. Le pas de temps $\Delta t$ doit être très petit, $\Delta t \le \frac{min(dx^2, dy^2)}{4 K_h}$. Sur une grille lat/lon, $dx$ devient très petit près des pôles, ce qui impose un $\Delta t$ minuscule.

---

### 1. Cas Simple : Domaine "Normal" (Limites Fermées)

Pour la diffusion, une limite "fermée" (un mur) implique un **flux nul** à travers le mur. C'est une condition de Neumann : $\frac{\partial C}{\partial n} = 0$ (où $n$ est la direction normale au mur).

-   **Comment l'implémenter ?** On utilise des "cellules fantômes" (ghost cells) dont la valeur est réglée pour assurer un flux nul.
-   **Exemple : Limite Ouest (colonne $i=1$)**
    -   La formule a besoin de $C^n(0, j)$ (la cellule fantôme à l'ouest).
    -   La condition de flux nul $\frac{\partial C}{\partial x} = 0$ se discretise en $\frac{C(1, j) - C(0, j)}{dx} = 0$.
    -   Cela implique : **$C(0, j) = C(1, j)$**.
-   **Application :**
    -   **Limite Ouest ($i=1$) :** On remplace $C^n(0, j)$ par $C^n(1, j)$ dans la formule.
    -   **Limite Est ($i=N$) :** On remplace $C^n(N+1, j)$ par $C^n(N, j)$.
    -   **Limite Sud ($j=1$) :** On remplace $C^n(i, 0)$ par $C^n(i, 1)$.
    -   **Limite Nord ($j=M$) :** On remplace $C^n(i, M+1)$ par $C^n(i, M)$.

Le calcul du Laplacien au bord Ouest $(i=1)$ devient (pour la partie en $x$) :
$\frac{C^n(2, j) - 2C^n(1, j) + C^n(0, j)}{dx^2} \rightarrow \frac{C^n(2, j) - 2C^n(1, j) + C^n(1, j)}{dx^2} = \frac{C^n(2, j) - C^n(1, j)}{dx^2}$

---

### 2. Cas 2 : Limites Périodiques (E/W) et Fermées (N/S)

On combine deux types de conditions aux limites :

-   **Limites Haut/Bas (Nord/Sud) Fermées :**

    -   Identique au Cas 1. C'est un mur à flux nul.
    -   $C^n(i, 0) = C^n(i, 1)$
    -   $C^n(i, M+1) = C^n(i, M)$

-   **Limites Gauche/Droite (Est/Ouest) Périodiques :**
    -   La grille "se referme" sur elle-même.
    -   **Limite Ouest ($i=1$) :** La formule a besoin de $C^n(0, j)$. La cellule à l'ouest de la colonne 1 est la dernière colonne, $N$.
        -   On impose : **$C^n(0, j) = C^n(N, j)$**.
    -   **Limite Est ($i=N$) :** La formule a besoin de $C^n(N+1, j)$. La cellule à l'est de la colonne $N$ est la première colonne, 1.
        -   On impose : **$C^n(N+1, j) = C^n(1, j)$**.

---

### 3. Cas 3 : Cas 2 + Terres (Murs Internes)

C'est une généralisation des limites fermées du Cas 1. Les "terres" sont simplement des **murs internes à flux nul**.

-   **Comment ça marche ?** On utilise un masque terre-mer (ex: `mask(i, j) = 0` si terre, `1` si mer).
-   Le calcul de $C^{n+1}$ n'est effectué que pour les cellules de mer (`mask(i, j) = 1`).
-   Lors du calcul du Laplacien pour une cellule de mer $(i, j)$, on doit vérifier si ses voisins sont de la mer ou de la terre.
-   **Logique de calcul pour $C^{n+1}(i, j)$ (cellule de mer) :**
    1.  On prépare les 4 valeurs voisines nécessaires au calcul :
    2.  **Voisin Ouest :** $C_{\text{ouest}} = C^n(i-1, j)$ **SI** `mask(i-1, j) == 1` (c'est de la mer).
        **SINON** (c'est de la terre), $C_{\text{ouest}} = C^n(i, j)$ (condition de flux nul).
    3.  **Voisin Est :** $C_{\text{est}} = C^n(i+1, j)$ **SI** `mask(i+1, j) == 1`.
        **SINON** $C_{\text{est}} = C^n(i, j)$.
    4.  **Voisin Sud :** $C_{\text{sud}} = C^n(i, j-1)$ **SI** `mask(i, j-1) == 1`.
        **SINON** $C_{\text{sud}} = C^n(i, j)$.
    5.  **Voisin Nord :** $C_{\text{nord}} = C^n(i, j+1)$ **SI** `mask(i, j+1) == 1`.
        **SINON** $C_{\text{nord}} = C^n(i, j)$.
    6.  On applique la formule en utilisant ces valeurs préparées :
        $C^{n+1}(i, j) = C^n(i, j) + \Delta t \cdot K_h \cdot \left[ \frac{C_{\text{est}} - 2C^n(i, j) + C_{\text{ouest}}}{dx(j)^2} + \frac{C_{\text{nord}} - 2C^n(i, j) + C_{\text{sud}}}{dy^2} \right]$

Cette logique gère automatiquement tous les cas : les cellules au milieu de l'océan, les cellules de bordure (périodiques) et les cellules côtières (contre la terre).
