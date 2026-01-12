# Matériel et Méthodes

## 1. Formalisme du Modèle Orienté Flux

### 1.1. Architecture en Graphe Acyclique Dirigé (DAG)

Nous modélisons la dynamique de la population comme un système dynamique ouvert structuré par un **Graphe Dirigé Acyclique (DAG)**. Soit $G = (V, E)$ le graphe où $V$ représente l'ensemble des nœuds et $E$ les dépendances de flux. L'architecture décompose le modèle en quatre types de nœuds :

1.  **Nœuds d'État ($S$)** : Variables conservées (Biomasse $B$, Production par âge $P$).
2.  **Nœuds de Forçage ($F$)** : Température, Production Primaire, Courants ($\mathbf{v}$), Coefficient de Diffusion ($D$).
3.  **Nœuds Intermédiaires ($I$)** : Variables transitoires calculées à la volée (ex: métabolisme thermique).
4.  **Nœuds Fonctionnels ($Op$)** : Opérateurs transformant des états et forçages en tendances.

### 1.2. Intégration des Processus Biologiques

Les processus biologiques (Recrutement, Production, Mortalité) sont implémentés sous forme de nœuds fonctionnels autonomes. Les équations constitutives qui régissent ces processus sont identiques à celles décrites et validées dans notre précédent travail sur SeapoPym v0.3 [Lehodey Jr. et al.].

L'innovation réside ici dans leur encapsulation : chaque processus devient un opérateur sans mémoire ("stateless") qui consomme l'état courant $S(t)$ et produit un flux instantané $\Phi_{bio}$. Par exemple, l'opérateur de mortalité ne modifie pas directement la biomasse, mais émet une tendance de perte que le contrôleur agrègera ultérieurement.

### 1.3. Unification du Transport (Advection-Diffusion) et de la Réaction

Le transport physique est traité comme un nœud fonctionnel $Op_{trans}$ standard, prenant en entrée les champs de vitesse $\mathbf{v}$ et de diffusion $D$. Il calcule le bilan net des flux (Advection + Diffusion) aux interfaces des mailles.

L'évolution de la biomasse $B$ est ainsi régie par la somme directe des sorties de tous les nœuds fonctionnels du graphe :

$$ \frac{dB}{dt} = \sum \Phi = \Phi*{bio}(S, F) + \Phi*{adv}(S, \mathbf{v}) + \Phi\_{diff}(S, D) $$

Cette formulation unifie conceptuellement la physique et la biologie : pour le contrôleur, l'advection n'est qu'un processus de flux parmi d'autres, traité avec la même priorité que la mortalité.

## 2. Discrétisation et Mise en Œuvre

### 2.1. Volumes Finis : Advection et Diffusion

Le domaine spatial est discrétisé par la méthode des Volumes Finis, assurant la conservation stricte de la masse.

-   **Advection** : Calculée via un schéma **Upwind** du premier ordre pour garantir la positivité et la stabilité (sous condition CFL).
-   **Diffusion** : Calculée via un schéma centré classique aux interfaces, représentant la dispersion turbulente sous-maille.

Le pas de temps est contraint par la condition de Courant-Friedrichs-Lewy (CFL) pour garantir la stabilité numérique :
$$ \mathrm{CFL} = \frac{|\mathbf{v}| \cdot \Delta t}{\Delta x} < 1 $$
Dans les expériences présentées, nous utilisons $\mathrm{CFL} = 0.5$ comme compromis entre stabilité et efficacité computationnelle.

L'équation locale discrétisée pour une cellule $i$ s'écrit :
$$ \frac{dB*i}{dt} = R(B_i) - \sum*{j \in voisins} (F*{adv, i \to j} + F*{diff, i \to j}) $$

### 2.2. Algorithme d'Exécution

Le cycle de vie de la simulation est orchestré par le Contrôleur :

```algorithm
Algorithm: SEAPOPYM Simulation Lifecycle

1. BUILD PHASE
   G = CreateGraph(Nodes={Biomasse, T, Currents, Diffusion, ...})
   Order = TopologicalSort(G)

2. EXECUTION LOOP (t_start -> t_end)
   while t < t_end:
       // A. Flux Computation (Parallel)
       for node in Order:
           if node is Functional:
               // Compute biologic OR physical (Adv/Diff) flux independently
               Run(node)

       // B. Global Integration (Euler)
       // Sum all fluxes: Bio + Advection + Diffusion
       B(t+dt) = B(t) + dt * Sum(Fluxes)

       t = t + dt
```

### 2.3. Implémentation Logicielle et Stack Technologique

L'architecture est implémentée entièrement en Python, profitant de la richesse de son écosystème scientifique pour concilier lisibilité et performance :

-   **Calcul Numérique** : La majorité des opérateurs biologiques (ex: mortalité, croissance) reposent sur la vectorisation native de **`numpy`**, suffisante pour les opérations locales matricielles. L'accélération "Just-In-Time" via **`numba`** est réservée aux noyaux itératifs critiques ne pouvant être vectorisés efficacement, comme les schémas de transport flux-limiter.
-   **Orchestration Hybride et Parallélisme** : Le Contrôleur exploite la structure du DAG via le backend **`Dask`** selon une stratégie hybride :
    1.  **Task Parallelism** : Exécution simultanée des tâches hétérogènes indépendantes (ex: advection et mortalité) pour maximiser l'occupation CPU.
    2.  **Data Parallelism** : Pour les tâches coûteuses et homogènes (ex: transport de production), le calcul est divisé en sous-tâches indépendantes selon la dimension `cohort`. Ce "chunking" permet de paralléliser la boucle dominante du modèle, dépassant les limites théoriques (loi d'Amdahl) imposées par une approche purement basée sur les tâches.
    Le backend supporte deux modes : le **ThreadPoolScheduler** (mémoire partagée, utilisé ici) et le **Distributed Client** (clusters multi-nœuds).
