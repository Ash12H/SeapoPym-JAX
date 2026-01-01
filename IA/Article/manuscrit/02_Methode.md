> Notes (Jules):
> - Peut être qu'on pourrait recadrer le file du "matériel". En créant un lien direct entre DAG/Processus biologique/implémentation. Ainsi on structurerait autour d'idée (biologie, transport). A voir si c'est pertinant ou non. Sinon on doit pouvoir étoffer un tout petit peu plus la lien entre DAG et graphe d'execution pour mettre en valeur l'importance du dAG pour le calcul performant.
> - Je pense qu'on doit différencier plus clairement le système de modélisation sous forme de DAG, du modèle LMTL représenté sous la forme d'un Blueprint qu'on propose ici.

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

$$ \frac{dB}{dt} = \sum \Phi = \Phi_{bio}(S, F) + \Phi_{adv}(S, \mathbf{v}) + \Phi_{diff}(S, D) $$

Cette formulation unifie conceptuellement la physique et la biologie : pour le contrôleur, l'advection n'est qu'un processus de flux parmi d'autres, traité avec la même priorité que la mortalité.

## 2. Discrétisation et Mise en Œuvre

### 2.1. Volumes Finis : Advection et Diffusion

Le domaine spatial est discrétisé par la méthode des Volumes Finis, assurant la conservation stricte de la masse.
*   **Advection** : Calculée via un schéma **Upwind** du premier ordre pour garantir la positivité et la stabilité (sous condition CFL).
*   **Diffusion** : Calculée via un schéma centré classique aux interfaces, représentant la dispersion turbulente sous-maille.

Le pas de temps est contraint par la condition de Courant-Friedrichs-Lewy (CFL) pour garantir la stabilité numérique :
$$ \mathrm{CFL} = \frac{|\mathbf{v}| \cdot \Delta t}{\Delta x} < 1 $$
Dans les expériences présentées, nous utilisons $\mathrm{CFL} = 0.5$ comme compromis entre stabilité et efficacité computationnelle.

L'équation locale discrétisée pour une cellule $i$ s'écrit :
$$ \frac{dB_i}{dt} = R(B_i) - \sum_{j \in voisins} (F_{adv, i \to j} + F_{diff, i \to j}) $$

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

*   **Calcul Numérique** : La majorité des opérateurs biologiques (ex: mortalité, croissance) reposent sur la vectorisation native de **`numpy`**, suffisante pour les opérations locales matricielles. L'accélération "Just-In-Time" via **`numba`** est réservée aux noyaux itératifs critiques ne pouvant être vectorisés efficacement, comme les schémas de transport flux-limiter.
*   **Orchestration Flexible et Parallélisme de Tâches** : Le Contrôleur propose plusieurs backends. Outre un exécuteur séquentiel, le backend **`Dask`** exploite la structure du DAG pour paralléliser les tâches indépendantes (Task Parallelism), par exemple en calculant simultanément l'advection et la mortalité. De plus, il supporte le traitement par tuiles ("Data Chunking"), permettant de simuler des domaines dépassant la mémoire vive disponible. Le backend Dask supporte deux modes d'exécution : le **ThreadPoolScheduler** pour l'exécution parallèle sur une machine unique (exploitant la mémoire partagée), et le **Distributed Client** pour les clusters multi-nœuds. Les résultats présentés dans cet article utilisent principalement le ThreadPoolScheduler, adapté aux workstations scientifiques.
