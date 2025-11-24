Voici un résumé des 7 composants clés que j'ai identifiés :

Blueprint (L'Architecte) :

C'est le "compilateur" du modèle. Il transforme des fonctions Python indépendantes en un graphe de dépendances cohérent (DAG).
Il gère le "câblage" des données (mapping des entrées/sorties) et définit le plan d'exécution pour le Controller.
Controller / Orchestrateur (Le Chef d'Orchestre) :

C'est le moteur d'exécution (Runtime). Il pilote la boucle temporelle ($t \rightarrow t+1$).
Il distribue les tâches aux workers, synchronise les étapes (Calcul -> Résolution -> IO) et gère le cycle de vie de la simulation.
Global State Manager (Le Conteneur de Données) :

Il définit la structure de la "Vérité Terrain" : un xarray.Dataset immuable contenant toutes les variables à l'instant $t$.
Il garantit la cohérence spatio-temporelle et facilite le partage de mémoire (Zero-Copy) via Ray.
Functional Group (Les Ouvriers) :

Ce sont les unités de calcul (Acteurs Ray) qui hébergent la logique scientifique (ex: Biologie, Transport).
Ils prennent l'état en lecture seule et retournent des tendances (dérivées) ou des diagnostics, sans jamais modifier l'état directement.
Forcing Manager (L'Approvisionneur) :

Il gère les données externes (météo, courants, etc.).
Il charge, interpole (temporellement) et injecte ces forçages dans le système au début de chaque pas de temps.
Time Integrator / Solver (Le Mathématicien) :

Il récupère toutes les tendances produites par les Functional Groups.
Il calcule l'état au temps $t+1$ en appliquant un schéma numérique (Euler, RK4, etc.) et crée le nouveau Dataset immuable.
Writer (L'Archiviste) :

Il gère la sauvegarde des résultats sur le disque de manière asynchrone (pour ne pas ralentir le calcul).
Il supporte des formats comme NetCDF ou Zarr.
En synthèse : C'est une architecture Flux de Données (Dataflow) où l'état est immuable. Le Controller orchestre le mouvement des données entre le Stockage (GSM), les Calculs (Functional Groups), le Solveur (Time Integrator) et le Disque (Writer), selon le plan établi par le Blueprint.
