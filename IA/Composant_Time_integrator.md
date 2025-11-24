Composant : Time integrator

Le composant logique suivant est le Time Integrator (aussi appelé Solver).
C'est la pièce maîtresse mathématique qui "ferme la boucle".
Nous avons défini qui prépare le plan (Blueprint), qui contient les données (GSM), qui calcule les variations (Workers). Il manque maintenant celui qui applique ces variations pour créer le futur.
Voici sa fiche technique.

Composant : Time Integrator (The Solver)

1. Rôle et Responsabilités

Le Time Integrator est le mathématicien du système. Il est responsable de l'évolution temporelle de la simulation.
Il intervient à la fin de la phase de calcul, une fois que tous les workers ont rendu leurs copies.
Ses missions sont :

-   L'Agrégation : Collecter et sommer les tendances (dérivées partielles) provenant de toutes les sources (Biologie, Physique, Transport).
-   L'Intégration Numérique : Calculer l'état $t+1$ en appliquant un schéma numérique (Euler, Runge-Kutta, etc.) sur l'état $t$.
-   La Garantie Physique (Sanity Check) : Appliquer des contraintes inviolables (ex: "Clipping" des valeurs négatives si la biomasse < 0).
-   La Création d'État : C'est la seule entité autorisée à instancier le nouveau xarray.Dataset pour le pas de temps suivant.

2. Inputs et Outputs

-   Entrées :
    -   current_state_ref : Référence vers l'état immuable au temps $t$.
    -   trends_refs_list : Liste des références vers les dictionnaires de tendances produits par les Groupes Fonctionnels et le Transport.
    -   dt : Le pas de temps (delta t) en secondes/jours.
-   Sortie :
    -   next_state_ref : Référence vers le nouvel état immuable au temps $t+1$ (dans le Ray Object Store).

3. Fonctionnement Interne

Le Solver est conçu comme une Tâche Distribuée (Remote Function) plutôt qu'un Acteur persistant, car il n'a pas besoin de mémoire entre deux pas de temps (sauf pour des schémas complexes comme Adam-Bashforth, mais restons simples pour l'instant).

L'Algorithme d'Agrégation (The Reducer)

C'est l'étape délicate où l'on fusionne les résultats.

1. Récupération : Le Solver télécharge (ou accède via mémoire partagée) les DataArrays de tendances.
2. Alignement : Il vérifie que toutes les tendances visent des variables existantes.
3. Sommation :$$\frac{dX}{dt}_{total} = \sum_{source} \frac{dX}{dt}_{source}$$Exemple : Tendance_Phyto_Total = Tendance_Croissance (Bio) + Tendance_Advection (Transport) - Tendance_Broutage (Zoo)

L'Algorithme d'Intégration (Schémas)

Le Solver peut être configuré avec différentes stratégies :

-   Euler Explicite (Le plus simple) :$$State_{t+1} = State_t + (\text{Tendances}_{Total} \times dt)$$
-   Runge-Kutta 4 (Le standard de précision) :Note : Pour faire du RK4, le Solver doit orchestrer des sous-pas de temps. Cela demande une interaction plus complexe avec le Controller (qui doit relancer les workers pour estimer les pentes intermédiaires $k1, k2, k3, k4$).Dans une première version, on se concentre souvent sur Euler ou Euler-Forward.

Post-Traitement (Physique)

Une fois le calcul mathématique fait, le Solver nettoie les résultats :

-   Positivity constraint : Si une variable est marquée comme "positive only" (ex: concentration), toute valeur $< 0$ (due à des erreurs numériques) est forcée à $0$.

4. API (Interface Ray)

Python

import ray
import xarray as xr

@ray.remote
def solve_step(current_state: xr.Dataset, trends_list: list, dt: float) -> xr.Dataset:
"""
Fonction stateless exécutée sur un noeud puissant (car manipulation de gros objets).
"""

    # 1. Copie de structure (On ne modifie pas l'ancien state)
    # On crée un nouveau dataset vide avec les mêmes coords
    next_state = current_state.copy(deep=False)

    # Dictionnaire temporaire pour accumuler les dérivées globales
    total_derivatives = {}

    # 2. Agrégation des Tendances
    # trends_list est une liste de dicts : [{'phyto': array_A}, {'phyto': array_B, 'zoo': array_C}]
    for worker_output in trends_list:
        for var_name, tendency_array in worker_output.items():

            if var_name not in total_derivatives:
                total_derivatives[var_name] = tendency_array
            else:
                # Xarray gère l'addition élément par élément + alignement
                total_derivatives[var_name] = total_derivatives[var_name] + tendency_array

    # 3. Application de l'équation (Euler simple ici)
    for var_name, dX_dt in total_derivatives.items():
        if var_name in next_state:
            # S_next = S_prev + dX/dt * dt
            next_state[var_name] = current_state[var_name] + (dX_dt * dt)

    # 4. Garde-fous (Exemple simple)
    # Pour toutes les variables, on clip à 0 si elles deviennent négatives
    # (En réalité, on le ferait seulement sur les variables de concentration)
    next_state = next_state.where(next_state >= 0, 0)

    # 5. Mise à jour du temps
    if 'time' in next_state.coords:
         # Logique d'incrément temporel
         pass

    return next_state

5. Interaction avec le Blueprint

Le Solver a besoin de savoir quelles variables sont des "State Variables" (à mettre à jour) et lesquelles sont des paramètres statiques (à ignorer).
Ces métadonnées sont implicitement portées par le Dataset (variables de données vs coordonnées), mais le Blueprint peut fournir une liste de "Variables Positives" pour configurer le garde-fou.

Résumé pour le développeur

Le Time Integrator est le comptable. Il prend le bilan d'ouverture ($t$), ajoute tous les mouvements de crédits/débits calculés par les départements (Workers), vérifie qu'aucun compte n'est à découvert (Positivité), et publie le bilan de clôture ($t+1$).
Il est critique qu'il soit efficace car il manipule des objets lourds (tout le state). Avec Ray, on essaie de l'exécuter sur le nœud qui a le plus de RAM ou là où se trouvent la majorité des données pour minimiser les transferts.
