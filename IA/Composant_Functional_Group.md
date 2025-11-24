Composant : Functional Group

Voici la fiche technique mise à jour du composant Functional Group.
Elle intègre désormais le fonctionnement "Tout Xarray" et la logique de Ports Nommés (Dictionnaire) pour gérer les entrées/sorties multiples et le câblage flexible.

Composant : Functional Group (The Compute Worker)

1. Rôle et Responsabilités

Le Functional Group est l'unité de travail distribuée du système (un Acteur Ray). Il incarne une entité scientifique (ex: "Phytoplancton", "Chimie des Carbonates", "Transport Hydro").
Il agit comme une "Boîte Noire" intelligente : il reçoit un état global en lecture seule, exécute une série de transformations mathématiques, et retourne des variations.
Ses missions sont :

-   Hébergement de la Logique : Il contient le pipeline d'exécution des XarrayUnits (fonctions pures).
-   Gestion des Paramètres : Il stocke la configuration statique (constantes $K_m$, taux, seuils) injectée au démarrage.
-   Abstraction des Données (Mapping) : Il traduit les noms des variables du Graphe Global vers les noms d'arguments attendus par les fonctions Python (et vice-versa).
-   Production de Tendances : Il retourne les dérivées ($dX/dt$) au système, sans jamais modifier l'état lui-même.

2. Structure Interne

L'Acteur maintient deux types d'objets :

-   Configuration (Stateful - Immuable) :
    -   params : Un dictionnaire/objet contenant les constantes physiques/biologiques spécifiques à cette instance (ex: {'max_growth_rate': 1.2}).
-   Pipeline d'Exécution (La Recette) :
    -   Une liste ordonnée de tâches (TaskDefinition). Chaque tâche contient :
        -   La référence vers la fonction XarrayUnit.
        -   input_mapping : Dictionnaire de routage Entrée (Nom Graphe $\rightarrow$ Argument Fonction).
        -   output_mapping : Dictionnaire de routage Sortie (Clé Retour $\rightarrow$ Nom Graphe).

3. Fonctionnement (Le Runtime)

À chaque pas de temps, le Controller appelle la méthode compute(state_ref). Voici le cycle interne :

1. Acquisition (Zero-Copy) :
    - L'Acteur résout state_ref pour accéder au xarray.Dataset global dans la mémoire partagée.
    - Il initialise un Contexte Local (mémoire tampon) pour stocker les résultats intermédiaires entre ses propres unités.
2. Exécution du Pipeline :Pour chaque unité de la liste :
    - A. Résolution des Entrées (Input Mapping) :Pour chaque argument de la fonction, l'Acteur regarde le input_mapping.
        - Il cherche la variable d'abord dans le Contexte Local (résultat précédent).
        - Sinon, il la cherche dans le State Global (Forçage ou État $t$).
        - Il extrait le DataArray correspondant (Slice ou Full selon le scope).
    - B. Calcul (Xarray pur) :Il exécute la fonction avec les données et injecte les paramètres statiques (\*\*self.params).
        - Attendu : La fonction retourne un Dictionnaire de DataArrays : {'key_A': data, 'key_B': data}.
    - C. Routage des Sorties (Output Mapping) :Il itère sur le dictionnaire retourné. Pour chaque clé interne (ex: 'carbon_flux'), il regarde le output_mapping pour trouver le nom de destination dans le graphe (ex: 'phyto_growth').Il stocke le résultat dans le Contexte Local sous ce nom de destination.
3. Tri Final (Marshalling) :Une fois toutes les unités exécutées, l'Acteur sépare les données du Contexte Local en deux paquets de retour :

    - trends : Les variables destinées au Solver (tous les flux modifiant l'état).
    - diagnostics : Les variables intermédiaires à sauvegarder (si configuré).

4. Gestion des Entrées/Sorties Multiples (Ports Nommés)

C'est la fonctionnalité clé pour la flexibilité.

-   Le Problème : La fonction photosynthesis a été codée pour retourner {'O2': ...} mais dans ce modèle précis, on veut que cela remplisse la variable Oxygen_Level_Layer_1.
-   La Solution : L'Acteur ne "devine" rien. Il applique strictement le dictionnaire de traduction fourni par le Blueprint.
    -   Si la fonction retourne une clé qui n'est pas dans le mapping $\rightarrow$ Erreur ou Ignoré (selon config).
    -   Si la fonction ne retourne pas une clé attendue $\rightarrow$ Erreur MissingOutput.

5. API (Pseudo-Code Ray Actor)

Python

import ray

@ray.remote
class FunctionalGroup:
def **init**(self, name, pipeline_def, parameters):
self.name = name
self.pipeline = pipeline_def # Liste de dicts définissant les étapes et mappings
self.params = parameters # Config scientifique (Pydantic model ou dict)

    def compute(self, state_ref):
        # 1. Accès au State (Lecture seule)
        global_state = ray.get(state_ref)
        local_context = {} # Scratchpad pour ce pas de temps

        trends = {}
        diagnostics = {}

        # 2. Boucle d'exécution
        for task in self.pipeline:
            func = task['func']

            # --- Résolution des Inputs ---
            args = {}
            for arg_name, source_var_name in task['input_mapping'].items():
                if source_var_name in local_context:
                    args[arg_name] = local_context[source_var_name]
                else:
                    args[arg_name] = global_state[source_var_name]

            # --- Exécution ---
            # On passe les args dynamiques + les params statiques
            results_dict = func(**args, **self.params)

            # --- Routage des Outputs ---
            for internal_key, graph_var_name in task['output_mapping'].items():
                data = results_dict[internal_key]

                # Mise en cache locale pour l'unité suivante du pipeline
                local_context[graph_var_name] = data

                # Classification pour le retour
                if task['is_trend']:
                     trends[graph_var_name] = data
                elif task['is_diagnostic']:
                     diagnostics[graph_var_name] = data

        return trends, diagnostics

Résumé pour le développeur

Le Functional Group est un interpréteur de pipeline.
Il découple totalement le code Python de la fonction (qui utilise des noms génériques comme temp, growth) de la structure du modèle (qui utilise des noms spécifiques comme SST, Diatom_growth).
Il assure l'interface entre le monde statique des paramètres et le monde dynamique des données Xarray.
