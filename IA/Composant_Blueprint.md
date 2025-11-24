Composant : Blueprint

C'est la pièce manquante qui rend ton système flexible et réellement utilisable pour de la modélisation complexe.
Voici la description mise à jour et définitive du composant Blueprint. Elle intègre désormais la logique de résolution de dépendances à 3 niveaux (Mapping Explicite > Namespacing > Nom par défaut).

Composant : The Blueprint (Graph Builder)

1. Rôle et Responsabilités

Le Blueprint est l'Architecte et Compilateur du modèle. Il permet de passer d'une collection de fonctions Python indépendantes à un scénario de simulation cohérent.
Sa responsabilité critique est le Câblage (Wiring). Il doit décider, pour chaque argument de chaque fonction, quelle donnée du système doit être injectée. Il offre pour cela une flexibilité totale entre l'automatisme (introspection) et le contrôle manuel (mapping).
Ses missions :

-   L'Enregistrement Hybride : Intégrer les unités avec leurs métadonnées (scope, paramètres).
-   La Résolution de Dépendances : Connecter les entrées/sorties en priorisant les directives explicites de l'utilisateur sur les noms de variables par défaut.
-   L'Analyse de Portée (Scoping) : Identifier les besoins spatiaux (Local vs Global/Stencil).
-   La Validation Topologique : Garantir l'intégrité du DAG final (pas de cycles).

2. Structure de Données Interne

Le Blueprint maintient un graphe networkx.DiGraph enrichi.

-   Nœuds (Nodes) :
    -   DataNode : Une variable nommée (ex: temperature, phyto.growth_rate).
    -   ComputeNode : Une tâche de calcul (XarrayUnit). Attributs clés :
        -   func : Pointeur vers la fonction.
        -   scope : 'local' ou 'global'.
        -   input_map : Dictionnaire de surcharge des entrées (ex: {'temp': 'metabolic_temp'}).
-   Arêtes (Edges) : Flux de données dirigés.

3. Logique de Résolution (Le Cœur du Système)

C'est ici que se gère ton cas d'usage (Température vs Température Métabolique). Lorsqu'une unité est ajoutée, le Blueprint détermine la source de chaque argument selon cet ordre de priorité strict :

1. Mapping Explicite (Surcharge) :
    - Question : L'utilisateur a-t-il dit : "Pour l'argument A, utilise la variable B" ?
    - Action : Si oui, on crée l'arête B -> Unit. On ignore le nom original A.
2. Namespacing (Contexte du Groupe) :
    - Question : Existe-t-il une variable préfixée par le groupe (ex: PhytoA.A) ?
    - Action : Si oui, on connecte PhytoA.A -> Unit.
3. Matching par Défaut (Global) :
   _ Question : Existe-t-il une variable globale nommée exactement A ?
   _ Action : Si oui, on connecte A -> Unit.
   Si aucune source n'est trouvée après ces 3 étapes : Erreur de compilation MissingInputError.

4. API (Interface Programmatique)

Python

class Blueprint:
def **init**(self):
self.graph = nx.DiGraph()

    def register_forcing(self, name: str, dims: tuple):
        """Déclare une source de données externe."""
        pass

    def register_unit(self, func, input_mapping: dict = None, output_name: str = None, scope: str = 'local'):
        """
        Enregistre une unité de calcul générique.

        Args:
            func: La fonction Python (XarrayUnit).
            input_mapping: Dict { 'nom_arg_fonction': 'nom_variable_dans_le_graphe' }.
                           Permet de rediriger 'temperature' vers 'temp_metabolique'.
            output_name: Nom personnalisé de la variable produite.
            scope: 'local' (element-wise) ou 'global' (transport/stencil).
        """
        # Logique de résolution appliquée ici
        pass

    def register_group(self, group_config):
        """
        Helper qui appelle register_unit pour toutes les fonctions d'un groupe,
        en appliquant automatiquement les préfixes de noms (Namespacing).
        """
        pass

    def build(self) -> ExecutionPlan:
        """Compile, valide les cycles, et retourne le plan d'exécution."""
        pass

5. Output : L'objet ExecutionPlan

Le résultat est une structure optimisée prête pour l'exécution :

-   task_graph : Liste ordonnée des tâches (NodeIDs).
-   data_routing : Table de correspondance pour l'Executor.
    -   Exemple pour ton cas : "Quand tu exécutes mortalité, injecte le DataArray temp_metabolique dans l'argument temperature".
-   scope_map : Instructions pour le découpage des données (Chunking vs Full Grid).

Avec cette définition, le Blueprint est complet. Il gère à la fois la rigueur de la structure (DAG) et la souplesse nécessaire à la modélisation scientifique (redirection de variables).
