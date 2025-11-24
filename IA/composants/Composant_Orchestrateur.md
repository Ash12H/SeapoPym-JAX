Composant : Orchestrateur

Le Blueprint est le "Compilateur" (qui crée le plan), le Controller est le "Runtime" (qui exécute le plan). C'est lui qui prend l'objet ExecutionPlan généré par le Blueprint et qui lui donne vie au fil du temps.
Voici la description technique de ce composant central.

Composant : Simulation Controller (The Engine)

1. Rôle et Responsabilités

Le Controller est le Chef d'Orchestre de la simulation. C'est un processus unique (le script principal Python) qui pilote toute l'infrastructure. Il ne touche jamais directement aux données lourdes (tableaux numpy), il ne manipule que des références (pointeurs Ray).
Ses missions sont :

-   L'Initialisation : Démarrer le cluster Ray, allouer les Acteurs (Workers) et placer l'État Initial ($t=0$) dans le Store.
-   La Boucle Temporelle : Gérer l'avancement du temps ($t$, $t+dt$, ...).
-   Le Dispatching : Distribuer les tâches définies dans l'ExecutionPlan aux workers disponibles.
-   La Synchronisation : Savoir quand attendre (Barrière) avant de passer à l'étape suivante (ex: attendre que la Bio soit finie avant de lancer le Transport).
-   Le Déclenchement IO : Ordonner au Writer de sauvegarder les données au bon moment.

2. Inputs

Pour démarrer, le Controller a besoin de deux choses :

1. ExecutionPlan (venant du Blueprint) : La liste des tâches, le graphe des dépendances et la carte des scopes (Local vs Global).
2. InitialState : Le xarray.Dataset de départ.

3. Structure Interne & Logique

Le Controller maintient l'état courant de la simulation sous forme de références.

-   current_state_ref : L'ID Ray de l'objet Dataset au temps $t$.
-   time_step : L'index temporel actuel.

La "Master Loop" (Algorithme Principal)

Voici ce que fait le Controller à chaque itération de la boucle while t < t_end :

1. Préparation des Forçages :
    - Il appelle le ForcingManager pour mettre à jour current_state_ref avec les données météo du jour.
2. Phase de Calcul (Dispatcher) :
    - Il parcourt la liste des tâches de l'ExecutionPlan.
    - Pour une unité LOCALE (Bio) :
        - Il ne bloque pas. Il envoie des requêtes asynchrones aux Workers Ray disponibles.
        - Il passe current_state_ref en argument.
    - Pour une unité GLOBALE (Transport) :
        - Il envoie une requête à l'Acteur dédié au transport avec current_state_ref.
3. Phase de Collecte (Gathering) :
    - Il récupère une liste de FutureObjects (les promesses de résultats) correspondant aux Tendances retournées par les workers.
    - Note : À ce stade, le Controller a juste des "tickets de retrait", il n'a pas téléchargé les données.
4. Phase de Résolution (Solver) :
    - Il appelle le composant Time Integrator.
    - Il lui passe : current_state_ref + liste_des_tendances_refs.
    - L'Integrateur lui renvoie : next_state_ref (L'ID du nouvel état $t+1$).
5. Phase de Rotation & IO :

    - Il envoie current_state_ref à l'Async Writer pour sauvegarde.
    - Il met à jour son pointeur : current_state_ref = next_state_ref.
    - Il libère la mémoire de l'ancien état (si la sauvegarde est finie).

6. Interaction avec Ray (La subtilité)

Le Controller doit gérer intelligemment la distribution selon le Scope défini dans le Blueprint :

-   Gestion des Unités Locales (Map Pattern) :Si le Blueprint dit que photosynthesis est local, le Controller peut utiliser ray.data ou une logique de "Scatter/Gather" pour découper l'appel sur plusieurs workers si la grille est immense.
-   Gestion des Unités Globales :Il envoie l'objet entier à l'Acteur Transport unique.

5. API (Interface)

Python

class SimulationController:
def **init**(self, blueprint_plan, config):
self.plan = blueprint_plan
self.store = RayObjectStoreWrapper() # Abstraction du store
self.solver = TimeIntegrator()
self.writer = AsyncWriter()

    def initialize(self, initial_dataset):
        """Place le dataset t=0 dans le store."""
        self.current_state_ref = self.store.put(initial_dataset)
        self.t = 0

    def run(self, steps: int):
        """Lance la boucle principale."""
        for _ in range(steps):
            self._step()

    def _step(self):
        # 1. Collecte des tendances (Dérivées)
        trends_refs = []

        for task in self.plan.task_sequence:
            # La logique de routing définie par le Blueprint est appliquée ici
            # ex: input_mapping dit que task 'mortality' prend 'temp_metabo'
            args_refs = self._resolve_args(task, self.current_state_ref)

            # Appel asynchrone (remote)
            result_ref = task.remote_func.remote(*args_refs)
            trends_refs.append(result_ref)

        # 2. Résolution (Intégration temporelle)
        # Le solver crée le t+1 sans que les données ne repassent par le Controller
        self.next_state_ref = self.solver.integrate.remote(
            self.current_state_ref,
            trends_refs,
            dt=self.dt
        )

        # 3. IO & Update
        self.writer.save.remote(self.current_state_ref)
        self.current_state_ref = self.next_state_ref

En résumé

Le Simulation Controller est la seule partie du code qui "sait quelle heure il est". Il est aveugle aux détails scientifiques (il ne sait pas ce qu'est le phytoplancton), mais il est expert en logistique : il s'assure que les données circulent du Stockage vers les Ouvriers, puis vers le Solveur, dans l'ordre strict dicté par le Blueprint.
Est-ce que cette séparation entre le Plan (Blueprint) et l'Exécution (Controller) te semble claire ?
