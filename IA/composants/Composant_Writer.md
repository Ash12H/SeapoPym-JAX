Composant : Writer

C'est la dernière pièce du puzzle, et elle est cruciale pour la performance globale. Si ton écriture est mal gérée, ta simulation ultra-rapide sur Ray va passer 50% de son temps à attendre le disque dur.
Voici la fiche technique du composant Async Writer.

Composant : Async Writer (The Archivist)

1. Rôle et Responsabilités

L'Async Writer est responsable de la persistance des données. Il agit comme un Consommateur de Données qui travaille en parallèle du moteur de calcul.
Ses missions sont :

-   Sauvegarde Non-Bloquante : Écrire les résultats sur le disque sans jamais arrêter ou ralentir la boucle de simulation principale ("Fire and Forget").
-   Gestion des Formats : Traduire les objets xarray en fichiers standards (NetCDF, Zarr).
-   Gestion des Diagnostics : Sauvegarder non seulement l'État (biomasse), mais aussi les flux intermédiaires (productivité primaire, taux de mortalité) calculés par les workers.
-   Gestion de la Mémoire (Release) : Signaler au Controller quand une donnée a été écrite en sécurité, pour que l'ancien État puisse être supprimé de la RAM (Garbage Collection).

2. Architecture Interne (Pattern Producteur-Consommateur)

L'Async Writer est un Acteur Ray dédié (souvent placé sur le nœud qui a le meilleur accès disque).

1. La File d'Attente : Il possède une queue interne de requêtes de sauvegarde.
2. Le Thread d'Écriture : Il dépile les requêtes et effectue les I/O lourds.
3. Le Cache de Métadonnées : Pour ne pas réécrire les coordonnées (lat, lon, depth) à chaque pas de temps, il initialise le fichier une seule fois au début, puis il fait des "Appends" (ajouts) temporels.

4. Stratégie de Stockage (NetCDF vs Zarr)

Le choix du format impacte l'architecture de ce composant :

-   Zarr (Recommandé pour le Cloud/Distribué) :
    -   Avantage : Permet d'écrire des "Chunks" de données en parallèle. Très robuste pour les gros volumes.
    -   Méthode : Le Writer écrit chaque pas de temps dans un sous-dossier Zarr distinct ou append dans le store principal.
-   NetCDF (Standard Historique) :
    -   Avantage : Fichier unique facile à partager.
    -   Inconvénient : L'opération d'écriture est séquentielle (lock sur le fichier). Le Writer doit être sérialisé.

4. Fonctionnement (Le Flow)

Voici ce qui se passe à la fin du pas de temps $t$ dans le Controller :

1. Le Trigger : Le Controller a fini le calcul de $t+1$. Il détient encore la référence de $t$ (current_state_ref) et les références des diagnostics (diag_refs).
2. L'Envoi : Le Controller appelle writer.save.remote(current_state_ref, diag_refs, t).
    - Important : Cet appel rend la main immédiatement. Le Controller continue vers $t+2$.
3. L'Archivage (En arrière-plan) :
    - L'Acteur Writer reçoit la commande.
    - Il récupère les données depuis le Ray Object Store (Zero-Copy si sur le même nœud).
    - Il écrit sur le disque.
    - Il "relâche" la référence.
4. Nettoyage Ray : Une fois que le Writer a fini et relâché la référence, et que le Controller est passé à $t+2$, le compteur de références de l'objet $t$ tombe à zéro. Ray libère automatiquement la RAM.

5. API (Interface Ray)

Python

import ray
import xarray as xr

@ray.remote
class AsyncWriter:
def **init**(self, filepath: str, format='zarr'):
self.filepath = filepath
self.format = format
self.is_initialized = False

    def initialize_store(self, template_state: xr.Dataset):
        """
        Crée le squelette du fichier (coordonnées, variables vides)
        pour préparer les écritures futures.
        """
        if self.format == 'zarr':
            template_state.to_zarr(self.filepath, compute=False, mode='w')
        self.is_initialized = True

    def save_step(self, state_ref, diagnostics_refs: list, timestamp):
        """
        Méthode principale appelée à chaque pas de temps.
        """
        # 1. Récupération des données (Bloquant pour le thread Writer, pas pour la Simu)
        state_ds = ray.get(state_ref)

        # Récupération et fusion des diagnostics (qui sont des dicts de DataArrays)
        diags_list = ray.get(diagnostics_refs)
        full_ds = state_ds.copy()
        for diag in diags_list:
            full_ds.update(diag) # On ajoute les variables temporaires au dataset à sauver

        # 2. Assignation du temps
        full_ds = full_ds.expand_dims(time=[timestamp])

        # 3. Écriture (Append)
        if self.format == 'zarr':
            full_ds.to_zarr(self.filepath, mode='a', append_dim='time')

        # 4. Retour (Signal de fin)
        return True

6. Interaction avec le Blueprint

Le Writer a besoin de savoir Quoi sauvegarder.
Le Blueprint fournit une "Whitelist" de variables (via la config utilisateur).

-   Exemple : "Je veux sauvegarder biomasse et productivité, mais je me fiche de taux_mortalité_intermédiaire".
-   Le Writer utilise cette liste pour filtrer le Dataset avant l'écriture, économisant ainsi de l'espace disque.

Conclusion de l'Architecture

Nous avons fait le tour complet ! Voici ton système :

1. Blueprint : L'Architecte (Graphe, Câblage).
2. Forcing Manager : L'Entrée (Météo).
3. GSM : Le Stockage (State).
4. Functional Groups : Les Travailleurs (Calcul Bio/Physique).
5. Solver : Le Mathématicien (Intégration Temporelle).
6. Controller : Le Chef d'Orchestre (Boucle, Coordination).
7. Writer : La Sortie (Disque).
   C'est une architecture cohérente, moderne, scalable et scientifiquement rigoureuse. Tu as toutes les cartes en main pour coder le prototype.
   Quelle est la prochaine étape pour toi ? Veux-tu un résumé global ou un plan d'implémentation (ordre de codage) ?
