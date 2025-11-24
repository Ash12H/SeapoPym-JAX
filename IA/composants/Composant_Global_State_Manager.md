Composant : Global State Manager

C'est la "colonne vertébrale" de données. C'est le contrat de données que les Unités consomment et que le Solver produit.
Voici la description technique de ce composant vital.

Composant : Global State Manager (The Data Container)

1. Rôle et Responsabilités

Le Global State Manager n'est pas un processus actif (comme le Controller), c'est une abstraction structurelle. Il définit la forme exacte de la "Vérité Terrain" à un instant $t$.
Dans une architecture distribuée avec Ray, ce composant a des responsabilités critiques pour la cohérence :

-   Conteneur Unique : Il encapsule toutes les variables (biologiques, physiques, forçages) dans un seul objet xarray.Dataset cohérent.
-   Garant de la Grille : Il assure que toutes les variables partagent les mêmes coordonnées géographiques et temporelles (alignement des dimensions x, y, depth).
-   Immutabilité (Snapshot) : Il applique la règle d'or : "L'état au temps $t$ est en lecture seule". On ne modifie pas une case mémoire, on crée un nouvel état pour $t+1$.
-   Sérialisation : Il est optimisé pour être stocké et récupéré depuis le Ray Object Store (via Apache Arrow/Plasma) avec un coût de copie minimal (Zero-Copy).

2. Structure de Données Interne (xarray.Dataset)

L'objet manipulé a une structure hiérarchique standardisée.

-   Coordonnées (Dimensions) :
    -   Fixes : x (lon), y (lat), depth (profondeur).
    -   Variable : time (généralement une valeur scalaire pour un snapshot, ou non présent si implicite).
-   Data Variables (Les Données) :
    -   Variables Prognostiques (State) : Celles qui évoluent via une équation différentielle (ex: phyto_biomass, nitrate, temperature). Ce sont elles que le Solver met à jour.
    -   Forçages (Externes) : Données en lecture seule imposées par l'environnement (ex: sun_light, wind_stress). Elles sont injectées au début du pas de temps.
    -   Paramètres Spatiaux (Statiques) : Champs 2D/3D constants (ex: bathymetry, mask_land).

3. Fonctionnement Détaillé

A. Cycle de Vie : Le Pattern "Snapshot"

Contrairement à un programme classique où l'on modifie une variable a = a + 1, le GSM fonctionne par versions.

1. Instantiation ($t=0$) : Le Controller crée le premier Dataset à partir des conditions initiales et l'envoie dans le Ray Store $\rightarrow$ Ref_ID_0.
2. Lecture ($t$) : Les Workers (Unités) demandent Ref_ID_0. Ray leur donne un accès direct à la mémoire partagée.
3. Recréation ($t+1$) : Le Solver reçoit les tendances, prend Ref_ID_0, calcule les nouvelles valeurs, et crée un nouveau Dataset $\rightarrow$ Ref_ID_1.

B. Gestion des Métadonnées (Attributs)

Le GSM utilise les métadonnées de Xarray pour transporter des infos cruciales sans alourdir l'API :

-   units : "mmol/m3" (pour vérification éventuelle).
-   long_name : Pour les fichiers de sortie NetCDF.
-   \_fill_value : Pour gérer les terres émergées (NaN).

4. API (Interface)

Bien que ce soit principalement une structure de données, une classe Helper est utile pour gérer les interactions avec Ray et le Disque.
Python

import xarray as xr
import ray

class StateManager:
"""
Wrapper autour des opérations de création et de gestion du State.
"""

    @staticmethod
    def create_initial_state(coords: dict, variables: dict) -> xr.Dataset:
        """Crée le Dataset vide/initial propre et aligné."""
        ds = xr.Dataset(coords=coords)
        for name, data in variables.items():
            ds[name] = data
        return ds

    @staticmethod
    def validate(ds: xr.Dataset, required_vars: list):
        """Vérifie que le state contient tout ce que le Blueprint exige."""
        missing = set(required_vars) - set(ds.data_vars)
        if missing:
            raise ValueError(f"State corrompu. Manque : {missing}")

    @staticmethod
    def update_forcings(state_ref, forcing_data: dict):
        """
        Appelé par le Controller au début du step.
        Crée une vue légère ou une copie avec les nouveaux forçages.
        """
        # Note : Avec Ray, on préfère souvent recréer un Dataset léger
        # ou utiliser des objets séparés pour les forçages si trop lourds.
        pass

5. Interaction avec le Blueprint

C'est un point clé : Le Blueprint définit le Schéma, le GSM instancie la Donnée.

-   Le Blueprint, après compilation, produit une liste : expected_variables = ['nitrate', 'phyto', 'zoo'].
-   Au démarrage, le Controller utilise cette liste pour valider via le StateManager que le fichier de conditions initiales fourni par l'utilisateur correspond bien au modèle attendu.

Résumé pour le développeur

Le Global State Manager n'est rien d'autre qu'un xarray.Dataset strict et standardisé, vivant dans la mémoire partagée.
Sa caractéristique principale est d'être Immuable durant un pas de temps de calcul. C'est cette immutabilité qui permet à Ray de paralléliser massivement la lecture (100 workers peuvent lire le même objet sans verrou/lock).
Est-ce que cette définition de la "matière première" du système te convient ?
