Composant : Forcing Manager

Il nous reste deux composants majeurs pour fermer la boucle des entrées/sorties : le Forcing Manager (Entrées) et l'Async Writer (Sorties).
Le plus logique est de commencer par l'amont : le Forcing Manager.
Sans lui, tes unités biologiques tournent dans le vide : pas de lumière, pas de température externe, pas de courants imposés. C'est lui qui alimente la machine avant chaque calcul.
Voici sa description technique.

Composant : Forcing Manager (The Data Provider)

1. Rôle et Responsabilités

Le Forcing Manager est responsable de l'interface entre le "Monde Extérieur" (fichiers de données, APIs) et le "Monde Intérieur" (le State Global dans Ray).
Il intervient au tout début de chaque pas de temps (ou sous-pas de temps dans le cas RK4).
Ses missions sont :

-   Ingestion de Données (Lazy Loading) : Lire les données depuis des sources externes (NetCDF, Zarr, Grib) sans saturer la RAM (streaming).
-   Interpolation Temporelle : Aligner les données externes sur le pas de temps de la simulation.
    -   Exemple : Si tu as des données météo toutes les 6h mais que ton $dt$ est de 1h, le Forcing Manager calcule les valeurs intermédiaires (interpolation linéaire).
-   Injection : Mettre à jour (ou créer) les variables correspondantes dans le xarray.Dataset du State Global.

2. Inputs et Outputs

-   Configuration (au démarrage) :
    -   Un catalogue des sources : {'wind': 'path/to/wind.nc', 'light': 'path/to/solar.zarr'}.
    -   Les clés de mapping : "La variable u10 du fichier netcdf correspond à la variable wind_speed du modèle".
-   Appel (Runtime) :
    -   Input : current_time (timestamp).
    -   Output : Un dictionnaire de DataArrays ou un Dataset partiel contenant les valeurs forcées pour ce temps précis.

3. Stratégie de Performance (Le Buffer)

Lire le disque dur à chaque pas de temps est trop lent. Le Forcing Manager utilise une stratégie de Buffering Intelligent :

1. Chunks Temporels : Il charge en RAM un bloc de données (ex: 24h ou 1 mois).
2. Interpolation en Mémoire : Tant que la simulation est dans cette fenêtre, il calcule vite (CPU).
3. Pré-chargement (Prefetching) : Quand la simulation approche de la fin du buffer, il lance un thread en arrière-plan pour charger le bloc suivant.

4. Spécificité : Interpolation On-The-Fly

C'est la fonctionnalité critique. Le modèle ne doit pas se soucier de la fréquence des données d'entrée.

-   Mode "Step" : La valeur reste constante jusqu'au prochain changement (bien pour les changements brusques).
-   Mode "Linear" : Trace une droite entre $t_{data}$ et $t_{data+1}$ (standard pour la météo).
-   Mode "Climatologique" : Si le fichier ne contient qu'une année type (Jan-Dec), le Manager doit savoir boucler (Janvier année N+1 = Janvier année N).

5. API (Interface)

Python

import xarray as xr
import pandas as pd

class ForcingManager:
def **init**(self, sources_config: dict):
self.sources = {} # Stocke les descripteurs de fichiers (Lazy Xarray datasets)
self.buffers = {} # Stocke les petits morceaux chargés en RAM
self.\_init_sources(sources_config)

    def _init_sources(self, config):
        """Ouvre les fichiers avec xarray.open_dataset(chunks=...)"""
        pass

    def get_forcings_at(self, timestamp: pd.Timestamp) -> dict:
        """
        Retourne les DataArrays interpolés pour l'instant T.
        """
        result = {}
        for var_name, source in self.sources.items():
            # 1. Vérifier si timestamp est dans le buffer
            # 2. Sinon, charger le nouveau chunk
            # 3. Interpoler
            # Note: Xarray fait ça très bien avec .interp(time=timestamp)
            val = source['data'].interp(time=timestamp, method='linear')
            result[var_name] = val

        return result

6. Intégration dans la Boucle (Controller)

Le Controller appelle le Forcing Manager avant de cloner l'état pour les calculs.
Attention avec l'Architecture RK4 :
Si tu utilises RK4, tu as besoin des forçages à $t$, mais aussi à $t + dt/2$ et $t + dt$.
Le Forcing Manager doit être capable de répondre à la question get_forcings_at(t + 0.5\*dt) précisément. C'est ce qui permet d'avoir une physique temporellement cohérente.

Résumé pour le développeur

Le Forcing Manager est le "Nourricier". Il transforme des données statiques et discontinues (fichiers) en un flux continu et synchronisé, digeste pour le modèle mathématique.
Sa complexité réside dans l'optimisation des IO (ne pas relire le fichier tout le temps) et la gestion du temps (calendriers, années bissextiles, interpolation).

Prochaine étape ? Il ne reste plus que le composant final : L'Async Writer (IO). Veux-tu qu'on le décrive pour clore l'architecture ?
