# Todo List

## Phase 1 : Refonte du Système de Forçages (Fondations) - ✅ COMPLÉTÉ
- [x] Créer le fichier `src/seapopym_message/forcing/source.py`.
- [x] Implémenter la classe `ForcingSource` :
    - [x] Constructeur acceptant un `xarray.DataArray`.
    - [x] Validation des données (vérification des dimensions minimales : temps, lat, lon).
    - [x] Gestion des métadonnées (unités, méthode d'interpolation).
- [x] Exposer la classe dans `src/seapopym_message/forcing/__init__.py`.
- [x] Modifier `src/seapopym_message/forcing/manager.py` :
    - [x] Mettre à jour `__init__` pour accepter un dictionnaire de `ForcingSource`.
    - [x] Mettre à jour `_interpolate_time` pour qu'elle soit agnostique aux dimensions spatiales/profondeur.
    - [x] Mettre à jour `prepare_timestep` pour retourner des tableaux JAX N-dimensionnels sans erreur.
- [x] Créer un test unitaire `tests/unit/forcing/test_source.py` pour `ForcingSource`.
- [x] Créer un test unitaire `tests/unit/forcing/test_manager_ndim.py` pour vérifier que le `ForcingManager` gère bien un champ 3D.

## Phase 2 : Architecture des Groupes Fonctionnels - ✅ COMPLÉTÉ
- [x] Créer le fichier `src/seapopym_message/core/group.py`.
- [x] Implémenter la classe `FunctionalGroup` (dataclass) :
    - [x] Attributs : `name`, `units`, `variable_map`, `params`.
- [x] Modifier `src/seapopym_message/core/unit.py` :
    - [x] Ajouter le support pour le binding dynamique (séparation nom de variable / nom dans le state).
- [x] Créer le module `src/seapopym_message/kernels/sensing.py` pour les Unités de Perception.
- [x] Implémenter `ExtractLayerUnit` (extraction d'une couche Z).
- [x] Implémenter `DielMigrationUnit` (moyenne pondérée jour/nuit).
- [x] Tests unitaires pour `FunctionalGroup` et les nouvelles unités.

## Phase 3 : Le "Kernel Compiler" - ✅ COMPLÉTÉ
- [x] Modifier le constructeur de `Kernel` dans `src/seapopym_message/core/kernel.py`.
    - [x] Accepter `list[FunctionalGroup]` en plus de `list[Unit]`.
    - [x] Implémenter la logique d'aplatissement : itérer sur les groupes, binder les unités, et créer une liste plate d'unités liées.
- [x] Ajouter la validation statique :
    - [x] Vérifier les dépendances sur le graphe global (déjà géré par `_check_dependencies` sur la liste plate).
- [x] Ajouter la visualisation du graphe (`visualize_graph`).
- [x] Mettre à jour les tests du Kernel.

## Phase 4 : Migration du Modèle Zooplancton
- [ ] Créer `src/seapopym_message/model/zooplankton.py` (ou modifier l'existant).
- [ ] Définir les `FunctionalGroup` pour le zooplancton (ex: `epipelagic`, `mesopelagic`, `migrant`).
- [ ] Configurer les `variable_map` pour chaque groupe.
- [ ] Créer un script de test/démonstration pour valider l'exécution multi-groupes.
