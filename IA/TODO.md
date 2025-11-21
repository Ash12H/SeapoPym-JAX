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

## Phase 2 : Architecture des Groupes Fonctionnels
- [ ] Créer le fichier `src/seapopym_message/core/group.py`.
- [ ] Implémenter la classe `FunctionalGroup` (dataclass) :
    - [ ] Attributs : `name`, `units`, `variable_map`, `params`.
- [ ] Modifier `src/seapopym_message/core/unit.py` :
    - [ ] Ajouter le support pour le binding dynamique (séparation nom de variable / nom dans le state).
- [ ] Créer le module `src/seapopym_message/kernels/sensing.py` pour les Unités de Perception.
- [ ] Implémenter `ExtractLayerUnit` (extraction d'une couche Z).
- [ ] Implémenter `DielMigrationUnit` (moyenne pondérée jour/nuit).
- [ ] Tests unitaires pour `FunctionalGroup` et les nouvelles unités.
