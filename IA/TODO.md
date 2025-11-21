# Todo List - Phase 1 : Refonte du Système de Forçages

Cette liste suit la roadmap définie dans `IA/02_Roadmap_MultiGroup.md`.

## 1. Création de `ForcingSource`
- [ ] Créer le fichier `src/seapopym_message/forcing/source.py`.
- [ ] Implémenter la classe `ForcingSource` :
    - [ ] Constructeur acceptant un `xarray.DataArray`.
    - [ ] Validation des données (vérification des dimensions minimales : temps, lat, lon).
    - [ ] Gestion des métadonnées (unités, méthode d'interpolation).
- [ ] Exposer la classe dans `src/seapopym_message/forcing/__init__.py`.

## 2. Mise à jour de `ForcingManager`
- [ ] Modifier `src/seapopym_message/forcing/manager.py` :
    - [ ] Mettre à jour `__init__` pour accepter un dictionnaire de `ForcingSource` (ou continuer avec `xr.Dataset` mais les wrapper en interne).
    - [ ] Mettre à jour `_interpolate_time` pour qu'elle soit agnostique aux dimensions spatiales/profondeur (Xarray le fait déjà, mais vérifier le retour).
    - [ ] Mettre à jour `prepare_timestep` pour retourner des tableaux JAX N-dimensionnels sans erreur.
- [ ] Vérifier la compatibilité avec Ray (sérialisation des gros tableaux).

## 3. Tests et Validation
- [ ] Créer un test unitaire `tests/unit/forcing/test_source.py` pour `ForcingSource`.
- [ ] Créer un test unitaire `tests/unit/forcing/test_manager_ndim.py` pour vérifier que le `ForcingManager` gère bien un champ 3D (ex: `(depth, lat, lon)`).

## 4. Nettoyage
- [ ] Vérifier que les types (Type Hinting) sont corrects.
- [ ] Lancer `ruff` et `mypy`.
