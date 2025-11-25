# Plan d'Implémentation : Forcing Manager

## Objectif
Implémenter le composant `ForcingManager` responsable de fournir les données de forçage (température, courants, production primaire, etc.) à la simulation pour chaque pas de temps.

## Philosophie
- **Abstraction des I/O** : Le `ForcingManager` ne charge pas de fichiers. Il reçoit un `xarray.Dataset` déjà chargé (ou lazy via Dask) par l'utilisateur.
- **Interpolation Temporelle** : Sa responsabilité principale est de fournir les données exactes pour un instant $t$, en interpolant si nécessaire entre les données disponibles.
- **Agnostique Spatialement** : Pour cette version, on suppose que les forçages sont déjà sur la bonne grille spatiale (celle définie par l'état initial).

## 1. Architecture

### 1.1. Nouveau Module
Création de `seapopym/forcing/` avec :
- `__init__.py`
- `core.py` : Contient la classe `ForcingManager`.

### 1.2. Classe `ForcingManager`

```python
class ForcingManager:
    def __init__(self, forcings: xr.Dataset):
        """
        Args:
            forcings: Dataset xarray contenant les variables de forçage.
                     Doit avoir une dimension 'time' compatible avec la simulation.
        """
        # Validation : Vérifier la présence de la dimension 'time'
        # Validation : Vérifier que les temps sont monotones
        pass

    def get_forcings(self, current_time: datetime) -> xr.Dataset:
        """
        Retourne les forçages interpolés pour le temps demandé.

        Logique :
        - Si current_time existe exactement dans les forçages -> .sel(time=t)
        - Sinon -> .interp(time=t) (linéaire par défaut)
        - Gestion des erreurs si hors bornes.
        """
        pass
```

### 1.3. Intégration dans le Controller

Modification de `SimulationController` :
- **`setup`** : Accepte un argument optionnel `forcings: xr.Dataset | None`.
- **`step`** :
    1. Si un `forcing_manager` est présent :
        - Appelle `forcing_manager.get_forcings(self._current_time)`.
        - Fusionne ces forçages dans `self.state` via `StateManager`.
    2. Continue l'exécution normale (calculs des groupes).

## 2. Étapes d'Implémentation

### Étape 1 : Module Forcing
- Créer la structure de fichiers.
- Implémenter `ForcingManager`.
- Tests unitaires :
    - Cas nominal : Temps exact.
    - Cas interpolation : Temps intermédiaire.
    - Cas erreur : Temps hors bornes.
    - Cas Dask : Vérifier que le lazy loading est préservé jusqu'au calcul.

### Étape 2 : Intégration Controller
- Modifier `SimulationController.setup` pour instancier le `ForcingManager`.
- Modifier `SimulationController.step` pour mettre à jour l'état.
- Mettre à jour les tests du Controller.

### Étape 3 : Validation End-to-End
- Mettre à jour le notebook de démo pour utiliser des forçages variables dans le temps (ex: température qui oscille).

## 3. Points d'Attention
- **Performance** : L'interpolation (`interp`) à chaque pas de temps peut être coûteuse si le graphe Dask devient trop gros.
    - *Optimisation future* : Si les pas de temps sont réguliers et alignés, on pourra optimiser pour éviter `interp`.
- **Noms des variables** : Les noms dans le dataset de forçage doivent correspondre aux noms attendus par le Blueprint (via `register_forcing`). Le `ForcingManager` ne renomme pas les variables, c'est la responsabilité de l'utilisateur de fournir les bons noms.
