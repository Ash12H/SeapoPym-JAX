# Plan d'Implémentation : Architecture Backend d'Exécution

## Objectif
Découpler la logique d'exécution du `SimulationController` en introduisant une abstraction `ComputeBackend`. Cela permettra de supporter plusieurs modes d'exécution (Séquentiel, Ray, Dask, etc.) sans modifier le cœur du contrôleur.

## 1. Architecture

### 1.1. Nouveau Module `seapopym/backend`
Création d'un nouveau package pour gérer les stratégies d'exécution.

Structure :
```
seapopym/backend/
├── __init__.py
├── base.py        # Interface abstraite (Protocol/ABC)
├── sequential.py  # Implémentation par défaut (Code actuel)
└── ray.py         # Implémentation distribuée (Futur)
```

### 1.2. Interface `ComputeBackend` (`base.py`)

```python
class ComputeBackend(ABC):
    @abstractmethod
    def execute(
        self,
        task_groups: list[tuple[str, list[ComputeNode]]],
        state: xr.Dataset
    ) -> dict[str, Any]:
        """
        Exécute une liste de groupes de tâches sur un état donné.

        Args:
            task_groups: Liste de (nom_groupe, liste_de_taches).
            state: État actuel du système (xarray Dataset).

        Returns:
            Dictionnaire des résultats {nom_variable: valeur}.
        """
        pass
```

### 1.3. Implémentation `SequentialBackend` (`sequential.py`)
C'est ici que nous déplacerons la boucle `for` qui se trouve actuellement dans `SimulationController.step`.

```python
class SequentialBackend(ComputeBackend):
    def execute(self, task_groups, state):
        results = {}
        # La logique actuelle de boucle sur les groupes et les tâches
        # ...
        return results
```

### 1.4. Modification du `SimulationController`
Le contrôleur va déléguer l'exécution à son backend.

```python
class SimulationController:
    def __init__(self, config, backend_type: str = "sequential"):
        # Factory simple pour choisir le backend
        if backend_type == "sequential":
            self.backend = SequentialBackend()
        elif backend_type == "ray":
            # self.backend = RayBackend() (Plus tard)
            pass

    def step(self):
        # ... (Forçages) ...

        # Délégation de l'exécution
        results = self.backend.execute(self.execution_plan.task_groups, self.state)

        # ... (Merge des résultats) ...
```

## 2. Étapes d'Implémentation

### Étape 1 : Création de l'interface et du Backend Séquentiel
1.  Créer `seapopym/backend/base.py`.
2.  Créer `seapopym/backend/sequential.py` et y déplacer la logique d'exécution actuelle.
3.  Exposer ces classes dans `seapopym/backend/__init__.py`.

### Étape 2 : Refactoring du Controller
1.  Modifier `SimulationController` pour accepter un argument `backend`.
2.  Remplacer la boucle d'exécution dans `step()` par `self.backend.execute()`.

### Étape 3 : Tests de Non-Régression
1.  Vérifier que tous les tests existants (Controller, Predator-Prey) passent sans modification.
2.  Ajouter un test unitaire simple pour `SequentialBackend`.

### Étape 4 : Préparation pour Ray (Placeholder)
1.  Créer un fichier `seapopym/backend/ray.py` vide ou avec une classe squelette pour préparer le terrain.

## 3. Avantages
- **Isolement** : Le Controller ne connaît plus les détails de l'exécution (boucles, threads, process).
- **Extensibilité** : Ajouter un backend "MPI" ou "Dask" se fait sans toucher au Controller.
- **Stabilité** : Le mode séquentiel reste le citoyen de première classe, facile à débugger.
