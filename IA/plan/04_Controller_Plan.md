# Plan d'Implémentation : Controller (Orchestrateur)

Ce document détaille l'implémentation du **Controller**, le chef d'orchestre de la simulation. Il assemble le Blueprint, le GSM et les Functional Groups pour exécuter la simulation.

## 1. Objectifs

Le Controller doit :
*   Être le point d'entrée unique pour l'utilisateur.
*   Gérer le cycle de vie de la simulation (Setup -> Run -> Cleanup).
*   Orchestrer les interactions entre les composants sans connaître leur logique interne.
*   Gérer la boucle temporelle.

## 2. Structure du Module

Création du package `seapopym.controller`.

*   `seapopym/controller/__init__.py` : Exports.
*   `seapopym/controller/core.py` : Classe `SimulationController`.
*   `seapopym/controller/configuration.py` : Dataclass `SimulationConfig`.

## 3. Implémentation

### 3.1. Configuration (`configuration.py`)
Une structure simple pour démarrer :
```python
@dataclass
class SimulationConfig:
    start_date: datetime
    end_date: datetime
    timestep: timedelta
    # Plus tard : paramètres I/O, chunks, etc.
```

### 3.2. SimulationController (`core.py`)

#### Initialisation
```python
class SimulationController:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.blueprint = Blueprint()
        self.state_manager = StateManager
        self.state: xr.Dataset | None = None
        self.functional_groups: list[FunctionalGroup] = []
```

#### Setup
```python
def setup(self, model_configuration_func: Callable[[Blueprint], None], initial_state: xr.Dataset):
    """
    1. Appelle la fonction utilisateur pour configurer le Blueprint.
    2. Construit le plan d'exécution (blueprint.build()).
    3. Valide et stocke l'état initial (GSM).
    4. Identifie les groupes uniques dans le plan et instancie les FunctionalGroup correspondants (Acteurs).
    """
```

#### Run Loop
```python
def run(self):
    """
    Boucle principale :
    1. Initialise le temps courant.
    2. While current_time < end_date:
        self.step()
        current_time += timestep
    """
```

#### Step
```python
def step(self):
    """
    Exécute un pas de temps :
    1. (TODO) Charger les forçages du jour.
    2. Pour chaque (group_name, tasks) dans execution_plan.task_groups :
        - Récupérer l'acteur 'group_name'.
        - results = group.compute(self.state, tasks=tasks)
        - self.state = StateManager.merge_forcings(self.state, results)
    3. self.state = StateManager.initialize_next_step(self.state)
    """
```
*Note : L'architecture est prête pour le parallélisme. Si les groupes sont indépendants (Operator Splitting), on pourra lancer les compute() en parallèle et merger à la fin.*

## 4. Tests Unitaires

Créer `tests/test_controller.py` :
*   **Setup** : Vérifier que le Blueprint est bien peuplé et les groupes créés.
*   **Step** : Vérifier qu'un pas de temps modifie l'état.
*   **Run** : Vérifier que la boucle tourne le bon nombre de fois.

## 5. Dépendances
*   `seapopym.blueprint`
*   `seapopym.gsm`
*   `seapopym.functional_group`
*   `xarray`

## Ordre de Développement
1.  `configuration.py`
2.  `core.py` (squelette + setup)
3.  `core.py` (run + step)
4.  Tests.
