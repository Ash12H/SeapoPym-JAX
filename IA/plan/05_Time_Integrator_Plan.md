# Plan d'Implémentation : Time Integrator

Ce document détaille l'implémentation du **Time Integrator**, responsable de la mise à jour physique de l'état global à partir des tendances calculées par les groupes fonctionnels.

## 1. Objectifs

Le Time Integrator résout le problème fondamental : **Comment faire évoluer l'état de $t$ à $t+\Delta t$ ?**

Il doit :
*   Recevoir l'état actuel $S(t)$ et les tendances $\frac{dS}{dt}$ de tous les groupes.
*   Appliquer un schéma d'intégration numérique (Euler explicite, RK4, etc.).
*   **Garantir la positivité** : Éviter les biomasses négatives.
*   **Gérer les conflits** : Si plusieurs groupes modifient la même variable, résoudre proprement.
*   Retourner le nouvel état $S(t+\Delta t)$.

## 2. Structure du Module

Création du package `seapopym.time_integrator`.

*   `seapopym/time_integrator/__init__.py` : Exports.
*   `seapopym/time_integrator/core.py` : Classe `TimeIntegrator`.
*   `seapopym/time_integrator/schemes.py` : Schémas d'intégration numérique.
*   `seapopym/time_integrator/constraints.py` : Gestion des contraintes (positivité, etc.).

## 2.1. Modifications du Blueprint (Prérequis)

Pour supporter les tendances explicites, le Blueprint doit être enrichi :

### `DataNode`
Ajouter un attribut pour marquer les tendances :
```python
@dataclass(frozen=True)
class DataNode:
    name: str
    dims: tuple | None = None
    is_tendency_of: str | None = None  # Nouveau : Si c'est une tendance, de quelle variable ?
```

### `ExecutionPlan`
Ajouter un mapping des tendances :
```python
@dataclass
class ExecutionPlan:
    task_groups: list[tuple[str, list[ComputeNode]]]
    initial_variables: list[str]
    produced_variables: list[str]
    tendency_map: dict[str, list[str]] = field(default_factory=dict)  # Nouveau
    # Ex: {"biomass": ["mortality_rate", "growth_rate", "transport_flux"]}
```

### `register_unit`
Ajouter un argument `output_tendencies` :
```python
bp.register_unit(
    compute_mortality,
    output_mapping={"rate": "mortality_rate"},
    output_tendencies={"rate": "biomass"}  # Cette sortie 'rate' est une tendance de 'biomass'
)
```

Le Blueprint construira automatiquement le `tendency_map` lors de `build()`.

## 3. Implémentation

### 3.1. Schémas d'Intégration (`schemes.py`)

**Euler Explicite avec Tendances Multiples** :
```python
def euler_forward(
    state: xr.Dataset,
    all_results: dict[str, xr.DataArray],
    tendency_map: dict[str, list[str]],
    dt: float
) -> xr.Dataset:
    """
    Schéma d'Euler explicite : S(t+dt) = S(t) + dt * sum(tendances)

    Args:
        state: État actuel.
        all_results: Tous les résultats des groupes (tendances + autres variables).
        tendency_map: Mapping {var_cible: [liste_tendances]}.
        dt: Pas de temps (en secondes).

    Returns:
        Nouvel état.
    """
    new_state = state.copy()

    for target_var, tendency_names in tendency_map.items():
        if target_var in state:
            # Somme de toutes les tendances affectant cette variable
            total_tendency = sum(
                all_results[t] for t in tendency_names if t in all_results
            )
            new_state[target_var] = state[target_var] + dt * total_tendency

    return new_state
```

### 3.2. Gestion des Contraintes (`constraints.py`)

Fonction pour garantir la positivité :
```python
def enforce_positivity(state: xr.Dataset, variables: list[str]) -> xr.Dataset:
    """
    Force les variables spécifiées à être >= 0.

    Args:
        state: L'état à contraindre.
        variables: Liste des variables qui doivent être positives (ex: biomasses).

    Returns:
        État avec contraintes appliquées.
    """
    constrained_state = state.copy()

    for var_name in variables:
        if var_name in constrained_state:
            constrained_state[var_name] = constrained_state[var_name].clip(min=0)

    return constrained_state
```

### 3.3. TimeIntegrator (`core.py`)

Classe principale :
```python
class TimeIntegrator:
    """
    Responsable de l'intégration temporelle de l'état.
    """

    def __init__(
        self,
        scheme: str = "euler",
        positive_vars: list[str] | None = None
    ):
        """
        Args:
            scheme: Nom du schéma d'intégration ('euler', 'rk4', etc.).
            positive_vars: Variables à contraindre >= 0 (ex: biomasses).
        """
        self.scheme = scheme
        self.positive_vars = positive_vars or []

    def integrate(
        self,
        state: xr.Dataset,
        all_results: dict[str, xr.DataArray],
        tendency_map: dict[str, list[str]],
        dt: float
    ) -> xr.Dataset:
        """
        Fait évoluer l'état de t à t+dt.

        Args:
            state: État à t.
            all_results: Tous les résultats calculés (tendances + diagnostics).
            tendency_map: Mapping des tendances vers leurs variables cibles.
            dt: Pas de temps (en secondes).

        Returns:
            État à t+dt.
        """
        # 1. Application du schéma numérique
        if self.scheme == "euler":
            new_state = euler_forward(state, all_results, tendency_map, dt)
        else:
            raise ValueError(f"Unknown scheme: {self.scheme}")

        # 2. Application des contraintes
        new_state = enforce_positivity(new_state, self.positive_vars)

        return new_state
```

## 4. Intégration avec le Controller

Le Controller doit :
1.  Instancier le `TimeIntegrator` dans `setup`.
2.  Dans `step()` :
    *   Collecter tous les résultats des groupes.
    *   Appeler `time_integrator.integrate(state, all_results, tendency_map, dt)`.
    *   Merger les variables non-tendances (diagnostics).

**Exemple dans `Controller.step()` :**
```python
def step(self) -> None:
    if self.state is None or self.execution_plan is None:
        raise RuntimeError("Simulation not set up.")

    # 1. Exécution des groupes
    all_results = {}
    for group_name, tasks in self.execution_plan.task_groups:
        group = self.groups[group_name]
        results = group.compute(self.state, tasks=tasks)
        all_results.update(results)

    # 2. Intégration temporelle (applique les tendances)
    dt = self.config.timestep.total_seconds()
    self.state = self.time_integrator.integrate(
        self.state,
        all_results,
        self.execution_plan.tendency_map,
        dt
    )

    # 3. Ajout des variables non-tendances (diagnostics, etc.)
    # On filtre pour ne pas réappliquer les tendances déjà intégrées
    tendency_vars = set(sum(self.execution_plan.tendency_map.values(), []))
    diagnostics = {k: v for k, v in all_results.items() if k not in tendency_vars}
    self.state = StateManager.merge_forcings(self.state, diagnostics)

    # 4. Préparation du pas suivant
    self.state = StateManager.initialize_next_step(self.state)
```

## 5. Tests Unitaires

Créer `tests/test_time_integrator.py` :
*   **Test Euler Simple** : Vérifier que $x(t+1) = x(t) + dt \times dx/dt$.
*   **Test Tendances Multiples** : Biomass avec mortalité (-0.1) et croissance (+0.2) -> tendance nette +0.1.
*   **Test Positivité** : Vérifier qu'une tendance négative forte ne rend pas la biomasse négative.

## 6. Extensions Futures

*   **RK4** : Pour une meilleure précision.
*   **Adaptive Timestepping** : Ajuster `dt` selon la stabilité numérique.
*   **Flux de Masse** : Log des flux entrants/sortants pour analyse.

## Ordre de Développement
1.  Modifier le Blueprint (DataNode, ExecutionPlan, register_unit).
2.  `schemes.py` : Euler explicite.
3.  `constraints.py` : Positivité.
4.  `core.py` : TimeIntegrator.
5.  Tests unitaires.
6.  Intégration dans le Controller.

Création du package `seapopym.time_integrator`.

*   `seapopym/time_integrator/__init__.py` : Exports.
*   `seapopym/time_integrator/core.py` : Classe `TimeIntegrator`.
*   `seapopym/time_integrator/schemes.py` : Schémas d'intégration numérique.
*   `seapopym/time_integrator/constraints.py` : Gestion des contraintes (positivité, etc.).

## 3. Implémentation

### 3.1. Schémas d'Intégration (`schemes.py`)

Commençons simple avec **Euler Explicite** :
```python
def euler_forward(state: xr.Dataset, tendencies: dict[str, xr.DataArray], dt: float) -> xr.Dataset:
    """
    Schéma d'Euler explicite : S(t+dt) = S(t) + dt * dS/dt

    Args:
        state: État actuel.
        tendencies: Dictionnaire {var_name: tendance}.
        dt: Pas de temps (en secondes).

    Returns:
        Nouvel état.
    """
    new_state = state.copy()

    for var_name, tendency in tendencies.items():
        if var_name in new_state:
            new_state[var_name] = state[var_name] + dt * tendency
        else:
            # Variable nouvellement créée (pas une tendance d'une variable existante)
            new_state[var_name] = tendency

    return new_state
```

### 3.2. Gestion des Contraintes (`constraints.py`)

Fonction pour garantir la positivité :
```python
def enforce_positivity(state: xr.Dataset, variables: list[str]) -> xr.Dataset:
    """
    Force les variables spécifiées à être >= 0.

    Args:
        state: L'état à contraindre.
        variables: Liste des variables qui doivent être positives (ex: biomasses).

    Returns:
        État avec contraintes appliquées.
    """
    constrained_state = state.copy()

    for var_name in variables:
        if var_name in constrained_state:
            constrained_state[var_name] = constrained_state[var_name].clip(min=0)

    return constrained_state
```

Optionnel : **Résolution des conflits par prorata** (si plusieurs groupes consomment la même ressource) :
```python
def resolve_conflicts(state: xr.Dataset, tendencies: dict[str, list[xr.DataArray]]) -> dict[str, xr.DataArray]:
    """
    Résout les conflits si plusieurs sources modifient la même variable.
    Applique un prorata si la somme des consommations dépasse la disponibilité.
    """
    # À implémenter si nécessaire
    pass
```

### 3.3. TimeIntegrator (`core.py`)

Classe principale :
```python
class TimeIntegrator:
    """
    Responsable de l'intégration temporelle de l'état.
    """

    def __init__(
        self,
        scheme: str = "euler",
        positive_vars: list[str] | None = None
    ):
        """
        Args:
            scheme: Nom du schéma d'intégration ('euler', 'rk4', etc.).
            positive_vars: Variables à contraindre >= 0 (ex: biomasses).
        """
        self.scheme = scheme
        self.positive_vars = positive_vars or []

    def integrate(
        self,
        state: xr.Dataset,
        tendencies: dict[str, xr.DataArray],
        dt: float
    ) -> xr.Dataset:
        """
        Fait évoluer l'état de t à t+dt.

        Args:
            state: État à t.
            tendencies: Tendances calculées par les groupes.
            dt: Pas de temps (en secondes).

        Returns:
            État à t+dt.
        """
        # 1. Application du schéma numérique
        if self.scheme == "euler":
            new_state = euler_forward(state, tendencies, dt)
        else:
            raise ValueError(f"Unknown scheme: {self.scheme}")

        # 2. Application des contraintes
        new_state = enforce_positivity(new_state, self.positive_vars)

        return new_state
```

## 4. Intégration avec le Controller

Le Controller doit :
1.  Instancier le `TimeIntegrator` à la configuration (dans `setup`).
2.  Dans `step()` :
    *   Collecter les résultats de tous les groupes.
    *   **Distinguer** les tendances (variables finissant par `_tendency` ou marquées) des variables directes.
    *   Appeler `time_integrator.integrate(state, tendencies, dt)`.
    *   Mettre à jour `self.state`.

**Exemple dans `Controller.step()` :**
```python
def step(self) -> None:
    if self.state is None or self.execution_plan is None:
        raise RuntimeError("Simulation not set up.")

    # 1. Exécution des groupes
    all_results = {}
    for group_name, tasks in self.execution_plan.task_groups:
        group = self.groups[group_name]
        results = group.compute(self.state, tasks=tasks)
        all_results.update(results)

    # 2. Séparation tendances / variables directes
    tendencies = {k: v for k, v in all_results.items() if k.endswith("_tendency")}
    direct_vars = {k: v for k, v in all_results.items() if not k.endswith("_tendency")}

    # 3. Intégration temporelle
    dt = self.config.timestep.total_seconds()
    self.state = self.time_integrator.integrate(self.state, tendencies, dt)

    # 4. Ajout des variables directes (diagnostics, etc.)
    self.state = StateManager.merge_forcings(self.state, direct_vars)

    # 5. Préparation du pas suivant
    self.state = StateManager.initialize_next_step(self.state)
```

## 5. Tests Unitaires

Créer `tests/test_time_integrator.py` :
*   **Test Euler Simple** : Vérifier que $x(t+1) = x(t) + dt \times dx/dt$.
*   **Test Positivité** : Vérifier qu'une tendance négative forte ne rend pas la biomasse négative.
*   **Test Intégration** : Simuler un cycle complet avec le Controller.

## 6. Extensions Futures

*   **RK4** : Pour une meilleure précision.
*   **Adaptive Timestepping** : Ajuster `dt` selon la stabilité numérique.
*   **Conservation de Masse** : Vérifier que $\sum \text{biomass} \approx \text{constant}$ (selon les flux).

## Ordre de Développement
1.  `schemes.py` : Euler explicite.
2.  `constraints.py` : Positivité.
3.  `core.py` : TimeIntegrator.
4.  Tests unitaires.
5.  Intégration dans le Controller.
