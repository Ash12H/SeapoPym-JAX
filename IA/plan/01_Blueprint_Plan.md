# Plan d'Implémentation : Composant Blueprint

Ce document détaille les étapes pour implémenter le composant **Blueprint**, responsable de la construction du graphe de dépendances et de la compilation du plan d'exécution de la simulation.

## 1. Structure du Module

Création de la structure de base pour le package `seapopym.blueprint`.

*   `seapopym/blueprint/__init__.py` : Exports principaux.
*   `seapopym/blueprint/core.py` : Classe `Blueprint` principale.
*   `seapopym/blueprint/nodes.py` : Définitions des noeuds du graphe (DataNode, ComputeNode).
*   `seapopym/blueprint/execution.py` : Définition de la classe `ExecutionPlan`.
*   `seapopym/blueprint/exceptions.py` : Erreurs spécifiques (MissingInputError, CycleError, etc.).

## 2. Définition des Structures de Données

### 2.1. Noeuds (Nodes)
Dans `nodes.py`, définir des dataclasses pour représenter les éléments du graphe.
*   **DataNode** : Représente une variable.
    *   `name`: str
    *   `dims`: tuple (optionnel, pour validation)
*   **ComputeNode** : Représente une unité de calcul.
    *   `func`: Callable
    *   `name`: str (nom unique de l'étape)
    *   `input_mapping`: dict (nom_arg -> nom_variable_graphe)
    *   `output_mapping`: dict (cle_retour -> nom_variable_graphe)
    *   `scope`: str ('local' ou 'global')

### 2.2. Plan d'Exécution
Dans `execution.py`, définir la structure de sortie.
*   **ExecutionPlan** :
```python
@dataclass
class ExecutionPlan:
    task_groups: list[tuple[str, list[ComputeNode]]] # Séquence ordonnée de (NomGroupe, [Tâches])
    initial_variables: list[str]
    produced_variables: list[str]
```

## 3. Implémentation du Blueprint (Core)

Dans `core.py`, implémenter la classe `Blueprint`.

### 3.1. Initialisation
*   Attribut `graph`: `networkx.DiGraph`.
*   Attribut `registered_variables`: Set[str] pour suivi rapide.

### 3.2. Enregistrement des Données (Forçages)
*   Méthode `register_forcing(name, dims)`.
*   Ajoute un `DataNode` au graphe.

### 3.3. Enregistrement des Unités (Compute)
*   Méthode `register_unit(func, input_mapping, output_name, scope)`.
*   **Introspection** : Utiliser `inspect.signature` pour récupérer les arguments de `func`.
*   **Résolution de Dépendances** (Le coeur logique) :
    *   Pour chaque argument de la fonction :
        1.  Vérifier `input_mapping` (Mapping Explicite).
        2.  (Si contexte de groupe) Vérifier préfixe (Namespacing).
        3.  Vérifier correspondance directe (Matching par défaut).
    *   Si trouvé : Créer arête `DataNode (Source) -> ComputeNode`.
    *   Si non trouvé : Lever `MissingInputError` (ou différer la validation au `build`).
*   **Sortie** :
    *   Déterminer le nom de sortie (`output_name` ou nom par défaut).
    *   Créer arête `ComputeNode -> DataNode (Sortie)`.

### 3.4. Enregistrement de Groupe
*   Méthode `register_group(group_prefix, units_list)`.
*   Appelle `register_unit` en injectant automatiquement des préfixes ou un contexte de nommage.

## 4. Compilation et Validation

### 4.1. Validation Topologique
*   Vérifier l'absence de cycles (`nx.is_directed_acyclic_graph`).
*   Vérifier que toutes les entrées requises sont connectées à une source (Forçage ou Sortie d'une autre unité).

### 4.2. Construction du Plan
*   Méthode `build() -> ExecutionPlan`.
*   Utiliser `nx.topological_sort` pour déterminer l'ordre d'exécution.
*   Remplir l'objet `ExecutionPlan`.

## 5. Tests Unitaires

Créer `tests/test_blueprint.py` pour couvrir :
*   Enregistrement simple (1 forçage -> 1 unité).
*   Chaînage (Unit A -> Unit B).
*   Mapping explicite (renommage d'entrée).
*   Détection de cycle (Erreur attendue).
*   Entrée manquante (Erreur attendue).
*   Ordre d'exécution correct.

## Ordre de Développement Suggéré

1.  `nodes.py` & `execution.py` (Structures)
2.  `core.py` (Squelette + `register_forcing`)
3.  `core.py` (`register_unit` avec introspection simple)
4.  `core.py` (Logique de résolution de dépendances avancée)
5.  `core.py` (`build` et validation)
6.  Tests complets.
