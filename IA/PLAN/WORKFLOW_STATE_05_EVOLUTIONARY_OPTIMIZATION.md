# Workflow State

## Informations générales

- **Projet** : Optimisation évolutionnaire (CMA-ES) et hybride
- **Étape courante** : 9. Clôture
- **Rôle actif** : Chef de projet
- **Dernière mise à jour** : 2026-01-29

## Résumé du besoin

### Objectif
Ajouter l'optimisation évolutionnaire (CMA-ES) au module `seapopym.optimization` avec trois modes d'utilisation : CMA-ES seul, gradient seul (existant), et hybride CMA-ES→Gradient.

## Décisions d'architecture

### Choix techniques

| Domaine | Choix | Justification |
|---------|-------|---------------|
| Lib évolutionnaire | evosax | JAX-natif, CMA-ES optimisé, bien maintenu |
| Stratégie par défaut | CMA-ES | Standard industriel, adapte covariance auto |
| Dépendance | Optionnelle | `pip install seapopym[optimization]` |
| Interface | Même pattern que `Optimizer` | Cohérence API |

## Todo List

| État | ID | Nom | Description | Dépendances | Résolution |
|------|-----|-----|-------------|-------------|------------|
| ☑ | T1 | Ajouter evosax | Modifier `pyproject.toml` | - | evosax>=0.1.0 ajouté dans [optimization] |
| ☑ | T2 | Créer evolutionary.py | EvolutionaryOptimizer avec CMA-ES | T1 | ~200 lignes, flatten/unflatten/bounds |
| ☑ | T3 | Créer hybrid.py | HybridOptimizer (CMA + gradient top K) | T2 | ~230 lignes, phase CMA puis gradient |
| ☑ | T4 | Modifier __init__.py | Exports conditionnels | T2, T3 | Import conditionnel avec message clair |
| ☑ | T5 | Tests evolutionary | Tests unitaires | T2 | 20 tests, 100% pass |
| ☑ | T6 | Tests hybrid | Tests unitaires | T3 | 18 tests, 100% pass |
| ☑ | T7 | Exemple comparatif | Grad vs CMA vs Hybride + visu 2D | T2, T3 | Démontre avantage CMA-ES |

## Historique des transitions

| De | Vers | Raison | Date |
|----|------|--------|------|
| - | 1. Initialisation | Début du projet | 2026-01-29 |
| 1. Initialisation | 2. Analyse | Besoin validé par l'utilisateur | 2026-01-29 |
| 2. Analyse | 3. Architecture | Analyse complétée | 2026-01-29 |
| 3. Architecture | 4. Planification | Architecture validée par l'utilisateur | 2026-01-29 |
| 4. Planification | 5. Execution | Todo list complétée | 2026-01-29 |
| 5. Execution | 6. Revue | Tâches T1-T4, T7 complétées | 2026-01-29 |
| 6. Revue | 8. Test | Revue OK (lint, format, types passés) | 2026-01-29 |
| 8. Test | 9. Clôture | T5, T6 complétés (38 tests, 100% pass) | 2026-01-29 |
