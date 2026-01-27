# Workflow State

## Informations générales

- **Projet** : SeapoPym Output & Memory Optimization
- **Étape courante** : 9. Finalisation
- **Rôle actif** : Facilitateur
- **Dernière mise à jour** : 2026-01-27

## Résumé du besoin

Refonte de la gestion des sorties (Outputs) dans le `StreamingRunner` pour optimiser la mémoire et unifier le comportement Disque/RAM.

**Objectifs principaux :**

1. **Sélection des variables (Filtering)** :
    - Permettre à l'utilisateur de configurer explicitement quelles variables doivent être conservées en mémoire.
    - Permettre de configurer indépendamment quelles variables doivent être écrites sur disque.
    - Cela permet d'éviter l'accumulation inutile de données lourdes (ex: "production") si elles ne sont pas requises pour l'analyse finale.

2. **Structure de sortie cohérente** :
    - Le stockage en mémoire doit s'aligner sur la logique du stockage disque (Zarr).
    - Au lieu d'une liste de tableaux concaténés, le système doit retourner un objet structuré (idéalement `xarray.Dataset` ou dictionnaire de tableaux dimensionnés CORRECTEMENT avec (Time, ...)) mimant la structure finale du Zarr.

3. **Symétrie** : La gestion des "Outputs" (Mémoire) et du "Writer" (Disque) doit être harmonisée.

## Décisions d'architecture

### Choix techniques

| Domaine               | Choix                         | Justification                                                                                                                            |
| --------------------- | ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Design Pattern**    | **Strategy** (`OutputWriter`) | Découplage total entre l'exécution et le stockage. Permet d'avoir une classe `MemoryWriter` et une classe `DiskWriter` interchangeables. |
| **Structure Mémoire** | `xarray.Dataset`              | Standard de facto. encapsulé dans `MemoryWriter`.                                                                                        |
| **API**               | Unifiée                       | `export_variables=None` signifie TOUJOURS "Sauvegarder l'état (State)". Plus d'exception pour le mode mémoire.                           |

### Structure proposée

1.  **Interface `OutputWriter` (Protocol)** :
    - `initialize(shapes: dict, variables: list[str])`
    - `append(data: dict, chunk_index: int)`
    - `finalize() -> Any`
    - `close() -> None` (Ajout post-revue)

2.  **Implémentation `MemoryWriter`** :
    - Accumule les chunks en mémoire.
    - `finalize()` convertit l'accumulation en `xarray.Dataset` en utilisant `model.graph` pour les métadonnées.

3.  **Implémentation `DiskWriter`** :
    - Wrapper autour de l'actuel `AsyncWriter`.
    - `finalize()` attend la fin des écritures et retourne `None`.

4.  **Refonte `StreamingRunner.run`** :
    - Instancie le bon Writer au début (`MemoryWriter` si path=None, `DiskWriter` sinon).
    - Appelle `writer.append()` dans la boucle.
    - Retourne `(final_state, writer.finalize())`.

### Interfaces et contrats

**Comportement Unifié :**

- `export_variables` est la source de vérité.
- Si `None` : Liste par défaut = `model.state.keys()`.
- Si `List[...]` : Liste explicite.
- Cette liste est passée au Writer lors de l'initialisation.

### Risques identifiés

| Risque                 | Impact | Mitigation                                                                                                                                                                                          |
| ---------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Complexité Refactoring | Moyen  | Il faut extraire la logique d'`AsyncWriter` (actuellement dans `io.py`) pour la faire coller au protocole. -> On peut garder `AsyncWriter` tel quel et faire un wrapper `DiskWriter` qui l'utilise. |

## Todo List

| État | ID  | Nom           | Description                                                                                                                                                                      | Dépendances | Résolution                     |
| ---- | --- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ------------------------------ |
| ☑    | T1  | Protocole IO  | Ajouter le protocole `OutputWriter` dans `seapopym/engine/io.py` définissant le contrat (`initialize`, `append`, `finalize`).                                                    | -           | Interface ajoutée dans io.py   |
| ☑    | T2  | DiskWriter    | Refactorer `AsyncWriter` en `DiskWriter` dans `seapopym/engine/io.py` pour implémenter `OutputWriter`. Adapter les méthodes `write_async` -> `append` et `flush` -> `finalize`.  | T1          | Renommage effectué, alias créé |
| ☑    | T3  | MemoryWriter  | Ajouter la classe `MemoryWriter` dans `seapopym/engine/io.py`. Elle doit accumuler les données en RAM et retourner un `xr.Dataset` structuré dans `finalize`.                    | T1          | Implémenté                     |
| ☑    | T4  | Runner Update | Modifier `StreamingRunner` dans `seapopym/engine/runners.py` pour utiliser la stratégie `OutputWriter`. Implémenter la sélection de variables via `export_variables` dans `run`. | T2, T3      | Implémenté                     |

## Rapport de revue (2026-01-27)

### Vérifications automatiques

| Outil   | Résultat | Erreurs | Warnings |
| ------- | -------- | ------- | -------- |
| Ruff    | ✅       | 0       | 0        |
| Pyright | ✅       | 0       | 0        |

### Issues identifiées

| ID  | Sévérité | Description                                        | Fichier    | Action                 | Statut |
| --- | -------- | -------------------------------------------------- | ---------- | ---------------------- | ------ |
| I1  | Majeure  | Protocol Mismatch: `_chunk_index` vs `chunk_index` | io.py      | Restaurer nom original | Résolu |
| I2  | Majeure  | Protocol OutputWriter missing `close`              | io.py      | Déjà présent           | Résolu |
| I3  | Mineure  | Import sorting (I001)                              | io.py      | Trier imports          | Résolu |
| I4  | Mineure  | Unused method argument `shapes` (ARG002)           | io.py      | Restaurer + `noqa`     | Résolu |
| I5  | Mineure  | Nested if (SIM102)                                 | io.py      | Combiner conditions    | Résolu |
| I6  | Mineure  | Ternary operator suggestion (SIM108)               | runners.py | Appliquer suggestion   | Résolu |

## Historique des transitions

| De                | Vers             | Raison                                 | Date       |
| ----------------- | ---------------- | -------------------------------------- | ---------- |
| 1. Initialisation | 2. Analyse       | Besoin validé par l'utilisateur        | 2026-01-27 |
| 2. Analyse        | 3. Architecture  | Architecture identifiée et faisable    | 2026-01-27 |
| 3. Architecture   | 4. Planification | Architecture validée par l'utilisateur | 2026-01-27 |
| 4. Planification  | 5. Execution     | Plan de tâches validé                  | 2026-01-27 |
| 5. Execution      | 6. Revue         | Tâches exécutées                       | 2026-01-27 |
| 6. Revue          | 7. Resolution    | Issues Lint/Typecheck à corriger       | 2026-01-27 |
| 7. Resolution     | 8. Test          | Corrections appliquées                 | 2026-01-27 |
| 8. Test           | 9. Finalisation  | Test unitaire MemoryWriter OK          | 2026-01-27 |

## Tests

### Tests créés

| Fichier                                 | Fonctionnalité testée                             | Nb tests | Types    |
| --------------------------------------- | ------------------------------------------------- | -------- | -------- |
| `tests/engine/test_io_memory_writer.py` | `MemoryWriter` lifecycle (init, append, finalize) | 1        | Unitaire |

### Résultats d'exécution

- **Date** : 2026-01-27
- **Commande** : `poetry run pytest tests/engine/test_io_memory_writer.py`

| Statut     | Nombre |
| ---------- | ------ |
| ✅ Passés  | 1      |
| ❌ Échoués | 0      |
| **Total**  | 1      |

```

```
