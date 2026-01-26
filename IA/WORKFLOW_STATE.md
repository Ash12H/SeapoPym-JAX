# Workflow State

## Informations générales

- **Projet** : seapopym-message - Système de vérification des unités
- **Étape courante** : 4. Planification
- **Rôle actif** : Planificateur
- **Dernière mise à jour** : 2026-01-26

## Résumé du besoin

Suite à l'implémentation des phases 1/2/3 (Blueprint, Compiler, Engine), des problèmes de cohérence dimensionnelle ont été identifiés, notamment dans `examples/predator_prey_0d.py` où une division manuelle par 86400 est nécessaire pour obtenir un résultat correct.

**Besoin principal :** Implémenter un système complet de vérification des unités au moment de la compilation, selon le principe d'un vrai compilateur qui valide la cohérence dimensionnelle de bout en bout.

### 3 axes de travail (par ordre de priorité)

#### 1. [URGENT] Vérification des unités à la compilation

**Problème :**
- Désalignement entre unités déclarées dans le Blueprint et unités attendues par le solveur
- Pas de vérification → erreurs silencieuses → résultats absurdes

**Solution :**
- Vérifier toute la chaîne : `params/forcings → fonction → tendencies/intermediate_data`
- Si pas d'unité spécifiée → `"dimensionless"`
- Documenter clairement quelle unité le solveur attend (investigation requise : dt en secondes ? tendencies en /s ?)
- Erreur de compilation si incohérence détectée avec message explicite

**Principe : PAS de conversion automatique** - L'utilisateur doit fournir les bonnes valeurs

#### 2. [IMPORTANT] Interpolation temporelle des forçages

**Problème :**
- Besoin de réduire dt pour contraintes de stabilité (CFL, Euler explicite)
- Forçages disponibles à pas de temps plus large (ex: données journalières, simulation horaire)

**Solution :**
- Interpolation configurable globale à la compilation (méthodes : `linear`, `nearest`, `ffill`)
- Configuration dans `execution.forcing_interpolation`
- Implémentation uniquement en phase de compilation (pas d'impact runtime JAX)

#### 3. [EXPLORATOIRE] Alternative outputs en mémoire

**Problème :**
- Overhead potentiel d'AsyncWriter pour petits modèles (notebooks, prototypage rapide)

**Solution :**
- Explorer une alternative "outputs en mémoire" si et seulement si :
  - Gain mesurable démontré
  - Complexité d'implémentation faible (pas de refactoring majeur)
  - Pas de dégradation des performances

### Périmètre IN

✅ Vérification dimensionnelle complète à la compilation
✅ Unités temporelles (s, d, h, etc.)
✅ Documentation claire des attentes du solveur
✅ Interpolation temporelle (phase compilation uniquement)
✅ Investigation du parsing actuel de `dt` et de son usage

### Périmètre OUT

❌ Conversion automatique d'unités
❌ Vérifications à l'exécution (runtime)
❌ Conversion d'unités non-temporelles (kg→g, m→km, etc.)
❌ Specs 4 et 5
❌ Support de bibliothèques externes (pint.Quantity) pour le moment

### Breaking changes acceptés

- Modification du format `dt` (ex: standardisation en secondes uniquement) : **OK**
- Modification des exemples existants : **OK**

## Rapport d'analyse

### Structure du projet

```
seapopym-message/
├── seapopym/           # Package principal
│   ├── blueprint/      # Définition des modèles (Blueprint, Config, validation)
│   ├── compiler/       # Compilation Blueprint→CompiledModel
│   ├── engine/         # Exécution (runners, step, backends, I/O)
│   └── functions/      # Fonctions biologiques réutilisables
├── tests/              # Tests unitaires et d'intégration (par module)
├── examples/           # Exemples end-to-end (predator_prey_0d, benchmarks)
├── IA/                 # Documentation méthode ASH
│   ├── METHODE/        # Workflow ASH
│   └── PLAN/           # Spécifications techniques
└── .venv/              # Environnement virtuel
```

**Organisation modulaire :** Architecture 3-couches (Blueprint → Compiler → Engine) bien séparée, patterns clairs.

### Technologies identifiées

- **Langage :** Python 3.10+ (minimum), 3.13 (dev actuel)
- **Backend compute :** JAX (pour autodiff et JIT) + NumPy (fallback)
- **Stack scientifique :** xarray, zarr, netCDF4, dask, pandas
- **Gestion unités :** pint, pint-xarray (déclaré dans dépendances mais non utilisé actuellement)
- **Graphes :** networkx (pour dependency graph)
- **Validation :** pydantic 2.x (schémas Blueprint/Config)
- **Build :** hatchling (PEP 517)
- **Tests :** pytest + pytest-cov + pytest-asyncio + pytest-benchmark
- **Linting :** ruff (très strict : E, W, F, I, B, C4, UP, ARG, SIM, S101)
- **Type checking :** pyright (mode basic)
- **Pre-commit hooks :** Configurés (.pre-commit-config.yaml)
- **Documentation :** mkdocs + mkdocs-material + mkdocstrings

### Patterns et conventions

#### Nommage
- **snake_case** pour fichiers, fonctions, variables
- **PascalCase** pour classes
- Namespace functions : `"namespace:function_name"` (ex: `"demo:prey_growth"`)

#### Architecture
- **3-phases pipeline :**
  1. Blueprint (déclaration modèle) + Config (valeurs concrètes)
  2. Compiler (validation, transformation, packaging)
  3. Engine (exécution temporelle avec scan JAX/NumPy)
- **Dependency graph** : NetworkX (nodes: DataNode + ComputeNode)
- **Function registry** : Décorateur `@functional(name=..., backend=...)`
- **Validation en 6 étapes** (validation.py) : Parse → Resolve → Signatures → Dims → Units → Graph

#### Code quality
- **Type hints** : Utilisés partout, checked par pyright
- **Docstrings** : Format Google style
- **Coverage** : Configuré (cov-report html/xml/term)
- **Tests markers** : `@pytest.mark.{slow,integration,unit,gpu}`

#### Gestion des données
- **Canonical dimensions :** `("T", "Y", "X", "C")` (compiler/model.py:21)
- **Transposition automatique** : Vers ordre canonique
- **NaN handling** : Remplacement par `fill_nan` (default: 0.0)
- **Async I/O** : AsyncWriter avec ThreadPoolExecutor (engine/io.py)

### Points d'attention

#### 1. Gestion des unités (🔴 URGENT - Problème principal)

**État actuel :**
- Unités déclarées dans Blueprint (`VariableDeclaration.units: str | None`)
- Validation basique par comparaison de strings (`validation.py:227-234`)
  - Génère des **warnings**, pas des erreurs
  - Commentaire : "Full Pint validation would require importing and checking dimensionality"
- **Aucune conversion automatique** : L'utilisateur doit manuellement diviser par 86400
- **Pas de vérification de cohérence dimensionnelle** entre sorties/entrées de fonctions

**Problème concret identifié :**
```python
# Blueprint (examples/predator_prey_0d.py:73)
"prey_growth_rate": {"units": "1/d"}

# Config (line 130) - division manuelle requise
"prey_growth_rate": {"value": 0.05 / 86400}  # User must convert to 1/s
```

**Cause racine :**
- Le solveur Euler multiplie les tendencies par `dt` en secondes (`engine/step.py:215`)
- Les tendencies doivent donc être en `unité/s`
- Mais aucune vérification ni conversion n'est effectuée

**Impact :**
- Erreurs silencieuses (résultats absurdes sans erreur de compilation)
- Charge cognitive élevée pour l'utilisateur (conversions manuelles)
- Documentation implicite (doit deviner que dt est en secondes)

**Dépendances disponibles :**
- `pint>=0.24.4` et `pint-xarray>=0.5.1` déjà dans pyproject.toml
- Tests legacy montrent qu'une intention existait (`tests/_legacy/test_units_parameters.py:54-89`)

#### 2. Parsing du timestep (🟡 IMPORTANT)

**État actuel :**
- Format string : `"1d"`, `"6h"`, `"30m"`, `"0.2d"`
- Parser simple : `compiler/compiler.py:24-48`
- Unités supportées : `{"s": 1, "m": 60, "h": 3600, "d": 86400}`
- Conversion en secondes (float) : `CompiledModel.dt: float = 86400.0`
- Utilisé par Euler : `new_state[var_name] = value + total_tendency * dt`

**Limitations :**
- Pas de support : `"month"`, `"year"`, `"week"`
- Pas de gestion d'interpolation temporelle si `dt` < pas de temps des forçages
- Comportement non documenté si désalignement temporel

**Code pertinent :**
```python
# compiler/compiler.py:33
units = {"s": 1, "m": 60, "h": 3600, "d": 86400}

# engine/runners.py:149-154 (slicing forcings)
if arr_np.ndim > 0 and arr_np.shape[0] == n_timesteps:
    sliced[name] = arr_np[start:end]
else:
    # Static forcing - broadcast to chunk length
    sliced[name] = np.broadcast_to(arr_np, (chunk_len,) + arr_np.shape)
```

**Pas d'interpolation actuellement** : Si forçages à pas de temps > dt, les forçages statiques sont broadcastés (répétés), pas interpolés.

#### 3. Outputs en mémoire (🟢 EXPLORATOIRE)

**État actuel :**
- Écriture obligatoire sur disque via `AsyncWriter` (Zarr)
- Streaming avec chunking temporel
- Async I/O avec ThreadPoolExecutor (2 workers par défaut)

**Alternative à explorer :**
- Option pour garder outputs en mémoire (petits modèles, notebooks)
- Trade-off : simplicité vs complexité de l'implémentation

**Complexité estimée :**
- Modification de `StreamingRunner.run()` (engine/runners.py:66-127)
- Ajout de conditionnels (if output_path is None)
- Impact potentiel sur performances si mal implémenté

#### 4. Validation des unités (État actuel détaillé)

**Fichier :** `seapopym/blueprint/validation.py:197-234`

**Logique actuelle :**
```python
def _validate_units(self, blueprint, result):
    for step in blueprint.process:
        metadata = result.resolved_functions[step.func]
        for arg_name, expected_unit in metadata.units.items():
            var_decl = all_vars[var_path]
            if var_decl.units != expected_unit:
                result.add_warning(...)  # ⚠️ WARNING only, not ERROR
```

**Limitations :**
- Simple comparaison de strings (`!=`)
- Pas de vérification dimensionnelle (ex: `"1/d"` vs `"1/s"` sont toutes deux valides mais incompatibles)
- Pas de vérification de la chaîne de dépendances (output fonction A → input fonction B)
- Pas de handling du cas `units=None` → assumé `"dimensionless"`

**Métadonnées des fonctions :**
- Stockées dans `FunctionMetadata.units: dict[str, str]`
- Déclarées via `@functional(units={...})`
- Exemple : `units={"biomass": "g", "rate": "1/d", "return": "g/d"}`

#### 5. NetworkX - Usage actuel

**Utilisation dans le projet :**
- Construction du dependency graph (validation.py:236-298)
- Tri topologique pour ordre d'exécution (engine/step.py:54)
- Types de nodes : `DataNode`, `ComputeNode`

**Pertinence maintenue :**
- Extensibilité future : détection de cycles, optimisation de graphe
- API claire et standard

#### 6. Tests et couverture

**État :**
- Tests organisés par module (blueprint/, compiler/, engine/)
- Fixtures réutilisables (tests/fixtures/)
- Tests legacy exclus (dossier _legacy/)
- Coverage configuré (html/xml/term reports)

**Manques identifiés :**
- Pas de tests pour la vérification dimensionnelle complète des unités
- Pas de tests pour l'interpolation temporelle (feature non implémentée)
- Exemples end-to-end utilisés comme tests d'intégration informels

#### 7. Documentation

**État :**
- mkdocs configuré (mkdocs.yml)
- Spécifications techniques dans IA/PLAN/
- Méthode ASH dans IA/METHODE/
- README.md vide (🔴)

**À améliorer :**
- Documentation des attentes du solveur (dt en secondes, tendencies en /s)
- Documentation des unités supportées pour dt parsing
- Guide utilisateur sur les unités

### Investigations complémentaires requises

Avant l'étape Architecture, clarifier :

1. **Interpolation temporelle actuelle :**
   - Que se passe-t-il si `n_timesteps * dt` ≠ longueur de `time_index` ?
   - Comportement actuel avec forçages journaliers + dt horaire ?

2. **Système pint :**
   - Pourquoi pint est-il déclaré en dépendance mais non utilisé ?
   - Tests legacy montrent une intention de l'utiliser - pourquoi abandonné ?

## Décisions d'architecture

### Principe directeur

**Philosophie : Compilateur strict, pas de conversion automatique**
- JAX n'est pas conscient des unités → Ce n'est pas son rôle de convertir
- L'utilisateur doit fournir les bonnes unités dès le Blueprint
- Le compilateur vérifie la cohérence et rejette si incohérent

### Convention temporelle universelle

**TOUS les temps sont en secondes dans le modèle :**
- `dt` : converti en secondes lors du parsing (déjà implémenté)
- Paramètres temporels : doivent être déclarés en `/s` dans le Blueprint
- Tendencies : doivent être en `unité/s` (ex: `individuals/s`)
- Solveur Euler : multiplie par `dt` en secondes

**Justification :** Cohérence interne, évite ambiguïtés, simplifie la validation.

### Choix techniques

| Domaine | Choix | Justification |
|---------|-------|---------------|
| **Validation unités** | Pint UnitRegistry avec comparaison stricte des formes canoniques | Standard Python, vérification robuste, déjà en dépendances. Comparaison stricte : pas de conversion implicite. |
| **Stratégie validation** | Erreurs à la compilation (pas warnings) | Fail-fast, expérience utilisateur claire, évite bugs silencieux |
| **Interpolation temporelle** | scipy.interpolate / xarray.interp | Déjà disponible, efficace, 4 méthodes supportées |
| **Méthodes interpolation** | `constant` (broadcast actuel), `nearest`, `linear`, `ffill` | Couvre tous les cas d'usage scientifiques |
| **Outputs in-memory** | Paramètre optionnel `output_path=None` | Simplicité (~40 lignes), backward-compatible, utile pour notebooks |
| **Breaking changes** | Acceptés avec documentation | Nécessaire pour cohérence, exemples mis à jour, migration guidée |

### Structure proposée

#### Nouveaux modules

```python
# seapopym/compiler/units.py
class UnitValidator:
    """Validation stricte des unités avec Pint."""

    def __init__(self, ureg: pint.UnitRegistry | None = None):
        """Initialize with optional custom registry."""

    def check_exact_match(self, unit1: str, unit2: str, context: str) -> None:
        """Vérifie identité stricte des formes canoniques Pint.

        Exemples:
            "m/s" == "meter/second"     → OK (même forme canonique)
            "m * s**-1" == "m/s"        → OK (même forme canonique)
            "m/s" == "km/s"             → ERREUR (formes différentes)
            "1/d" == "1/s"              → ERREUR (formes différentes)
        """

    def validate_process_chain(
        self,
        blueprint: Blueprint,
        resolved_functions: dict[str, FunctionMetadata]
    ) -> list[UnitError]:
        """Valide toute la chaîne : inputs → fonction → outputs."""
```

#### Modules modifiés

```python
# seapopym/compiler/compiler.py
class Compiler:
    def __init__(
        self,
        backend: Backend = "jax",
        forcing_interpolation: str = "constant",  # NOUVEAU
        ...
    ):
        """
        Args:
            forcing_interpolation: Method for temporal interpolation
                - "constant": Broadcast static forcings (default, existing behavior)
                - "nearest": Nearest neighbor for under-sampled forcings
                - "linear": Linear interpolation
                - "ffill": Forward fill
        """

    def _prepare_forcings(self, config, dim_mapping, shapes):
        """Prepare forcings with optional interpolation.

        Logic:
        1. If forcing has no T dimension → broadcast (constant)
        2. If forcing.shape[0] == n_timesteps → slice directly
        3. If forcing.shape[0] < n_timesteps → interpolate
        """

# seapopym/blueprint/schema.py
class ExecutionParams(BaseModel):
    dt: str = "1d"
    forcing_interpolation: str = "constant"  # NOUVEAU
    output_path: str | None = None

# seapopym/blueprint/validation.py
class BlueprintValidator:
    def _validate_units(self, blueprint, result):
        """Use UnitValidator for strict checking.

        Changes:
        - Warnings → Errors
        - String comparison → Pint canonical form comparison
        - Chain validation: output of A → input of B
        """

# seapopym/engine/runners.py
class StreamingRunner:
    def run(self, output_path: str | Path | None = None) -> tuple[State, Outputs | None]:
        """Execute with optional disk output.

        Args:
            output_path: If None, accumulate outputs in memory and return.
                        If provided, write to Zarr asynchronously.

        Returns:
            (final_state, outputs_dict_or_None)
        """

# seapopym/compiler/exceptions.py
class UnitError(ValidationError):
    """Raised when units are incompatible or incorrect."""
```

### Arborescence des modifications

```
seapopym/
├── compiler/
│   ├── units.py              # NOUVEAU - Validation stricte Pint
│   ├── compiler.py           # MODIFIÉ - Interpolation temporelle
│   └── exceptions.py         # MODIFIÉ - UnitError
├── blueprint/
│   ├── validation.py         # MODIFIÉ - Intégrer UnitValidator
│   └── schema.py             # MODIFIÉ - forcing_interpolation
└── engine/
    └── runners.py            # MODIFIÉ - output_path=None optionnel

tests/
├── compiler/
│   └── test_units.py         # NOUVEAU - Tests validation unités
└── engine/
    └── test_runners.py       # MODIFIÉ - Tests in-memory mode

examples/
├── predator_prey_0d.py       # MODIFIÉ - Unités en /s
└── predator_prey_2d_benchmark.py  # MODIFIÉ - Unités en /s
```

### Interfaces et contrats

#### Interface 1 : UnitValidator

```python
# Entrée : Blueprint + resolved_functions
# Sortie : list[UnitError] (vide si valide)
# Règle : Formes canoniques doivent être identiques (pas juste compatibles)

validator = UnitValidator()
errors = validator.validate_process_chain(blueprint, resolved_functions)
if errors:
    raise errors[0]  # Fail-fast
```

#### Interface 2 : Forcing Interpolation

```python
# Cas 1 : Forcing statique (pas de dim T)
#   → Broadcast (comportement actuel conservé)

# Cas 2 : Forcing temporel aligné (shape[0] == n_timesteps)
#   → Slicing direct (comportement actuel conservé)

# Cas 3 : Forcing sous-échantillonné (shape[0] < n_timesteps)
#   → Interpolation selon méthode configurée (NOUVEAU)
```

#### Interface 3 : In-Memory Outputs

```python
# output_path = "/path/to/output.zarr" (défaut actuel)
#   → AsyncWriter, écriture Zarr, return (state, None)

# output_path = None (nouveau)
#   → Accumulation en mémoire, return (state, outputs_dict)
```

### Règles de validation des unités

#### Règle 1 : Unités par défaut
- Si `units=None` dans VariableDeclaration → `"dimensionless"`
- Pint : `ureg.dimensionless` (unité sans dimension)

#### Règle 2 : Validation des inputs
Pour chaque ProcessStep :
```python
for arg_name, var_path in step.inputs.items():
    var_unit = blueprint.get_variable(var_path).units
    expected_unit = function_metadata.units[arg_name]

    validator.check_exact_match(var_unit, expected_unit, context=f"{step.func}.{arg_name}")
```

#### Règle 3 : Validation des outputs
```python
for out_key, out_spec in step.outputs.items():
    func_return_unit = function_metadata.units["return"]
    target_unit = blueprint.get_variable(out_spec.target).units

    if target_unit is not None:  # Si déclaré
        validator.check_exact_match(func_return_unit, target_unit, ...)
```

#### Règle 4 : Validation des tendencies
```python
if out_spec.type == "tendency":
    # La tendency DOIT avoir une dimension temporelle en /s
    if not func_return_unit.dimensionality == "[time]^-1 * [...]":
        raise UnitError("Tendencies must have time dimension in seconds (/s)")
```

#### Règle 5 : Chaîne de dépendances
```python
# Sortie de fonction A utilisée comme entrée de fonction B
output_A = compute_node_A.output_mapping["result"]  # Ex: "derived.temperature"
input_B = compute_node_B.input_mapping["temp"]      # Ex: "derived.temperature"

if output_A == input_B:
    unit_A = function_A.units["return"]
    unit_B = function_B.units["temp"]
    validator.check_exact_match(unit_A, unit_B, context="chain A→B")
```

### Risques identifiés

| Risque | Impact | Probabilité | Mitigation |
|--------|--------|-------------|------------|
| **Pint ajoute overhead compilation** | Faible | Faible | Validation uniquement au compile-time, aucun impact runtime. Benchmark si nécessaire. |
| **Breaking changes cassent workflows existants** | Haut | Certain | Documentation migration, exemples mis à jour, messages d'erreur explicites. |
| **Interpolation consomme beaucoup de mémoire** | Moyen | Moyen | Interpolation par chunk si nécessaire, option désactivable. |
| **Outputs in-memory cause OOM** | Haut | Moyen | Documentation claire sur limites, warning si taille > seuil. |
| **Pint ne reconnaît pas unités custom** | Faible | Faible | Possibilité d'étendre le registry, fallback documenté. |
| **Forme canonique Pint différente attendue** | Moyen | Faible | Documentation des formes acceptées, tests exhaustifs. |

### Breaking changes documentés

#### 1. Validation des unités (Warnings → Errors)

**Ancien comportement :**
```python
# Blueprint
"parameters": {"rate": {"units": "1/d"}}

# Config
"rate": {"value": 0.05}

# Résultat : Warning dans les logs, exécution continue
```

**Nouveau comportement :**
```python
# Résultat : UnitError à la compilation
UnitError: Parameter 'rate' unit mismatch in 'demo:growth'
  Blueprint declares: '1/d' (1/day)
  Function expects: '1/s' (1/second)
  → Units must be identical (not just compatible)
  → Update Blueprint to declare '1/s' or change function signature
```

#### 2. Convention temporelle universelle

**Ancien comportement :**
- Unités temporelles ambiguës (1/d, 1/h, etc.)
- Conversion manuelle par l'utilisateur (division par 86400)

**Nouveau comportement :**
- **TOUS les temps en secondes**
- Blueprint doit déclarer en `/s`
- Erreur si autre unité temporelle détectée

**Migration :**
```python
# Avant
"parameters": {
    "prey_growth_rate": {"units": "1/d"}
}
config = {"prey_growth_rate": {"value": 0.05 / 86400}}

# Après
"parameters": {
    "prey_growth_rate": {"units": "1/s"}
}
config = {"prey_growth_rate": {"value": 0.05 / 86400}}  # Même valeur numérique
```

#### 3. Fichiers impactés

**Exemples à mettre à jour :**
- `examples/predator_prey_0d.py` : Lignes 73-76 (Blueprint), ligne 130-135 (Config)
- `examples/predator_prey_2d_benchmark.py` : Même pattern

**Tests potentiellement impactés :**
- `tests/blueprint/test_validation.py` : Tests avec unités temporelles
- Tous les fixtures utilisant des unités de temps

#### 4. Guide de migration

**Pour les utilisateurs existants :**

1. **Audit du Blueprint :**
   - Identifier tous les paramètres avec dimension temporelle
   - Remplacer `"1/d"`, `"1/h"` par `"1/s"`
   - Remplacer `"d"`, `"h"` par `"s"`

2. **Vérifier les Config :**
   - Les valeurs numériques doivent correspondre aux nouvelles unités
   - Si avant `0.05 / 86400` pour `"1/d"` → garder `0.05 / 86400` pour `"1/s"`

3. **Exécuter validation :**
   ```python
   from seapopym.compiler import compile_model

   try:
       compiled = compile_model(blueprint, config, validate=True)
   except UnitError as e:
       print(f"Unit mismatch: {e}")
       # Corriger et réessayer
   ```

4. **Tests :**
   - Tous les tests doivent passer après migration
   - Nouveaux tests de validation ajoutés automatiquement

### Implémentation par priorité

#### Phase 1 - Validation unités (URGENT)
1. Créer `seapopym/compiler/units.py` avec `UnitValidator`
2. Ajouter `UnitError` dans `exceptions.py`
3. Modifier `validation.py` pour intégrer `UnitValidator`
4. Écrire tests unitaires (`test_units.py`)
5. Mettre à jour exemples (predator_prey_*.py)
6. Documentation migration

#### Phase 2 - Interpolation temporelle (IMPORTANT)
1. Ajouter `forcing_interpolation` dans `ExecutionParams`
2. Modifier `Compiler._prepare_forcings()` pour interpolation
3. Tests d'interpolation (linéaire, nearest, ffill)
4. Documentation configuration

#### Phase 3 - Outputs in-memory (EXPLORATOIRE)
1. Modifier `StreamingRunner.run()` pour supporter `output_path=None`
2. Implémenter accumulation et concatenation
3. Tests de charge mémoire
4. Documentation limites

## Todo List

_À compléter après l'étape 4 (Planification)_

## Historique des transitions

| De                | Vers            | Raison                               | Date       |
| ----------------- | --------------- | ------------------------------------ | ---------- |
| 1. Initialisation | 2. Analyse      | Besoin validé par l'utilisateur      | 2026-01-26 |
| 2. Analyse        | 3. Architecture | Analyse complétée                    | 2026-01-26 |
