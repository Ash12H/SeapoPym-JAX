# Feuille de Route : Architecture Multi-Groupes et Multi-Forçages

Cette feuille de route détaille les étapes de développement pour transformer `seapopym-message` en un framework multi-espèces générique.

## Phase 1 : Refonte du Système de Forçages (Fondations)

**Changement de stratégie :** Le `ForcingManager` reste "agnostique" et se contente de fournir les données brutes (potentiellement 3D/4D). La logique de réduction (ex: moyenne jour/nuit) est déplacée dans des **Unités de Perception** au sein du Kernel.

### 1.1. Abstraction des Sources de Données
- [ ] Créer une classe `ForcingSource` qui encapsule un `xarray.DataArray`.
- [ ] Implémenter la gestion des dimensions pour supporter les champs N-dimensionnels (ex: `(depth, lat, lon)`).

### 1.2. Mise à jour du `ForcingManager`
- [ ] Modifier `ForcingManager` pour qu'il accepte et distribue ces champs N-dimensionnels sans chercher à les réduire.
- [ ] S'assurer que le passage de données via Ray (Zero-Copy) est optimal pour les gros tableaux 3D.

---

## Phase 2 : Architecture des Groupes Fonctionnels

### 2.1. Classe `FunctionalGroup`
- [ ] Créer la classe `FunctionalGroup` (dataclass).
- [ ] Attributs :
    - `name`: str (ex: "tuna")
    - `units`: list[Unit] (les comportements, incluant la perception)
    - `variable_map`: dict[str, str] (mapping interne -> externe)
    - `params`: dict[str, float] (paramètres spécifiques)

### 2.2. Refonte de la classe `Unit`
- [ ] Séparer la définition des variables (signature) de leur nom dans le state global.
- [ ] Ajouter un mécanisme pour "binder" une Unit à un Groupe lors de la construction du Kernel.

### 2.3. Création d'Unités de Perception (Sensing Units)
- [ ] Créer des unités génériques pour manipuler les forçages :
    - `ExtractLayerUnit` : Extrait une couche 2D d'un champ 3D à un index donné.
    - `DielMigrationUnit` : Calcule une moyenne pondérée entre deux couches (Jour/Nuit) en fonction de la durée du jour.
    - Ces unités écriront leur résultat dans des variables temporaires du State (ex: `tuna/temp_effective`) qui seront consommées par les unités biologiques suivantes.

---

## Phase 3 : Le "Kernel Compiler"

L'objectif est de transformer le Kernel pour qu'il comprenne les Groupes et valide l'intégrité du système.

### 3.1. Construction du DAG (Graphe de Dépendances)
- [ ] Modifier le constructeur de `Kernel` pour accepter une liste de `FunctionalGroup`.
- [ ] **Validation Statique :**
    - Vérifier la cohérence des dimensions.
    - Vérifier l'absence de caractères illégaux dans les noms de variables.
    - Vérifier que toutes les dépendances (inputs) sont satisfaites (soit par le state, soit par un forçage).
- [ ] **Visualisation :**
    - Ajouter une méthode `Kernel.visualize_graph(output_file)` pour générer une image du graphe (ex: via Graphviz).

### 3.2. Exécution
- [ ] Implémenter la logique d'aplatissement (Namespacing avec `/`).
    - Ex: `biomass` -> `tuna/biomass`.
- [ ] Vérifier que l'exécution topologique fonctionne sur le graphe élargi.

---

## Phase 4 : Migration du Modèle Zooplancton

### 4.1. Définition des Groupes
- [ ] Configurer un `FunctionalGroup` nommé "zooplankton".
- [ ] Ajouter une unité de perception simple (ex: `ExtractLayerUnit` si on utilise une température 3D, ou rien si 2D).
- [ ] Mapper ses variables :
    - `biomass` -> `zooplankton/biomass`
    - `production` -> `zooplankton/production`

### 4.2. Validation
- [ ] Valider manuellement que les résultats sont identiques à l'ancienne version (non-régression).

---

## Phase 5 : Extension Multi-Couches

### 5.1. Implémentation du cas "Jour/Nuit"
- [ ] Utiliser l'unité `DielMigrationUnit` créée en Phase 2.
- [ ] Configurer le groupe pour utiliser cette unité afin de calculer sa `temperature_effective`.

### 5.2. Simulation Multi-Groupes
- [ ] Créer une simulation avec 2 groupes de zooplancton :
    - Groupe A : Surface (0-100m)
    - Groupe B : Profond (200-500m) ou Migrant.

---

## Phase 6 : Transport (Mise à jour)

### 6.1. Transport Multi-Champs
- [ ] S'assurer que le `TransportWorker` peut gérer dynamiquement la liste des champs générée par les groupes (ex: `tuna/biomass`, `shark/biomass`).
