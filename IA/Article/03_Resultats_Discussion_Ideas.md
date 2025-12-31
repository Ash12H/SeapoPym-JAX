# Plan des Expériences de Validation (Résultats)

Cette section liste les expériences numériques conçues pour valider les différents aspects de la nouvelle architecture DAG (Exactitude, Couplage, Performance).

## 1. Validation de l'Exactitude Numérique (Unitaire)
**Objectif :** Valider séparément les composants biologiques et physiques par rapport à des solutions de référence.

*   **Expérience 1.1 : Biologie (Réplication v0.3)**
    *   *Protocole* : Simulation 0D d'une population sous température constante.
    *   *Comparaison* : Solution DAG vs Solution Analytique ($B_{eq}$) vs Résultats SeapoPym v0.3.
    *   *Validation visée* : Convergence identique, prouvant la non-régression par rapport à l'implémentation précédente.

*   **Expérience 1.2 : Physique (Transport Analytique)**
    *   *Protocole* : Advection 1D/2D d'une distribution connue (ex: Gaussienne).
    *   *Comparaison* : Solution Numérique vs Solution Analytique Exacte et Vérification de la Conservation de la Masse.
    *   *Validation visée* : Quantification de l'erreur numérique (diffusion num.) et preuve de la conservation stricte.

## 2. Validation du Couplage Transport-Réaction
**Objectif :** Démontrer que l'architecture unifiée gère correctement l'interaction simultanée (sans biais de "Time Splitting").

*   **Expérience 2.1 : Le "Patch Test" Advecté-Réatif (Théorique)**
    *   *Protocole* : Advection d'une biomasse soumise à une mortalité exponentielle connue.
    *   *Comparaison* : Simulation DAG vs Solution Théorique ($Solution_{transport}(x,t) \times e^{-\lambda t}$).
    *   *Validation visée* : Correspondance parfaite, validant l'algorithme de somme des flux.

## 3. Validation de la Performance et de l'Ingénierie
**Objectif :** Démontrer la scalabilité sur un cas réaliste (pas de fonctions fictives).

*   **Expérience 3.1 : Scalabilité sur Modèle Réel (LMTL Global)**
    *   *Protocole* : Exécuter le modèle complet (Biologie + Transport Global 1/12°) (ou une sous partie significative).
    *   *Mesure* : "Weak Scaling" (Augmenter la taille du domaine avec le nombre de cœurs) ou "Strong Scaling" (Domaine fixe, plus de cœurs).
    *   *Validation visée* : Montrer que le surcoût du DAG est amorti par la parallélisation efficace des tâches lourdes (Transport, Recrutement multi-cohortes).

---
# Discussion (Idées Clés)

*   **Extensibilité** : L'ajout d'un prédateur se fait par ajout de nœud, sans refonte du code.
*   **Arbitrage Précision/Coût** : Le schéma explicite (DAG) est plus coûteux en pas de temps (CFL faible) que les schémas implicites, mais permet cette parallélisation massive et cette modularité.
*   **Intégration** : Capacité à remplacer des modules de SEAPODYM C++ progressivement ou à servir de "Digital Twin" pour tester des hypothèses rapides.
