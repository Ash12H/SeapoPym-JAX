# Workflow State — Brique 3 : Intégration BlackJAX NUTS

## Informations generales

- **Projet** : SeapoPym v0.2 — Chaîne CMA-ES → NUTS
- **Brique** : 3/5 — Intégration BlackJAX NUTS
- **Etape courante** : 1. Initialisation
- **Role actif** : Facilitateur
- **Derniere mise a jour** : 2026-02-13

## Objectif global

Implémenter les briques techniques manquantes pour la chaîne d'estimation CMA-ES → NUTS décrite dans l'article 2 de SeapoPym, et produire des exemples tests démontrant leur fonctionnement.

## Resume du besoin

Intégrer le sampler NUTS de BlackJAX dans SeapoPym. Connecter la `log_posterior` (brique 2) et son gradient à `blackjax.nuts`. Gérer le warmup (window adaptation), extraire les échantillons et les diagnostics (divergences, acceptance rate). Créer un runner cohérent avec l'architecture existante. Produire un exemple test (modèle jouet → corner plot).

**Périmètre :**
- IN : Wrapper BlackJAX NUTS, warmup, diagnostics, exemple test avec corner plot
- OUT : Pas de pipeline CMA-ES → NUTS (brique 5)

**Dépendances amont :** Brique 2 (log-postérieure)
**Dépendances aval :** Brique 5 (pipeline)

## Decisions d'architecture

[A définir à l'étape 3]

## Todo List

[A définir à l'étape 4]

## Historique des transitions

| De | Vers | Raison | Date |
|----|------|--------|------|
