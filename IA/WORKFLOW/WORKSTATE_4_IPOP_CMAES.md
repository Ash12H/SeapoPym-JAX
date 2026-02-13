# Workflow State — Brique 4 : IPOP-CMA-ES

## Informations generales

- **Projet** : SeapoPym v0.2 — Chaîne CMA-ES → NUTS
- **Brique** : 4/5 — IPOP-CMA-ES (restarts automatisés)
- **Etape courante** : 1. Initialisation
- **Role actif** : Facilitateur
- **Derniere mise a jour** : 2026-02-13

## Objectif global

Implémenter les briques techniques manquantes pour la chaîne d'estimation CMA-ES → NUTS décrite dans l'article 2 de SeapoPym, et produire des exemples tests démontrant leur fonctionnement.

## Resume du besoin

Automatiser les restarts CMA-ES avec populations croissantes (IPOP-CMA-ES, Auger & Hansen 2005) au-dessus du wrapper `evolutionary.py` existant. Collecter les modes distincts en filtrant par distance dans l'espace des paramètres et par valeur de loss. Produire un exemple test sur une fonction bimodale synthétique.

**Périmètre :**
- IN : IPOP-CMA-ES wrapper, détection et collecte des modes distincts, exemple test bimodal
- OUT : Pas de couplage avec NUTS (brique 5)

**Dépendances amont :** Aucune (indépendant des briques 1-2-3)
**Dépendances aval :** Brique 5 (pipeline)

## Decisions d'architecture

[A définir à l'étape 3]

## Todo List

[A définir à l'étape 4]

## Historique des transitions

| De | Vers | Raison | Date |
|----|------|--------|------|
