# Workflow State — Brique 5 : Pipeline CMA-ES → NUTS

## Informations generales

- **Projet** : SeapoPym v0.2 — Chaîne CMA-ES → NUTS
- **Brique** : 5/5 — Pipeline CMA-ES → NUTS
- **Etape courante** : 1. Initialisation
- **Role actif** : Facilitateur
- **Derniere mise a jour** : 2026-02-13

## Objectif global

Implémenter les briques techniques manquantes pour la chaîne d'estimation CMA-ES → NUTS décrite dans l'article 2 de SeapoPym, et produire des exemples tests démontrant leur fonctionnement.

## Resume du besoin

Assembler les briques 3 (NUTS) et 4 (IPOP-CMA-ES) en un pipeline complet : IPOP-CMA-ES identifie les modes, puis NUTS échantillonne autour de chaque mode. Produire un exemple test sur LMTL 0D (twin experiment) démontrant la chaîne de bout en bout avec corner plots.

**Périmètre :**
- IN : Orchestration CMA-ES → NUTS, initialisation NUTS depuis les modes, N chaînes par mode, exemple LMTL 0D
- OUT : Pas d'expériences pour l'article (0D vs 2D, cas géographiques)

**Dépendances amont :** Briques 3 (NUTS) et 4 (IPOP-CMA-ES)
**Dépendances aval :** Aucune (dernière brique)

## Decisions d'architecture

[A définir à l'étape 3]

## Todo List

[A définir à l'étape 4]

## Historique des transitions

| De | Vers | Raison | Date |
|----|------|--------|------|
