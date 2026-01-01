# Modifications Suggérées — Introduction et Méthodes

Suite à la finalisation des résultats expérimentaux, voici les ajustements proposés pour assurer la cohérence entre les sections.

---

## Modifications pour l'Introduction (01_Introduction_Draft.md)

### Ajout suggéré — Fin du paragraphe "Proposition"

Après la phrase actuelle sur les benchmarks, ajouter :

> *"Plus spécifiquement, nous validons (1) la reproduction des solutions analytiques pour les processus biologiques et le transport, (2) la convergence numérique du couplage transport-biologie, (3) la non-régression par rapport aux implémentations existantes (SeapoPym v0.3 et Seapodym-LMTL), et (4) la complexité algorithmique linéaire O(N) du système."*

### Cohérence vérifiée ✓

L'introduction actuelle correspond bien aux résultats :
- ✓ Mention de SeapoPym v0.3 (comparaison planifiée)
- ✓ Mention du transport comme défi (validé par nos expériences)
- ✓ Promesse de benchmarks (12 figures produites)

---

## Modifications pour Matériel et Méthodes (02_Methode_DAG.md)

### Ajout suggéré — Section 2.3 (Stack Technologique)

La section actuelle mentionne Dask mais ne précise pas les schedulers utilisés. Ajouter :

> *"Le backend Dask supporte deux modes d'exécution : le **ThreadPoolScheduler** pour l'exécution parallèle sur une machine unique (exploitant la mémoire partagée), et le **Distributed Client** pour les clusters multi-nœuds. Les résultats présentés dans cet article utilisent principalement le ThreadPoolScheduler, adapté aux workstations scientifiques."*

### Ajout suggéré — Section 2.1 (Volumes Finis)

Préciser la condition CFL avec la valeur utilisée :

> *"Le pas de temps est contraint par la condition CFL pour garantir la stabilité : $\text{CFL} = \frac{u \cdot \Delta t}{\Delta x} < 1$. Dans nos expériences, nous utilisons CFL = 0.5 comme compromis entre stabilité et efficacité."*

### Cohérence vérifiée ✓

Le reste de la section Méthodes correspond aux résultats :
- ✓ Description du DAG (validé par Fig 1A, 1B)
- ✓ Schéma Upwind mentionné (validé par Fig 2A, 3D)
- ✓ Conservation de masse (validée par Fig 2B)
- ✓ Parallélisme Dask (validé par Fig 4C, 4E)

---

## Section Discussion — Points à Développer

### 1. Limites Identifiées

> *"Le speedup Strong Scaling est limité (~1.7×) par la dominance du module de transport (88% du temps de calcul). Conformément à la Loi d'Amdahl, une parallélisation inter-tâches ne peut dépasser ~1.2× dans cette configuration."*

### 2. Recommandations

> *"Plusieurs pistes d'optimisation sont envisageables :*
> - *Parallélisation intra-tâche du transport via chunking spatial (Dask Arrays)*
> - *Décomposition de domaine pour les très grandes grilles*
> - *Utilisation de schémas implicites pour relaxer la contrainte CFL*"

### 3. Perspectives

> *"L'architecture DAG ouvre la voie à l'intégration de nouveaux processus (prédation, migration verticale) sous forme de nœuds fonctionnels, sans modification du cœur du contrôleur. Cette modularité facilite également le couplage avec des modèles externes (biogéochimie, hydrodynamique)."*

---

## Checklist de Cohérence

| Élément | Intro | Méthodes | Résultats | Status |
|---------|-------|----------|-----------|--------|
| DAG | ✓ Mentionné | ✓ Décrit | ✓ Validé indirectement | OK |
| SeapoPym v0.3 | ✓ Mentionné | ✓ Référencé | [ En attente ] | À compléter |
| Transport FVM | ✓ Mentionné | ✓ Décrit | ✓ Validé (Fig 2) | OK |
| Couplage | ✓ Promis | ✓ Expliqué | ✓ Validé (Fig 3) | OK |
| Parallélisme | ✓ Promis | ✓ Décrit | ✓ Analysé (Fig 4) | OK |
| Conservation masse | — | ✓ Mentionné | ✓ Validé (Fig 2B) | OK |
| Complexité O(N) | — | — | ✓ Mesuré (Fig 4A) | À ajouter en Méthodes |

---

## Actions Requises

1. **Intro** : Ajouter 1 phrase précisant les 4 axes de validation
2. **Méthodes** : Ajouter précisions sur CFL et schedulers Dask
3. **Résultats** : Compléter les placeholders après comparaisons LMTL/SeapoPym
4. **Discussion** : Rédiger à partir des points ci-dessus
