# Rapport de Bug : Incohérence Temporelle avec Interpolation des Forçages

## Description du Problème

En essayant de faire tourner une simulation 0D avec un pas de temps (`dt`) plus fin que la résolution temporelle des forçages d'entrée, nous avons observé que **la simulation s'arrête prématurément**.

### Exemple Concret

- **Données d'entrée** : Forçages journaliers sur 20 ans ($N_{data} = 7300$ valeurs).
- **Configuration** : `dt = "0.05d"` (20 pas par jour) et `forcing_interpolation = "linear"`.
- **Comportement Attendu** : Le modèle interpole les forçages pour générer $7300 \times 20 = 146,000$ pas de temps et couvre les 20 ans.
- **Comportement Observé** : Le modèle s'exécute sur 7300 pas seulement ($\approx 1$ an à ce $dt$) et s'arrête.

L'interpolation temporelle, bien qu'activée dans la configuration, ne semble pas prise en compte pour déterminer la durée totale de la simulation.

## Analyse Technique

L'analyse du code source (`seapopym/compiler`) révèle la cause racine :

1. **Inférence des Formes (`infer_shapes`)** :
   Le compilateur détermine la taille de la dimension temporelle (`T`) en inspectant la forme des données d'entrée (`config.forcings`).
    - Il lit que les forçages ont une taille de **7300**.
    - Il fixe donc la dimension canonique `T = 7300` dans le modèle compilé.

2. **Logique d'Interpolation (`_prepare_forcings`)** :
   L'interpolation n'est déclenchée que si la taille du tableau d'entrée diffère de la dimension `T` inférée.

    ```python
    # Dans compiler.py
    n_timesteps = shapes.get("T", 1)  # Vaut 7300 (inféré depuis les fichiers)
    if arr.shape[0] != n_timesteps:   # 7300 == 7300 -> Faux
        interpolate(...)
    ```

    Comme `T` a été déduit des fichiers eux-mêmes, la condition est toujours fausse quand tous les fichiers ont la même résolution.

3. **Conséquence** :
   Le modèle est compilé avec `T=7300`. Le solveur exécute 7300 itérations. Comme `dt=0.05d`, cela représente seulement $7300 \times 0.05 = 365$ jours physiques.

## Pistes de Solution (Pour l'Agent ASH)

Le nœud du problème est que **la durée de la simulation est implicitement dictée par la taille des fichiers**, alors qu'elle devrait pouvoir être définie explicitement (par `time_range` ou `duration`) pour permettre le sur-échantillonnage (interpolation) ou sous-échantillonnage.

### Questions à résoudre :

1. Comment définir la **dimension `T` cible** indépendamment des fichiers d'entrée ?
    - Utiliser `config.execution.time_range` ?
    - Ajouter un champ `n_timesteps` ?
2. Comment gérer le conflit dans `infer_shapes` qui lève actuellement une `GridAlignmentError` si les tailles ne correspondent pas ?
    - Faut-il assouplir la validation pour la dimension `T` ?
3. Quelle est la priorité entre "Taille des données" et "Configuration Explicite" ?

## Reproduction

Le script `examples/full_model_0d.py` permet de reproduire le problème.

- Avec `dt="1d"`, tout fonctionne (20 ans simulés).
- Avec `dt="0.05d"` + `forcing_interpolation="linear"`, la simulation s'arrête après 1 an.
