# Rapport de Tests : Chunking et Efficacité Dask

**Date** : 12 Janvier 2026
**Statut Global** : ⚠️ Correctness validée mais problèmes d'efficacité détectés.

## ✅ Ce qui fonctionne (Correctness)

Tous les tests de validité numérique passent. Le découpage des données (chunking) ne fausse pas les calculs.

| Module        | Fonction              | Test Chunking Cohorte | Test Chunking Spatial  | Test Chunking Temps |
| :------------ | :-------------------- | :-------------------: | :--------------------: | :-----------------: |
| **LMTL**      | `production_dynamics` |          ✅           |           ✅           |         N/A         |
| **LMTL**      | `mortality_tendency`  |          N/A          |           ✅           |         N/A         |
| **LMTL**      | `recruitment_age`     |          N/A          |           ✅           |         ✅          |
| **Transport** | `transport_xarray`    |          N/A          |           ✅           |         ✅          |
| **Transport** | `transport_numba`     |          N/A          | ❌ (Limitation connue) |         ✅          |

_Note : Le transport Numba ne supporte pas le chunking spatial sur les dimensions internes (lat/lon), ce qui est attendu._

## ❌ Ce qui ne marche pas (Efficiency)

Des opérations de **rechunking implicite** ont été détectées. Cela signifie que Dask doit réorganiser la mémoire entre les workers à chaque étape, ce qui est très coûteux et nuit à la scalabilité.

### 1. Module LMTL (`compute_production_dynamics`)

-   **Symptôme** : Rechunking détecté lors du calcul.
-   **Cause probable** : L'utilisation de `.shift()` ou `.reindex()` pour aligner les cohortes ou calculer `d_tau` (durée des cohortes).
-   **Impact** : Graph de tâches inutilement complexe, risque de saturation mémoire sur cluster.

### 2. Module Transport

-   **`compute_transport_xarray`** : ❌ **PROBLÈME DÉTECTÉ**. Rechunking implicite présent.
-   **`compute_transport_numba`** : ✅ **VALIDE**. Pas de rechunking implicite détecté. C'est la version recommandée pour la production.

## 🚀 Recommandations

1.  **LMTL** : Remplacer `reindex/shift` par des opérations de padding ou broadcasting manuel pour `d_tau`.
2.  **Transport** : Utiliser `map_overlap` ou `pad` explicitement avant les calculs de flux Xarray pour éviter le rechunking automatique désordonné.
