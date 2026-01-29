# Workflow State - Migration Scripts Article

## Informations générales

- **Projet** : Migration des scripts article/ vers nouvelle architecture SeapoPym
- **Étape courante** : 5. Exécution
- **Rôle actif** : Développeur
- **Dernière mise à jour** : 2026-01-29

## Résumé du besoin

Migrer les anciens scripts de benchmarking/validation (article/notebooks/) vers la nouvelle architecture SeapoPym avec backend JAX.

## Todo List

### Phase 0 : Préparation

| État | ID | Nom | Description | Dépendances |
|------|----|-----|-------------|-------------|
| ☑ | T0.1 | Déplacer article/_legacy | Déplacer le contenu actuel de `article/` vers `article/_legacy/` | - |
| ☑ | T0.2 | Créer structure article/ | Créer nouvelle structure : `article/notebooks/`, `article/data/`, `article/figures/` | T0.1 |

### Phase 1 : Scripts Biologie 0D

| État | ID | Nom | Description | Dépendances |
|------|----|-----|-------------|-------------|
| ☑ | T1.1 | Migrer article_01a | Migrer `article_01a_bio_0d_temperature_scan.py` - Test bio 0D avec scan température | T0.2 |

### Phase 2 : Comparaison versions

| État | ID | Nom | Description | Dépendances |
|------|----|-----|-------------|-------------|
| ☑ | T2.1 | Migrer article_02a | Migrer `article_02a_comparison_seapopym_v0.3.py` - Comparaison avec ancienne version | T1.1 |

### Phase 3 : Transport 2D

| État | ID | Nom | Description | Dépendances |
|------|----|-----|-------------|-------------|
| ☑ | T3.1 | Créer benchmark Zalesak JAX | Créer `examples/transport_zalesak_jax.py` | - |
| ☑ | T3.2 | Migrer article_03a | Migrer `article_03a_transport_2d_zalesak.py` vers JAX (déjà fait: T3.1) | T3.1 |

### Phase 4 : Performance / Scaling

| État | ID | Nom | Description | Dépendances |
|------|----|-----|-------------|-------------|
| ☐ | T4.1 | Migrer article_04a | Migrer `article_04a_weak_scaling.py` - Tests de scaling | T0.2 |
| ☐ | T4.2 | Migrer article_04b | Migrer `article_04b_time_decomposition.py` - Décomposition temporelle | T4.1 |

### Phase 5 : Simulation Pacific

| État | ID | Nom | Description | Dépendances |
|------|----|-----|-------------|-------------|
| ☑ | T5.1 | Migrer article_05a | Migrer `article_05a_prepare_data_pacific.py` - Préparation données | T0.2 |
| ☑ | T5.2 | Migrer article_05b | Migrer `article_05b_simulation_pacific.py` - Simulation principale (JAX) | T5.1 |
| ☑ | T5.3 | Migrer article_05c | Migrer `article_05c_prepare_seapodym_forcings.py` - Préparation forçages | T5.1 |
| ☑ | T5.4 | Évaluer article_05d | Script externe (binaire C++), copié avec chemins mis à jour | T5.2 |
| ☑ | T5.5 | Migrer article_05e | Migrer `article_05e_postprocess_seapodym.py` - Post-traitement | T5.2 |
| ☑ | T5.6 | Migrer article_05f | Migrer `article_05f_comparison_pacific.py` - Comparaison résultats (JAX) | T5.5 |

### Phase 6 : Bonus

| État | ID | Nom | Description | Dépendances |
|------|----|-----|-------------|-------------|
| ☑ | T6.1 | Migrer article_06 | Migrer `article_06_bonus_animation_pacific.py` - Animation (JAX) | T5.6 |

## Ordre de priorité suggéré

1. **T0.1-T0.2** : Préparation structure
2. **T3.2** : Valider transport (déjà fait avec T3.1)
3. **T1.1** : Bio 0D (base pour le reste)
4. **T5.1-T5.2** : Simulation Pacific (cas d'usage principal)
5. **T4.1-T4.2** : Performance (pour l'article)
6. **T2.1, T5.3-T5.6, T6.1** : Le reste

## Notes techniques

- Les scripts legacy utilisent `seapopym.transport.numba_kernels` → remplacer par `seapopym.functions.transport`
- Les scripts legacy utilisent Numba guvectorize → remplacer par JAX vmap/jit
- Conserver les mêmes paramètres de test pour comparaison valide

## Fonctions LMTL créées

Fichier : `seapopym/functions/lmtl.py`

| Fonction | Description | Ajoutée pour |
|----------|-------------|--------------|
| `threshold_temperature` | Seuil température minimum | T1.1 |
| `gillooly_temperature` | Normalisation Gillooly T/(1+T/273) | T1.1 |
| `recruitment_age` | Âge de recrutement τ_r | T1.1 |
| `mortality_tendency` | Tendance mortalité | T1.1 |
| `npp_injection` | Injection NPP → cohort 0 | T1.1 |
| `aging_flow` | Flux vieillissement cohortes | T1.1 |
| `recruitment_flow` | Flux recrutement → biomasse | T1.1 |
| `day_length` | Durée du jour (latitude, DOY) | T2.1 |
| `layer_weighted_mean` | Moyenne pondérée jour/nuit sur Z | T2.1 |

## Résultats de validation

| Script | Métrique clé | Valeur | Statut |
|--------|--------------|--------|--------|
| article_01a | Erreur max | 0.001% | ✅ |
| article_02a | Corrélation vs v0.3 | 0.9996 | ✅ |
| article_02a | Erreur L2 | 2.5% | ✅ |

## Scripts migrés Phase 5-6

| Script | Description | Changements clés |
|--------|-------------|------------------|
| article_05a | Préparation données Pacific | xarray standard, chemins mis à jour |
| article_05b | Simulation Pacific | Blueprint/Config/compile_model/StreamingRunner, fonctions JAX |
| article_05c | Préparation forçages NetCDF | xarray standard, chemins mis à jour |
| article_05d | Benchmark C++ externe | Script externe, chemins mis à jour uniquement |
| article_05e | Post-traitement Seapodym | xarray standard |
| article_05f | Comparaison résultats | Adapté pour fichiers JAX (_jax.zarr) |
| article_06 | Animation GIF | Adapté pour fichiers JAX |

## Historique des transitions

| De | Vers | Raison | Date |
|----|------|--------|------|
| - | 4. Planification | Nouveau workflow créé | 2026-01-29 |
| 5. Exécution | 5. Exécution | Phase 5-6 complétées | 2026-01-29 |
