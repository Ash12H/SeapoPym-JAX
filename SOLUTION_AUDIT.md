 #1 — Supprimer StepError

  Problème : StepError était définie dans exceptions.py et exportée dans __init__.py, mais jamais raise nulle part
  dans le projet. C'est du code mort qui encombre l'API publique.

  Solution : Suppression de la classe dans exceptions.py et de l'import + l'entrée __all__ dans __init__.py. J'ai
  aussi nettoyé le docstring du module (supprimé la ligne - E301: StepError).

  Pas de difficulté particulière — c'était un grep pour confirmer l'absence d'usage, puis deux éditions simples.

  ---
  #2 — Supprimer stub netcdf + paramètre format

  Problème : DiskWriter acceptait format="zarr" (défaut) ou format="netcdf", mais _init_netcdf() faisait
  immédiatement raise EngineIOError(...). Ce paramètre ajoutait des branches if/else inutiles dans initialize(),
  _write_chunk(), et le constructeur — du code mort qui prétend supporter un format qui n'existe pas.

  Solution : Suppression de :
  - Le paramètre format du constructeur et self.format
  - La méthode _init_netcdf()
  - Toutes les branches if self.format == "zarr" / else — le code appelle directement les méthodes zarr

  Pas d'erreur. Le seul choix était de garder ou non le format comme extension future. J'ai supprimé : si un jour on
  ajoute netcdf, on ajoutera le code à ce moment. Garder un stub qui raise n'apporte rien.

  ---
  #3 — Supprimer branche "intermediates"

  Problème : Dans step.py, _resolve_inputs avait :
  elif category in ("intermediates", "derived"):
  Le blueprint n'utilise que "derived." comme préfixe. "intermediates" est un alias fantôme jamais utilisé.

  Solution : Remplacé par elif category == "derived": — une seule ligne modifiée.

  Pas de difficulté. J'ai confirmé par grep que "intermediates." n'apparaît dans aucun blueprint ou test du projet.

  ---
  #4 — Simplifier heuristique time_dim dans MemoryWriter

  Problème : MemoryWriter.finalize() cherchait le nom de la dimension temporelle avec :
  time_dim = next((d for d in ["T", "time", "t"] if d in coords), "T")
  Or le système utilise exclusivement "T" (canonical dims). Les fallbacks "time" et "t" ne servaient jamais, et
  masquaient un potentiel bug silencieux si un coord "time" existait par erreur.

  Solution : Remplacé par le littéral "T" directement :
  dims = ("T",) + dims
  Plus de variable intermédiaire time_dim.

  Pas de difficulté. Le test test_io_memory_writer.py utilisait justement "time" comme dim name — ce qui était
  corrigé en parallèle par la tâche #7.

  ---
  #5 — Documenter le clamp ≥ 0 dans Euler

  Problème : _integrate_euler fait :
  new_state[var_name] = jnp.maximum(value + total * dt, 0.0)
  Ce maximum(..., 0.0) est une hypothèse physique hardcodée (la biomasse ne peut pas être négative) mais il n'y avait
   aucun commentaire expliquant pourquoi. Quelqu'un qui lit le code pourrait le considérer comme un bug ou le
  supprimer.

  Solution : Ajout d'un commentaire de 4 lignes expliquant :
  1. Que les variables d'état sont physiquement non-négatives
  2. Qu'Euler explicite peut overshooter sous zéro quand la tendance de perte est grande
  3. Que le clamp est volontaire

  Pas de choix alternatif. On ne touche pas au comportement, on documente.

  ---
  #6 — Factoriser fixtures dans conftest.py

  Problème : clean_registry, simple_blueprint, et simple_config étaient copiés quasi-identiquement dans test_step.py,
   test_runners.py, et test_integration.py. Triple maintenance, risque de divergence.

  Solution : Création de tests/engine/conftest.py avec les trois fixtures. Pytest les découvre automatiquement.
  Suppression des duplicatas des trois fichiers.

  Choix important : test_runners.py avait un simple_config différent (30 timesteps au lieu de 10, growth_rate=0.001
  au lieu de 0.1) parce que les tests de chunking ont besoin de plus de pas de temps. Plutôt que de modifier le
  simple_config partagé (ce qui casserait les tests de step), j'ai créé une fixture locale runner_config dans la
  classe TestStreamingRunner. Le simple_config du conftest reste celui de test_step.py (10 timesteps), et
  test_runners.py utilise son propre runner_config quand nécessaire. Le test test_invalid_chunk_size utilise le
  simple_config partagé (il n'a pas besoin de 30 steps).

  test_integration.py n'utilisait pas simple_blueprint/simple_config de toute façon (chaque test E2E crée son propre
  blueprint). Seul clean_registry est repris du conftest.

  ---
  #7 — Corriger dims dans test_io_memory_writer.py

  Problème : Le mock model utilisait ("time", "lat", "lon") comme dims et coords. Or le système réel utilise ("T",
  "Y", "X"). Le test passait mais ne testait pas le comportement réel — il testait un chemin qui n'existe plus en
  production (surtout après la simplification #4).

  Solution : Remplacé dans le mock :
  - coords: "time" → "T", "lat" → "Y", "lon" → "X"
  - DataNode.dims: ("time", "lat", "lon") → ("Y", "X") (les dims déclarées n'incluent pas T, c'est le writer qui
  l'ajoute)
  - assert: ds["biomass"].dims == ("T", "Y", "X"), ds["biomass"].isel(T=...), etc.

  Subtilité : Les DataNode.dims dans le vrai système déclarent les dims spatiales (sans T). C'est
  MemoryWriter.finalize() qui ajoute T en tête quand data.ndim == len(dims) + 1. J'ai donc corrigé le mock pour
  refléter ça : dims=("Y", "X") et non dims=("T", "Y", "X").

  ---
  #8 — Corriger _init_zarr : dims réelles par variable

  Problème : _init_zarr hardcodait ["Y", "X"] pour le shape de toutes les variables :
  var_shape = (0,) + tuple(shapes.get(d, 1) for d in ["Y", "X"] if d in shapes)
  Une variable avec dims (T, C, Y, X) aurait un shape (0, Y, X) — la dimension C serait perdue. Les données écrites
  auraient un shape incompatible.

  Solution : Ajout d'un paramètre var_dims: dict[str, tuple[str, ...]] à initialize() et _init_zarr(). Chaque
  variable utilise ses propres dims pour calculer son shape :
  if var_dims and var_name in var_dims:
      dims = var_dims[var_name]
      spatial_dims = tuple(d for d in dims if d != "T")
  else:
      spatial_dims = tuple(d for d in ["Y", "X"] if d in shapes)
  Le fallback ["Y", "X"] est gardé pour la rétrocompatibilité si var_dims n'est pas fourni.

  Côté OutputWriter (Protocol) : J'ai ajouté var_dims comme paramètre optionnel à initialize(). MemoryWriter
  l'accepte mais l'ignore (del var_dims) puisqu'elle résout les dims elle-même via _resolve_variable_dims().

  Côté StreamingRunner : J'ai ajouté le code qui résout var_dims à partir de model.data_nodes (même logique que
  MemoryWriter._resolve_variable_dims() mais faite en amont), puis le passe à writer.initialize().

  Choix : J'aurais pu mettre la résolution des dims dans le DiskWriter lui-même, mais ça aurait exigé de lui passer
  le CompiledModel complet (comme MemoryWriter). Le plan prévoyait de passer le mapping au writer, ce qui garde
  DiskWriter découplé du modèle.

  ---
  #9 — Écrire les coords dans le zarr store

  Problème : DiskWriter.initialize() recevait coords mais faisait del coords. Le store zarr produit ne contenait
  aucune métadonnée : pas de coordonnées de latitude, longitude, temps. Le fichier zarr était inutilisable sans
  connaître les coords par un autre moyen.

  Solution : Deux choses :
  1. Écriture des coords : Chaque dimension coord est écrite comme un dataset dans le zarr store à l'initialisation.
  2. Attribut _ARRAY_DIMENSIONS : Ajouté sur chaque variable — c'est la convention zarr/xarray pour que
  xr.open_zarr() sache quelles dimensions correspondent à quels axes.

  Erreur rencontrée : L'API zarr v3 (installée dans ce projet) exige shape comme argument obligatoire de
  create_dataset(), même si on passe data=. Mon premier code faisait :
  self.store.create_dataset(dim_name, data=coord_np, dtype=coord_np.dtype)
  Ce qui a échoué avec TypeError: missing 1 required keyword-only argument: 'shape'. J'ai corrigé en créant le
  dataset avec shape puis en écrivant les données séparément :
  ds = self.store.create_dataset(dim_name, shape=coord_np.shape, dtype=coord_np.dtype)
  ds[:] = coord_np
  C'est la seule erreur de tout l'audit — détectée au premier run de tests et corrigée immédiatement.

  ---
  #10 — Simplifier le threading de DiskWriter

  Problème : DiskWriter utilisait max_workers=2 par défaut avec un ThreadPoolExecutor, mais un _write_lock
  sérialisait toutes les écritures zarr. Avoir 2 threads ne donne aucun parallélisme réel d'écriture — le seul
  bénéfice est l'overlap entre le compute JAX (qui libère le GIL) et l'I/O.

  Solution : Changé le défaut à max_workers=1. Un seul thread worker suffit pour l'overlap compute/IO. Le paramètre
  max_workers est conservé dans l'API (pour les cas avancés), mais la valeur par défaut reflète la réalité : le lock
  sérialise de toute façon.

  Choix : Le plan proposait soit passer à 1 thread, soit documenter la limitation. J'ai fait les deux — le défaut est
   1, et le docstring de la classe et du constructeur expliquent le pattern (overlap compute/IO, pas parallélisme
  d'écriture).

  ---
  #11 — Améliorer couverture step.py

  Problème : step.py était à ~60% de couverture. Branches non testées :
  - _transpose_vmap_output (cas tuple et cas scalar/ndim mismatch)
  - _resolve_inputs branche directe (sans préfixe de catégorie, len(parts) < 2)
  - _handle_compute_outputs multi-output + erreurs
  - _apply_mask avec un vrai array (seul le cas mask=1.0 no-op était testé)
  - Le clamp ≥ 0 dans Euler

  Solution : Ajout de 15 nouveaux tests dans test_step.py :

  Classe: TestTransposeVmapOutput
  Tests ajoutés: 4 tests — single array, tuple, ndim mismatch, scalar
  ────────────────────────────────────────
  Classe: TestHandleComputeOutputs
  Tests ajoutés: 4 tests — single, multi, wrong type, wrong count
  ────────────────────────────────────────
  Classe: TestApplyMask
  Tests ajoutés: 2 tests — no-op (1.0) et array réel
  ────────────────────────────────────────
  Classe: TestResolveInputs
  Tests ajoutés: 6 tests — direct ref dans state/forcings/params/intermediates, not found, unknown category
  ────────────────────────────────────────
  Classe: TestIntegrateEuler
  Tests ajoutés: 1 test — clamp non-négatif

  Résultat : step.py passe de ~60% à 92%. Les 8% restants sont les lignes 58-61 et 96-98 (le code path des core_dims
  avec vmap, qui nécessite un blueprint avec des fonctions ayant des core dims — testé indirectement par
  test_vectorize.py mais pas couvert dans le scope de test_step.py seul) et la ligne 213 (la branche "derived" dans
  _resolve_inputs après suppression de "intermediates" — testée via les tests E2E mais la ligne spécifique n'est pas
  flaggée).